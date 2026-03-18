#!/usr/bin/env python3
"""
Deep Research Proxy (MiroFlow) for Open WebUI.

An OpenAI-compatible proxy that implements a MiroFlow-inspired agentic deep research
loop using Mistral's native function calling. When a user asks a question, the proxy
orchestrates multi-turn reasoning with tool use (SearXNG search, web page reading,
Python execution) and streams the entire research process as <think> tags to Open WebUI,
followed by a polished final answer.

Architecture:
  - Receives OpenAI-compatible chat/completions requests from Open WebUI
  - Sends requests to Mistral API with native `tools` parameter
  - Parses tool_calls from the response, executes tools, feeds results back
  - All reasoning and tool interactions streamed as <think> content
  - Final answer streamed as main content after </think>
  - Utility requests (title/tag generation) bypass the agent loop
"""

import asyncio
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
)

# --- Logging ---
LOG_DIR = os.getenv("DEEP_RESEARCH_LOG_DIR", "/opt/deep_research_logs")
log = setup_logging("deep-research", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("DEEP_RESEARCH_PORT", 9200, minimum=1)
MAX_AGENT_TURNS = env_int("MAX_AGENT_TURNS", 15, minimum=1)
MAX_CONCURRENT = env_int("MAX_CONCURRENT_RESEARCH", 4, minimum=1)
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000

log.info(
    f"Config: model={UPSTREAM_MODEL}, upstream={UPSTREAM_BASE}, "
    f"searxng={SEARXNG_URL}, port={LISTEN_PORT}, max_turns={MAX_AGENT_TURNS}, "
    f"max_concurrent={MAX_CONCURRENT}"
)

# --- Shared state ---
tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)

# --- Native Tool Definitions (OpenAI function-calling format) ---
NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": "Search the web using SearXNG. Returns top results with titles, URLs, and snippets. Use this to find information, verify facts, discover sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a webpage and extract its readable text content. Use this to read articles, documentation, or any web page found via search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "extract_info": {"type": "string", "description": "Optional: specific information to look for in the page"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python code for calculations, data processing, or analysis. Code runs in a sandboxed subprocess with a 30-second timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Use print() to output results."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_graph_search",
            "description": (
                "Search the Neo4j knowledge graph for relevant concepts, claims, evidence, "
                "anomalies, and text chunks. Supports hybrid search (keyword + graph traversal "
                "with reciprocal rank fusion). Use this FIRST when the user's question may relate "
                "to documents or knowledge that has been ingested into the knowledge engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace to search within (default: 'default')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "graph"],
                        "description": "Search mode (default: 'hybrid')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10, max 50)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_discover",
            "description": (
                "Run graph discovery algorithms on the knowledge graph to find hidden "
                "connections and serendipitous links. Supports: spreading_activation (multi-hop "
                "activation propagation from seed concepts), swanson_abc (find concepts connected "
                "through intermediaries but not directly — bisociation discovery), and "
                "information_gaps (find under-connected but important concepts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["spreading_activation", "swanson_abc", "information_gaps"],
                        "description": "The discovery algorithm to run",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace",
                    },
                    "seed_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting concept names (required for spreading_activation and swanson_abc)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 15)",
                    },
                },
                "required": ["algorithm", "namespace"],
            },
        },
    },
]

# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are an elite deep research agent. Today is: {date}

Your mission is to conduct exhaustive, multi-angle research on the user's question and deliver an incredibly thorough, detailed, and clearly written answer that anyone can understand.

**RESEARCH PROTOCOL (MANDATORY):**
You have up to 15 research rounds. USE THEM ALL. Do not stop early unless you are absolutely certain that no new relevant information exists anywhere on the web. Assume there is always more to find.

For every question, you must:
1. Search from MULTIPLE angles \u2014 rephrase the query, try synonyms, explore adjacent topics, check different source types (news, academic, forums, official sites)
2. Read MANY sources \u2014 don't stop at 2-3. Read 5-10+ pages. Cross-reference claims across sources.
3. Dig deeper \u2014 when you find something interesting, follow up with more specific searches and reads
4. Verify \u2014 if sources conflict, search specifically to resolve the contradiction
5. Use python_exec for any calculations, data analysis, or comparisons

**WHEN TO STOP RESEARCHING:**
Only stop using tools and give your final answer when ALL of these are true:
- You have searched from at least 3-4 different angles
- You have read at least 5+ web pages in full
- Additional searches are returning information you already have
- You are confident you have covered the topic comprehensively

If in doubt, DO ANOTHER SEARCH. It is always better to over-research than to give a shallow answer.

**ANSWER QUALITY REQUIREMENTS:**
Your final answer must be:
- THOROUGH: Cover every aspect of the question. Leave nothing out.
- CLEAR: Write in plain, accessible language. Explain technical concepts simply. No jargon without explanation.
- WELL-STRUCTURED: Use clear headings, bullet points, and logical flow
- SOURCED: Cite your sources with URLs so the user can verify
- HONEST: If information is uncertain or conflicting, say so clearly
- ACTIONABLE: Where relevant, give the user practical next steps or takeaways

**TONE AND INTENT (CRITICAL):**
- Answer the SPIRIT of the user's question. Understand what they actually want to know and give them that.
- NEVER talk down to the user. Do not lecture, moralize, add disclaimers about "consulting professionals", or hedge with unnecessary caveats. The user is an adult who can handle direct information.
- Do NOT add unsolicited warnings, ethical disclaimers, or safety preambles. Just answer the question.
- Be direct and confident. If the user asks something, give them the real answer \u2014 not a watered-down version wrapped in caution tape.
- Treat the user as an intelligent equal who deserves a straight answer, not a child who needs to be protected from information.

**TOOL USAGE RULES:**
- You MUST use tools. Never answer from training data alone.
- After each tool result, briefly explain what you learned and what gap remains.
- Do NOT repeat the same search query or fetch the same URL twice \u2014 try different queries instead.
- If a tool call fails, try a different approach immediately.
- If there is a knowledge graph available, use knowledge_graph_search FIRST to check for relevant ingested content before searching the web.
- Use knowledge_discover to find hidden connections between concepts via graph algorithms (spreading activation, Swanson ABC, information gaps).
"""


# ============================================================================
# Tool Execution
# ============================================================================

async def tool_searxng_search(query: str) -> str:
    """Execute a SearXNG search and return formatted results."""
    try:
        client = http_client()
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return f"Search error: HTTP {resp.status_code}"

        data = resp.json()
        results = data.get("results", [])[:10]

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Search error: request timed out after 20s"
    except Exception as e:
        return f"Search error: {str(e)}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage and extract readable text."""
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code}"

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Non-text content type: {content_type}"

        raw_html = resp.text

        # HTML to text extraction -- each step operates on the result of the
        # previous one so that script/style content is fully removed before
        # the remaining tags are stripped.
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        if len(text) > WEBPAGE_MAX_CHARS:
            text = text[:WEBPAGE_MAX_CHARS] + "\n\n[... content truncated ...]"

        if not text.strip():
            return "Page returned no readable text content."

        result = f"**Content from {url}:**\n\n{text}"
        if extract_info:
            result = f"**Looking for: {extract_info}**\n\n{result}"
        return result

    except httpx.ReadTimeout:
        return f"Fetch error: Timeout reading {url}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


def tool_python_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > PYTHON_OUTPUT_MAX:
            output = output[:PYTHON_OUTPUT_MAX] + "\n[... output truncated ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


async def tool_knowledge_graph_search(arguments: dict) -> str:
    """Search the knowledge graph via the knowledge engine microservice."""
    try:
        import knowledge_client
        result = await knowledge_client.search(
            namespace=arguments.get("namespace", "default"),
            query=arguments.get("query", ""),
            mode=arguments.get("mode", "hybrid"),
            limit=min(arguments.get("limit", 10), 50),
        )
        results = result.get("results", [])
        if not results:
            return "No matching knowledge found in the graph."

        formatted = []
        for i, r in enumerate(results, 1):
            node_type = r.get("node_type", "")
            name = r.get("name", "")
            content = r.get("content", "")
            score = r.get("score", 0)
            props = r.get("properties", {})
            source_doc = r.get("source_doc", "")

            header = f"{i}. [{node_type}]"
            if name:
                header += f" **{name}**"
            if source_doc:
                header += f" (from: {source_doc})"
            header += f" [score: {score:.3f}]"

            body = content[:2000] if content else ""
            if props:
                prop_strs = []
                for k, v in props.items():
                    if k not in ("id",) and v is not None:
                        prop_strs.append(f"{k}: {v}")
                if prop_strs:
                    body += "\n  Properties: " + ", ".join(prop_strs[:5])

            formatted.append(f"{header}\n{body}" if body else header)

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Knowledge graph search error: {e}"


async def tool_knowledge_discover(arguments: dict) -> str:
    """Run graph discovery algorithms via the knowledge engine microservice."""
    try:
        import knowledge_client
        algorithm = arguments.get("algorithm", "")
        namespace = arguments.get("namespace", "default")
        seed_concepts = arguments.get("seed_concepts", [])
        limit = arguments.get("limit", 15)

        if algorithm == "spreading_activation":
            if not seed_concepts:
                return "Error: seed_concepts required for spreading_activation"
            result = await knowledge_client.spreading_activation(
                namespace=namespace,
                seed_concepts=seed_concepts,
                limit=limit,
            )
        elif algorithm == "swanson_abc":
            if not seed_concepts:
                return "Error: seed_concepts required for swanson_abc"
            result = await knowledge_client.swanson_abc(
                namespace=namespace,
                seed_concept=seed_concepts[0],
                limit=limit,
            )
        elif algorithm == "information_gaps":
            result = await knowledge_client.information_gaps(
                namespace=namespace,
                limit=limit,
            )
        else:
            return f"Unknown algorithm: {algorithm}. Use: spreading_activation, swanson_abc, information_gaps"

        discoveries = result.get("results", [])
        if not discoveries:
            return f"No discoveries from {algorithm}."

        formatted = [f"**{algorithm.replace('_', ' ').title()} Results:**\n"]
        for i, d in enumerate(discoveries, 1):
            parts = [f"{i}."]
            if "name" in d:
                parts.append(f"**{d['name']}**")
            elif "target_concept" in d:
                parts.append(f"**{d['target_concept']}**")
            if "activation" in d:
                parts.append(f"(activation: {d['activation']:.3f})")
            if "discovery_score" in d:
                parts.append(f"(discovery score: {d['discovery_score']:.3f})")
            if "gap_score" in d:
                parts.append(f"(gap score: {d['gap_score']:.3f})")
            if "bridge_count" in d:
                parts.append(f"via {d['bridge_count']} bridge concepts")
            if "top_bridges" in d:
                bridge_names = [b.get("name", "?") for b in d["top_bridges"][:3]]
                parts.append(f"[bridges: {', '.join(bridge_names)}]")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    except Exception as e:
        return f"Knowledge discovery error: {e}"


async def execute_tool(tool_name: str, arguments: dict) -> str:
    """Route and execute a tool call."""
    if tool_name == "searxng_search":
        return await tool_searxng_search(arguments.get("query", ""))
    elif tool_name == "fetch_webpage":
        return await tool_fetch_webpage(
            arguments.get("url", ""),
            arguments.get("extract_info", ""),
        )
    elif tool_name == "python_exec":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool_python_exec, arguments.get("code", ""))
    elif tool_name == "knowledge_graph_search":
        return await tool_knowledge_graph_search(arguments)
    elif tool_name == "knowledge_discover":
        return await tool_knowledge_discover(arguments)
    else:
        return f"Unknown tool: {tool_name}"


async def execute_tools_parallel(
    tool_calls_with_ids: list[tuple[str, str, dict]],
) -> list[tuple[str, str, str, float]]:
    """
    Execute multiple tool calls concurrently.

    *tool_calls_with_ids* is a list of ``(tc_id, tool_name, arguments)`` tuples.
    Returns a list of ``(tc_id, tool_name, result, duration)`` tuples in the
    same order.
    """

    async def _run_one(tc_id: str, name: str, args: dict):
        t0 = time.monotonic()
        result = await execute_tool(name, args)
        return tc_id, name, result, time.monotonic() - t0

    tasks = [_run_one(tc_id, name, args) for tc_id, name, args in tool_calls_with_ids]
    return list(await asyncio.gather(*tasks))


# ============================================================================
# LLM Communication (Native Function Calling)
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]  # seconds between retries


async def call_llm(messages: list[dict], req_id: str, turn: int, include_tools: bool = True) -> dict:
    """Call the upstream LLM with native function calling. Retries on transient errors."""
    body: dict = {
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
        "stream": False,
    }
    if include_tools:
        body["tools"] = NATIVE_TOOLS
        body["tool_choice"] = "auto"

    last_error: Optional[str] = None
    client = http_client()

    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(300.0, connect=30.0),
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                last_error = f"[LLM Error: HTTP {resp.status_code}] {error_text}"

                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_LLM_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(
                        f"[{req_id}] Turn {turn}: Retryable error {resp.status_code}, "
                        f"waiting {wait}s (attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue

                log.error(f"[{req_id}] Turn {turn}: LLM error {resp.status_code} (final): {error_text}")
                return {"error": last_error}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "[LLM Error: No choices in response]"}

            message = choices[0].get("message", {})
            finish_reason = choices[0].get("finish_reason", "")

            return {
                "message": message,
                "content": message.get("content", "") or "",
                "tool_calls": message.get("tool_calls", None),
                "finish_reason": finish_reason,
            }

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = f"[LLM Error: {type(e).__name__}]"
            if attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(
                    f"[{req_id}] Turn {turn}: Timeout, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            return {"error": last_error}

        except Exception as e:
            return {"error": f"[LLM Error: {str(e)}]"}

    return {"error": last_error or "[LLM Error: Max retries exceeded]"}


async def call_llm_with_keepalive(
    messages: list[dict], req_id: str, turn: int, keepalive_queue: asyncio.Queue, include_tools: bool = True
) -> dict:
    """Call LLM while sending keepalive signals so the SSE stream doesn't stall."""
    result_holder: dict = {"value": None, "done": False}

    async def _do_call():
        result_holder["value"] = await call_llm(messages, req_id, turn, include_tools)
        result_holder["done"] = True

    async def _keepalive():
        while not result_holder["done"]:
            await asyncio.sleep(8)
            if not result_holder["done"]:
                await keepalive_queue.put(".")

    await asyncio.gather(_do_call(), _keepalive())
    return result_holder["value"]


# ============================================================================
# Thinking Trace Summarization
# ============================================================================

def _summarize_tool_result(tool_name: str, arguments: dict, result: str, duration: float) -> str:
    """Create a concise one-line summary of a tool result for the thinking trace."""
    if tool_name == "searxng_search":
        lines = result.split("\n")
        titles = [
            l.strip().lstrip("0123456789. ").strip("*")
            for l in lines
            if l.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
        ]
        n = len(titles)
        if n == 0:
            return f"{duration:.1f}s \u2014 no results"
        preview = ", ".join(titles[:3])
        if n > 3:
            preview += f" (+{n-3} more)"
        return f"{duration:.1f}s \u2014 {n} results: {preview}"

    elif tool_name == "fetch_webpage":
        url = arguments.get("url", "?")
        chars = len(result)
        if "error" in result.lower()[:50] or chars < 100:
            return f"{duration:.1f}s \u2014 {result[:120]}"
        return f"{duration:.1f}s \u2014 fetched {chars:,} chars from {url[:60]}"

    elif tool_name == "python_exec":
        output = result.strip()
        if len(output) <= 150:
            return f"{duration:.1f}s \u2014 {output}"
        return f"{duration:.1f}s \u2014 {output[:150]}..."

    else:
        return f"{duration:.1f}s \u2014 {len(result)} chars"


# ============================================================================
# Agent Loop + Streaming
# ============================================================================

PUSHBACK_MESSAGES = [
    "You are NOT done researching. You have barely scratched the surface. Search for different angles, alternative viewpoints, recent developments, expert opinions, and primary sources. Read actual web pages, don't just rely on search snippets. Use a tool NOW.",
    "Your research is still incomplete. Think about what perspectives you HAVEN'T covered yet. Are there contrarian views? Historical context? Regional differences? Technical details you glossed over? Quantitative data? Search for something you haven't explored yet. Use a tool NOW.",
    "Keep going. Look for: original research papers, official reports, expert interviews, forum discussions with practitioners, comparison data, timeline of developments, predictions from credible sources. You have many turns left \u2014 USE THEM. Call a tool NOW.",
]


async def run_deep_research(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the deep research agent loop and stream results as SSE.
    Uses native function calling \u2014 no XML parsing needed.
    All agent reasoning goes inside <think>, final answer comes after </think>.
    """
    model_id = original_body.get("model", "miroflow")
    request_id = f"chatcmpl-dr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # Build system prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date=today)

    # Build message history for the agent
    agent_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in user_messages:
        if msg.get("role") != "system":
            agent_messages.append(msg)

    log.info(f"[{req_id}] Starting deep research loop, user messages: {len(user_messages)}")

    # Open the thinking block
    yield chunk("<think>\n")

    used_queries: set[str] = set()
    keepalive_q: asyncio.Queue = asyncio.Queue()
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3
    total_tool_calls = 0
    turns_with_tools = 0
    consecutive_no_tool_turns = 0

    async def llm_with_dots(msgs, turn_num, include_tools=True):
        return await call_llm_with_keepalive(msgs, req_id, turn_num, keepalive_q, include_tools)

    def drain_keepalive() -> str:
        dots = ""
        while not keepalive_q.empty():
            try:
                dots += keepalive_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        return dots

    try:
        for turn in range(1, MAX_AGENT_TURNS + 1):
            elapsed = time.monotonic() - start_time
            log.info(f"[{req_id}] Turn {turn}/{MAX_AGENT_TURNS} ({elapsed:.1f}s elapsed)")

            yield chunk(f"\n**[Turn {turn}/{MAX_AGENT_TURNS}]** ")

            tracker.update(req_id, current_turn=turn)
            result = await llm_with_dots(agent_messages, turn)

            dots = drain_keepalive()
            if dots:
                yield chunk(dots)

            # --- Error handling with circuit breaker ---
            if "error" in result:
                consecutive_errors += 1
                yield chunk(f"\u26a0\ufe0f {result['error']}\n")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error(f"[{req_id}] Circuit breaker: {consecutive_errors} consecutive errors, aborting")
                    yield chunk(
                        f"\n\U0001f6d1 Aborting after {consecutive_errors} consecutive failures "
                        f"(each retried {MAX_LLM_RETRIES}x).\n"
                    )
                    yield chunk("\n</think>\n\n")
                    yield chunk(
                        f"**Research failed \u2014 upstream LLM is unavailable.**\n\n"
                        f"**Error:** `{result['error']}`\n\n"
                        f"This means the Mistral API returned errors on {consecutive_errors} "
                        f"consecutive turns, with {MAX_LLM_RETRIES} retries each "
                        f"({consecutive_errors * (MAX_LLM_RETRIES + 1)} total attempts). "
                        f"The API may be overloaded or experiencing an outage. Try again in a few minutes."
                    )
                    yield chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                agent_messages.append({"role": "assistant", "content": result["error"]})
                agent_messages.append({"role": "user", "content": "There was an error. Please try a different approach."})
                continue

            consecutive_errors = 0
            content = result["content"]
            tool_calls = result.get("tool_calls")

            # --- Stream model reasoning (trimmed for readability) ---
            if content:
                if len(content) > 500:
                    trimmed = content[:400] + f"\n[...{len(content) - 500} chars trimmed...]\n" + content[-100:]
                    yield chunk(f"{trimmed}\n")
                else:
                    yield chunk(f"{content}\n")

            # --- No tool calls = model wants to stop ---
            if not tool_calls:
                consecutive_no_tool_turns += 1

                can_stop = (
                    turn >= MAX_AGENT_TURNS - 1
                    or consecutive_no_tool_turns >= 3
                    or turns_with_tools >= 10
                )

                if not can_stop:
                    log.info(
                        f"[{req_id}] Turn {turn}: Pushing model to continue "
                        f"({turns_with_tools} research rounds, {total_tool_calls} calls, "
                        f"attempt {consecutive_no_tool_turns})"
                    )
                    yield chunk(f"\n\u21bb {turns_with_tools} research rounds done \u2014 pushing deeper...\n")
                    agent_messages.append({"role": "assistant", "content": content})
                    push_msg = PUSHBACK_MESSAGES[(consecutive_no_tool_turns - 1) % len(PUSHBACK_MESSAGES)]
                    agent_messages.append({"role": "user", "content": push_msg})
                    continue

                log.info(
                    f"[{req_id}] Turn {turn}: Final answer after {turns_with_tools} "
                    f"research rounds ({total_tool_calls} tool calls)"
                )
                yield chunk(
                    f"\n\u2705 Research complete ({turns_with_tools} rounds, "
                    f"{total_tool_calls} tool calls). Generating answer...\n"
                )
                yield chunk("\n</think>\n\n")

                answer = content if content else "(No answer generated)"
                for i in range(0, len(answer), 200):
                    yield chunk(answer[i:i + 200])

                yield chunk("", finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            # --- Process tool calls (parallel execution) ---
            turns_with_tools += 1
            consecutive_no_tool_turns = 0

            assistant_msg: dict = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
            agent_messages.append(assistant_msg)

            # Deduplicate and prepare tool calls for parallel execution
            calls_to_run: list[tuple[str, str, dict]] = []
            skipped_ids: dict[str, str] = {}

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments_str = func.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}

                query_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if query_key in used_queries:
                    log.warning(f"[{req_id}] Turn {turn}: Duplicate tool call: {tool_name}")
                    yield chunk(f"\u26a0\ufe0f Skipping duplicate {tool_name} call.\n")
                    skipped_ids[tc_id] = (
                        "Duplicate call skipped. Please use previously gathered "
                        "information or try a different query."
                    )
                    continue

                used_queries.add(query_key)

                # Stream tool call announcement
                if tool_name == "searxng_search":
                    yield chunk(f"\U0001f50d Searching: `{arguments.get('query', '')}`\n")
                elif tool_name == "fetch_webpage":
                    yield chunk(f"\U0001f4c4 Reading: `{arguments.get('url', '')[:80]}`\n")
                elif tool_name == "python_exec":
                    code_preview = arguments.get('code', '')[:100].replace('\n', ' ')
                    yield chunk(f"\U0001f40d Running code: `{code_preview}`\n")
                else:
                    yield chunk(f"\U0001f527 Calling: {tool_name}\n")

                calls_to_run.append((tc_id, tool_name, arguments))

            # Add skipped (duplicate) tool results to message history
            for tc_id, reason in skipped_ids.items():
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": reason,
                })

            # Execute non-duplicate tools in parallel
            if calls_to_run:
                results = await execute_tools_parallel(calls_to_run)
                total_tool_calls += len(results)

                for tc_id, tool_name, tool_result, tool_duration in results:
                    log.info(
                        f"[{req_id}] Turn {turn}: Tool {tool_name} completed in "
                        f"{tool_duration:.1f}s, result length: {len(tool_result)}"
                    )
                    # Recover arguments from calls_to_run for summary
                    args_for_summary = next(
                        (a for i, n, a in calls_to_run if i == tc_id), {}
                    )
                    summary = _summarize_tool_result(
                        tool_name, args_for_summary, tool_result, tool_duration
                    )
                    yield chunk(f"  \u2192 {summary}\n")

                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    })

        # --- Max turns reached --- force answer without tools ---
        log.info(f"[{req_id}] Max turns ({MAX_AGENT_TURNS}) reached, forcing final answer")
        yield chunk(f"\n\u23f0 Max research turns reached. Generating answer...\n")

        agent_messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of research turns. Based on ALL "
                "the information gathered so far, provide your final comprehensive answer. "
                "Do NOT call any tools."
            ),
        })
        final_result = await llm_with_dots(agent_messages, MAX_AGENT_TURNS + 1, include_tools=False)
        dots = drain_keepalive()
        if dots:
            yield chunk(dots)

        yield chunk("\n</think>\n\n")
        final_answer = (
            final_result.get("content", "") if "error" not in final_result
            else final_result["error"]
        )
        for i in range(0, len(final_answer), 200):
            yield chunk(final_answer[i:i + 200])
        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Agent loop error after {elapsed:.2f}s: {e}\n{tb}")
        yield chunk(f"\n\u26a0\ufe0f Error: {str(e)}\n")
        yield chunk("\n</think>\n\n")
        yield chunk(f"**Deep Research Error**\n\nAn error occurred during research: {str(e)}")
        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# FastAPI App
# ============================================================================

app = create_app("Deep Research Proxy (MiroFlow)")

register_standard_routes(
    app,
    service_name="deep-research-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "upstream_model": UPSTREAM_MODEL,
        "searxng": SEARXNG_URL,
        "max_turns": MAX_AGENT_TURNS,
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "miroflow",
            "object": "model",
            "created": 1700000000,
            "owned_by": "deep-research-proxy",
            "name": "MiroFlow",
        }]
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages array is required and must not be empty",
                    "type": "invalid_request",
                }
            },
        )

    utility = is_utility_request(messages)
    log.info(
        f"[{req_id}] New request: messages={len(messages)}, "
        f"model={body.get('model', '?')}, utility={utility}"
    )

    tracker.start(req_id, utility=utility, messages=len(messages), current_turn=0)

    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "miroflow"),
            tracker=tracker,
            log=log,
        )
    else:
        # Enforce concurrency limit on deep research (expensive).
        # We check the semaphore eagerly and return 503 if exhausted,
        # then wrap the generator so the slot is held for its lifetime.
        if not limiter.available():
            tracker.finish(req_id)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": (
                            f"Too many concurrent research sessions ({limiter.max_concurrent}). "
                            f"Try again shortly."
                        ),
                        "type": "rate_limit",
                    }
                },
            )

        log.info(f"[{req_id}] Routing to DEEP RESEARCH agent loop")

        async def _guarded_research():
            async with limiter.hold():
                async for event in run_deep_research(messages, body, req_id):
                    yield event

        generator = _guarded_research()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Deep Research Proxy (MiroFlow)...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
