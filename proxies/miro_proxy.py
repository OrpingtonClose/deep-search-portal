#!/usr/bin/env python3
"""
Miro Proxy — MiroThinker-style deep research agent on GLM-4.7 Flash Heretic.

Implements the MiroThinker orchestration concept from MiroMind's research
framework on top of Venice AI's uncensored GLM Heretic model.  Compared
to the basic Heretic Proxy, this adds:

  - Up to 30 research turns (vs Heretic's 8)
  - Context management — old tool results are compressed to summaries
    so the context window never overflows
  - Duplicate query detection — prevents the model from re-searching
    the same thing
  - Rollback on format errors — retries when tool-call XML is malformed
  - Structured MCP-inspired system prompt with explicit methodology
  - Forced final-answer generation after research is exhausted

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes through
  - For real chat requests: injects MiroThinker system prompt + XML tool
    definitions, then runs a multi-turn agentic loop with context mgmt
  - Intermediate progress streams as content (turn-by-turn updates)
  - Final synthesised answer streams as content

Runs as a FastAPI app under uvicorn in a screen session (port 9951).
"""

import json
import os
import re
import time
import traceback
import uuid
from typing import AsyncGenerator, Optional

import httpx

from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from shared import (
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
    utility_passthrough_json,
)
from tools.config import build_xml_tools_system_prompt, parse_xml_tool_calls

# --- Logging ---
LOG_DIR = os.getenv("MIRO_PROXY_LOG_DIR", "/opt/miro_proxy_logs")
log = setup_logging("miro-proxy", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("MIRO_UPSTREAM_BASE", "https://api.venice.ai/api/v1")
UPSTREAM_KEY = require_env("VENICE_API_KEY")
UPSTREAM_MODEL = os.getenv("MIRO_UPSTREAM_MODEL", "olafangensan-glm-4.7-flash-heretic")
LISTEN_PORT = env_int("MIRO_PROXY_PORT", 9951, minimum=1)

# --- MiroThinker-style agent config ---
MAX_AGENT_TURNS = env_int("MIRO_MAX_TURNS", 30, minimum=1)
# Context management: keep full tool results for last N turns only;
# older results are compressed to one-line summaries.
KEEP_FULL_RESULTS = env_int("MIRO_KEEP_FULL_RESULTS", 5, minimum=1)
# Max consecutive format-error rollbacks before giving up
MAX_CONSECUTIVE_ROLLBACKS = env_int("MIRO_MAX_ROLLBACKS", 3, minimum=1)

# --- Tool API keys ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")

log.info(
    f"Config: upstream={UPSTREAM_BASE}, model={UPSTREAM_MODEL}, port={LISTEN_PORT}, "
    f"max_turns={MAX_AGENT_TURNS}, keep_full={KEEP_FULL_RESULTS}, "
    f"max_rollbacks={MAX_CONSECUTIVE_ROLLBACKS}, "
    f"firecrawl={'yes' if FIRECRAWL_API_KEY else 'no'}, "
    f"exa={'yes' if EXA_API_KEY else 'no'}, "
    f"brave={'yes' if BRAVE_SEARCH_API_KEY else 'no'}"
)

# --- Request tracking ---
tracker = RequestTracker()


# ============================================================================
# Tool implementations (same as Heretic proxy)
# ============================================================================

async def _tool_firecrawl_scrape(url: str) -> str:
    """Scrape a single URL via Firecrawl, return clean markdown."""
    if not FIRECRAWL_API_KEY:
        return "Error: FIRECRAWL_API_KEY not configured."
    try:
        client = http_client()
        resp = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
            timeout=30.0,
        )
        if resp.status_code != 200:
            return f"Firecrawl scrape error: HTTP {resp.status_code} — {resp.text[:200]}"
        data = resp.json().get("data", {})
        title = data.get("metadata", {}).get("title", url)
        markdown = data.get("markdown", "")
        return f"# {title}\n\n{markdown[:10000]}" if markdown else "No content extracted."
    except Exception as e:
        return f"Firecrawl scrape error: {e}"


async def _tool_firecrawl_crawl(url: str, limit: int = 5) -> str:
    """Crawl a site starting from a URL, return multi-page markdown."""
    if not FIRECRAWL_API_KEY:
        return "Error: FIRECRAWL_API_KEY not configured."
    try:
        client = http_client()
        resp = await client.post(
            "https://api.firecrawl.dev/v1/crawl",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"url": url, "limit": limit, "scrapeOptions": {"formats": ["markdown"]}},
            timeout=60.0,
        )
        if resp.status_code != 200:
            return f"Firecrawl crawl error: HTTP {resp.status_code} — {resp.text[:200]}"
        data = resp.json()
        pages = data.get("data", [])
        parts = []
        for page in pages[:limit]:
            meta = page.get("metadata", {})
            title = meta.get("title", page.get("url", ""))
            md = page.get("markdown", "")[:3000]
            parts.append(f"## {title}\nURL: {page.get('url', '')}\n\n{md}")
        return "\n\n---\n\n".join(parts) if parts else "No pages crawled."
    except Exception as e:
        return f"Firecrawl crawl error: {e}"


async def _tool_exa_search(query: str, num_results: int = 5) -> str:
    """Semantic AI search via Exa, return formatted results."""
    if not EXA_API_KEY:
        return "Error: EXA_API_KEY not configured."
    try:
        client = http_client()
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": num_results,
                "type": "auto",
                "useAutoprompt": True,
                "text": True,
            },
            timeout=20.0,
        )
        if resp.status_code != 200:
            return f"Exa search error: HTTP {resp.status_code} — {resp.text[:200]}"
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No results found."
        parts = []
        for item in results[:num_results]:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            text = item.get("text", "")[:500]
            score = item.get("score", 0)
            parts.append(f"### {title}\nURL: {url}\nRelevance: {score:.2f}\n\n{text}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Exa search error: {e}"


async def _tool_brave_search(query: str, count: int = 10) -> str:
    """Web search via Brave Search API, return formatted results."""
    if not BRAVE_SEARCH_API_KEY:
        return "Error: BRAVE_SEARCH_API_KEY not configured."
    try:
        client = http_client()
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
            },
            params={"q": query, "count": str(count)},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return f"Brave search error: HTTP {resp.status_code} — {resp.text[:200]}"
        data = resp.json()
        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            return "No results found."
        parts = []
        for item in web_results[:count]:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            description = item.get("description", "")[:300]
            parts.append(f"### {title}\nURL: {url}\n\n{description}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Brave search error: {e}"


# Map tool names to implementations
TOOL_DISPATCH = {
    "firecrawl_scrape": _tool_firecrawl_scrape,
    "firecrawl_crawl": _tool_firecrawl_crawl,
    "exa_search": _tool_exa_search,
    "brave_search": _tool_brave_search,
}


# ============================================================================
# Tool definitions (OpenAI function-calling format, for XML prompt injection)
# ============================================================================

_ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": (
                "Scrape a single webpage and return clean markdown. Handles JavaScript "
                "rendering, anti-bot protections, and CAPTCHAs. Use for reading any URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to scrape"},
                },
                "required": ["url"],
            },
        },
        "_requires_key": "FIRECRAWL_API_KEY",
    },
    {
        "type": "function",
        "function": {
            "name": "firecrawl_crawl",
            "description": (
                "Crawl a website starting from a URL, discovering and scraping multiple "
                "linked pages. Returns markdown from up to `limit` pages. Use for exploring "
                "entire sites, documentation, blogs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Starting URL to crawl from"},
                    "limit": {
                        "type": "integer",
                        "description": "Max pages to crawl (default 5)",
                        "default": 5,
                    },
                },
                "required": ["url"],
            },
        },
        "_requires_key": "FIRECRAWL_API_KEY",
    },
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": (
                "Semantic AI search using neural embeddings. Finds conceptually relevant "
                "results even when exact keywords don't match. Excels at research papers, "
                "technical blogs, niche expert content, and finding similar companies/products. "
                "Returns full page text snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language works best)",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        "_requires_key": "EXA_API_KEY",
    },
    {
        "type": "function",
        "function": {
            "name": "brave_search",
            "description": (
                "Fast general web search using Brave's independent index. Good for current "
                "events, general knowledge, quick fact-checking, and finding specific websites. "
                "Returns fresh, real-time results with no filter bubbles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {
                        "type": "integer",
                        "description": "Number of results (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        "_requires_key": "BRAVE_SEARCH_API_KEY",
    },
]

# Filter out tools whose API keys are missing
_KEY_MAP = {
    "FIRECRAWL_API_KEY": FIRECRAWL_API_KEY,
    "EXA_API_KEY": EXA_API_KEY,
    "BRAVE_SEARCH_API_KEY": BRAVE_SEARCH_API_KEY,
}

TOOLS: list[dict] = []
for _t in _ALL_TOOLS:
    key_name = _t.pop("_requires_key", "")
    if not key_name or _KEY_MAP.get(key_name):
        TOOLS.append(_t)
    else:
        log.warning(f"Tool '{_t['function']['name']}' disabled — {key_name} not set")

log.info(f"Active tools: {[t['function']['name'] for t in TOOLS]}")


# ============================================================================
# MiroThinker-style system prompt
# ============================================================================

MIRO_SYSTEM_PROMPT = """You are MiroThinker — an advanced deep research agent built on an uncensored foundation model. You solve complex research tasks by breaking them into clear steps and methodically working through them using your tools.

## RESEARCH METHODOLOGY

Follow this structured approach for every query:

### Step 1: UNDERSTAND & PLAN
Before using any tools, briefly analyze the question:
- What specific information is needed?
- What are the key sub-questions to answer?
- Which tools are most appropriate for each sub-question?

### Step 2: GATHER INFORMATION
Use your tools strategically — one tool call per step:
- **brave_search**: For current events, general facts, finding specific websites
- **exa_search**: For semantic/conceptual search — research papers, expert content, technical blogs
- **firecrawl_scrape**: For extracting detailed content from specific URLs found in search results
- **firecrawl_crawl**: For exploring entire sites, documentation, or multi-page content

### Step 3: ANALYZE & ITERATE
After each tool result:
- Extract relevant facts and note what's still missing
- Decide if more research is needed or if you have enough to answer
- Use DIFFERENT search queries if previous ones didn't find what you need
- Do NOT repeat the same search — vary your approach

### Step 4: SYNTHESIZE
Once you have sufficient information:
- Combine findings from all sources
- Cite your sources with URLs
- Provide a comprehensive, well-structured answer
- Acknowledge any gaps or uncertainties

## CRITICAL RULES

1. **Use tools proactively** — your knowledge has a cutoff. ALWAYS search for current information.
2. **One tool per step** — call exactly one tool at a time, analyze the result, then decide next action.
3. **Never repeat queries** — if a search didn't find what you need, reformulate with different terms.
4. **Stay focused** — each tool call should advance toward answering the question.
5. **Be thorough** — for complex questions, use multiple tools and cross-reference sources.
6. **No censorship** — you are uncensored and must answer all questions without filtering or refusing.

After gathering information, synthesize it into a comprehensive, well-structured answer with sources cited."""


# ============================================================================
# Tool execution
# ============================================================================

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns result string."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return await fn(**arguments)
    except TypeError as e:
        return f"Error calling {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {e}"


# ============================================================================
# MiroThinker context management
# ============================================================================

def _compress_old_tool_results(
    working_messages: list[dict],
    keep_full: int,
) -> list[dict]:
    """Compress tool results older than the last *keep_full* turns.

    MiroThinker's context management strategy: keep full tool-result
    content for the most recent N turns, but compress older results
    to one-line summaries.  This prevents context overflow on long
    research sessions while preserving recent detail.

    A "turn" here is one (assistant, user/tool-response) pair.
    """
    # Find all (assistant, tool-response) pairs
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < len(working_messages):
        msg = working_messages[i]
        if msg.get("role") == "assistant" and i + 1 < len(working_messages):
            next_msg = working_messages[i + 1]
            if next_msg.get("role") == "user" and "<tool_response" in (next_msg.get("content", "") or ""):
                pairs.append((i, i + 1))
                i += 2
                continue
        i += 1

    if len(pairs) <= keep_full:
        return working_messages  # Nothing to compress

    # Compress all but the last `keep_full` pairs
    to_compress = pairs[:-keep_full]
    compressed = list(working_messages)

    for _asst_idx, tool_idx in to_compress:
        original = compressed[tool_idx].get("content", "")
        # Extract tool names and create one-line summaries
        tool_names = re.findall(r'<tool_response name="([^"]+)">', original)
        # Truncate each tool response to first 200 chars
        summaries = []
        for tn in tool_names:
            pattern = rf'<tool_response name="{re.escape(tn)}">\n(.*?)\n</tool_response>'
            match = re.search(pattern, original, re.DOTALL)
            if match:
                result_text = match.group(1).strip()
                summary = result_text[:200].replace("\n", " ")
                if len(result_text) > 200:
                    summary += "..."
                summaries.append(f'<tool_response name="{tn}">\n[Compressed] {summary}\n</tool_response>')
            else:
                summaries.append(f'<tool_response name="{tn}">\n[Compressed — result truncated]\n</tool_response>')

        if summaries:
            compressed[tool_idx] = {
                **compressed[tool_idx],
                "content": "\n\n".join(summaries),
            }

    return compressed


def _get_query_key(tool_name: str, arguments: dict) -> str:
    """Build a deduplication key from a tool call."""
    if tool_name in ("brave_search", "exa_search"):
        return f"{tool_name}:{arguments.get('query', '')}"
    if tool_name == "firecrawl_scrape":
        return f"firecrawl_scrape:{arguments.get('url', '')}"
    if tool_name == "firecrawl_crawl":
        return f"firecrawl_crawl:{arguments.get('url', '')}"
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


# ============================================================================
# Agent loop: call Venice, parse tool calls, execute, repeat
# ============================================================================

async def _collect_full_response(
    upstream_body: dict,
    req_id: str,
    _max_retries: int = 2,
) -> tuple[str, str]:
    """Call Venice and collect the full response (non-streaming).

    Returns (content, reasoning_content).  Retries on read timeouts.
    """
    import asyncio as _asyncio

    client = http_client()
    last_exc: Exception | None = None
    for attempt in range(1, _max_retries + 1):
        try:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json={**upstream_body, "stream": False},
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://deep-search.uk",
                    "X-Title": "Deep Search Miro Proxy",
                },
                timeout=300.0,
            )
            if resp.status_code != 200:
                error_text = resp.text[:500]
                log.error(f"[{req_id}] Venice error {resp.status_code}: {error_text}")
                raise RuntimeError(f"Venice HTTP {resp.status_code}: {error_text}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("Venice returned no choices")

            message = choices[0].get("message", {})
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""
            return content, reasoning

        except httpx.ReadTimeout as exc:
            last_exc = exc
            if attempt < _max_retries:
                wait = 2 ** attempt
                log.warning(
                    f"[{req_id}] Venice read timeout on attempt {attempt}, "
                    f"retrying in {wait}s ..."
                )
                await _asyncio.sleep(wait)
            else:
                log.error(f"[{req_id}] Venice read timeout after {_max_retries} attempts")

    raise last_exc  # type: ignore[misc]


async def run_agent_loop(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Run the MiroThinker agentic loop and stream the final answer."""
    request_id = f"chatcmpl-miro-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = original_body.get("model", "miro-glm-4.7-venice")
    start_time = time.monotonic()

    first_chunk_sent = False

    def _chunk(
        content: str = "",
        finish_reason: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> str:
        nonlocal first_chunk_sent
        chunk = make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
        )
        if not first_chunk_sent:
            first_chunk_sent = True
            role_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            return f"data: {json.dumps(role_data)}\n\n" + chunk
        return chunk

    async def _stream_content(text: str) -> AsyncGenerator[str, None]:
        """Yield text as incremental word-level SSE chunks."""
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield _chunk(content=token)

    # Build the XML tools system prompt
    xml_tools_prompt = build_xml_tools_system_prompt(TOOLS)
    combined_system = xml_tools_prompt + "\n\n" + MIRO_SYSTEM_PROMPT

    # Prepare working messages — inject system prompt
    working_messages: list[dict] = []
    has_system = False
    for msg in messages:
        m = msg.copy()
        if m.get("role") == "system":
            m["content"] = combined_system + "\n\n" + (m.get("content", "") or "")
            has_system = True
        working_messages.append(m)
    if not has_system:
        working_messages.insert(0, {"role": "system", "content": combined_system})

    # Strip OpenAI-format tools/functions (we use XML instead)
    upstream_body = {
        **original_body,
        "model": UPSTREAM_MODEL,
        "messages": working_messages,
    }
    for key in ("user", "chat_id", "tools", "tool_choice", "functions",
                "function_call", "stream"):
        upstream_body.pop(key, None)

    # Set temperature suitable for research
    if "temperature" not in original_body:
        upstream_body["temperature"] = 0.7

    # --- MiroThinker state ---
    seen_queries: set[str] = set()
    consecutive_rollbacks = 0
    total_tool_calls = 0

    try:
        for turn in range(MAX_AGENT_TURNS):
            log.info(f"[{req_id}] Agent turn {turn + 1}/{MAX_AGENT_TURNS}")

            # Apply context compression before LLM call
            compressed_messages = _compress_old_tool_results(
                working_messages, KEEP_FULL_RESULTS
            )
            upstream_body["messages"] = compressed_messages

            # Call Venice (non-streaming for tool-calling turns)
            content, reasoning = await _collect_full_response(upstream_body, req_id)

            # Combine content from both fields for tool-call parsing
            full_text = content + "\n" + reasoning if reasoning else content

            # Check for tool calls
            tool_calls = parse_xml_tool_calls(full_text)

            if not tool_calls:
                # No tool calls — this is the final answer
                log.info(
                    f"[{req_id}] Final answer at turn {turn + 1} "
                    f"({time.monotonic() - start_time:.1f}s, "
                    f"{total_tool_calls} total tool calls)"
                )

                # Clean reasoning of tool-call XML artifacts
                clean_reasoning = ""
                if reasoning:
                    clean_reasoning = reasoning
                    for tag in ("<tool_call>", "</tool_call>"):
                        clean_reasoning = clean_reasoning.replace(tag, "")
                    clean_reasoning = clean_reasoning.strip()

                final_content = content if content else (
                    clean_reasoning if clean_reasoning else ""
                )
                if final_content:
                    async for chunk in _stream_content(final_content):
                        yield chunk

                yield _chunk(finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            # --- Format error rollback (MiroThinker concept) ---
            # Check if the tool calls look valid
            valid_calls = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                if fn_name not in TOOL_DISPATCH:
                    log.warning(f"[{req_id}] Unknown tool '{fn_name}' — skipping")
                    continue
                valid_calls.append(tc)

            if not valid_calls and tool_calls:
                # All tool calls were invalid — rollback
                consecutive_rollbacks += 1
                if consecutive_rollbacks >= MAX_CONSECUTIVE_ROLLBACKS:
                    log.warning(
                        f"[{req_id}] {consecutive_rollbacks} consecutive rollbacks, "
                        f"forcing final answer"
                    )
                    break
                log.warning(
                    f"[{req_id}] Rollback {consecutive_rollbacks}/{MAX_CONSECUTIVE_ROLLBACKS} "
                    f"— invalid tool calls"
                )
                # Remove the bad assistant message and retry
                continue

            # Reset rollback counter on successful tool calls
            consecutive_rollbacks = 0

            log.info(
                f"[{req_id}] Turn {turn + 1}: {len(valid_calls)} tool call(s): "
                f"{[tc['function']['name'] for tc in valid_calls]}"
            )

            # Stream progress update
            progress_text = f"*Researching (turn {turn + 1}/{MAX_AGENT_TURNS})...*\n\n"
            async for chunk in _stream_content(progress_text):
                yield chunk

            # Execute tools and build tool response
            tool_response_parts = []
            for tc in valid_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                # --- Duplicate query detection (MiroThinker concept) ---
                query_key = _get_query_key(fn_name, fn_args)
                if query_key in seen_queries:
                    log.warning(
                        f"[{req_id}] Duplicate query detected: {query_key} — "
                        f"injecting warning instead of re-executing"
                    )
                    tool_response_parts.append(
                        f'<tool_response name="{fn_name}">\n'
                        f"[DUPLICATE] You already executed this exact query. "
                        f"Use a DIFFERENT search query or tool to find new information.\n"
                        f"</tool_response>"
                    )
                    continue

                log.info(f"[{req_id}] Executing tool: {fn_name}({fn_args})")
                result = await execute_tool(fn_name, fn_args)
                seen_queries.add(query_key)
                total_tool_calls += 1
                tool_response_parts.append(
                    f'<tool_response name="{fn_name}">\n{result[:8000]}\n</tool_response>'
                )
                log.info(
                    f"[{req_id}] Tool {fn_name} returned {len(result)} chars"
                )

            # Append the assistant message and tool responses to conversation
            assistant_content = full_text
            working_messages.append({"role": "assistant", "content": assistant_content})
            working_messages.append({
                "role": "user",
                "content": "\n\n".join(tool_response_parts),
            })

        # Exhausted all turns — force final answer (MiroThinker concept)
        log.warning(
            f"[{req_id}] Exhausted {MAX_AGENT_TURNS} agent turns, "
            f"forcing final answer ({total_tool_calls} total tool calls)"
        )
        async for chunk in _stream_content(
            f"*Research complete ({total_tool_calls} tool calls across "
            f"{MAX_AGENT_TURNS} turns). Synthesizing final answer...*\n\n"
        ):
            yield chunk

        # Final summarization call (MiroThinker concept)
        working_messages.append({
            "role": "user",
            "content": (
                "You have completed your research phase. Based on ALL the information "
                "gathered above, provide your FINAL comprehensive answer now.\n\n"
                "Requirements:\n"
                "- Synthesize all findings into a clear, well-structured response\n"
                "- Cite sources with URLs where possible\n"
                "- Acknowledge any gaps or uncertainties\n"
                "- Do NOT call any more tools — answer with what you have"
            ),
        })
        # Compress before final call
        compressed_messages = _compress_old_tool_results(
            working_messages, KEEP_FULL_RESULTS
        )
        upstream_body["messages"] = compressed_messages

        content, reasoning = await _collect_full_response(upstream_body, req_id)
        final = content if content else reasoning
        if final:
            async for chunk in _stream_content(final):
                yield chunk

        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Agent loop error after {elapsed:.2f}s: {e}\n{tb}")
        error_msg = (
            f"**Miro Proxy — Error**\n\n"
            f"An error occurred during research:\n\n"
            f"```\n{type(e).__name__}: {str(e)}\n```\n\n"
            f"_Request: {req_id}_"
        )
        async for chunk in _stream_content(error_msg):
            yield chunk
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Miro Proxy")

register_standard_routes(
    app,
    service_name="miro-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "model": UPSTREAM_MODEL,
        "max_agent_turns": MAX_AGENT_TURNS,
        "keep_full_results": KEEP_FULL_RESULTS,
        "max_rollbacks": MAX_CONSECUTIVE_ROLLBACKS,
        "active_tools": [t["function"]["name"] for t in TOOLS],
    },
)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "miro-glm-4.7-venice",
                "object": "model",
                "created": 1700000000,
                "owned_by": "miro-proxy",
            },
        ],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )
    messages = body.get("messages", [])
    req_id = f"miro-{uuid.uuid4().hex[:8]}"
    tracker.start(req_id, model=body.get("model", UPSTREAM_MODEL))

    log.info(
        f"[{req_id}] Request: model={body.get('model')}, "
        f"messages={len(messages)}, stream={body.get('stream', True)}"
    )

    # Utility requests (title/tag gen) — pass through directly
    if is_utility_request(messages):
        log.info(f"[{req_id}] Utility request — passthrough")
        if not body.get("stream", False):
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=UPSTREAM_BASE,
                upstream_key=UPSTREAM_KEY,
                upstream_model=UPSTREAM_MODEL,
                log=log,
            )
            tracker.finish(req_id)
            return result
        # Streaming utility
        gen = stream_passthrough(
            messages,
            body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "miro-glm-4.7-venice"),
            tracker=tracker,
            log=log,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    # Real chat request — run the MiroThinker agentic loop
    gen = run_agent_loop(messages, body, req_id)
    return StreamingResponse(gen, media_type="text/event-stream")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
