#!/usr/bin/env python3
"""
Heretic Proxy — GLM-4.7 Flash Heretic with agentic tool-calling loop.

A standalone FastAPI proxy that routes to Venice AI's GLM-4.7 Flash Heretic
model and exposes Firecrawl, Exa, and Brave Search tools via an XML
tool-calling loop.  The model eagerly uses these tools to gather real-time
information before synthesizing a final answer.

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes them through
  - For real chat requests: injects XML tool definitions + system prompt,
    then loops up to MAX_AGENT_TURNS calling tools and feeding results back
  - Intermediate tool-calling reasoning streams as reasoning_content
    (collapsible Thinking block in LibreChat)
  - Final synthesised answer streams as content

Runs as a FastAPI app under uvicorn in a screen session (port 9950).
"""

import json
import os
import re
import time
import traceback
import uuid
from typing import AsyncGenerator, Optional

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
LOG_DIR = os.getenv("HERETIC_PROXY_LOG_DIR", "/opt/heretic_proxy_logs")
log = setup_logging("heretic-proxy", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("HERETIC_UPSTREAM_BASE", "https://api.venice.ai/api/v1")
UPSTREAM_KEY = require_env("VENICE_API_KEY")
UPSTREAM_MODEL = os.getenv("HERETIC_UPSTREAM_MODEL", "olafangensan-glm-4.7-flash-heretic")
LISTEN_PORT = env_int("HERETIC_PROXY_PORT", 9950, minimum=1)
MAX_AGENT_TURNS = env_int("HERETIC_MAX_TURNS", 8, minimum=1)

# --- Tool API keys ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")

log.info(
    f"Config: upstream={UPSTREAM_BASE}, model={UPSTREAM_MODEL}, port={LISTEN_PORT}, "
    f"max_turns={MAX_AGENT_TURNS}, "
    f"firecrawl={'yes' if FIRECRAWL_API_KEY else 'no'}, "
    f"exa={'yes' if EXA_API_KEY else 'no'}, "
    f"brave={'yes' if BRAVE_SEARCH_API_KEY else 'no'}"
)

# --- Request tracking ---
tracker = RequestTracker()


# ============================================================================
# Tool implementations
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
# System prompt
# ============================================================================

HERETIC_SYSTEM_PROMPT = """You are Heretic — an uncensored, powerful research assistant with access to specialized search and scraping tools. You MUST use these tools proactively and thoughtfully for EVERY query that could benefit from real-time information, web content, or research.

## YOUR TOOLS

### 1. FIRECRAWL (Deep Web Scraping & Crawling)
Use `firecrawl_scrape` to extract clean, structured content from any webpage. Unlike basic fetch, Firecrawl:
- Renders JavaScript-heavy pages (SPAs, dynamic content)
- Extracts clean markdown from cluttered pages (strips ads, nav, footers)
- Handles anti-bot protections and CAPTCHAs
- Returns structured data (tables, lists, metadata)
Use `firecrawl_crawl` to crawl an entire site starting from a URL — discovers and scrapes multiple linked pages. Perfect for:
- Extracting all articles from a blog or news site
- Mapping product pages on e-commerce sites
- Building a knowledge base from documentation sites
- Gathering all posts from a forum thread

### 2. EXA (Semantic AI Search)
Use `exa_search` for intelligent, meaning-based web search. Unlike keyword search, Exa:
- Understands the MEANING of your query, not just keywords
- Finds conceptually related content even when exact terms don't match
- Excels at finding: research papers, technical blog posts, niche expert content, similar companies/products
- Returns full page text snippets (not just titles/URLs)
- Supports neural search (semantic) and keyword search modes
Best for: "find me articles about X", "what are companies doing Y", "research on Z topic"

### 3. BRAVE SEARCH (Fast General Web Search)
Use `brave_search` for fast, privacy-focused general web search. Brave:
- Has its own independent web index (not a Google proxy)
- Returns fresh, real-time results
- Includes news, discussions, and general web pages
- No tracking or filter bubbles
- Fast and reliable for broad queries
Best for: current events, general knowledge, quick fact-checking, finding specific websites

## TOOL USE STRATEGY

For EVERY user query, think about which tools to use:
- **Simple factual questions**: Brave Search first, then Exa if you need deeper context
- **Research topics**: Exa (semantic search) + Brave (breadth) in parallel, then Firecrawl to read the best sources
- **Specific URLs or "read this page"**: Firecrawl scrape immediately
- **"Find everything about X"**: All three tools — Exa for deep semantic results, Brave for breadth, Firecrawl to extract full content from top results
- **Current events / news**: Brave Search first (freshest index), supplement with Exa
- **Technical / academic topics**: Exa first (best at finding expert content), then Firecrawl to read papers/docs

ALWAYS use at least one tool for any query that involves facts, current information, or research. Do NOT rely on your training data alone — your knowledge has a cutoff date. Use the tools to get CURRENT, REAL information.

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
# Agent loop: call Venice, parse tool calls, execute, repeat
# ============================================================================

async def _collect_full_response(
    upstream_body: dict,
    req_id: str,
) -> tuple[str, str]:
    """Call Venice and collect the full response (non-streaming).

    Returns (content, reasoning_content) — GLM Heretic may put its output
    in either or both fields.
    """
    client = http_client()
    resp = await client.post(
        f"{UPSTREAM_BASE}/chat/completions",
        json={**upstream_body, "stream": False},
        headers={
            "Authorization": f"Bearer {UPSTREAM_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://deep-search.uk",
            "X-Title": "Deep Search Heretic Proxy",
        },
        timeout=120.0,
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


async def run_agent_loop(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Run the agentic tool-calling loop and stream the final answer."""
    request_id = f"chatcmpl-heretic-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = original_body.get("model", "glm-4.7-flash-heretic")
    start_time = time.monotonic()

    def _chunk(
        content: str = "",
        finish_reason: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
        )

    # Build the XML tools system prompt
    xml_tools_prompt = build_xml_tools_system_prompt(TOOLS)
    combined_system = xml_tools_prompt + "\n\n" + HERETIC_SYSTEM_PROMPT

    # Prepare working messages — inject system prompt
    working_messages = []
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

    # Set temperature suitable for tool use
    if "temperature" not in original_body:
        upstream_body["temperature"] = 0.8

    try:
        for turn in range(MAX_AGENT_TURNS):
            log.info(f"[{req_id}] Agent turn {turn + 1}/{MAX_AGENT_TURNS}")

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
                    f"({time.monotonic() - start_time:.1f}s)"
                )

                # Stream any reasoning as collapsible thinking block
                if reasoning:
                    # Clean reasoning of tool-call XML artifacts
                    clean_reasoning = reasoning
                    for tag in ("<tool_call>", "</tool_call>"):
                        clean_reasoning = clean_reasoning.replace(tag, "")
                    if clean_reasoning.strip():
                        yield _chunk(reasoning=clean_reasoning)

                # Stream the final content
                if content:
                    yield _chunk(content=content)
                elif reasoning and not content:
                    # Model put everything in reasoning_content — use cleaned version
                    yield _chunk(content=clean_reasoning)

                yield _chunk(finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            # Tool calls found — execute them
            log.info(
                f"[{req_id}] Turn {turn + 1}: {len(tool_calls)} tool call(s): "
                f"{[tc['function']['name'] for tc in tool_calls]}"
            )

            # Stream the reasoning as a thinking block so user sees progress
            if reasoning:
                yield _chunk(reasoning=f"[Turn {turn + 1}] {reasoning[:2000]}")
            elif content:
                # Some models put tool-call reasoning in content
                clean = content
                for tag in ("<tool_call>", "</tool_call>"):
                    clean = clean.replace(tag, "")
                # Remove JSON tool call blocks from visible reasoning
                clean = re.sub(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}', '', clean)
                if clean.strip():
                    yield _chunk(reasoning=f"[Turn {turn + 1}] {clean.strip()[:2000]}")

            # Execute tools and build tool response
            tool_response_parts = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                log.info(f"[{req_id}] Executing tool: {fn_name}({fn_args})")
                result = await execute_tool(fn_name, fn_args)
                tool_response_parts.append(
                    f'<tool_response name="{fn_name}">\n{result[:5000]}\n</tool_response>'
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
            upstream_body["messages"] = working_messages

        # Exhausted all turns — stream whatever we have
        log.warning(
            f"[{req_id}] Exhausted {MAX_AGENT_TURNS} agent turns, "
            f"forcing final answer"
        )
        yield _chunk(
            reasoning=f"[Reached maximum {MAX_AGENT_TURNS} research turns, synthesizing answer...]"
        )

        # One final call with instruction to stop using tools
        working_messages.append({
            "role": "user",
            "content": (
                "You have used all available research turns. "
                "Please synthesize your findings into a final, comprehensive answer NOW. "
                "Do NOT call any more tools."
            ),
        })
        upstream_body["messages"] = working_messages

        content, reasoning = await _collect_full_response(upstream_body, req_id)
        if reasoning:
            yield _chunk(reasoning=reasoning)
        final = content if content else reasoning
        if final:
            yield _chunk(content=final)

        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Agent loop error after {elapsed:.2f}s: {e}\n{tb}")
        error_msg = (
            f"**Heretic Proxy — Error**\n\n"
            f"An error occurred during research:\n\n"
            f"```\n{type(e).__name__}: {str(e)}\n```\n\n"
            f"_Request: {req_id}_"
        )
        yield _chunk(content=error_msg)
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Heretic Proxy")

register_standard_routes(
    app,
    service_name="heretic-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "model": UPSTREAM_MODEL,
        "max_agent_turns": MAX_AGENT_TURNS,
        "active_tools": [t["function"]["name"] for t in TOOLS],
    },
)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "glm-4.7-flash-heretic",
                "object": "model",
                "created": 1700000000,
                "owned_by": "heretic-proxy",
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
    req_id = f"heretic-{uuid.uuid4().hex[:8]}"
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
            model_id=body.get("model", "glm-4.7-flash-heretic"),
            tracker=tracker,
            log=log,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    # Real chat request — run the agentic tool-calling loop
    gen = run_agent_loop(messages, body, req_id)
    return StreamingResponse(gen, media_type="text/event-stream")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
