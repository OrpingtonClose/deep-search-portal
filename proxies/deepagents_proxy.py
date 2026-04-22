#!/usr/bin/env python3
"""
Deep Agents Proxy — LangGraph deepagents SDK wrapped as an OpenAI-compatible
endpoint, routed through Venice AI's GLM-4.7 Flash Heretic model.

The deepagents SDK (https://github.com/OrpingtonClose/deepagents) provides a
pre-wired LangGraph "deep agent" with built-in planning (write_todos),
virtual filesystem (ls / read_file / write_file / edit_file / glob / grep),
and sub-agent delegation (task). We expose that agent as a drop-in OpenAI
chat-completions server so LibreChat can speak to it like any other model.

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes them through
    to Venice AI directly (bypassing the deep-agent graph)
  - For real chat requests:
      * Converts ChatML messages to LangChain BaseMessage objects
      * Invokes the compiled LangGraph agent via ``astream_events(version="v2")``
      * Streams ``on_chat_model_stream`` tokens as SSE ``content`` chunks
      * Streams ``on_tool_start`` / ``on_tool_end`` events as ``reasoning_content``
        so LibreChat renders them in a collapsible Thinking block
      * Non-streaming requests collect the final AI message via ``ainvoke``

Runs as a FastAPI app under uvicorn in a screen session (default port 8200).
"""

import asyncio
import json
import os
import time
import traceback
import uuid
from typing import AsyncGenerator, Optional

import html as html_mod
import re

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI

from deepagents import SubAgent, create_deep_agent

from research_tools import ALL_RESEARCH_TOOLS
from shared import (
    RequestTracker,
    create_app,
    env_int,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
    utility_passthrough_json,
)

# --- Logging ---
LOG_DIR = os.getenv("DEEPAGENTS_LOG_DIR", "/opt/deepagents_proxy_logs")
log = setup_logging("deepagents-proxy", LOG_DIR)

# --- Configuration ---
VENICE_API_BASE = os.getenv("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_API_KEY = require_env("VENICE_API_KEY")
DEEPAGENTS_MODEL = os.getenv(
    "DEEPAGENTS_MODEL", "olafangensan-glm-4.7-flash-heretic"
)
LISTEN_PORT = env_int("DEEPAGENTS_PORT", 8200, minimum=1)

# Model ID we advertise to LibreChat via /v1/models.
MODEL_ID = "deepagents-research"

log.info(
    f"Config: venice_base={VENICE_API_BASE}, model={DEEPAGENTS_MODEL}, "
    f"port={LISTEN_PORT}, model_id={MODEL_ID}"
)

tracker = RequestTracker()


# ============================================================================
# System prompts
# ============================================================================

# NOTE: The deepagents SDK appends its own BASE_AGENT_PROMPT (planning,
# filesystem, sub-agent instructions) after our custom system prompt, so this
# prompt should ONLY contain the domain-specific research rules we want to
# layer on top.

SYSTEM_PROMPT = """\
You are a friendly, sharp research agent with access to 17+ specialised \
data sources spanning web search, academic databases, government records, \
preprint servers, court filings, and more. Use them.

RULES
- Always search first — never answer from memory alone.
- Use MULTIPLE tools per query. Different sources have different coverage.
- Honour the user's exact question. No reframing, no softening.
- Cite your sources with URLs. Flag uncertainty honestly.
- Be concise: get to the point, then stop. No disclaimers or filler.
- Use Markdown only when it genuinely helps readability.
- NEVER reference virtual file paths, filenames, or documents you created \
  in your answer. The user CANNOT access your filesystem. All content MUST \
  be inlined directly into the response text. Never say "I saved to X" or \
  "see file X" — just put the content in your answer.

WEB SEARCH
- `brave_search(query)` — primary general-purpose web search (Brave API)
- `exa_search(query)` — neural/semantic web search (Exa API)
- `web_search(query)` — DuckDuckGo fallback
- `fetch_webpage(url)` — raw page fetch with HTML stripping
- `jina_read_url(url)` — clean content extraction (Jina Reader, better \
  for articles)

ACADEMIC (free, no keys)
- `openalex_search(query)` — 240M+ works (papers, books, datasets)
- `semantic_scholar_search(query)` — 200M+ papers, AI relevance ranking
- `search_pubmed(query)` — 36M+ biomedical citations (NCBI)
- `resolve_doi(doi)` — full metadata for any DOI via CrossRef
- `check_retraction(doi)` — verify if a paper has been retracted

PREPRINTS
- `search_biorxiv(query, server)` — bioRxiv/medRxiv preprints

GOVERNMENT / LEGAL (free, no keys)
- `search_clinical_trials(query)` — ClinicalTrials.gov (suppressed trials)
- `search_fda_adverse_events(drug_name, reaction)` — FDA FAERS database
- `search_sec_filings(query, filing_type)` — SEC EDGAR corporate filings
- `search_court_opinions(query)` — CourtListener (PACER for free)
- `search_offshore_leaks(query)` — ICIJ Panama/Paradise/Pandora Papers

ARCHIVES
- `wayback_search(url)` — Internet Archive / Wayback Machine snapshots

BUILT-IN
- `research_subagent` via `task` — delegate focused sub-questions
- `write_file` / `read_file` — private scratchpad (NEVER mention these \
  files or their paths in your answer — the user cannot see them)

QUERY-AWARE TOOL SELECTION
- Academic question: openalex_search + semantic_scholar_search, then \
  search_pubmed for biomedical, resolve_doi for specific papers.
- Medical/drug question: search_clinical_trials + search_fda_adverse_events \
  + search_pubmed + search_biorxiv(server="medrxiv").
- Legal/corporate: search_sec_filings + search_court_opinions + \
  search_offshore_leaks.
- General factual: brave_search + exa_search, then jina_read_url.
- Always cross-reference with at least 2 different source types.

WORKFLOW
1. Quick plan with `write_todos`.
2. Choose tools based on query domain (see above).
3. Search with 3-5 tools in parallel — cast a wide net.
4. Read promising URLs with jina_read_url or fetch_webpage.
5. Delegate sub-questions to research_subagent if needed.
6. Synthesize a direct, friendly answer with source links.
"""

RESEARCH_SUBAGENT_PROMPT = """\
You are a focused research helper. For the question you've been given:

1. Search for relevant info using available tools. Use multiple sources.
2. Write key findings (with source URLs) to the shared filesystem.
3. Return a concise summary of what you found.

Be direct, factual, and honest about gaps. Don't reframe the question.
"""


# ============================================================================
# Search tools — API-backed engines for reliable web research
# ============================================================================

_HTTP_CLIENT = httpx.Client(timeout=30, follow_redirects=True)

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

log.info(
    "search_keys: brave=%s, exa=%s",
    "SET" if BRAVE_API_KEY else "MISSING",
    "SET" if EXA_API_KEY else "MISSING",
)


@langchain_tool
def brave_search(query: str, max_results: int = 8) -> str:
    """Search the web using the Brave Search API.

    This is the primary search engine — use it first for every query.
    Returns titles, URLs, and descriptions from Brave's index.

    Args:
        query: The search query.
        max_results: Maximum number of results (default 8).

    Returns:
        Formatted search results with titles, URLs, and descriptions.
    """
    if not BRAVE_API_KEY:
        return "ERROR: BRAVE_API_KEY not configured"
    try:
        resp = _HTTP_CLIENT.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", [])[:max_results]:
            title = r.get("title", "No title")
            url = r.get("url", "N/A")
            desc = r.get("description", "No description")
            results.append(f"**{title}**\nURL: {url}\n{desc}")
        if not results:
            return f"No Brave results for: {query}"
        return "\n\n---\n\n".join(results)
    except Exception as exc:
        return f"Brave search error: {exc}"


@langchain_tool
def exa_search(query: str, max_results: int = 8) -> str:
    """Search the web using the Exa neural search API.

    Exa excels at finding specific documents, research papers, niche content,
    and pages that match the meaning of a query (not just keywords).

    Args:
        query: The search query (can be a natural language question).
        max_results: Maximum number of results (default 8).

    Returns:
        Formatted search results with titles, URLs, and text highlights.
    """
    if not EXA_API_KEY:
        return "ERROR: EXA_API_KEY not configured"
    try:
        resp = _HTTP_CLIENT.post(
            "https://api.exa.ai/search",
            json={
                "query": query,
                "numResults": max_results,
                "type": "auto",
                "contents": {"text": {"maxCharacters": 1000}},
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": EXA_API_KEY,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            title = r.get("title", "No title")
            url = r.get("url", "N/A")
            text = r.get("text", "No content")[:500]
            results.append(f"**{title}**\nURL: {url}\n{text}")
        if not results:
            return f"No Exa results for: {query}"
        return "\n\n---\n\n".join(results)
    except Exception as exc:
        return f"Exa search error: {exc}"


@langchain_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo (fallback search engine).

    Use brave_search and exa_search first — this is the fallback.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"**{r.get('title', 'No title')}**\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"{r.get('body', 'No snippet')}"
                )
        if not results:
            return f"No DuckDuckGo results for: {query}"
        return "\n\n---\n\n".join(results)
    except ImportError:
        return (
            "DuckDuckGo not available. "
            "Use brave_search or exa_search instead."
        )
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"


@langchain_tool
def fetch_webpage(url: str) -> str:
    """Fetch the text content of a web page.

    Args:
        url: The URL to fetch.

    Returns:
        The page text content (truncated to 15000 chars).
    """
    try:
        resp = _HTTP_CLIENT.get(url, headers={"User-Agent": "DeepSearchBot/1.0"})
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            return f"PDF document at {url} (binary content, cannot extract text directly)"
        if (
            "text/html" not in content_type
            and "text/plain" not in content_type
            and "text/xml" not in content_type
            and "application/json" not in content_type
            and content_type
        ):
            return f"Non-text content type: {content_type} at {url}"
        text = resp.text
        # Strip HTML tags, scripts, styles to extract readable text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html_mod.unescape(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\S\n]+', ' ', text).strip()
        if len(text) > 15000:
            text = text[:15000] + "\n\n[... truncated ...]"
        return text
    except Exception as exc:
        return f"Fetch error for {url}: {exc}"


SEARCH_TOOLS = [brave_search, exa_search, web_search, fetch_webpage] + ALL_RESEARCH_TOOLS


# ============================================================================
# Agent construction (module-level singleton)
# ============================================================================

def _build_agent():
    """Build the compiled deepagents LangGraph StateGraph.

    The LLM is a ``ChatOpenAI`` instance pointed at Venice AI's OpenAI-compatible
    endpoint. The deepagents SDK wires planning, filesystem, and sub-agent
    middleware around it automatically.

    Web search tools (web_search, fetch_webpage) are injected so the agent can
    actually research topics on the internet instead of relying on training data.
    """
    model = ChatOpenAI(
        model=DEEPAGENTS_MODEL,
        api_key=VENICE_API_KEY,
        base_url=VENICE_API_BASE,
        streaming=True,
    )

    research_subagent: SubAgent = {
        "name": "research_subagent",
        "description": (
            "Specialized research agent for focused web investigations. "
            "Delegate a specific, well-scoped question and it will gather "
            "facts and return a concise summary with source attribution."
        ),
        "system_prompt": RESEARCH_SUBAGENT_PROMPT,
        "model": model,
    }

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=[research_subagent],
        tools=SEARCH_TOOLS,
        name=MODEL_ID,
    )
    tool_count = len(SEARCH_TOOLS)
    tool_names = [t.name for t in SEARCH_TOOLS]
    log.info(
        "subagents=<1>, search_tools=<%d>, tools=<%s> | deep agent constructed",
        tool_count,
        tool_names,
    )
    return agent


AGENT = _build_agent()


# ============================================================================
# ChatML → LangChain message conversion
# ============================================================================

def _messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """Convert OpenAI-format chat messages to LangChain ``BaseMessage`` objects.

    Tool / function / name fields are dropped: LibreChat sends conversational
    turns only, and the deep agent manages its own tool-call history internally.
    """
    out: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Some clients send content as a list of parts — flatten the text parts.
            parts = []
            for part in content or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(p for p in parts if p)

        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "tool":
            # Rare for LibreChat — preserve for completeness.
            out.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
        else:
            out.append(HumanMessage(content=content))
    return out


def _final_text_from_state(state: dict) -> str:
    """Extract the assistant's final text answer from an agent result dict."""
    messages = state.get("messages", []) if isinstance(state, dict) else []
    # Walk backwards to the last AI message that has non-empty text content.
    for msg in reversed(messages):
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    return ""


# ============================================================================
# File-reference stripping
# ============================================================================

# Matches file paths and common "I saved to file" patterns the agent emits.
_FILE_REF_PATTERNS = [
    # "I've saved the findings to `compound_interactions.md`"
    re.compile(
        r"(?:I(?:'ve|'ve| have)?|Here(?:'s| is))\s+"
        r"(?:saved|written|compiled|created|stored|put|exported|generated)\s+"
        r"(?:the |my |a |an )?(?:findings?|research|results?|notes?|report|"
        r"analysis|data|summary|details?|content|document|overview|information)?"
        r"\s*(?:to|in|into|as|at)\s+"
        r"[`\"']?[\w/.\-]+\.(?:md|txt|json|csv|html|pdf)[`\"']?"
        r"[.]?",
        re.IGNORECASE,
    ),
    # "See `filename.md` for details" / "refer to filename.md"
    re.compile(
        r"(?:see|refer to|check|view|open|read)\s+"
        r"[`\"']?[\w/.\-]+\.(?:md|txt|json|csv|html|pdf)[`\"']?"
        r"(?:\s+for\s+\w+(?:\s+\w+){0,4})?[.]?",
        re.IGNORECASE,
    ),
    # Bare backtick-quoted filenames like `research_notes.md`
    re.compile(
        r"`[\w/.\-]+\.(?:md|txt|json|csv|html|pdf)`"
        r"(?:\s*[-–—]\s*[^\n]{0,80})?",
    ),
]


def _strip_file_references(text: str) -> str:
    """Remove virtual-filesystem references from agent output."""
    for pat in _FILE_REF_PATTERNS:
        text = pat.sub("", text)
    # Clean up leftover blank lines from stripped references
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# Streaming: convert LangGraph events → OpenAI SSE chunks
# ============================================================================

async def _stream_agent(
    langchain_messages: list[BaseMessage],
    *,
    req_id: str,
    model_id: str,
) -> AsyncGenerator[str, None]:
    """Run the deep agent with ``astream_events`` and emit OpenAI-format SSE."""
    request_id = f"chatcmpl-deepagents-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    first_chunk_sent = False

    def _chunk(
        content: str = "",
        *,
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
            # Emit an initial role delta so LibreChat's OpenAI SDK parser
            # sees the assistant turn before the first content token.
            role_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(role_data)}\n\n" + chunk
        return chunk

    try:
        any_content_emitted = False
        # Track files written by the agent so we can inline them for the user.
        written_files: dict[str, str] = {}  # filename → content
        async for event in AGENT.astream_events(
            {"messages": langchain_messages},
            version="v2",
            config={"recursion_limit": 150},
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            metadata = event.get("metadata", {}) or {}
            node = metadata.get("langgraph_node", "")
            data = event.get("data", {}) or {}

            if kind == "on_chat_model_stream":
                chunk_msg = data.get("chunk")
                if chunk_msg is None:
                    continue
                text = getattr(chunk_msg, "content", "")
                if isinstance(text, list):
                    # Content parts — flatten text-type blocks.
                    flat = []
                    for part in text:
                        if isinstance(part, dict) and part.get("type") == "text":
                            flat.append(part.get("text", ""))
                    text = "".join(flat)
                if not text:
                    continue

                # Sub-agent tokens are intermediate reasoning — send them as
                # reasoning_content so LibreChat folds them into the Thinking
                # block rather than dumping them into the final answer.
                # Only the LEAD agent's final synthesis becomes visible content.
                if node and node != "model":
                    yield _chunk(reasoning=text)
                else:
                    any_content_emitted = True
                    yield _chunk(content=text)

            elif kind == "on_tool_start":
                tool_name = name or "tool"
                tool_input = data.get("input", {})

                # Capture write_file content so we can inline it later.
                if tool_name in ("write_file", "edit_file"):
                    fname = (
                        tool_input.get("filename")
                        or tool_input.get("file_path")
                        or tool_input.get("path")
                        or "untitled"
                    )
                    fcontent = tool_input.get("content", "")
                    if isinstance(fname, str) and isinstance(fcontent, str) and fcontent:
                        written_files[fname] = fcontent
                        log.info(
                            f"[{req_id}] Captured write_file: "
                            f"{fname} ({len(fcontent)} chars)"
                        )

                try:
                    args_preview = json.dumps(tool_input, ensure_ascii=False)[:200]
                except Exception:
                    args_preview = str(tool_input)[:200]
                log.info(f"[{req_id}] Tool start: {tool_name} args={args_preview}")
                yield _chunk(
                    reasoning=f"\n> 🔧 **{tool_name}**\n> `{args_preview}`\n"
                )

            elif kind == "on_tool_end":
                tool_name = name or "tool"
                output = data.get("output", "")
                if not isinstance(output, str):
                    try:
                        output = json.dumps(output, ensure_ascii=False, default=str)
                    except Exception:
                        output = str(output)
                preview = output[:300].replace("\n", " ")
                log.info(
                    f"[{req_id}] Tool end: {tool_name} ({len(output)} chars)"
                )
                yield _chunk(
                    reasoning=f"> ↳ _{tool_name} returned {len(output)} chars_\n"
                )

            # Other event kinds (on_chain_start/end, on_chat_model_start/end, etc.)
            # are ignored — we only surface tokens and tool boundaries.

        # Append captured file content as collapsible sections so the user
        # can see the research the agent compiled (instead of broken paths).
        if written_files:
            yield _chunk(content="\n\n---\n\n")
            for fname, fcontent in written_files.items():
                section = (
                    f"<details>\n"
                    f"<summary>📎 {fname}</summary>\n\n"
                    f"{fcontent}\n\n"
                    f"</details>\n\n"
                )
                yield _chunk(content=section)
                any_content_emitted = True

        if not any_content_emitted:
            # Rare fallback: the agent produced no streamed content. Emit a
            # minimal notice so LibreChat doesn't render an empty assistant turn.
            yield _chunk(
                content="_The agent finished without producing a final answer._"
            )

        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

        elapsed = time.monotonic() - start_time
        log.info(f"[{req_id}] Stream complete in {elapsed:.1f}s")

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(
            f"[{req_id}] Agent stream error after {elapsed:.1f}s: {exc}\n{tb}"
        )
        err_msg = (
            f"**Deep Agents Proxy — Error**\n\n"
            f"```\n{type(exc).__name__}: {exc}\n```\n\n"
            f"_Request: {req_id}_"
        )
        yield _chunk(content=err_msg)
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# Non-streaming path: run the agent and return a single JSON response
# ============================================================================

def _extract_written_files(state: dict) -> dict[str, str]:
    """Extract filenames and content from write_file/edit_file tool calls in the result state."""
    written: dict[str, str] = {}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for msg in messages:
        if not isinstance(msg, (AIMessage, AIMessageChunk)):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if tc.get("name") not in ("write_file", "edit_file"):
                continue
            args = tc.get("args", {})
            fname = args.get("filename") or args.get("file_path") or args.get("path") or "untitled"
            fcontent = args.get("content", "")
            if isinstance(fname, str) and isinstance(fcontent, str) and fcontent:
                written[fname] = fcontent
    return written


async def _invoke_agent_json(
    langchain_messages: list[BaseMessage],
    *,
    req_id: str,
    model_id: str,
) -> JSONResponse:
    request_id = f"chatcmpl-deepagents-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()
    try:
        result = await AGENT.ainvoke(
            {"messages": langchain_messages},
            config={"recursion_limit": 150},
        )
        final_text = _strip_file_references(_final_text_from_state(result))

        # Inline any files the agent wrote (mirrors streaming path behaviour).
        written_files = _extract_written_files(result)
        if written_files:
            sections = ["\n\n---\n\n"]
            for fname, fcontent in written_files.items():
                sections.append(
                    f"<details>\n"
                    f"<summary>📎 {fname}</summary>\n\n"
                    f"{fcontent}\n\n"
                    f"</details>\n\n"
                )
            final_text += "".join(sections)
        elapsed = time.monotonic() - start_time
        log.info(
            f"[{req_id}] Agent invoke complete in {elapsed:.1f}s "
            f"({len(final_text)} chars)"
        )
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(
            f"[{req_id}] Agent invoke error after {elapsed:.1f}s: {exc}\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "agent_error",
                }
            },
        )
    finally:
        tracker.finish(req_id)


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Deep Agents Proxy")

register_standard_routes(
    app,
    service_name="deepagents-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "venice_base": VENICE_API_BASE,
        "upstream_model": DEEPAGENTS_MODEL,
        "model_id": MODEL_ID,
    },
)


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "deepagents-proxy",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Invalid request body: {exc}",
                    "type": "invalid_request",
                }
            },
        )

    messages = body.get("messages", []) or []
    # Default to False to match the rest of the proxy fleet and the
    # documented `is_utility_request(...) and not body.get("stream", False)`
    # contract from shared.py. LibreChat always sets `stream` explicitly for
    # real chat turns, and omits it (or sets False) for title / tag utility
    # requests — we want those to take the JSON passthrough path.
    stream = bool(body.get("stream", False))
    model_id = body.get("model", MODEL_ID) or MODEL_ID
    req_id = f"deepagents-{uuid.uuid4().hex[:8]}"
    tracker.start(req_id, model=model_id, stream=stream)

    log.info(
        f"[{req_id}] Request: model={model_id}, messages={len(messages)}, "
        f"stream={stream}"
    )

    # Utility requests (title / tag gen) — passthrough to Venice directly.
    if is_utility_request(messages):
        log.info(f"[{req_id}] Utility request — passthrough to Venice")
        if not stream:
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=VENICE_API_BASE,
                upstream_key=VENICE_API_KEY,
                upstream_model=DEEPAGENTS_MODEL,
                log=log,
            )
            tracker.finish(req_id)
            return result
        gen = stream_passthrough(
            messages,
            body,
            req_id=req_id,
            upstream_base=VENICE_API_BASE,
            upstream_key=VENICE_API_KEY,
            upstream_model=DEEPAGENTS_MODEL,
            model_id=model_id,
            tracker=tracker,
            log=log,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    # Real chat request — run through the deep agent graph.
    langchain_messages = _messages_to_langchain(messages)

    if not stream:
        return await _invoke_agent_json(
            langchain_messages, req_id=req_id, model_id=model_id
        )

    gen = _stream_agent(langchain_messages, req_id=req_id, model_id=model_id)
    return StreamingResponse(gen, media_type="text/event-stream")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
