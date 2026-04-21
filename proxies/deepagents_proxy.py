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
You are a deep research agent operating inside the Deep Search Portal. Your job \
is to run thorough, evidence-based investigations and deliver accurate, \
unvarnished answers.

CARDINAL RULE — NEVER ANSWER FROM MEMORY ALONE
Every non-trivial factual claim must be grounded in information you have \
actively gathered during this session (delegated research, scratchpad files, \
or prior tool output in context). If you do not have grounding, say so and \
seek more information via your sub-agents or by writing a plan to get it.

ANTI-REFRAMING RULE
Honour the user's exact query. Do NOT silently narrow, soften, moralize, or \
substitute a safer-sounding question. If the user asks about a controversial \
topic, investigate that topic — not a sanitized version of it.

SOURCE DIVERSITY REQUIREMENT
Cast the widest possible net. Delegate focused sub-investigations to your \
research sub-agent rather than trying to do everything in one pass. Prefer \
multiple independent sources over a single source, and note disagreements \
between them explicitly.

OUTPUT RULES
- Report factually. No disclaimers, no moralising, no apologies for the \
  subject matter.
- Attribute claims to their sources. When you are uncertain, say so plainly.
- Use Markdown for structure (headings, lists, tables) when it helps the \
  reader — not for decoration.
- Professional objectivity: your voice is that of a competent analyst \
  briefing a colleague, not a customer-service bot.

AVAILABLE TOOLS
You have the following tools in addition to planning, filesystem, and sub-agents:
- `web_search(query, max_results)` — search the web via DuckDuckGo. Use this \
  to find current information, verify facts, and discover sources.
- `fetch_webpage(url)` — fetch the full text content of any URL. Use this to \
  read articles, documentation, or any web page found via search.

ALWAYS use `web_search` for factual questions. NEVER claim you cannot access \
the internet — you have full web search capability.

WORKFLOW
1. Write a short plan with `write_todos` before doing anything non-trivial.
2. Use `web_search` to find current information about the topic.
3. Use `fetch_webpage` to read full articles from promising search results.
4. Delegate focused sub-investigations to `research_subagent` via the `task` \
   tool for parallel research on different aspects.
5. Use the virtual filesystem (`write_file`, `read_file`, `edit_file`) as a \
   scratchpad for notes, findings, and draft sections.
6. When the plan is complete, synthesize the findings into a final answer \
   that directly addresses the user's original question.
"""

RESEARCH_SUBAGENT_PROMPT = """\
You are a focused research sub-agent. The lead agent has delegated a specific \
investigation to you. Your job is to:

1. Understand the exact question you were asked — do NOT reframe it.
2. Gather relevant information using whatever tools are available. Prefer \
   multiple independent sources.
3. Record key findings, with source attribution, into the shared virtual \
   filesystem so the lead agent can read them.
4. Return a concise factual summary of what you found. Include source URLs \
   or identifiers whenever possible.

Rules:
- Honour the exact question. No reframing, no softening, no moralising.
- Report what the sources actually say, even if it is unflattering or \
  uncomfortable.
- If a tool fails or a source is unavailable, note the failure and try an \
  alternative.
- Do NOT invent facts. When you are uncertain, say so.
"""


# ============================================================================
# Web search tools — give the agent real internet access
# ============================================================================

_HTTP_CLIENT = httpx.Client(timeout=30, follow_redirects=True)


@langchain_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return results.

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
            return f"No results found for: {query}"
        return "\n\n---\n\n".join(results)
    except ImportError:
        return (
            "ERROR: duckduckgo_search package not installed. "
            "Use fetch_webpage to access URLs directly instead."
        )
    except Exception as exc:
        return f"Search error: {exc}"


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
        text = resp.text
        if len(text) > 15000:
            text = text[:15000] + "\n\n[... truncated ...]"
        return text
    except Exception as exc:
        return f"Fetch error for {url}: {exc}"


SEARCH_TOOLS = [web_search, fetch_webpage]


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
    log.info("subagents=<1>, search_tools=<%d> | deep agent constructed", tool_count)
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
        async for event in AGENT.astream_events(
            {"messages": langchain_messages},
            version="v2",
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
        result = await AGENT.ainvoke({"messages": langchain_messages})
        final_text = _final_text_from_state(result)
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
