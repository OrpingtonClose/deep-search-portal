# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for the Venice uncensored research agent (Strands SDK).

Exposes the Strands agent as an HTTP API with:
- POST /query — single-turn query (single-agent mode)
- POST /query/multi — single-turn query (planner + researcher mode)
- POST /v1/chat/completions — OpenAI-compatible (LibreChat integration)
- GET  /v1/models — OpenAI-compatible model list
- GET  /health — health check
- GET  /tools — list loaded tools
- GET  /logs/{request_id} — human-readable activity log for a request
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ── Observability: import from deep-search-portal when available ──────
# The strands_observability module lives in deep-search-portal/proxies/
# and is added to PYTHONPATH by scripts/start_strands_agent.sh.
# If unavailable, we skip external observability (metrics still logged locally).
try:
    from strands_observability import (
        extract_usage,
        format_inline_log,
        get_request_log,
        setup_strands_sdk_logging,
        store_request_log,
        write_metrics_jsonl,
    )
    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False

# ── Thought refinement middleware ──────────────────────────────────────
# Transforms chaotic agent reasoning into user-friendly status updates.
try:
    from thought_refiner import refine_async, REFINER_ENABLED
    _HAS_REFINER = True
except ImportError:
    _HAS_REFINER = False

logger = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────

_single_agent = None
_multi_agent = None
_deep_agent = None
_mcp_clients: list = []
_multi_researcher = None
_agent_lock = threading.Lock()


# ── Observability wrappers ────────────────────────────────────────────
# When strands_observability (from deep-search-portal) is available,
# delegate to it.  Otherwise fall back to minimal local-only logging.

def _write_metrics_jsonl(req_id: str, model: str, query: str, elapsed: float,
                         metrics_summary: dict | None, tool_events: list[dict]) -> None:
    """Write per-request metrics to JSONL.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        write_metrics_jsonl(req_id, model, query, elapsed, metrics_summary, tool_events)
    else:
        # Minimal fallback: just log a summary line
        logger.info(
            "[metrics] %s model=%s elapsed=%.1fs tools=%d",
            req_id, model, elapsed, len(tool_events),
        )


def _format_inline_log(tool_events: list[dict], elapsed: float, query: str = "",
                       model: str = "", reasoning: str = "",
                       metrics: dict | None = None) -> str:
    """Format activity log for inline display.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        return format_inline_log(
            tool_events, elapsed,
            query=query, model=model, reasoning=reasoning, metrics=metrics,
        )
    # Minimal fallback: just a tool count summary
    tool_names = [e.get("tool", "?") for e in tool_events]
    summary = ", ".join(tool_names) if tool_names else "(no tools)"
    return f"\n\n---\n*{len(tool_events)} tool calls in {elapsed:.1f}s: {summary}*"


def _store_log(req_id: str, entry: dict) -> None:
    """Store per-request activity log.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        store_request_log(req_id, entry)


def _get_log(req_id: str) -> dict | None:
    """Retrieve a stored request log."""
    if _HAS_OBSERVABILITY:
        return get_request_log(req_id)
    return None


def _extract_usage(metrics_summary: dict | None) -> dict[str, int]:
    """Extract OpenAI-compatible usage dict from metrics."""
    if _HAS_OBSERVABILITY:
        return extract_usage(metrics_summary)
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if metrics_summary and metrics_summary.get("accumulated_usage"):
        u = metrics_summary["accumulated_usage"]
        usage = {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
        }
    return usage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create agents. Shutdown: close MCP connections."""
    global _single_agent, _multi_agent, _deep_agent, _multi_researcher, _mcp_clients

    from agent import (
        _enter_mcp_clients,
        _setup_otel,
        create_deep_agent_instance,
        create_multi_agent,
        create_single_agent,
    )
    from tools import get_all_mcp_clients

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Structured JSON logging for Strands SDK internals ──
    # Delegates to deep-search-portal's strands_observability module when available.
    if _HAS_OBSERVABILITY:
        setup_strands_sdk_logging()
    else:
        logger.info("strands_observability not available — SDK debug logging disabled")

    _setup_otel()

    # Enter MCP clients once and share tools between both agents
    try:
        _mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(_mcp_clients)
    except Exception:
        logger.exception("Failed to initialise MCP tools")
        tool_list = []
        _mcp_clients = []

    try:
        _single_agent, _ = create_single_agent(
            tool_list=tool_list, mcp_clients=_mcp_clients
        )
        logger.info(
            "Single agent ready — %d tools",
            len(_single_agent.tool_registry.get_all_tools_config()),
        )
    except Exception:
        logger.exception("Failed to create single agent")

    try:
        _multi_agent, _multi_researcher, _ = create_multi_agent(
            tool_list=tool_list, mcp_clients=_mcp_clients
        )
        logger.info("Multi agent ready")
    except Exception:
        logger.exception("Failed to create multi agent")

    try:
        _deep_agent, _ = create_deep_agent_instance(
            tool_list=tool_list, mcp_clients=_mcp_clients
        )
        logger.info("Deep agent ready")
    except Exception:
        logger.exception("Failed to create deep agent")

    yield

    # Shutdown: close MCP connections (once)
    from agent import _cleanup_mcp

    _cleanup_mcp(_mcp_clients)
    logger.info("MCP connections closed")


app = FastAPI(
    title="Strands Venice Agent API",
    description="Venice uncensored research agent — Strands Agents SDK",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Request / Response models ────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., description="The research query to send to the agent")


class QueryResponse(BaseModel):
    query: str
    response: str
    mode: str
    elapsed_seconds: float


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""


class ChatCompletionRequest(BaseModel):
    model: str = "strands-venice-single"
    messages: list[ChatMessage] = []
    stream: bool = False

    model_config = {"extra": "allow"}


# ── Helper: extract user message from ChatML messages ────────────────


def _extract_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


def _chatml_to_strands(messages: list[ChatMessage]) -> list[dict]:
    """Convert LibreChat ChatML messages to Strands Converse format.

    LibreChat sends the full conversation history on each request.  We
    convert user/assistant turns into the Strands internal format so the
    agent has full conversational context.

    Strands format:  ``{"role": "user"|"assistant", "content": [{"text": "..."}]}``
    """
    result = []
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            continue
        content = msg.content
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content) if content else ""
        if text.strip():
            result.append({"role": msg.role, "content": [{"text": text}]})
    return result


def _load_conversation_history(
    agent, messages: list[ChatMessage], researcher=None
) -> str:
    """Load conversation history into the agent and return the latest user message.

    Converts all previous turns from the ChatML request into Strands
    format and injects them into ``agent.messages``.  The latest user
    message is extracted and returned (it will be passed to ``agent()``
    which adds it to messages internally).

    For multi-agent, the researcher's messages are always cleared (it
    starts fresh for each delegation from the planner).
    """
    strands_messages = _chatml_to_strands(messages)

    # The last message should be the new user query — exclude it from
    # history since agent() will add it.
    if strands_messages and strands_messages[-1]["role"] == "user":
        history = strands_messages[:-1]
        user_message = strands_messages[-1]["content"][0]["text"]
    else:
        # Last strands message is not a user message (unusual — e.g.
        # trailing assistant message).  Find the last user message by
        # index and slice: history = everything before it.  Messages
        # after it (assistant responses) are omitted since the agent
        # will re-process the user message via agent().
        user_message = _extract_user_message(messages)
        last_user_idx = None
        for i in range(len(strands_messages) - 1, -1, -1):
            if strands_messages[i]["role"] == "user" and strands_messages[i]["content"][0]["text"] == user_message:
                last_user_idx = i
                break
        history = strands_messages[:last_user_idx] if last_user_idx is not None else strands_messages

    agent.messages.clear()
    agent.messages.extend(history)

    if researcher is not None:
        researcher.messages.clear()

    return user_message


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "single_agent": _single_agent is not None,
        "multi_agent": _multi_agent is not None,
        "deep_agent": _deep_agent is not None,
    }


@app.get("/tools")
async def list_tools():
    """List all loaded tools."""
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    tools = _single_agent.tool_registry.get_all_tools_config()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": name,
                "description": spec.get("description", "")
                if isinstance(spec, dict)
                else "",
            }
            for name, spec in tools.items()
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query_single(req: QueryRequest):
    """Send a query to the single-agent (all tools directly available)."""
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Single agent not initialised")

    start = time.time()
    req_id = f"query-{uuid.uuid4().hex[:12]}"
    with _agent_lock:
        from agent import reset_budget

        _single_agent.messages.clear()
        reset_budget()
        try:
            response = _single_agent(req.query)
            metrics = None
            try:
                metrics = response.metrics.get_summary()
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = round(time.time() - start, 2)
    _write_metrics_jsonl(req_id, _MODEL_SINGLE, req.query, elapsed, metrics, [])

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="single",
        elapsed_seconds=elapsed,
    )


@app.post("/query/multi", response_model=QueryResponse)
def query_multi(req: QueryRequest):
    """Send a query to the multi-agent (planner delegates to researcher)."""
    if _multi_agent is None:
        raise HTTPException(status_code=503, detail="Multi agent not initialised")

    start = time.time()
    req_id = f"query-{uuid.uuid4().hex[:12]}"
    with _agent_lock:
        from agent import reset_budget

        _multi_agent.messages.clear()
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
        reset_budget()
        try:
            response = _multi_agent(req.query)
            metrics = None
            try:
                metrics = response.metrics.get_summary()
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = round(time.time() - start, 2)
    _write_metrics_jsonl(req_id, _MODEL_MULTI, req.query, elapsed, metrics, [])

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="multi",
        elapsed_seconds=elapsed,
    )


# ── OpenAI-compatible endpoints (for LibreChat integration) ──────────

_MODEL_SINGLE = "strands-venice-single"
_MODEL_MULTI = "strands-venice-multi"
_MODEL_DEEP = "strands-venice-deep"


@app.get("/v1/models")
async def openai_models():
    """Return available models in OpenAI list format."""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _MODEL_SINGLE,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "strands-venice-agent",
                },
                {
                    "id": _MODEL_MULTI,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "strands-venice-agent",
                },
                {
                    "id": _MODEL_DEEP,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "strands-deep-agents",
                },
            ],
        }
    )


def _dispatch_agent(
    model: str,
    user_message: str,
    chat_messages: list[ChatMessage] | None = None,
) -> tuple[str, dict | None]:
    """Run the appropriate agent.  **Caller must already hold _agent_lock**.

    When *chat_messages* is provided (from the OpenAI-compatible endpoint),
    conversation history is loaded into the agent so it has full context
    from previous turns.  Otherwise messages are simply cleared (legacy
    ``/query`` endpoints).

    Returns (text_answer, metrics_summary).  If the agent result has no text
    content (e.g. it ended on a tool call), falls back to the captured
    streamed text via ``stream_capture.response_text``.
    """
    from agent import reset_budget, stream_capture

    metrics_summary = None

    if model == _MODEL_MULTI:
        if _multi_agent is None:
            raise RuntimeError("Multi agent not initialised")
        if chat_messages:
            user_message = _load_conversation_history(
                _multi_agent, chat_messages, researcher=_multi_researcher
            )
        else:
            _multi_agent.messages.clear()
            if _multi_researcher is not None:
                _multi_researcher.messages.clear()
        reset_budget()
        agent_result = _multi_agent(user_message)
        result = str(agent_result)
        try:
            metrics_summary = agent_result.metrics.get_summary()
        except Exception:
            pass
    elif model == _MODEL_DEEP:
        if _deep_agent is None:
            raise RuntimeError("Deep agent not initialised")
        if chat_messages:
            user_message = _load_conversation_history(
                _deep_agent, chat_messages
            )
        else:
            _deep_agent.messages.clear()
        reset_budget()
        agent_result = _deep_agent(user_message)
        result = str(agent_result)
        try:
            metrics_summary = agent_result.metrics.get_summary()
        except Exception:
            pass
    elif _single_agent is not None:
        if chat_messages:
            user_message = _load_conversation_history(
                _single_agent, chat_messages
            )
        else:
            _single_agent.messages.clear()
        reset_budget()
        agent_result = _single_agent(user_message)
        result = str(agent_result)
        try:
            metrics_summary = agent_result.metrics.get_summary()
        except Exception:
            pass
    else:
        raise RuntimeError("No agent initialised")

    # Fallback: if the agent result has no text, use captured text.
    # Venice GLM with reasoning:high may send answer as reasoningText only.
    if not result.strip() and (stream_capture.response_text or stream_capture.reasoning_text):
        result = "".join(stream_capture.response_text) or "".join(stream_capture.reasoning_text)
    return result, metrics_summary


def _run_agent(
    model: str,
    user_message: str,
    chat_messages: list[ChatMessage] | None = None,
) -> tuple[str, dict | None]:
    """Dispatch to the correct agent under lock (convenience wrapper)."""
    from agent import stream_capture

    with _agent_lock:
        stream_capture.activate()
        try:
            return _dispatch_agent(model, user_message, chat_messages=chat_messages)
        finally:
            stream_capture.deactivate()


def _openai_chunk(req_id: str, model: str, content: str, finish: bool = False) -> str:
    """Format a single SSE chunk in OpenAI streaming format."""
    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Accepts standard OpenAI request format. Routes to single or multi
    agent based on the ``model`` field. Supports both streaming (SSE)
    and non-streaming responses.

    When streaming, tokens are emitted in real-time as the agent thinks
    and searches. A per-request activity log is stored and accessible
    at ``GET /logs/{request_id}``.
    """
    from agent import stream_capture

    model = body.model
    stream = body.stream
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    user_message = _extract_user_message(body.messages)
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No user message found",
                    "type": "invalid_request_error",
                }
            },
        )

    logger.info(
        "[%s] model=%s messages=%d stream=%s query=%.100s",
        req_id,
        model,
        len(body.messages),
        stream,
        user_message,
    )

    # Capture messages for session persistence (used by both paths)
    chat_messages = body.messages

    if stream:
        # ── Streaming mode ───────────────────────────────────────
        # activate() and deactivate() happen inside _agent_lock so
        # no concurrent request can clear the capture state.
        # A threading.Event signals the SSE generator once the queue
        # is ready (i.e. the lock has been acquired).
        result_holder: dict = {
            "text": None, "error": None,
            "tool_events": [], "streamed_text": "",
        }
        queue_holder: dict = {"q": None}
        queue_ready = threading.Event()

        def _agent_thread():
            with _agent_lock:
                token_q = stream_capture.activate()
                queue_holder["q"] = token_q
                queue_ready.set()
                try:
                    text, metrics = _dispatch_agent(model, user_message, chat_messages=chat_messages)
                    result_holder["text"] = text
                    result_holder["metrics"] = metrics
                except Exception as exc:
                    logger.exception("Agent error in streaming [%s]", req_id)
                    result_holder["error"] = str(exc)
                finally:
                    # Snapshot captured data while still under lock
                    result_holder["tool_events"] = list(stream_capture.tool_events)
                    result_holder["streamed_text"] = "".join(stream_capture.response_text)
                    result_holder["reasoning_text"] = "".join(stream_capture.reasoning_text)
                    stream_capture.deactivate()

        thread = threading.Thread(target=_agent_thread, daemon=True)
        thread.start()

        async def _generate_sse():
            loop = asyncio.get_event_loop()
            # Wait until agent thread has acquired the lock and created the queue
            await loop.run_in_executor(None, queue_ready.wait)
            token_queue = queue_holder["q"]

            # ── Streaming presentation state ──
            # We buffer thinking tokens and refine them via a fast LLM
            # before emitting, so the user sees concise status updates
            # instead of chaotic chain-of-thought.
            #
            # Layout:
            #   > 💭 **Thinking** — refined summary
            #   🔧 **Tool:** brave_web_search — `"query text..."`
            #   🔧 **Tool:** firecrawl_scrape — `"https://..."`
            #   ---
            #   <final answer text>
            thinking_buffer: list[str] = []  # Buffer raw thinking tokens
            has_answer_text = False     # True once we've emitted answer text
            tool_count = 0             # Number of tool calls emitted
            last_event_type = None     # Track event transitions

            async def _flush_thinking(is_final: bool = False):
                """Refine and emit buffered thinking tokens.

                When the refiner is available, the raw thinking is sent to
                a fast LLM for summarisation.  The refined version is emitted
                inside a blockquote.  If refinement fails or is disabled,
                the raw thinking is emitted as before.

                Args:
                    is_final: True when this is the last thinking block
                        (agent finished with only thinking, no answer text).
                        In that case, emit as plain text since the thinking
                        IS the answer.
                """
                if not thinking_buffer:
                    return

                raw_thinking = "".join(thinking_buffer)
                thinking_buffer.clear()

                # If thinking-only (no answer text follows), emit as plain
                # text — the reasoning IS the answer.
                if is_final:
                    if _HAS_REFINER:
                        refined = await refine_async(raw_thinking)
                    else:
                        refined = raw_thinking
                    yield _openai_chunk(req_id, model, refined)
                    return

                # Normal case: thinking followed by tools/answer.
                # Refine and wrap in a collapsible block.
                if _HAS_REFINER:
                    refined = await refine_async(raw_thinking)
                    # Emit as italic status text — unobtrusive but informative
                    yield _openai_chunk(
                        req_id, model,
                        f"*💭 {refined}*\n\n"
                    )
                else:
                    # No refiner — emit truncated raw thinking
                    truncated = raw_thinking[:500]
                    if len(raw_thinking) > 500:
                        truncated += "…"
                    yield _openai_chunk(
                        req_id, model,
                        f"*💭 {truncated}*\n\n"
                    )

            while True:
                try:
                    item = await loop.run_in_executor(
                        None, lambda: token_queue.get(timeout=5)
                    )
                except queue.Empty:
                    # Keep connection alive during long tool executions
                    yield ": heartbeat\n\n"
                    continue

                if item is None:
                    # Agent finished — flush any remaining thinking.
                    # If no answer text was emitted, thinking IS the answer.
                    async for chunk in _flush_thinking(is_final=not has_answer_text):
                        yield chunk
                    break

                event_type, data = item

                if event_type == "thinking":
                    # Buffer thinking tokens — they'll be refined and
                    # emitted when the block closes (tool/text/finish).
                    thinking_buffer.append(data)

                elif event_type == "tool":
                    # Flush (refine + emit) buffered thinking before tool
                    async for chunk in _flush_thinking():
                        yield chunk

                    tool_count += 1
                    tool_name = data.get("tool", "unknown")
                    tool_input = data.get("input", "")
                    # Extract a short preview of the tool input
                    input_preview = tool_input[:120].replace("\n", " ")
                    if len(tool_input) > 120:
                        input_preview += "…"
                    yield _openai_chunk(
                        req_id, model,
                        f"🔧 *Using {tool_name}*\n\n"
                    )

                elif event_type == "text":
                    # Flush (refine + emit) buffered thinking before answer
                    async for chunk in _flush_thinking():
                        yield chunk
                    if not has_answer_text and tool_count > 0:
                        yield _openai_chunk(req_id, model, "\n---\n\n**Answer:**\n\n")
                    has_answer_text = True
                    yield _openai_chunk(req_id, model, data)

                last_event_type = event_type

            # If agent errored and produced no streamed text, send error.
            # Check both response_text and reasoning_text since Venice GLM
            # may send the answer entirely as reasoning tokens.
            has_streamed = result_holder["streamed_text"] or result_holder.get("reasoning_text", "")
            if result_holder["error"] and not has_streamed:
                yield _openai_chunk(
                    req_id, model, f"\n\nError: {result_holder['error']}"
                )

            # Append inline activity log at end of response
            elapsed = round(time.time() - start_time, 2)
            # Thinking is always emitted inline (refined or raw) via
            # _flush_thinking — never duplicate it in the footer.
            reasoning_for_log = ""
            inline_log = _format_inline_log(
                result_holder["tool_events"], elapsed,
                query=user_message, model=model,
                reasoning=reasoning_for_log,
                metrics=result_holder.get("metrics"),
            )
            yield _openai_chunk(req_id, model, inline_log)

            yield _openai_chunk(req_id, model, "", finish=True)
            yield "data: [DONE]\n\n"

            # Store activity log (reads from snapshot, not live capture)
            _store_log(
                req_id,
                {
                    "query": user_message,
                    "model": model,
                    "response": result_holder.get("text", ""),
                    "error": result_holder.get("error"),
                    "tool_events": result_holder["tool_events"],
                    "streamed_text": result_holder["streamed_text"],
                    "elapsed": round(time.time() - start_time, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Write metrics to JSONL log file
            _write_metrics_jsonl(
                req_id, model, user_message, elapsed,
                result_holder.get("metrics"),
                result_holder["tool_events"],
            )

        return StreamingResponse(_generate_sse(), media_type="text/event-stream")

    # ── Non-streaming mode ───────────────────────────────────────
    # Offload to a thread so the asyncio event loop stays responsive
    # for health checks, SSE heartbeats, and new request acceptance.
    def _sync_non_streaming():
        with _agent_lock:
            stream_capture.activate()
            try:
                answer, metrics = _dispatch_agent(model, user_message, chat_messages=chat_messages)
            except Exception as exc:
                logger.exception("Agent error in /v1/chat/completions [%s]", req_id)
                raise
            finally:
                # Snapshot captured data while still under lock
                captured_tool_events = list(stream_capture.tool_events)
                captured_all_text = "".join(stream_capture.response_text)
                captured_reasoning = "".join(stream_capture.reasoning_text)
                stream_capture.deactivate()
        return answer, metrics, captured_tool_events, captured_all_text, captured_reasoning

    try:
        answer, metrics, captured_tool_events, captured_all_text, captured_reasoning = await asyncio.to_thread(
            _sync_non_streaming
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "server_error"}},
        )

    elapsed = round(time.time() - start_time, 2)
    # Thinking is always shown inline (refined or truncated) — never
    # duplicate it in the footer.
    reasoning_for_log = ""
    inline_log = _format_inline_log(
        captured_tool_events, elapsed,
        query=user_message, model=model,
        reasoning=reasoning_for_log,
        metrics=metrics,
    )

    # Build a pleasant output with thinking, tool actions, and answer.
    parts: list[str] = []

    # Wrap reasoning in a collapsible block if present AND distinct from answer.
    # When Venice GLM sends only reasoning tokens, _dispatch_agent promotes
    # reasoning to the answer text — in that case don't duplicate it.
    reasoning_is_answer = (
        captured_reasoning.strip()
        and answer.strip() == captured_reasoning.strip()
    )
    if captured_reasoning.strip() and not reasoning_is_answer:
        # Refine the thinking via fast LLM if available
        if _HAS_REFINER:
            refined_reasoning = await refine_async(captured_reasoning)
        else:
            refined_reasoning = captured_reasoning[:500]
            if len(captured_reasoning) > 500:
                refined_reasoning += "…"
        parts.append(f"*💭 {refined_reasoning}*\n\n")

    # Show tool calls as unobtrusive italic lines
    if captured_tool_events:
        for ev in captured_tool_events:
            tool_name = ev.get("tool", "unknown")
            parts.append(f"🔧 *Using {tool_name}*\n\n")
        parts.append("\n---\n\n**Answer:**\n\n")

    parts.append(answer)
    parts.append(inline_log)
    answer_with_log = "".join(parts)

    _store_log(
        req_id,
        {
            "query": user_message,
            "model": model,
            "response": answer,
            "error": None,
            "tool_events": captured_tool_events,
            "streamed_text": captured_all_text,
            "elapsed": round(time.time() - start_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Write metrics to JSONL log file
    _write_metrics_jsonl(
        req_id, model, user_message, elapsed,
        metrics, captured_tool_events,
    )

    # Extract token usage from metrics if available
    usage_data = _extract_usage(metrics)

    return JSONResponse(
        {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_with_log},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_data,
        }
    )


# ── Public activity log endpoint ─────────────────────────────────────


@app.get("/logs/{request_id}", response_class=HTMLResponse)
async def get_request_log_page(request_id: str):
    """Human-readable HTML page showing what the agent did during a request."""
    entry = _get_log(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Log not found")

    # Build tool events table
    tool_rows = ""
    for i, ev in enumerate(entry.get("tool_events", []), 1):
        t = datetime.fromtimestamp(ev["time"], tz=timezone.utc).strftime("%H:%M:%S")
        tool_rows += (
            f"<tr><td>{i}</td><td><code>{html.escape(ev['tool'])}</code></td>"
            f"<td><pre>{html.escape(ev.get('input', ''))}</pre></td>"
            f"<td>{t}</td></tr>\n"
        )

    if not tool_rows:
        tool_rows = '<tr><td colspan="4">No tool calls recorded</td></tr>'

    escaped_query = html.escape(entry.get("query", ""))
    escaped_response = html.escape(entry.get("response", "") or "")
    escaped_streamed = html.escape(entry.get("streamed_text", "") or "")
    error_block = ""
    if entry.get("error"):
        error_block = (
            f'<div style="background:#fee;padding:12px;border-radius:6px;">'
            f"<strong>Error:</strong> {html.escape(entry['error'])}</div>"
        )

    page = f"""\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Agent Log — {html.escape(request_id)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; font-size: 1.4em; }}
  h2 {{ color: #8b949e; font-size: 1.1em; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
  th {{ background: #161b22; color: #8b949e; }}
  tr:nth-child(even) {{ background: #161b22; }}
  pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 0.85em; }}
  code {{ color: #79c0ff; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
  .response {{ background: #161b22; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.6; }}
  .thinking {{ background: #1c2128; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; color: #8b949e; font-size: 0.9em; max-height: 600px; overflow-y: auto; }}
</style>
</head><body>
<h1>Agent Activity Log</h1>
<p class="meta">
  <strong>Request:</strong> <code>{html.escape(request_id)}</code><br>
  <strong>Model:</strong> <code>{html.escape(entry.get('model', ''))}</code><br>
  <strong>Time:</strong> {html.escape(entry.get('timestamp', ''))}<br>
  <strong>Elapsed:</strong> {entry.get('elapsed', 0)}s
</p>
{error_block}
<h2>Query</h2>
<div class="response">{escaped_query}</div>

<h2>Tool Calls ({len(entry.get('tool_events', []))})</h2>
<table>
<tr><th>#</th><th>Tool</th><th>Input</th><th>Time</th></tr>
{tool_rows}
</table>

<h2>Agent Thinking (streamed tokens)</h2>
<details>
<summary>Click to expand ({len(entry.get('streamed_text', ''))} chars)</summary>
<div class="thinking">{escaped_streamed}</div>
</details>

<h2>Final Response</h2>
<div class="response">{escaped_response}</div>
</body></html>"""

    return HTMLResponse(page)
