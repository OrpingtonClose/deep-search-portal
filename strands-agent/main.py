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
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import log_viewer
from conversation import (
    ChatMessage,
    extract_user_message,
    load_conversation_history,
)
from plugins.tool_display import format_footer, tool_label
from streaming import generate_sse

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

logger = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────

_single_agent = None
_multi_agent = None
_deep_agent = None
_mcp_clients: list = []
_multi_researcher = None
_agent_lock = threading.Lock()

# Lazy-loaded plugin references (set in lifespan from agent module)
_stream_capture = None
_thought_refiner = None
_budget_plugin = None


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
    # Fallback: use ToolDisplayPlugin's footer formatter
    return format_footer(tool_events, elapsed)


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
    global _stream_capture, _thought_refiner, _budget_plugin

    from agent import (
        _enter_mcp_clients,
        _setup_otel,
        budget_plugin,
        create_deep_agent_instance,
        create_multi_agent,
        create_single_agent,
        stream_capture,
        thought_refiner,
    )
    from tools import get_all_mcp_clients

    # Store plugin references for use in request handlers
    _stream_capture = stream_capture
    _thought_refiner = thought_refiner
    _budget_plugin = budget_plugin

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
    version="0.3.0",
    lifespan=lifespan,
)
# Wire up extracted log viewer router
log_viewer.configure(_get_log)
app.include_router(log_viewer.router)


# ── Request / Response models ────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., description="The research query to send to the agent")


class QueryResponse(BaseModel):
    query: str
    response: str
    mode: str
    elapsed_seconds: float


class ChatCompletionRequest(BaseModel):
    model: str = "strands-venice-single"
    messages: list[ChatMessage] = []
    stream: bool = False

    model_config = {"extra": "allow"}


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
        from agent import reset_plugins

        _single_agent.messages.clear()
        reset_plugins()
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
        from agent import reset_plugins

        _multi_agent.messages.clear()
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
        reset_plugins()
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
    from agent import reset_plugins

    metrics_summary = None

    if model == _MODEL_MULTI:
        if _multi_agent is None:
            raise RuntimeError("Multi agent not initialised")
        if chat_messages:
            user_message = load_conversation_history(
                _multi_agent, chat_messages, researcher=_multi_researcher
            )
        else:
            _multi_agent.messages.clear()
            if _multi_researcher is not None:
                _multi_researcher.messages.clear()
        reset_plugins()
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
            user_message = load_conversation_history(
                _deep_agent, chat_messages
            )
        else:
            _deep_agent.messages.clear()
        reset_plugins()
        agent_result = _deep_agent(user_message)
        result = str(agent_result)
        try:
            metrics_summary = agent_result.metrics.get_summary()
        except Exception:
            pass
    elif _single_agent is not None:
        if chat_messages:
            user_message = load_conversation_history(
                _single_agent, chat_messages
            )
        else:
            _single_agent.messages.clear()
        reset_plugins()
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
    if not result.strip() and _stream_capture and (_stream_capture.response_text or _stream_capture.reasoning_text):
        result = "".join(_stream_capture.response_text) or "".join(_stream_capture.reasoning_text)
    return result, metrics_summary


def _run_agent(
    model: str,
    user_message: str,
    chat_messages: list[ChatMessage] | None = None,
) -> tuple[str, dict | None]:
    """Dispatch to the correct agent under lock (convenience wrapper)."""
    with _agent_lock:
        _stream_capture.activate()
        try:
            return _dispatch_agent(model, user_message, chat_messages=chat_messages)
        finally:
            _stream_capture.deactivate()


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
    model = body.model
    stream = body.stream
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    user_message = extract_user_message(body.messages)
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
                token_q = _stream_capture.activate()
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
                    result_holder["tool_events"] = [
                        {k: v for k, v in ev.items() if k != "_tool_use_ref"}
                        for ev in _stream_capture.tool_events
                    ]
                    result_holder["streamed_text"] = "".join(_stream_capture.response_text)
                    result_holder["reasoning_text"] = "".join(_stream_capture.reasoning_text)
                    _stream_capture.deactivate()

        thread = threading.Thread(target=_agent_thread, daemon=True)
        thread.start()

        async def _sse_wrapper():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, queue_ready.wait)
            token_queue = queue_holder["q"]

            async for chunk in generate_sse(
                req_id=req_id,
                model=model,
                token_queue=token_queue,
                thought_refiner=_thought_refiner,
                result_holder=result_holder,
                start_time=start_time,
                format_inline_log=_format_inline_log,
                user_message=user_message,
            ):
                yield chunk

            # Store activity log and metrics after streaming completes
            elapsed = round(time.time() - start_time, 2)
            _store_log(
                req_id,
                {
                    "query": user_message,
                    "model": model,
                    "response": result_holder.get("text", ""),
                    "error": result_holder.get("error"),
                    "tool_events": result_holder["tool_events"],
                    "streamed_text": result_holder["streamed_text"],
                    "elapsed": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            _write_metrics_jsonl(
                req_id, model, user_message, elapsed,
                result_holder.get("metrics"),
                result_holder["tool_events"],
            )

        return StreamingResponse(_sse_wrapper(), media_type="text/event-stream")

    # ── Non-streaming mode ───────────────────────────────────────
    def _sync_non_streaming():
        with _agent_lock:
            _stream_capture.activate()
            try:
                answer, metrics = _dispatch_agent(model, user_message, chat_messages=chat_messages)
            except Exception:
                logger.exception("Agent error in /v1/chat/completions [%s]", req_id)
                raise
            finally:
                captured_tool_events = [
                    {k: v for k, v in ev.items() if k != "_tool_use_ref"}
                    for ev in _stream_capture.tool_events
                ]
                captured_all_text = "".join(_stream_capture.response_text)
                captured_reasoning = "".join(_stream_capture.reasoning_text)
                _stream_capture.deactivate()
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
    reasoning_for_log = ""
    inline_log = _format_inline_log(
        captured_tool_events, elapsed,
        query=user_message, model=model,
        reasoning=reasoning_for_log,
        metrics=metrics,
    )

    # Build formatted response with thinking, tools, and answer
    parts: list[str] = []
    reasoning_is_answer = (
        captured_reasoning.strip()
        and answer.strip() == captured_reasoning.strip()
    )
    if captured_reasoning.strip() and not reasoning_is_answer:
        if _thought_refiner and _thought_refiner.is_available:
            refined_reasoning = await _thought_refiner.refine_async(captured_reasoning)
        else:
            refined_reasoning = captured_reasoning[:500]
            if len(captured_reasoning) > 500:
                refined_reasoning += "…"
        parts.append(f"<details>\n<summary>💭 Thinking</summary>\n\n{refined_reasoning}\n\n</details>\n\n")

    if captured_tool_events:
        for ev in captured_tool_events:
            label = tool_label(ev.get("tool", "unknown"), ev.get("input", ""))
            parts.append(f"🔧 *{label}*\n\n")
        parts.append("\n---\n\n")

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
            "elapsed": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_metrics_jsonl(req_id, model, user_message, elapsed, metrics, captured_tool_events)

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
