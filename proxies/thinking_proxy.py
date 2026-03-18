#!/usr/bin/env python3
"""
Thinking Proxy for Mistral Large.

An OpenAI-compatible proxy that sits between Open WebUI and the Mistral API.
It injects a "think step-by-step" system instruction, then wraps the model's
reasoning output in <think>...</think> tags before streaming it back to the client.

The proxy implements a two-phase response:
  Phase 1 (Thinking): The model is asked to reason step-by-step in a structured way.
                       The proxy streams this wrapped in <think>...</think>.
  Phase 2 (Answer):   The model's final answer is streamed normally.

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes them through WITHOUT thinking
  - For real chat requests: injects thinking instructions, streams via state machine
  - All responses are ALWAYS streamed (SSE) regardless of client's stream param

Runs as a FastAPI app under uvicorn in a screen session.
"""

import json
import os
import time
import traceback
import uuid
from typing import Annotated, Any, AsyncGenerator, Optional, TypedDict

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from langgraph.graph import END, START, StateGraph

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
)

# --- Logging ---
LOG_DIR = os.getenv("THINKING_PROXY_LOG_DIR", "/opt/thinking_proxy_logs")
log = setup_logging("thinking-proxy", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
LISTEN_PORT = env_int("THINKING_PROXY_PORT", 9100, minimum=1)

log.info(f"Config: upstream={UPSTREAM_BASE}, model={UPSTREAM_MODEL}, port={LISTEN_PORT}")

# --- Request tracking ---
tracker = RequestTracker()

# The thinking instruction injected into the system prompt
THINKING_INSTRUCTION = """
IMPORTANT — STRUCTURED THINKING PROTOCOL:

You MUST structure every response in exactly two clearly labeled sections:

<THINKING>
[Your detailed step-by-step reasoning goes here. Break down the problem, consider different angles, 
evaluate evidence, check your logic, explore alternatives. Be thorough and genuine in your analysis.
This section should reflect your actual reasoning process — not a performance.]
</THINKING>

<ANSWER>
[Your final, polished response to the user goes here. This is what the user primarily cares about.
It should be comprehensive, well-structured, and directly address their question.]
</ANSWER>

You MUST always include both sections. The THINKING section comes first, then the ANSWER section.
Never skip the THINKING section. Never merge them. Always use the exact tags shown above.
"""


def inject_thinking_prompt(messages: list[dict]) -> list[dict]:
    """Inject the thinking instruction into the message list."""
    messages = [m.copy() for m in messages]

    has_system = False
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i]["content"] = m["content"] + "\n\n" + THINKING_INSTRUCTION
            has_system = True
            break

    if not has_system:
        messages.insert(0, {"role": "system", "content": THINKING_INSTRUCTION})

    return messages


async def stream_thinking_response(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the upstream response, transforming <THINKING>/<ANSWER> tags
    into <think>/<content> format that Open WebUI renders natively.

    Uses LangGraph for traced request preparation, then streams the
    upstream response through a token-level state machine.
    """
    request_id = f"chatcmpl-think-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    token_count = 0
    start_time = time.monotonic()
    finish_sent = False  # Guard against duplicate finish chunks

    # --- Phase 1: Build request via LangGraph (traced) ---
    graph_input: dict[str, Any] = {
        "req_id": req_id,
        "messages": messages,
        "original_body": original_body,
        "upstream_body": {},
        "model_id": original_body.get("model", UPSTREAM_MODEL),
        "tools_stripped": False,
        "token_count": 0,
        "final_phase": "",
        "elapsed": 0.0,
        "error": "",
        "progress_log": [],
        "phase": "build_request",
    }
    config = {"configurable": {"thread_id": req_id}}
    graph_state = await _thinking_graph.ainvoke(graph_input, config=config)

    upstream_body = graph_state["upstream_body"]
    model_id = graph_state["model_id"]

    def make_chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # State machine for parsing the stream
    buffer = ""
    phase = "pre_think"  # pre_think -> thinking -> between -> answering -> done
    think_opened = False
    think_closed = False

    try:
        client = http_client()
        log.info(f"[{req_id}] Connecting to upstream: {UPSTREAM_BASE}/chat/completions")

        async with client.stream(
            "POST",
            f"{UPSTREAM_BASE}/chat/completions",
            json=upstream_body,
            headers={
                "Authorization": f"Bearer {UPSTREAM_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://deep-search.uk",
                "X-Title": "Deep Search Thinking Proxy",
            },
        ) as resp:
            elapsed_connect = time.monotonic() - start_time
            log.info(
                f"[{req_id}] Upstream responded: status={resp.status_code}, "
                f"connect_time={elapsed_connect:.2f}s"
            )

            if resp.status_code != 200:
                error_body = await resp.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:1000]
                log.error(f"[{req_id}] Upstream error {resp.status_code}: {error_text}")

                error_msg = (
                    f"**Thinking Proxy Error**\n\n"
                    f"Upstream returned HTTP {resp.status_code}.\n\n"
                    f"```\n{error_text}\n```\n\n"
                    f"_Model: {UPSTREAM_MODEL} via {UPSTREAM_BASE}_"
                )
                yield make_chunk(error_msg)
                yield make_chunk("", finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            log.info(f"[{req_id}] Streaming started, phase={phase}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    elapsed_total = time.monotonic() - start_time
                    log.info(
                        f"[{req_id}] Stream complete: tokens={token_count}, "
                        f"total_time={elapsed_total:.2f}s, final_phase={phase}"
                    )

                    # Flush remaining buffer before closing
                    if buffer.strip():
                        if think_opened and not think_closed:
                            yield make_chunk(buffer)
                        else:
                            yield make_chunk(buffer)
                        buffer = ""

                    # Ensure we close thinking tag if still open
                    if think_opened and not think_closed:
                        yield make_chunk("\n</think>\n\n")
                    if not finish_sent:
                        yield make_chunk("", finish_reason="stop")
                        finish_sent = True
                    yield "data: [DONE]\n\n"
                    return

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as e:
                    log.warning(f"[{req_id}] Bad JSON chunk: {e} — payload: {payload[:200]}")
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                token = delta.get("content", "")

                if not token:
                    fr = choices[0].get("finish_reason")
                    if fr:
                        log.info(f"[{req_id}] Finish reason received: {fr}")
                        # Flush buffer
                        if buffer.strip():
                            yield make_chunk(buffer)
                            buffer = ""
                        if think_opened and not think_closed:
                            yield make_chunk("\n</think>\n\n")
                            think_closed = True
                        if not finish_sent:
                            yield make_chunk("", finish_reason="stop")
                            finish_sent = True
                    continue

                token_count += 1
                buffer += token

                # --- State machine ---

                if phase == "pre_think":
                    if "<THINKING>" in buffer:
                        idx = buffer.index("<THINKING>") + len("<THINKING>")
                        remaining = buffer[idx:]
                        buffer = ""
                        yield make_chunk("<think>\n")
                        think_opened = True
                        log.info(
                            f"[{req_id}] Phase: pre_think -> thinking "
                            f"(found <THINKING> at token {token_count})"
                        )
                        if remaining.strip():
                            if "</THINKING>" in remaining:
                                think_part = remaining[:remaining.index("</THINKING>")]
                                buffer = remaining[remaining.index("</THINKING>"):]
                                if think_part.strip():
                                    yield make_chunk(think_part)
                                phase = "between"
                                log.info(f"[{req_id}] Phase: thinking -> between (immediate close)")
                            else:
                                yield make_chunk(remaining)
                                buffer = ""
                        phase = "thinking" if phase == "pre_think" else phase
                    else:
                        if len(buffer) > 200 and "<THINK" not in buffer:
                            log.warning(
                                f"[{req_id}] Model not following protocol after "
                                f"200 chars, forcing thinking phase"
                            )
                            yield make_chunk("<think>\n")
                            think_opened = True
                            yield make_chunk(buffer)
                            buffer = ""
                            phase = "thinking"

                elif phase == "thinking":
                    if "</THINKING>" in buffer:
                        idx = buffer.index("</THINKING>")
                        think_content = buffer[:idx]
                        buffer = buffer[idx + len("</THINKING>"):]
                        if think_content:
                            yield make_chunk(think_content)
                        yield make_chunk("\n</think>\n\n")
                        think_closed = True
                        phase = "between"
                        log.info(f"[{req_id}] Phase: thinking -> between (at token {token_count})")
                    elif len(buffer) > 50:
                        emit = buffer[:-20]
                        buffer = buffer[-20:]
                        if emit:
                            yield make_chunk(emit)

                elif phase == "between":
                    if "<ANSWER>" in buffer:
                        idx = buffer.index("<ANSWER>") + len("<ANSWER>")
                        remaining = buffer[idx:]
                        buffer = ""
                        log.info(f"[{req_id}] Phase: between -> answering (at token {token_count})")
                        if remaining.strip():
                            if "</ANSWER>" in remaining:
                                answer_part = remaining[:remaining.index("</ANSWER>")]
                                if answer_part.strip():
                                    yield make_chunk(answer_part)
                                phase = "done"
                                log.info(f"[{req_id}] Phase: answering -> done (immediate close)")
                            else:
                                yield make_chunk(remaining)
                                buffer = ""
                                phase = "answering"
                        else:
                            phase = "answering"
                    elif len(buffer) > 200:
                        log.warning(
                            f"[{req_id}] No <ANSWER> tag found after 200 chars, "
                            f"streaming directly"
                        )
                        emit = buffer[:-20]
                        buffer = buffer[-20:]
                        if emit.strip():
                            yield make_chunk(emit)
                        phase = "answering"

                elif phase == "answering":
                    if "</ANSWER>" in buffer:
                        idx = buffer.index("</ANSWER>")
                        answer_content = buffer[:idx]
                        buffer = ""
                        if answer_content:
                            yield make_chunk(answer_content)
                        phase = "done"
                        log.info(f"[{req_id}] Phase: answering -> done (at token {token_count})")
                    elif len(buffer) > 50:
                        emit = buffer[:-20]
                        buffer = buffer[-20:]
                        if emit:
                            yield make_chunk(emit)

                elif phase == "done":
                    buffer = ""

    except httpx.ConnectError as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Connection error after {elapsed:.2f}s: {e}")
        error_msg = (
            f"**Thinking Proxy \u2014 Connection Error**\n\n"
            f"Could not connect to upstream: `{UPSTREAM_BASE}`\n\n"
            f"```\n{str(e)}\n```"
        )
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except httpx.ReadTimeout as e:
        elapsed = time.monotonic() - start_time
        log.error(
            f"[{req_id}] Read timeout after {elapsed:.2f}s "
            f"(tokens so far: {token_count}): {e}"
        )
        timeout_msg = (
            f"\n\n**Thinking Proxy \u2014 Timeout**\n\n"
            f"Upstream stopped responding after {elapsed:.1f}s "
            f"({token_count} tokens received).\n"
            f"The model may be overloaded. Try again."
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(timeout_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except httpx.TimeoutException as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Timeout after {elapsed:.2f}s: {e}")
        error_msg = (
            f"**Thinking Proxy \u2014 Timeout**\n\n"
            f"Request timed out after {elapsed:.1f}s.\n\n"
            f"```\n{str(e)}\n```"
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Unhandled exception after {elapsed:.2f}s: {e}\n{tb}")
        error_msg = (
            f"**Thinking Proxy \u2014 Internal Error**\n\n"
            f"An unexpected error occurred:\n\n"
            f"```\n{type(e).__name__}: {str(e)}\n```\n\n"
            f"_Check proxy logs for details (request: {req_id})_"
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# LangGraph State & Graph for Thinking Pipeline
# ============================================================================


def _think_append_log(left: list[str], right: list[str]) -> list[str]:
    """Reducer: append new progress entries."""
    return left + right


class ThinkingState(TypedDict):
    """LangGraph state for the thinking proxy pipeline."""
    req_id: str
    messages: list[dict]
    original_body: dict
    # Prepared by build_request node
    upstream_body: dict
    model_id: str
    tools_stripped: bool
    # Result from stream_transform node
    token_count: int
    final_phase: str
    elapsed: float
    error: str
    # Progress log for traceability
    progress_log: Annotated[list[str], _think_append_log]
    phase: str  # "build_request", "stream_transform", "done"


async def think_node_build_request(state: ThinkingState) -> dict:
    """Prepare the upstream request body with thinking injection."""
    messages = state["messages"]
    original_body = state["original_body"]
    req_id = state["req_id"]

    upstream_body = {
        **original_body,
        "model": UPSTREAM_MODEL,
        "messages": inject_thinking_prompt(messages),
        "stream": True,
    }
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    tools_stripped = any(
        k in original_body for k in ("tools", "tool_choice", "functions", "function_call")
    )

    log.info(
        f"[{req_id}] THINKING upstream request: model={UPSTREAM_MODEL}, "
        f"messages={len(messages)}, stream=True, tools_stripped={tools_stripped}"
    )
    log.debug(f"[{req_id}] User message (last): {messages[-1].get('content', '')[:200]}")

    return {
        "upstream_body": upstream_body,
        "model_id": original_body.get("model", UPSTREAM_MODEL),
        "tools_stripped": tools_stripped,
        "progress_log": [f"Built upstream request (tools_stripped={tools_stripped})"],
        "phase": "stream_transform",
    }


def build_thinking_graph() -> Any:
    """Build the thinking proxy LangGraph.

    Graph topology (2-node pipeline)::

        START -> build_request -> END

    Note: The streaming transformation phase (stream_transform) is not included
    in the graph because it is inherently a token-level streaming operation that
    must be implemented as an async generator. The graph provides traceability
    for the request preparation phase. The stream_transform phase is traced
    separately within stream_thinking_response().
    """
    graph = StateGraph(ThinkingState)
    graph.add_node("build_request", think_node_build_request)
    graph.add_edge(START, "build_request")
    graph.add_edge("build_request", END)
    return graph.compile()


_thinking_graph = build_thinking_graph()


# ============================================================================
# FastAPI App
# ============================================================================

app = create_app("Thinking Proxy")

register_standard_routes(
    app,
    service_name="thinking-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "upstream_model": UPSTREAM_MODEL,
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Return the thinking model in OpenAI format."""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "mistral-large-thinking",
            "object": "model",
            "created": 1700000000,
            "owned_by": "thinking-proxy",
            "name": "Mistral Large (Thinking)",
        }]
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests with thinking injection."""
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        log.error(f"[{req_id}] Failed to parse request body: {e}")
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

    client_wants_stream = body.get("stream", False)
    utility = is_utility_request(messages)

    log.info(
        f"[{req_id}] New request: client_stream={client_wants_stream}, "
        f"messages={len(messages)}, model={body.get('model', '?')}, utility={utility}"
    )

    # ALWAYS stream — the thinking proxy needs streaming for the state machine
    if not client_wants_stream:
        log.info(f"[{req_id}] Overriding stream=false -> stream=true (proxy always streams)")

    tracker.start(req_id, stream=True, utility=utility, messages=len(messages))

    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH (utility request — no thinking injection)")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", UPSTREAM_MODEL),
            tracker=tracker,
            log=log,
            extra_headers={
                "HTTP-Referer": "https://deep-search.uk",
                "X-Title": "Deep Search Thinking Proxy",
            },
        )
    else:
        log.info(f"[{req_id}] Routing to THINKING pipeline")
        generator = stream_thinking_response(messages, body, req_id)

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
    log.info("Starting Thinking Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
