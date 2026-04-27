#!/usr/bin/env python3
"""
Kimi Proxy — Routes to a self-hosted Kimi K2.6 Heretic (GGUF) on RunPod.

A lightweight FastAPI proxy that forwards OpenAI-compatible requests to a
Kimi K2.6 Heretic instance running on RunPod via llama-server.  The proxy
handles:
  - Request forwarding with SSE streaming passthrough
  - Pod status awareness (helpful error when pod is down)
  - Utility request detection (title/tag gen — passed through)
  - Health endpoint reporting upstream availability

The RunPod pod is managed separately via scripts/runpod/manage_kimi.py.
Set KIMI_RUNPOD_URL to the pod's proxy endpoint.

Runs as a FastAPI app under uvicorn in a screen session (port 9960).
"""

import json
import os
import time
import traceback
import uuid
from typing import AsyncGenerator

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
    setup_logging,
)

# --- Logging ---
LOG_DIR = os.getenv("KIMI_PROXY_LOG_DIR", "/opt/kimi_proxy_logs")
log = setup_logging("kimi-proxy", LOG_DIR)

# --- Configuration ---
# The RunPod pod endpoint URL. Get it from: python manage_kimi.py endpoint
# Format: https://<pod-id>-8000.proxy.runpod.net/v1
_raw_upstream = os.getenv("KIMI_RUNPOD_URL", "").rstrip("/")
UPSTREAM_BASE = _raw_upstream                       # includes /v1 for API calls
UPSTREAM_ROOT = _raw_upstream.removesuffix("/v1")    # without /v1 for health checks
LISTEN_PORT = env_int("KIMI_PROXY_PORT", 9960, minimum=1)
UPSTREAM_TIMEOUT = env_int("KIMI_UPSTREAM_TIMEOUT", 300, minimum=10)

MODEL_ID = "kimi-k26-heretic"

if not UPSTREAM_BASE:
    log.warning(
        "KIMI_RUNPOD_URL not set. Proxy will return errors until configured. "
        "Run: python scripts/runpod/manage_kimi.py endpoint"
    )

log.info(
    f"Config: upstream={UPSTREAM_BASE or '(not set)'}, port={LISTEN_PORT}, "
    f"timeout={UPSTREAM_TIMEOUT}s"
)

# --- Request tracking ---
tracker = RequestTracker()


# ============================================================================
# Upstream health check
# ============================================================================

async def _check_upstream() -> dict:
    """Check if the RunPod Kimi server is reachable."""
    if not UPSTREAM_BASE:
        return {"status": "not_configured", "message": "KIMI_RUNPOD_URL not set"}
    try:
        client = http_client()
        resp = await client.get(
            f"{UPSTREAM_ROOT}/health",
            timeout=10.0,
        )
        if resp.status_code == 200:
            return {"status": "ok", "message": "Kimi server is healthy"}
        return {"status": "unhealthy", "message": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"status": "offline", "message": "Pod appears to be stopped. Run: python manage_kimi.py start"}
    except httpx.TimeoutException:
        return {"status": "loading", "message": "Pod is starting or model is loading"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# Streaming forwarder
# ============================================================================

async def _forward_streaming(
    body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Forward a streaming request to the RunPod Kimi server."""
    body_copy = {**body}
    body_copy["stream"] = True
    created = int(time.time())
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    client = http_client()
    try:
        async with client.stream(
            "POST",
            f"{UPSTREAM_BASE}/chat/completions",
            json=body_copy,
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=30.0),
        ) as resp:
            if resp.status_code != 200:
                error_body = (await resp.aread()).decode("utf-8", errors="replace")
                log.error(f"[{req_id}] Upstream error: {resp.status_code} — {error_body[:500]}")
                error_chunk = make_sse_chunk(
                    content=f"[Kimi server error: HTTP {resp.status_code}]",
                    request_id=cmpl_id,
                    created=created,
                    model_id=MODEL_ID,
                    finish_reason="stop",
                )
                yield error_chunk
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if not line.strip():
                    yield "\n"
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    # Rewrite model name to our canonical ID
                    try:
                        chunk = json.loads(data_str)
                        chunk["model"] = MODEL_ID
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        yield f"{line}\n"
                else:
                    yield f"{line}\n"

    except httpx.ConnectError:
        log.error(f"[{req_id}] Cannot connect to Kimi server — pod may be stopped")
        yield make_sse_chunk(
            content=(
                "[Kimi K2.6 Heretic server is offline. "
                "Start it with: python scripts/runpod/manage_kimi.py start]"
            ),
            request_id=cmpl_id,
            created=created,
            model_id=MODEL_ID,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
    except httpx.TimeoutException:
        log.error(f"[{req_id}] Upstream timeout after {UPSTREAM_TIMEOUT}s")
        yield make_sse_chunk(
            content="[Kimi server timeout — model may still be loading]",
            request_id=cmpl_id,
            created=created,
            model_id=MODEL_ID,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
    except Exception as e:
        log.error(f"[{req_id}] Forward error: {traceback.format_exc()}")
        yield make_sse_chunk(
            content=f"[Kimi proxy error: {e}]",
            request_id=cmpl_id,
            created=created,
            model_id=MODEL_ID,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
    finally:
        tracker.finish(req_id)


async def _forward_json(body: dict, req_id: str) -> JSONResponse:
    """Forward a non-streaming request to the RunPod Kimi server."""
    body_copy = {**body}
    body_copy["stream"] = False

    client = http_client()
    try:
        resp = await client.post(
            f"{UPSTREAM_BASE}/chat/completions",
            json=body_copy,
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=30.0),
        )
        if resp.status_code != 200:
            tracker.finish(req_id)
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": {"message": f"Upstream error: {resp.text[:500]}", "type": "upstream_error"}},
            )
        result = resp.json()
        result["model"] = MODEL_ID
        tracker.finish(req_id)
        return JSONResponse(content=result)

    except httpx.ConnectError:
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "Kimi K2.6 Heretic server is offline. Start it with: python scripts/runpod/manage_kimi.py start",
                    "type": "service_unavailable",
                }
            },
        )
    except Exception as e:
        tracker.finish(req_id)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "proxy_error"}},
        )


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Kimi Proxy")

register_standard_routes(
    app,
    service_name="kimi-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE or "(not configured)",
        "model": MODEL_ID,
    },
)


@app.get("/v1/models")
async def list_models():
    upstream = await _check_upstream()
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1700000000,
                "owned_by": "kimi-proxy",
                "meta": {
                    "description": "Kimi K2.6 Heretic (abliterated/uncensored) — self-hosted GGUF on RunPod",
                    "upstream_status": upstream["status"],
                },
            },
        ],
    })


@app.get("/v1/kimi/status")
async def kimi_status():
    """Extra endpoint to check RunPod pod health."""
    upstream = await _check_upstream()
    return JSONResponse({
        "model": MODEL_ID,
        "upstream_url": UPSTREAM_BASE or "(not configured)",
        **upstream,
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not UPSTREAM_BASE:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        "KIMI_RUNPOD_URL not configured. "
                        "Set it to your RunPod pod endpoint. "
                        "Get it with: python scripts/runpod/manage_kimi.py endpoint"
                    ),
                    "type": "service_unavailable",
                }
            },
        )

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )

    messages = body.get("messages", [])
    req_id = f"kimi-{uuid.uuid4().hex[:8]}"
    tracker.start(req_id, model=MODEL_ID)

    log.info(
        f"[{req_id}] Request: model={body.get('model')}, "
        f"messages={len(messages)}, stream={body.get('stream', True)}"
    )

    # Utility requests — forward directly (llama-server handles them fine)
    # Real chat — also forward directly (no agentic loop needed here;
    # Kimi's native tool-calling happens inside llama-server)
    if body.get("stream", True):
        gen = _forward_streaming(body, req_id)
        return StreamingResponse(gen, media_type="text/event-stream")
    else:
        return await _forward_json(body, req_id)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
