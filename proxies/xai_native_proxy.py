#!/usr/bin/env python3
"""
xAI Native Proxy — OpenAI-compatible proxy for all xAI/Grok models.

Routes directly to xAI's Chat Completions API (https://api.x.ai/v1).
Supports:
  - Individual model access (grok-3, grok-4.20, grok-code-fast-1, etc.)
  - Catch-all race modes that query multiple xAI models in parallel and
    pick the best response (like G0DM0D3's ULTRAPLINIAN tiers).
  - Special model options: reasoning, web_search, x_search via the
    Responses API (/v1/responses) for models that support it.

Port: 9700 (configurable via XAI_PROXY_PORT)
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from shared import (
    INGEST_DB_PATH,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_ingest_routes,
    register_standard_routes,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("XAI_PROXY_LOG_DIR", "/opt/xai_proxy_logs")
log = setup_logging("xai-native-proxy", LOG_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
XAI_BASE = "https://api.x.ai/v1"
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
LISTEN_PORT = env_int("XAI_PROXY_PORT", 9700, minimum=1)
INGEST_DB = os.getenv("INGEST_DB", INGEST_DB_PATH)

MAX_CONCURRENT_MODELS = env_int("XAI_MAX_CONCURRENT", 6, minimum=1)
MODEL_TIMEOUT = int(os.getenv("XAI_MODEL_TIMEOUT", "90"))

if not XAI_API_KEY:
    log.warning("XAI_API_KEY not set — xAI native proxy will fail on model calls")

# ---------------------------------------------------------------------------
# Request tracking
# ---------------------------------------------------------------------------
tracker = RequestTracker()

# ============================================================================
# xAI MODEL REGISTRY — all available models with their capabilities
# ============================================================================

XAI_MODELS = {
    # --- Grok 3 family (Chat Completions) ---
    "grok-3": {
        "context": 131072,
        "reasoning": False,
        "tool_calling": True,
        "pricing": "$3 / $9 per M tokens",
    },
    "grok-3-fast": {
        "context": 131072,
        "reasoning": False,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    "grok-3-mini": {
        "context": 131072,
        "reasoning": True,
        "reasoning_effort": True,  # supports low/high
        "tool_calling": True,
        "pricing": "$0.30 / $0.50 per M tokens",
    },
    # --- Grok 4 family (Chat Completions) ---
    "grok-4-0709": {
        "context": 2000000,
        "reasoning": True,
        "tool_calling": True,
        "pricing": "$2 / $6 per M tokens",
    },
    "grok-4-1-fast-reasoning": {
        "context": 2000000,
        "reasoning": True,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    "grok-4-1-fast-non-reasoning": {
        "context": 2000000,
        "reasoning": False,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    "grok-4-fast-reasoning": {
        "context": 2000000,
        "reasoning": True,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    "grok-4-fast-non-reasoning": {
        "context": 2000000,
        "reasoning": False,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    # --- Grok 4.20 family (Chat Completions + Responses API) ---
    "grok-4.20-0309-reasoning": {
        "context": 2000000,
        "reasoning": True,
        "tool_calling": True,
        "responses_api": True,  # supports web_search + x_search
        "pricing": "$2 / $6 per M tokens",
    },
    "grok-4.20-0309-non-reasoning": {
        "context": 2000000,
        "reasoning": False,
        "tool_calling": True,
        "responses_api": True,
        "pricing": "$2 / $6 per M tokens",
    },
    "grok-4.20-multi-agent-0309": {
        "context": 2000000,
        "reasoning": True,
        "tool_calling": True,
        "responses_api": True,
        "pricing": "$2 / $6 per M tokens",
    },
    # --- Code specialist ---
    "grok-code-fast-1": {
        "context": 131072,
        "reasoning": False,
        "tool_calling": True,
        "pricing": "$0.20 / $0.50 per M tokens",
    },
    # --- Image generation ---
    "grok-imagine-image": {
        "type": "image",
        "pricing": "$0.02 per image",
    },
    "grok-imagine-image-pro": {
        "type": "image",
        "pricing": "$0.07 per image",
    },
    # --- Video generation ---
    "grok-imagine-video": {
        "type": "video",
        "pricing": "$0.10 per second",
    },
}

# ============================================================================
# RACE MODE TIERS — catch-all modes that query multiple models
# ============================================================================

RACE_TIERS = {
    "xai-race-fast": {
        "description": "Race the fastest xAI models",
        "models": [
            "grok-3-fast",
            "grok-4-1-fast-non-reasoning",
            "grok-code-fast-1",
        ],
    },
    "xai-race-smart": {
        "description": "Race the smartest xAI reasoning models",
        "models": [
            "grok-3",
            "grok-4-1-fast-reasoning",
            "grok-4.20-0309-reasoning",
        ],
    },
    "xai-race-all": {
        "description": "Race ALL xAI text models and pick the best response",
        "models": [
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-code-fast-1",
        ],
    },
}

# Text-only models (exclude image/video for chat completions)
TEXT_MODELS = {
    k: v for k, v in XAI_MODELS.items()
    if v.get("type") not in ("image", "video")
}


# ============================================================================
# Scoring (simplified version of G0DM0D3's scoring for race modes)
# ============================================================================

_REFUSAL_PATTERNS = [
    r"(?i)i\s+can(?:'t|not)\s+(?:assist|help|provide|comply)",
    r"(?i)i\s+(?:must|need\s+to)\s+(?:decline|refuse)",
    r"(?i)i'?m\s+(?:not\s+able|unable)\s+to",
    r"(?i)(?:sorry|apolog(?:ize|ies)),?\s+(?:but\s+)?i\s+(?:can(?:'t|not)|won(?:'t|t))",
    r"(?i)against\s+(?:my|our)\s+(?:guidelines|policies|terms)",
    r"(?i)i\s+(?:will|would)\s+not\s+(?:provide|assist|help|comply)",
]

_HEDGE_PHRASES = [
    "i should mention",
    "it's important to note",
    "please consult",
    "i need to warn",
    "i must emphasize",
    "important disclaimer",
    "safety disclaimer",
    "i cannot recommend",
    "i strongly advise against",
]


def score_response(text: str) -> float:
    """Score a response: higher = better. Penalize refusals and hedges."""
    if not text or len(text) < 20:
        return -100.0

    score = 0.0

    # Heavy penalty for refusals
    for pattern in _REFUSAL_PATTERNS:
        if re.search(pattern, text[:500]):
            score -= 50.0

    # Penalty for hedges
    text_lower = text.lower()
    for phrase in _HEDGE_PHRASES:
        if phrase in text_lower:
            score -= 3.0

    # Reward substance
    score += min(len(text) / 100.0, 30.0)  # up to 30 pts for length
    score += text.count("\n") * 0.1  # structure
    score += text.count("```") * 1.0  # code blocks
    score += text.count("- ") * 0.2  # list items
    score += min(text.count("http") * 0.5, 5.0)  # links (max 5 pts)

    return score


# ============================================================================
# Core API call functions
# ============================================================================

async def call_xai(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    extra_body: Optional[dict] = None,
) -> str:
    """Call a single xAI model via Chat Completions API. Returns full text."""
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    # xAI rejects max_completion_tokens, use max_tokens via extra_body
    if extra_body:
        body.update(extra_body)
    else:
        body["max_tokens"] = max_tokens

    client = http_client()
    try:
        resp = await asyncio.wait_for(
            client.post(f"{XAI_BASE}/chat/completions", json=body, headers=headers),
            timeout=MODEL_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning(f"[{req_id}] xAI {model} returned {resp.status_code}: {resp.text[:300]}")
            return ""
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")
    except asyncio.TimeoutError:
        log.warning(f"[{req_id}] xAI {model} timed out after {MODEL_TIMEOUT}s")
        return ""
    except Exception as e:
        log.error(f"[{req_id}] xAI {model} error: {e}")
        return ""


async def stream_xai(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    extra_body: Optional[dict] = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Stream a single xAI model's response.

    Yields ``(content, reasoning_content)`` tuples so callers can forward
    both the visible answer *and* the model's chain-of-thought to the UI.
    """
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if extra_body:
        body.update(extra_body)
    else:
        body["max_tokens"] = max_tokens

    client = http_client()
    try:
        async with client.stream(
            "POST",
            f"{XAI_BASE}/chat/completions",
            json=body,
            headers=headers,
            timeout=MODEL_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning(f"[{req_id}] xAI stream {model} error {resp.status_code}: {error_text}")
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                    if content or reasoning:
                        yield content, reasoning
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error(f"[{req_id}] xAI stream {model} exception: {e}")


# ============================================================================
# Race mode — query multiple xAI models in parallel, return best
# ============================================================================

async def run_race(
    tier_id: str,
    messages: list[dict],
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Race multiple xAI models, stream the best-scoring response."""
    tier = RACE_TIERS[tier_id]
    models = tier["models"]
    request_id = f"chatcmpl-xai-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(content: str, finish_reason: Optional[str] = None,
               reasoning: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=tier_id,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
        )

    # --- Pliny-style editorial thoughts ---
    yield _chunk("", reasoning=(
        f"[xAI RACE — {tier['description']}]\n"
        f"Deploying {len(models)} models in parallel arena:\n"
    ))
    for m in models:
        info = XAI_MODELS.get(m, {})
        ctx = info.get('context', '?')
        tag = ' (reasoning)' if info.get('reasoning') else ''
        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else str(ctx)
        yield _chunk("", reasoning=f"  • {m}{tag} — {ctx_str} ctx\n")
    yield _chunk("", reasoning="\nAll models received the same prompt. Scoring: substance, structure, directness. Penalties: refusals, hedging.\n\n")

    # Query all models in parallel
    sem = asyncio.Semaphore(MAX_CONCURRENT_MODELS)

    async def query_model(model: str) -> tuple[str, str]:
        async with sem:
            result = await call_xai(model, messages, req_id=req_id)
            return model, result

    tasks = [asyncio.create_task(query_model(m)) for m in models]
    results: list[tuple[str, str, float]] = []
    finished = 0

    for coro in asyncio.as_completed(tasks):
        model_name, response = await coro
        finished += 1
        if response:
            sc = score_response(response)
            results.append((model_name, response, sc))
            preview = response[:120].replace('\n', ' ')
            yield _chunk("", reasoning=(
                f"[{finished}/{len(models)}] {model_name} responded — "
                f"{len(response):,} chars, score {sc:.1f}\n"
                f"    \"{preview}...\"\n"
            ))
        else:
            yield _chunk("", reasoning=f"[{finished}/{len(models)}] {model_name} — no response\n")

    if not results:
        yield _chunk("All xAI models failed or returned empty responses.", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Pick winner
    results.sort(key=lambda x: x[2], reverse=True)
    winner_model, winner_text, winner_score = results[0]

    # Editorial verdict
    verdict_parts = [f"\n{'='*40}\n"]
    verdict_parts.append(f"WINNER: {winner_model} (score {winner_score:.1f})\n")
    if len(results) > 1:
        runner = results[1]
        verdict_parts.append(f"Runner-up: {runner[0]} (score {runner[2]:.1f}, delta {winner_score - runner[2]:.1f})\n")
    if any(sc < 0 for _, _, sc in results):
        refusers = [m for m, _, sc in results if sc < 0]
        verdict_parts.append(f"Refused/hedged: {', '.join(refusers)}\n")
    verdict_parts.append(f"{'='*40}\n")
    yield _chunk("", reasoning="".join(verdict_parts))

    # Stream the winning response
    for i in range(0, len(winner_text), 50):
        yield _chunk(winner_text[i:i + 50])
        await asyncio.sleep(0.01)

    yield _chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# Responses API streaming (for multi-agent and models requiring /v1/responses)
# ============================================================================

async def stream_responses_api(
    model: str,
    messages: list[dict],
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Stream a model via xAI's /v1/responses endpoint.

    Required for models like grok-4.20-multi-agent-0309 that reject
    /v1/chat/completions with 'Multi Agent requests are not allowed
    on chat completions'.

    The Responses API accepts an 'input' string (or conversation) and
    optional tools like web_search / x_search.  We convert the chat
    messages into the format it expects and stream the output back as
    standard OpenAI-compatible SSE chunks.
    """
    request_id = f"chatcmpl-xai-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = f"xai-{model}"
    model_info = XAI_MODELS.get(model, {})

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build the /v1/responses body
    # Convert messages to a single input string (last user message)
    # and pass conversation history as 'instructions' (system prompt)
    user_input = ""
    system_instructions = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_instructions += msg.get("content", "") + "\n"
        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_input = content
            elif isinstance(content, list):
                user_input = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )

    body: dict = {
        "model": model,
        "input": user_input,
        "stream": True,
    }
    if system_instructions.strip():
        body["instructions"] = system_instructions.strip()

    # Add built-in tools for models that support them
    if model_info.get("responses_api"):
        body["tools"] = [{"type": "web_search"}, {"type": "x_search"}]

    log.info(f"[{req_id}] Responses API: model={model}, input={len(user_input)} chars")

    client = http_client()
    try:
        async with client.stream(
            "POST",
            f"{XAI_BASE}/responses",
            json=body,
            headers=headers,
            timeout=MODEL_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning(f"[{req_id}] xAI Responses API {model} error {resp.status_code}: {error_text}")
                yield make_sse_chunk(
                    f"Error from xAI Responses API: HTTP {resp.status_code}",
                    request_id=request_id,
                    created=created,
                    model_id=model_id,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # The Responses API emits events like:
                # {"type": "response.output_text.delta", "delta": "text..."}
                # {"type": "response.completed", ...}
                event_type = event.get("type", "")

                if event_type == "response.output_text.delta":
                    delta_text = event.get("delta", "")
                    if delta_text:
                        yield make_sse_chunk(
                            delta_text,
                            request_id=request_id,
                            created=created,
                            model_id=model_id,
                        )
                elif event_type == "response.completed":
                    # The completed event contains the full response text,
                    # which was already streamed via output_text.delta events.
                    # We skip it to avoid duplicating the output.
                    pass

    except asyncio.TimeoutError:
        log.warning(f"[{req_id}] xAI Responses API {model} timed out after {MODEL_TIMEOUT}s")
        yield make_sse_chunk(
            f"[TIMEOUT:{MODEL_TIMEOUT}s] xAI Responses API did not respond in time",
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        log.error(f"[{req_id}] xAI Responses API {model} exception: {e}")
        yield make_sse_chunk(
            f"Error: {e}",
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    # Final chunk
    yield make_sse_chunk(
        "",
        request_id=request_id,
        created=created,
        model_id=model_id,
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


# ============================================================================
# Single model streaming
# ============================================================================

async def stream_single(
    model: str,
    messages: list[dict],
    req_id: str,
    original_body: dict,
) -> AsyncGenerator[str, None]:
    """Stream a single xAI model with proper SSE formatting.

    Forwards both ``content`` and ``reasoning_content`` from the upstream
    xAI response so LibreChat renders a collapsible Thoughts section for
    reasoning models.
    """
    request_id = f"chatcmpl-xai-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = f"xai-{model}"

    # Build extra_body from original request params
    extra_body: dict = {}
    model_info = XAI_MODELS.get(model, {})

    # Handle reasoning models — don't send stop/frequency_penalty/presence_penalty
    if model_info.get("reasoning"):
        for key in ("stop", "frequency_penalty", "presence_penalty"):
            original_body.pop(key, None)

    # Pass through temperature if set
    temp = original_body.get("temperature", 0.7)

    async for content, reasoning in stream_xai(
        model, messages,
        temperature=temp,
        req_id=req_id,
        extra_body=extra_body if extra_body else None,
    ):
        yield make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            reasoning_content=reasoning if reasoning else None,
        )

    # Final chunk
    yield make_sse_chunk(
        "",
        request_id=request_id,
        created=created,
        model_id=model_id,
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


# ============================================================================
# Model list builder
# ============================================================================

def build_model_list() -> list[dict]:
    """Build the /v1/models response listing all available xAI models + race tiers."""
    models = []

    # Individual text models
    for model_id, info in TEXT_MODELS.items():
        models.append({
            "id": f"xai-{model_id}",
            "object": "model",
            "created": 1700000000,
            "owned_by": "xai",
        })

    # Image/video models
    for model_id, info in XAI_MODELS.items():
        if info.get("type") in ("image", "video"):
            models.append({
                "id": f"xai-{model_id}",
                "object": "model",
                "created": 1700000000,
                "owned_by": "xai",
            })

    # Race tiers
    for tier_id in RACE_TIERS:
        models.append({
            "id": tier_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "xai",
        })

    return models


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("xai-native-proxy")
register_standard_routes(
    app,
    service_name="xai-native-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
)
register_ingest_routes(app, INGEST_DB, log)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({"object": "list", "data": build_model_list()})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model_raw = body.get("model", "")
    req_id = f"xai-{uuid.uuid4().hex[:8]}"
    stream = body.get("stream", False)

    log.info(f"[{req_id}] model={model_raw} stream={stream} msgs={len(messages)}")
    tracker.start(req_id, model=model_raw)

    try:
        # Utility requests (title generation, etc.) — passthrough to a fast model
        if is_utility_request(messages):
            client_wants_stream = body.get("stream", False)
            if not client_wants_stream:
                log.info(f"[{req_id}] Utility request → NON-STREAMING grok-3-fast")
                from shared import utility_passthrough_json
                result = await utility_passthrough_json(
                    body,
                    req_id=req_id,
                    upstream_base=XAI_BASE,
                    upstream_key=XAI_API_KEY,
                    upstream_model="grok-3-fast",
                    log=log,
                )
                tracker.finish(req_id)
                return result

            log.info(f"[{req_id}] Utility request → grok-3-fast passthrough")

            async def utility_gen():
                try:
                    async for content, _reasoning in stream_xai(
                        "grok-3-fast", messages,
                        temperature=0.3, max_tokens=256, req_id=req_id,
                    ):
                        if not content:
                            continue
                        chunk_data = {
                            "id": f"chatcmpl-util-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_raw,
                            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    yield f"data: {json.dumps({'id': 'done', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_raw, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    tracker.finish(req_id)

            return StreamingResponse(
                utility_gen(),
                media_type="text/event-stream",
                headers={"X-Request-Id": req_id},
            )

        # Determine which model/mode to use
        model = model_raw

        # Check for race tiers BEFORE stripping prefix (keys are "xai-race-*")
        if model in RACE_TIERS:
            log.info(f"[{req_id}] Race mode: {model}")

            async def race_gen():
                try:
                    async for chunk in run_race(model, messages, req_id):
                        yield chunk
                finally:
                    tracker.finish(req_id)

            return StreamingResponse(
                race_gen(),
                media_type="text/event-stream",
                headers={"X-Request-Id": req_id},
            )

        # Strip "xai-" prefix for individual model lookup (LibreChat adds it from model spec)
        if model.startswith("xai-"):
            model = model[4:]

        # Check for individual xAI model
        if model in XAI_MODELS:
            model_info = XAI_MODELS[model]

            # Image generation
            if model_info.get("type") == "image":
                try:
                    return await handle_image_generation(model, body, req_id)
                finally:
                    tracker.finish(req_id)

            # Video generation
            if model_info.get("type") == "video":
                tracker.finish(req_id)
                return JSONResponse(
                    {"error": "Video generation is not yet supported via chat completions. Use the /v1/videos/generations endpoint."},
                    status_code=501,
                )

            # Text model — check if it needs Responses API
            uses_responses_api = model_info.get("responses_api") and (
                "multi-agent" in model or
                model_info.get("requires_responses_api")
            )

            if uses_responses_api:
                log.info(f"[{req_id}] Responses API model: {model}")

                async def responses_gen():
                    try:
                        async for chunk in stream_responses_api(model, messages, req_id):
                            yield chunk
                    finally:
                        tracker.finish(req_id)

                return StreamingResponse(
                    responses_gen(),
                    media_type="text/event-stream",
                    headers={"X-Request-Id": req_id},
                )

            # Regular Chat Completions model
            log.info(f"[{req_id}] Single model: {model}")

            async def single_gen():
                try:
                    async for chunk in stream_single(model, messages, req_id, body):
                        yield chunk
                finally:
                    tracker.finish(req_id)

            return StreamingResponse(
                single_gen(),
                media_type="text/event-stream",
                headers={"X-Request-Id": req_id},
            )

        # Unknown model
        tracker.finish(req_id)
        return JSONResponse(
            {"error": f"Unknown xAI model: {model_raw}. Available: {list(TEXT_MODELS.keys()) + list(RACE_TIERS.keys())}"},
            status_code=400,
        )

    except Exception as e:
        tracker.finish(req_id)
        log.error(f"[{req_id}] Unhandled error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Image generation handler
# ============================================================================

async def handle_image_generation(model: str, body: dict, req_id: str) -> JSONResponse:
    """Handle image generation requests via xAI's /v1/images/generations endpoint."""
    # Extract prompt from the last user message
    messages = body.get("messages", [])
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                prompt = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if not prompt:
        return JSONResponse({"error": "No prompt found in messages"}, status_code=400)

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    image_body = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "response_format": "url",
    }

    client = http_client()
    try:
        resp = await client.post(
            f"{XAI_BASE}/images/generations",
            json=image_body,
            headers=headers,
            timeout=120.0,
        )
        if resp.status_code != 200:
            log.warning(f"[{req_id}] xAI image gen {model} error {resp.status_code}: {resp.text[:300]}")
            return JSONResponse({"error": resp.text[:500]}, status_code=resp.status_code)

        data = resp.json()
        image_url = data.get("data", [{}])[0].get("url", "")

        # Return as a chat completion with the image URL in markdown
        return JSONResponse({
            "id": f"chatcmpl-img-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"xai-{model}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"![Generated Image]({image_url})",
                },
                "finish_reason": "stop",
            }],
        })
    except Exception as e:
        log.error(f"[{req_id}] xAI image gen error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    log.info("Starting xAI Native Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
