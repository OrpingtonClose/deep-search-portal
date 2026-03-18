"""
Shared utilities for Deep Search Portal proxies.

Provides common infrastructure used by both the Thinking Proxy and the
Deep Research Proxy: logging, configuration, SSE helpers, HTTP client
management, passthrough streaming, utility-request detection, and
standard FastAPI endpoints (health, logs).
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse


# ============================================================================
# Utility-request detection
# ============================================================================

UTILITY_PATTERNS = [
    "generate a concise",
    "generate 1-3 broad tags",
    "generate a title",
    "### task:\ngenerate",
    "create a concise title",
    "generate a search query",
    "autocomplete",
]


def is_utility_request(messages: list[dict]) -> bool:
    """Detect automated utility requests from Open WebUI (title/tag gen, etc.)."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            for pattern in UTILITY_PATTERNS:
                if pattern in content_lower:
                    return True
    return False


# ============================================================================
# Logging
# ============================================================================

def setup_logging(service_name: str, log_dir: str) -> logging.Logger:
    """
    Configure root logging with console (INFO) and rotating file (DEBUG)
    handlers, then return a named logger for *service_name*.

    Safe to call once per process — guards against duplicate handlers when
    the module is re-imported.
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.root
    root.setLevel(logging.DEBUG)

    # Only add handlers if none are attached yet (prevents duplicates on reimport).
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(console)

        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "proxy.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(fh)

    return logging.getLogger(service_name)


# ============================================================================
# Configuration helpers
# ============================================================================

def require_env(name: str) -> str:
    """Return an environment variable or raise with a clear message."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return val


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an env var as int, clamping to *minimum*."""
    raw = os.getenv(name, "")
    if not raw:
        return max(default, minimum)
    try:
        return max(int(raw), minimum)
    except ValueError:
        logging.getLogger("config").warning(
            f"Invalid integer for {name}={raw!r}, using default {default}"
        )
        return max(default, minimum)


# ============================================================================
# Shared HTTP client (connection-pooled, created once per process)
# ============================================================================

# Populated by the lifespan context manager — see ``create_app``.
_http_client: Optional[httpx.AsyncClient] = None


def http_client() -> httpx.AsyncClient:
    """Return the shared httpx client. Raises if called before lifespan start."""
    if _http_client is None:
        raise RuntimeError("HTTP client not initialised — is the app lifespan running?")
    return _http_client


# ============================================================================
# SSE helpers
# ============================================================================

def make_sse_chunk(
    content: str,
    *,
    request_id: str,
    created: int,
    model_id: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Build a single SSE ``data:`` line in OpenAI chat-completion chunk format."""
    delta: dict = {}
    if content:
        delta["content"] = content

    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(data)}\n\n"


# ============================================================================
# Active-request tracking
# ============================================================================

class RequestTracker:
    """Thread-safe (single-event-loop) tracker for in-flight requests."""

    def __init__(self) -> None:
        self._active: dict[str, dict] = {}

    def start(self, req_id: str, **meta: object) -> None:
        self._active[req_id] = {
            "started": datetime.now(timezone.utc).isoformat(),
            **meta,
        }

    def update(self, req_id: str, **fields: object) -> None:
        if req_id in self._active:
            self._active[req_id].update(fields)

    def finish(self, req_id: str) -> None:
        self._active.pop(req_id, None)

    @property
    def count(self) -> int:
        return len(self._active)

    @property
    def details(self) -> dict[str, dict]:
        return dict(self._active)


# ============================================================================
# Concurrency limiter
# ============================================================================

class ConcurrencyLimiter:
    """Async semaphore wrapper for limiting expensive concurrent operations."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def available(self) -> bool:
        """Return True if at least one slot is free (non-blocking check)."""
        return self._sem._value > 0  # noqa: SLF001

    @asynccontextmanager
    async def hold(self):
        """Acquire a slot for the duration of the ``async with`` block."""
        async with self._sem:
            yield


# ============================================================================
# Passthrough streaming (shared between both proxies)
# ============================================================================

async def stream_passthrough(
    messages: list[dict],
    original_body: dict,
    *,
    req_id: str,
    upstream_base: str,
    upstream_key: str,
    upstream_model: str,
    model_id: str,
    tracker: RequestTracker,
    log: logging.Logger,
    extra_headers: Optional[dict[str, str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Forward a request to the upstream LLM without any agent / thinking logic.
    Used for utility requests (title generation, tag generation, etc.).
    Always streams the response as SSE.
    """
    request_id = f"chatcmpl-pass-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def _chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # Build upstream body — strip Open-WebUI-specific keys
    upstream_body = {
        **original_body,
        "model": upstream_model,
        "messages": messages,
        "stream": True,
    }
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    log.info(f"[{req_id}] PASSTHROUGH upstream: model={upstream_model}, msgs={len(messages)}")

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        client = http_client()
        async with client.stream(
            "POST",
            f"{upstream_base}/chat/completions",
            json=upstream_body,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:1000]
                log.error(f"[{req_id}] Passthrough upstream error {resp.status_code}: {error_text}")
                yield _chunk(f"Error: {error_text[:200]}", finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(payload)
                        chunk["model"] = model_id
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        pass

            # Stream ended without [DONE] — send it ourselves
            yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Passthrough error after {elapsed:.2f}s: {e}")
        yield _chunk(f"Error: {str(e)[:200]}", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# Standard FastAPI endpoints (health + logs)
# ============================================================================

def register_standard_routes(
    app: FastAPI,
    *,
    service_name: str,
    log_dir: str,
    tracker: RequestTracker,
    health_extras: Optional[dict] = None,
) -> None:
    """
    Register ``/health`` and ``/logs`` endpoints on *app*.

    *health_extras* is a dict of additional key/value pairs to include
    in the health response (e.g. upstream URL, model, searxng URL).
    """

    @app.get("/health")
    async def health():
        info = {
            "status": "ok",
            "service": service_name,
            **(health_extras or {}),
            "active_requests": tracker.count,
            "active_details": tracker.details,
        }
        return JSONResponse(info)

    @app.get("/logs")
    async def get_logs(lines: int = 100):
        log_path = os.path.join(log_dir, "proxy.log")
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
                return JSONResponse({
                    "total_lines": len(all_lines),
                    "returned": min(lines, len(all_lines)),
                    "lines": all_lines[-lines:],
                })
        except FileNotFoundError:
            return JSONResponse({"error": "Log file not found"}, status_code=404)


# ============================================================================
# App factory with lifespan (manages shared httpx client)
# ============================================================================

def create_app(title: str) -> FastAPI:
    """
    Create a FastAPI app with a lifespan that manages the shared httpx client.
    The client uses connection pooling (max 20 keepalive, 100 total connections)
    and generous timeouts suited for LLM API calls.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _http_client
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
            follow_redirects=True,
        )
        yield
        await _http_client.aclose()
        _http_client = None

    return FastAPI(title=title, lifespan=lifespan)
