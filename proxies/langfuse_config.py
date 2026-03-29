"""Langfuse Observability — centralized configuration and callback factory.

Provides a single initialization point for Langfuse tracing across all proxy
services (persistent research, deep research, Mistral Real).  Each research
run gets its own trace with a deterministic trace ID (derived from the
request ID) so the Langfuse dashboard URL can be emitted as the very first
SSE message to the user.

Environment variables (all optional — tracing is a no-op when unconfigured):
  LANGFUSE_PUBLIC_KEY   — project public key from Langfuse
  LANGFUSE_SECRET_KEY   — project secret key from Langfuse
  LANGFUSE_BASE_URL     — self-hosted Langfuse URL (default: https://cloud.langfuse.com)
  LANGFUSE_ENABLED      — set to "false" to disable tracing at runtime
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger("langfuse_config")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() not in ("false", "0", "no")

# Derived: is Langfuse actually usable?
_langfuse_available = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_ENABLED)

# ---------------------------------------------------------------------------
# Lazy singleton — initialised on first use
# ---------------------------------------------------------------------------

_langfuse_client: Optional[Any] = None
_init_attempted = False


def _get_langfuse() -> Optional[Any]:
    """Return the global Langfuse client, creating it on first call.

    Returns ``None`` when Langfuse is not configured or import fails.
    """
    global _langfuse_client, _init_attempted

    if _init_attempted:
        return _langfuse_client

    _init_attempted = True

    if not _langfuse_available:
        log.info("Langfuse tracing disabled (missing keys or LANGFUSE_ENABLED=false)")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            base_url=LANGFUSE_BASE_URL,
        )
        _langfuse_client.auth_check()
        log.info("Langfuse tracing enabled — base_url=%s", LANGFUSE_BASE_URL)
        return _langfuse_client

    except Exception as exc:
        log.warning("Langfuse initialisation failed (tracing disabled): %s", exc)
        _langfuse_client = None
        return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True when Langfuse tracing is active."""
    return _get_langfuse() is not None


def create_trace_id(req_id: str) -> str:
    """Create a deterministic trace ID from a request ID.

    Uses Langfuse's ``create_trace_id`` with the req_id as seed so the
    trace URL can be computed *before* the pipeline starts.
    """
    try:
        from langfuse import Langfuse
        return Langfuse.create_trace_id(seed=req_id)
    except Exception:
        # Fallback: use the req_id directly
        return req_id


def get_trace_url(trace_id: str) -> str:
    """Return the full Langfuse dashboard URL for a given trace.

    Returns an empty string if Langfuse is not configured.
    """
    client = _get_langfuse()
    if client is None:
        return ""

    try:
        return client.get_trace_url(trace_id)
    except Exception as exc:
        log.warning("Failed to get trace URL: %s", exc)
        return ""


def create_callback_handler(
    trace_id: str,
    session_id: str = "",
    user_id: str = "",
    tags: Optional[list[str]] = None,
) -> Optional[Any]:
    """Create a Langfuse LangChain CallbackHandler for a research run.

    Returns ``None`` if Langfuse is not configured so callers can safely
    skip it when building the callbacks list.

    The ``trace_id`` should come from :func:`create_trace_id` so it matches
    the URL already sent to the user.
    """
    if not is_enabled():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        trace_context: dict[str, Any] = {"trace_id": trace_id}
        if session_id:
            trace_context["session_id"] = session_id
        if user_id:
            trace_context["user_id"] = user_id
        if tags:
            trace_context["tags"] = tags

        handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
            trace_context=trace_context,
        )
        return handler

    except Exception as exc:
        log.warning("Failed to create Langfuse callback handler: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-request trace registry — allows any module to create child spans
# on the current request's Langfuse trace without plumbing the handler
# through every function signature.
# ---------------------------------------------------------------------------

_trace_ids: dict[str, str] = {}  # req_id → trace_id


def register_trace(req_id: str, trace_id: str) -> None:
    """Associate a request ID with its Langfuse trace ID.

    Called once when the pipeline starts so that downstream helpers
    (``start_span``, ``end_span``) can attach spans to the correct trace.
    """
    _trace_ids[req_id] = trace_id


def unregister_trace(req_id: str) -> None:
    """Remove the trace association for a completed request."""
    _trace_ids.pop(req_id, None)


def start_span(
    req_id: str,
    name: str,
    *,
    input: Any = None,
    metadata: Optional[dict] = None,
) -> Optional[Any]:
    """Create a new Langfuse span on the current request's trace.

    Returns the span object (call ``.end()`` when done) or ``None`` if
    Langfuse is not available.  Safe to call unconditionally.
    """
    client = _get_langfuse()
    if client is None:
        return None
    trace_id = _trace_ids.get(req_id)
    if not trace_id:
        return None
    try:
        span = client.span(
            trace_id=trace_id,
            name=name,
            input=input,
            metadata=metadata or {},
        )
        return span
    except Exception as exc:
        log.debug("Failed to create Langfuse span %s: %s", name, exc)
        return None


def end_span(
    span: Optional[Any],
    *,
    output: Any = None,
    level: str = "DEFAULT",
    status_message: str = "",
) -> None:
    """End a Langfuse span.  No-op if *span* is ``None``."""
    if span is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if level != "DEFAULT":
            kwargs["level"] = level
        if status_message:
            kwargs["status_message"] = status_message
        span.end(**kwargs)
    except Exception as exc:
        log.debug("Failed to end Langfuse span: %s", exc)


def flush() -> None:
    """Flush any pending Langfuse events.  Safe to call even when disabled."""
    client = _get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


def shutdown() -> None:
    """Gracefully shut down the Langfuse client."""
    global _langfuse_client, _init_attempted
    if _langfuse_client is not None:
        try:
            _langfuse_client.shutdown()
        except Exception:
            pass
        _langfuse_client = None
    _init_attempted = False
