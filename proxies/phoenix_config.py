"""Arize Phoenix Observability — OpenTelemetry-based tracing for LangGraph pipelines.

Provides automatic instrumentation of all LangChain/LangGraph calls via
OpenInference.  Once ``initialize()`` is called, every LLM invocation,
tool call, and graph node execution is captured as OpenTelemetry spans
and sent to the Phoenix collector.

Phoenix is 100% free to self-host with no feature limitations.  The
Agent Graph visualization shows the full StateGraph execution as an
interactive node-and-edge diagram in realtime.

Environment variables (all optional — tracing is a no-op when unconfigured):
  PHOENIX_COLLECTOR_ENDPOINT  — gRPC endpoint (default: http://localhost:4317)
  PHOENIX_BASE_URL            — Phoenix UI URL (default: http://localhost:6006)
  PHOENIX_PROJECT_NAME        — project name in Phoenix (default: deep-search)
  PHOENIX_ENABLED             — set to "false" to disable tracing at runtime
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("phoenix_config")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317"
)
PHOENIX_BASE_URL = os.getenv("PHOENIX_BASE_URL", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "deep-search")
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() not in (
    "false", "0", "no",
)

# ---------------------------------------------------------------------------
# Lazy singleton — initialised on first use
# ---------------------------------------------------------------------------

_initialized = False
_tracer_provider: Optional[object] = None


def initialize() -> bool:
    """Initialize Phoenix OpenTelemetry tracing (idempotent).

    Registers the OpenInference LangChain instrumentor which automatically
    captures all LangChain/LangGraph calls.  Safe to call multiple times —
    subsequent calls are no-ops.

    Returns True if tracing is active, False otherwise.
    """
    global _initialized, _tracer_provider

    if _initialized:
        return _tracer_provider is not None

    _initialized = True

    if not PHOENIX_ENABLED:
        log.info("Phoenix tracing disabled (PHOENIX_ENABLED=false)")
        return False

    try:
        from phoenix.otel import register

        _tracer_provider = register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=PHOENIX_COLLECTOR_ENDPOINT,
            auto_instrument=True,
        )
        log.info(
            "Phoenix tracing enabled — collector=%s, UI=%s, project=%s",
            PHOENIX_COLLECTOR_ENDPOINT,
            PHOENIX_BASE_URL,
            PHOENIX_PROJECT_NAME,
        )
        return True

    except Exception as exc:
        log.warning("Phoenix initialisation failed (tracing disabled): %s", exc)
        _tracer_provider = None
        return False


def is_enabled() -> bool:
    """Return True when Phoenix tracing is active."""
    return initialize()


def get_dashboard_url() -> str:
    """Return the Phoenix dashboard base URL.

    Returns an empty string if Phoenix is not configured.
    """
    if not is_enabled():
        return ""
    return PHOENIX_BASE_URL


def get_project_url() -> str:
    """Return the direct URL to the project's traces in Phoenix."""
    if not is_enabled():
        return ""
    return f"{PHOENIX_BASE_URL}/projects/{PHOENIX_PROJECT_NAME}"


def shutdown() -> None:
    """Gracefully shut down the Phoenix tracer provider."""
    global _tracer_provider, _initialized
    if _tracer_provider is not None:
        try:
            if hasattr(_tracer_provider, "shutdown"):
                _tracer_provider.shutdown()
        except Exception:
            pass
        _tracer_provider = None
    _initialized = False
