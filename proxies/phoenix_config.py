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

import functools
import logging
import os
from typing import Any, Callable, Optional

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

# Per-request root span context — keyed by req_id.
# Populated by start_pipeline_span(), consumed by traced_node().
_root_contexts: dict[str, Any] = {}

# Per-request node status tracking for live graph visualization.
# Each entry: {"active": set[str], "completed": set[str]}
_node_status: dict[str, dict[str, set[str]]] = {}


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


def get_tracer(name: str = "deep-search.pipeline") -> Any:
    """Return an OpenTelemetry tracer for manual span creation.

    Returns a real Tracer when Phoenix is enabled, or a no-op object
    whose ``start_as_current_span`` context manager does nothing.
    """
    if not is_enabled():
        return _NoOpTracer()

    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


class _NoOpTracer:
    """Dummy tracer that produces no-op context managers."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> "_NoOpSpanCtx":
        return _NoOpSpanCtx()

    def start_span(self, name: str, **kwargs: Any) -> "_NoOpSpan":
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span returned by _NoOpTracer.start_span."""

    def end(self) -> None:
        pass

    def get_span_context(self) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any) -> None:
        pass


class _NoOpSpanCtx:
    """No-op context manager returned by _NoOpTracer."""

    def __enter__(self) -> "_NoOpSpanCtx":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    async def __aenter__(self) -> "_NoOpSpanCtx":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Pipeline-level root span management
# ---------------------------------------------------------------------------

def get_node_status() -> dict[str, dict[str, set[str]]]:
    """Return current node status for all active pipelines.

    Used by the ``/graph/active`` endpoint to highlight nodes in the
    live StateGraph visualization.
    """
    return _node_status


def start_pipeline_span(req_id: str, user_query: str) -> Any:
    """Create a long-lived root span for the entire pipeline run.

    The span is stored in ``_root_contexts`` keyed by req_id.  Each
    ``@traced_node`` decorated function reads this context to create
    child spans, producing the hierarchical trace tree:

        persistent_research_pipeline
          +- comprehend
          |    +- ChatOpenAI (auto-instrumented LLM call)
          +- retrieve
          +- tree_research.init_tree
          +- tree_research.explore
          +- entities
          +- verify
          +- reflect
          +- persist
          +- synthesize

    Returns the span object (caller must call ``end_pipeline_span``
    when the pipeline finishes).
    """
    if not is_enabled():
        return _NoOpSpan()

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("deep-search.pipeline")
        span = tracer.start_span(
            "persistent_research_pipeline",
            attributes={
                "graph.name": "persistent_research_pipeline",
                "graph.node.id": "persistent_research_pipeline",
                "graph.node.display_name": "Research Pipeline",
                "metadata.langgraph_node": "persistent_research_pipeline",
                "metadata.langgraph_step": 0,
                "req_id": req_id,
                "user_query": user_query[:200],
            },
        )
        # Store the context with this span active so child spans can parent to it
        ctx = trace.set_span_in_context(span)
        _root_contexts[req_id] = ctx
        _node_status[req_id] = {"active": set(), "completed": set()}
        log.debug("[%s] Started pipeline root span", req_id)
        return span

    except Exception as exc:
        log.warning("Failed to create pipeline root span: %s", exc)
        return _NoOpSpan()


def end_pipeline_span(req_id: str, span: Any) -> None:
    """End the root pipeline span and clean up the stored context."""
    try:
        span.end()
    except Exception:
        pass
    _root_contexts.pop(req_id, None)
    _node_status.pop(req_id, None)


def traced_node(node_name: str) -> Callable:
    """Decorator that wraps a LangGraph node function in an OTel span.

    The span is created as a child of the root pipeline span (stored
    in ``_root_contexts`` by ``start_pipeline_span``).  This preserves
    the parent-child hierarchy even though LangGraph runs each node in
    a separate async context.

    Usage::

        @traced_node("comprehend")
        async def pdr_node_comprehend(state):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: Any) -> Any:
            if not is_enabled():
                return await fn(state)

            # Phase 1: OTel setup — if this fails, run fn without tracing
            span = None
            token = None
            otel_ctx_mod = None
            try:
                from opentelemetry import context as otel_ctx_mod, trace

                req_id = state.get("req_id", "") if isinstance(state, dict) else ""
                parent_ctx = _root_contexts.get(req_id)

                tracer = trace.get_tracer("deep-search.pipeline")

                iteration = state.get("research_iterations", 0) if isinstance(state, dict) else 0

                # Create child span under the pipeline root
                span = tracer.start_span(
                    node_name,
                    context=parent_ctx,
                    attributes={
                        "graph.node.id": node_name,
                        "graph.node.parent_id": "persistent_research_pipeline",
                        "graph.node.display_name": node_name.replace("_", " ").title(),
                        "graph.name": "persistent_research_pipeline",
                        "graph.node": node_name,
                        "graph.iteration": iteration,
                        "metadata.langgraph_node": node_name,
                        "metadata.langgraph_step": iteration,
                    },
                )

                # Activate this span as current so auto-instrumented LLM calls
                # become children of this node span
                ctx = trace.set_span_in_context(span)
                token = otel_ctx_mod.attach(ctx)
            except Exception:
                # OTel setup failed — run without tracing
                return await fn(state)

            # Track node as active for live graph visualization
            if req_id and req_id in _node_status:
                _node_status[req_id]["active"].add(node_name)
                _node_status[req_id]["completed"].discard(node_name)

            # Phase 2: run the actual node function under the span
            try:
                result = await fn(state)
                return result
            except Exception as exc:
                from opentelemetry.trace import StatusCode
                span.set_status(StatusCode.ERROR, str(exc))
                raise
            finally:
                span.end()
                otel_ctx_mod.detach(token)
                # Mark node as completed for live graph visualization
                if req_id and req_id in _node_status:
                    _node_status[req_id]["active"].discard(node_name)
                    _node_status[req_id]["completed"].add(node_name)

        return wrapper
    return decorator


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
    _root_contexts.clear()
    _node_status.clear()
