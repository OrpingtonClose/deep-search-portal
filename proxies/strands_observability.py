"""Strands Agent Observability — JSONL metrics, enriched activity logs, structured logging.

Self-contained observability module for the Strands Venice research agent.
Designed to be imported by the strands-agent FastAPI server
(in deep-search-portal/strands-agent/) or used standalone for metrics analysis.

All output is structured for LLM consumption — an AI analyst can read the
JSONL metrics file and recommend concrete improvements to agent behavior.

Collected data per request:
  - Request identity (id, model, query, timestamp)
  - Wall-clock elapsed time
  - Tool execution timeline (tool name, input summary)
  - AgentResult.metrics summary (cycles, token usage, tool success/error rates)
  - Strands SDK internal debug logs (separate JSONL file)

Storage:
  - /var/log/strands-metrics.jsonl   — one JSON line per request (~1 KB each)
  - /var/log/strands-agent-debug.jsonl — Strands SDK internal logs
  - Both files managed by logrotate (see config/strands-logrotate.conf)
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("strands_observability")

# ── Configuration ─────────────────────────────────────────────────────

METRICS_JSONL_PATH = os.getenv(
    "STRANDS_METRICS_LOG",
    "/var/log/strands-metrics.jsonl",
)

SDK_DEBUG_JSONL_PATH = os.getenv(
    "STRANDS_DEBUG_LOG",
    "/var/log/strands-agent-debug.jsonl",
)

# Maximum number of per-request activity logs kept in memory (ring buffer).
MAX_REQUEST_LOGS = int(os.getenv("STRANDS_MAX_REQUEST_LOGS", "200"))


# ── Per-request activity log ring buffer ──────────────────────────────

_request_logs: OrderedDict[str, dict] = OrderedDict()


def store_request_log(req_id: str, entry: dict) -> None:
    """Store a per-request activity log entry in the in-memory ring buffer."""
    _request_logs[req_id] = entry
    while len(_request_logs) > MAX_REQUEST_LOGS:
        _request_logs.popitem(last=False)


def get_request_log(req_id: str) -> dict | None:
    """Retrieve a stored request log by ID, or None if evicted / not found."""
    return _request_logs.get(req_id)


# ── JSONL metrics writer ─────────────────────────────────────────────


def trim_metrics(metrics_summary: dict | None) -> dict | None:
    """Strip bulky fields from AgentResult.metrics.get_summary() to keep
    JSONL compact.  The ``traces`` field contains full model outputs and
    can be 100s of KB.  Agent invocations are trimmed to just usage data
    (per-cycle message bodies are dropped).

    Returns a new dict (the original is not mutated), or None if input
    is None.
    """
    if not metrics_summary:
        return None

    trimmed = {k: v for k, v in metrics_summary.items() if k != "traces"}

    if "agent_invocations" in trimmed:
        trimmed_invocations = []
        for inv in trimmed["agent_invocations"]:
            trimmed_invocations.append({
                "usage": inv.get("usage"),
                "cycles": [
                    {
                        "event_loop_cycle_id": c.get("event_loop_cycle_id"),
                        "usage": c.get("usage"),
                    }
                    for c in inv.get("cycles", [])
                ],
            })
        trimmed["agent_invocations"] = trimmed_invocations

    return trimmed


def write_metrics_jsonl(
    req_id: str,
    model: str,
    query: str,
    elapsed: float,
    metrics_summary: dict | None,
    tool_events: list[dict],
    *,
    metrics_path: str | None = None,
) -> None:
    """Append a single JSON line to the metrics log file.

    Parameters
    ----------
    req_id : str
        Unique request identifier.
    model : str
        Model name used for the request (e.g. ``strands-venice-single``).
    query : str
        The user's query (truncated to 500 chars in the record).
    elapsed : float
        Wall-clock elapsed seconds for the request.
    metrics_summary : dict | None
        Output of ``AgentResult.metrics.get_summary()`` (will be trimmed).
    tool_events : list[dict]
        List of tool event dicts captured during execution.
    metrics_path : str | None
        Override the default JSONL file path.
    """
    path = metrics_path or METRICS_JSONL_PATH
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": req_id,
        "model": model,
        "query": query[:500] if query else "",
        "elapsed_s": elapsed,
        "tool_events": [
            {"tool": e.get("tool", ""), "input": (e.get("input") or "")[:200]}
            for e in tool_events
        ],
        "metrics": trim_metrics(metrics_summary),
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        log.warning("Failed to write metrics JSONL to %s", path, exc_info=True)


# ── Enriched inline activity log formatter ────────────────────────────


def format_inline_log(
    tool_events: list[dict],
    elapsed: float,
    *,
    query: str = "",
    model: str = "",
    reasoning: str = "",
    metrics: dict | None = None,
) -> str:
    """Format a minimal, user-friendly research summary footer.

    Renders a single-line italic footer showing time taken and number of
    sources consulted.  All developer-facing metrics (tokens, cycles,
    cache stats, tool timelines) are omitted — they are still written to
    the JSONL metrics log for debugging.
    """
    parts: list[str] = []

    # ── Thinking section (inline, not collapsible) ──
    if reasoning and reasoning.strip():
        parts.append(
            f"\n\n---\n**💭 Thinking**\n\n{reasoning.strip()}"
        )

    # ── Minimal footer — just time and source count ──
    # Only show footer when tools were actually used (research happened).
    # Simple Q&A with no tools doesn't need a footer.
    unique_tools = {ev.get("tool", "") for ev in tool_events}
    unique_tools.discard("")
    n_sources = len(unique_tools)
    if n_sources > 0:
        time_str = f"{elapsed:.0f}s" if elapsed >= 1 else "<1s"
        if n_sources == 1:
            footer = f"*Researched using 1 source in {time_str}*"
        else:
            footer = f"*Researched using {n_sources} sources in {time_str}*"
        parts.append(f"\n\n---\n{footer}\n")

    return "".join(parts)


# ── Structured JSON logging for Strands SDK internals ─────────────────


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — suitable for JSONL files."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            },
            default=str,
        )


def setup_strands_sdk_logging(
    *,
    debug_path: str | None = None,
    level: int = logging.DEBUG,
    sdk_modules: tuple[str, ...] = (
        "strands",
        "strands.tools",
        "strands.event_loop",
        "strands.models",
    ),
) -> logging.FileHandler | None:
    """Configure structured JSON file logging for Strands SDK internals.

    Attaches a ``_JsonFormatter`` file handler to the specified SDK logger
    modules.  Returns the file handler on success, or None on failure.

    Parameters
    ----------
    debug_path : str | None
        Path to the debug JSONL file.  Defaults to ``SDK_DEBUG_JSONL_PATH``.
    level : int
        Logging level for SDK loggers (default ``DEBUG``).
    sdk_modules : tuple[str, ...]
        Logger names to attach the handler to.
    """
    path = debug_path or SDK_DEBUG_JSONL_PATH
    try:
        fh = logging.FileHandler(path)
        fh.setFormatter(_JsonFormatter())
        fh.setLevel(level)
        for mod in sdk_modules:
            logger = logging.getLogger(mod)
            logger.setLevel(level)
            logger.addHandler(fh)
        log.info("Strands SDK debug logging → %s", path)
        return fh
    except Exception:
        log.warning("Could not set up Strands JSON logging at %s", path, exc_info=True)
        return None


# ── Convenience: build OpenAI-compatible usage dict from metrics ──────


def extract_usage(metrics_summary: dict | None) -> dict[str, int]:
    """Extract OpenAI-compatible ``usage`` dict from AgentResult metrics.

    Returns ``{"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}``.
    """
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if metrics_summary and metrics_summary.get("accumulated_usage"):
        u = metrics_summary["accumulated_usage"]
        usage = {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
        }
    return usage
