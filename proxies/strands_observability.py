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
    """Format a detailed activity log as markdown sections.

    Renders:
    1. **Thinking** — the model's full reasoning chain (inline, not collapsible)
    2. **Activity Log** — tool execution timeline + metrics (collapsible ``<details>``)

    The output is designed to be appended to the agent's response text in
    LibreChat (or any markdown-capable chat UI).
    """
    parts: list[str] = []

    # ── Thinking section (inline, not collapsible) ──
    if reasoning and reasoning.strip():
        parts.append(
            f"\n\n---\n**💭 Thinking**\n\n{reasoning.strip()}"
        )

    # ── Activity log section ──
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_lines = [
        "=== Strands Agent Activity Log ===",
        f"Timestamp : {ts}",
        f"Model     : {model or 'unknown'}",
        f"Elapsed   : {elapsed:.1f}s",
        f"Tool calls: {len(tool_events)}",
        f"Query     : {query[:200] if query else 'N/A'}",
    ]

    # ── Metrics from AgentResult ──
    if metrics:
        log_lines.append("")
        log_lines.append("--- Performance Metrics ---")
        usage = metrics.get("accumulated_usage") or {}
        if usage:
            log_lines.append(f"  Input tokens : {usage.get('inputTokens', 'N/A')}")
            log_lines.append(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")
            log_lines.append(f"  Total tokens : {usage.get('totalTokens', 'N/A')}")
            if usage.get("cacheReadInputTokens"):
                log_lines.append(f"  Cache read   : {usage['cacheReadInputTokens']}")
            if usage.get("cacheWriteInputTokens"):
                log_lines.append(f"  Cache write  : {usage['cacheWriteInputTokens']}")
        latency = metrics.get("accumulated_metrics") or {}
        if latency.get("latencyMs"):
            log_lines.append(f"  Model latency: {latency['latencyMs']}ms")
        cycles = metrics.get("total_cycles")
        if cycles is not None:
            log_lines.append(f"  Agent cycles : {cycles}")
        duration = metrics.get("total_duration")
        if duration is not None:
            log_lines.append(f"  Total duration: {duration:.2f}s")

        # Tool-level metrics from AgentResult
        tool_usage = metrics.get("tool_usage") or {}
        if tool_usage:
            log_lines.append("")
            log_lines.append("--- Tool Metrics ---")
            for tname, tstats in tool_usage.items():
                calls = tstats.get("total_calls", "?")
                success = tstats.get("successful_calls", "?")
                errors = tstats.get("errors", "?")
                avg_time = tstats.get("average_execution_time") or 0
                log_lines.append(
                    f"  {tname}: {calls} calls, {success} ok, {errors} err, avg {avg_time:.2f}s"
                )

    log_lines.append("")
    log_lines.append("--- Tool Execution Timeline ---")

    if not tool_events:
        log_lines.append("  (no tool calls)")
    else:
        start_time = tool_events[0].get("time") if tool_events else None
        for i, ev in enumerate(tool_events, 1):
            tool_name = ev.get("tool", "unknown")
            tool_input = ev.get("input", "")
            t = ev.get("time")
            offset = f"+{t - start_time:.1f}s" if (start_time is not None and t is not None) else ""
            log_lines.append(f"  [{i}] {offset:>8s}  {tool_name}")
            if tool_input and tool_input != "{}":
                for line in tool_input[:300].split("\n"):
                    log_lines.append(f"              {line}")
                if len(tool_input) > 300:
                    log_lines.append("              ... (truncated)")

    log_lines.append("")
    log_lines.append("=== End of Log ===")
    log_content = "\n".join(log_lines)

    parts.append(
        f"\n\n---\n📄 **agent-activity-log.txt** ({len(tool_events)} tools, {elapsed:.1f}s)\n\n"
        f"```yaml\n{log_content}\n```"
    )

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
