"""Langfuse Observability Dashboards — query Langfuse Metrics API and render HTML.

Provides a self-contained observability dashboard for the Deep Search Portal
research pipeline.  Two data sources:

  1. **Langfuse Metrics API** (``GET /api/public/metrics``) — when Langfuse is
     configured with ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``.
     Queries traces/observations for latency, cost, model usage, error rates.

  2. **Local metrics JSON files** — always available.  Reads from
     ``RESEARCH_METRICS_DIR`` (default ``/opt/persistent_research_logs/metrics``)
     for per-session research metrics collected by ``research_metrics.py``.

The dashboard is rendered as a self-contained HTML page with inline CSS and
vanilla JS charts (no external dependencies).  It is served by the persistent
proxy at ``/research/dashboard``.

Environment variables:
  LANGFUSE_PUBLIC_KEY   — Langfuse project public key
  LANGFUSE_SECRET_KEY   — Langfuse project secret key
  LANGFUSE_BASE_URL     — Langfuse host (default: https://cloud.langfuse.com)
  RESEARCH_METRICS_DIR  — local metrics directory
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import html as html_mod

import httpx

log = logging.getLogger("langfuse_dashboards")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

METRICS_DIR = os.getenv(
    "RESEARCH_METRICS_DIR",
    "/opt/persistent_research_logs/metrics",
)


def _langfuse_configured() -> bool:
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


# ---------------------------------------------------------------------------
# Langfuse Metrics API queries
# ---------------------------------------------------------------------------


def _langfuse_query(query: dict[str, Any], timeout: float = 15.0) -> list[dict]:
    """Execute a query against the Langfuse Metrics API v1.

    Returns a list of result rows, or an empty list on error.
    """
    if not _langfuse_configured():
        return []

    url = f"{LANGFUSE_BASE_URL.rstrip('/')}/api/public/metrics"
    try:
        resp = httpx.get(
            url,
            params={"query": json.dumps(query)},
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as exc:
        log.warning("Langfuse metrics query failed: %s", exc)
        return []


def _time_range(days: int = 7) -> tuple[str, str]:
    """Return (from_ts, to_ts) ISO strings for the last N days."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")


def query_trace_volume(days: int = 7) -> list[dict]:
    """Traces per day over the last N days."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "traces",
        "metrics": [{"measure": "count", "aggregation": "count"}],
        "dimensions": [],
        "timeDimension": {"granularity": "day"},
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
    })


def query_trace_latency(days: int = 7) -> list[dict]:
    """Trace latency (p50, p95, avg) per day."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "traces",
        "metrics": [
            {"measure": "latency", "aggregation": "p50"},
            {"measure": "latency", "aggregation": "p95"},
            {"measure": "latency", "aggregation": "avg"},
        ],
        "dimensions": [],
        "timeDimension": {"granularity": "day"},
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
    })


def query_model_usage(days: int = 7) -> list[dict]:
    """Token usage and cost grouped by model."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "observations",
        "metrics": [
            {"measure": "count", "aggregation": "count"},
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": [{"field": "providedModelName"}],
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "orderBy": [{"field": "totalCost_sum", "direction": "desc"}],
    })


def query_observation_latency_by_name(days: int = 7) -> list[dict]:
    """Observation latency (p50, p95) grouped by observation name."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "observations",
        "metrics": [
            {"measure": "latency", "aggregation": "p50"},
            {"measure": "latency", "aggregation": "p95"},
            {"measure": "count", "aggregation": "count"},
        ],
        "dimensions": [{"field": "name"}],
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "orderBy": [{"field": "count_count", "direction": "desc"}],
    })


def query_error_rates(days: int = 7) -> list[dict]:
    """Observations at ERROR level grouped by name."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "observations",
        "metrics": [{"measure": "count", "aggregation": "count"}],
        "dimensions": [{"field": "name"}],
        "filters": [
            {"column": "level", "operator": "=", "value": "ERROR", "type": "string"},
        ],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "orderBy": [{"field": "count_count", "direction": "desc"}],
    })


def query_cost_over_time(days: int = 7) -> list[dict]:
    """Total cost per day."""
    from_ts, to_ts = _time_range(days)
    return _langfuse_query({
        "view": "observations",
        "metrics": [{"measure": "totalCost", "aggregation": "sum"}],
        "dimensions": [],
        "timeDimension": {"granularity": "day"},
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
    })


# ---------------------------------------------------------------------------
# Local metrics aggregation
# ---------------------------------------------------------------------------


def _load_all_local_metrics() -> list[dict]:
    """Load all local metrics JSON files, sorted newest first."""
    metrics_dir = Path(METRICS_DIR)
    if not metrics_dir.exists():
        return []

    results: list[dict] = []
    for f in sorted(metrics_dir.glob("*.json"), reverse=True):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except Exception:
            pass
    return results


def aggregate_local_metrics() -> dict[str, Any]:
    """Aggregate local metrics into dashboard-friendly summaries."""
    all_metrics = _load_all_local_metrics()
    if not all_metrics:
        return {
            "total_sessions": 0,
            "sessions": [],
            "avg_duration_secs": 0,
            "avg_conditions": 0,
            "avg_confidence": 0,
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "model_usage": {},
            "tool_usage": {},
            "recommendations_summary": {},
        }

    total = len(all_metrics)
    durations = [m.get("total_duration_secs", 0) for m in all_metrics]
    conditions_counts = [
        m.get("quality", {}).get("total_conditions", 0) for m in all_metrics
    ]
    confidences = [
        m.get("quality", {}).get("avg_condition_confidence", 0) for m in all_metrics
    ]

    # Model usage across sessions
    model_usage: dict[str, dict[str, float]] = {}
    for m in all_metrics:
        for model_name, stats in m.get("llm_calls", {}).get("summary_by_model", {}).items():
            if model_name not in model_usage:
                model_usage[model_name] = {"calls": 0, "total_duration": 0}
            model_usage[model_name]["calls"] += stats.get("count", 0)
            model_usage[model_name]["total_duration"] += stats.get("total_duration_secs", 0)

    # Tool usage across sessions
    tool_usage: dict[str, dict[str, float]] = {}
    for m in all_metrics:
        for tool_name, stats in m.get("tool_calls", {}).get("summary_by_tool", {}).items():
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {"calls": 0, "errors": 0, "total_duration": 0}
            tool_usage[tool_name]["calls"] += stats.get("count", 0)
            tool_usage[tool_name]["errors"] += stats.get("errors", 0)
            tool_usage[tool_name]["total_duration"] += stats.get("total_duration_secs", 0)

    # Recommendations frequency
    rec_summary: dict[str, int] = {}
    for m in all_metrics:
        for rec in m.get("recommendations", []):
            cat = rec.get("category", "unknown")
            rec_summary[cat] = rec_summary.get(cat, 0) + 1

    # Per-session summaries (last 20)
    sessions = []
    for m in all_metrics[:20]:
        sessions.append({
            "session_id": m.get("session_id", ""),
            "query": m.get("query", "")[:100],
            "started_at": m.get("started_at", ""),
            "duration_secs": m.get("total_duration_secs", 0),
            "conditions": m.get("quality", {}).get("total_conditions", 0),
            "confidence": m.get("quality", {}).get("avg_condition_confidence", 0),
            "llm_calls": m.get("llm_calls", {}).get("total_calls", 0),
            "tool_calls": m.get("tool_calls", {}).get("total_calls", 0),
            "domains": m.get("sources", {}).get("domain_count", 0),
        })

    return {
        "total_sessions": total,
        "sessions": sessions,
        "avg_duration_secs": round(sum(durations) / total, 1) if total else 0,
        "avg_conditions": round(sum(conditions_counts) / total, 1) if total else 0,
        "avg_confidence": round(sum(confidences) / total, 3) if total else 0,
        "total_llm_calls": sum(
            m.get("llm_calls", {}).get("total_calls", 0) for m in all_metrics
        ),
        "total_tool_calls": sum(
            m.get("tool_calls", {}).get("total_calls", 0) for m in all_metrics
        ),
        "model_usage": model_usage,
        "tool_usage": tool_usage,
        "recommendations_summary": rec_summary,
    }


# ---------------------------------------------------------------------------
# HTML Dashboard Renderer
# ---------------------------------------------------------------------------


def _safe(text: str) -> str:
    """HTML-escape a string."""
    return html_mod.escape(text, quote=True)


def _format_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    mins = secs / 60
    if mins < 60:
        return f"{mins:.1f}m"
    hours = mins / 60
    return f"{hours:.1f}h"


def _format_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def render_dashboard_html(days: int = 7) -> str:
    """Render the full observability dashboard as a self-contained HTML page.

    Queries both Langfuse (if configured) and local metrics to build a
    comprehensive view of system health.
    """
    langfuse_available = _langfuse_configured()

    # Fetch Langfuse data (parallel would be nice but sync is fine here)
    lf_trace_volume = query_trace_volume(days) if langfuse_available else []
    lf_model_usage = query_model_usage(days) if langfuse_available else []
    lf_obs_latency = query_observation_latency_by_name(days) if langfuse_available else []
    lf_errors = query_error_rates(days) if langfuse_available else []
    lf_cost = query_cost_over_time(days) if langfuse_available else []
    lf_trace_latency = query_trace_latency(days) if langfuse_available else []

    # Local metrics
    local = aggregate_local_metrics()

    # Build the HTML
    langfuse_status = "Connected" if langfuse_available else "Not configured"
    langfuse_badge_color = "#16a34a" if langfuse_available else "#dc2626"

    # --- Langfuse sections ---
    langfuse_sections = ""
    if langfuse_available:
        langfuse_sections = _render_langfuse_sections(
            lf_trace_volume, lf_model_usage, lf_obs_latency,
            lf_errors, lf_cost, lf_trace_latency, days,
        )

    # --- Local metrics sections ---
    local_sections = _render_local_sections(local)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Search Portal — Observability Dashboard</title>
<style>
  :root {{
    --bg: #0f172a;
    --surface: #1e293b;
    --surface2: #334155;
    --border: #475569;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #3b82f6;
    --accent2: #8b5cf6;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --orange: #f97316;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 1.5rem;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.15rem; color: var(--accent); margin: 1.5rem 0 0.75rem; }}
  h3 {{ font-size: 0.95rem; color: var(--text-muted); margin: 1rem 0 0.5rem; }}
  .header {{
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1rem;
  }}
  .header-right {{ text-align: right; font-size: 0.8rem; color: var(--text-muted); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; color: white;
  }}
  .grid {{ display: grid; gap: 1rem; }}
  .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; text-align: center;
  }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
  .stat-label {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
  }}
  th, td {{
    padding: 0.5rem 0.75rem; text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  td {{ color: var(--text); }}
  .bar-container {{
    background: var(--surface2); border-radius: 4px; height: 20px;
    overflow: hidden; position: relative;
  }}
  .bar {{
    height: 100%; border-radius: 4px; transition: width 0.3s;
    display: flex; align-items: center; padding-left: 6px;
    font-size: 0.7rem; font-weight: 600; color: white;
    min-width: 20px;
  }}
  .bar-blue {{ background: var(--accent); }}
  .bar-purple {{ background: var(--accent2); }}
  .bar-green {{ background: var(--green); }}
  .bar-red {{ background: var(--red); }}
  .bar-orange {{ background: var(--orange); }}
  .section {{ margin-bottom: 2rem; }}
  .section-divider {{
    border: none; border-top: 1px solid var(--border);
    margin: 2rem 0;
  }}
  .no-data {{ color: var(--text-muted); font-style: italic; padding: 1rem; text-align: center; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .refresh-link {{
    display: inline-block; margin-top: 1rem;
    padding: 6px 12px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-size: 0.8rem; cursor: pointer;
  }}
  .refresh-link:hover {{ background: var(--border); text-decoration: none; }}
  .truncate {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Deep Search Portal — Observability</h1>
      <span style="font-size: 0.85rem; color: var(--text-muted);">
        Last {days} days &middot; Generated {now_str}
      </span>
    </div>
    <div class="header-right">
      <div>
        Langfuse: <span class="badge" style="background:{langfuse_badge_color}">{langfuse_status}</span>
      </div>
      <div style="margin-top:4px;">
        Local sessions: <strong>{local['total_sessions']}</strong>
      </div>
      <a href="?days={days}" class="refresh-link" style="margin-top:8px;">Refresh</a>
    </div>
  </div>

  <!-- KPI Cards -->
  <div class="grid grid-4" style="margin-bottom:1.5rem;">
    <div class="stat-card">
      <div class="stat-value">{local['total_sessions']}</div>
      <div class="stat-label">Research Sessions</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{_format_duration(local['avg_duration_secs'])}</div>
      <div class="stat-label">Avg Duration</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{local['avg_conditions']:.0f}</div>
      <div class="stat-label">Avg Findings / Session</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{local['avg_confidence']:.1%}</div>
      <div class="stat-label">Avg Confidence</div>
    </div>
  </div>

  <div class="grid grid-4" style="margin-bottom:1.5rem;">
    <div class="stat-card">
      <div class="stat-value">{local['total_llm_calls']}</div>
      <div class="stat-label">Total LLM Calls</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{local['total_tool_calls']}</div>
      <div class="stat-label">Total Tool Calls</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{len(local['model_usage'])}</div>
      <div class="stat-label">Models Used</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{len(local['tool_usage'])}</div>
      <div class="stat-label">Tools Available</div>
    </div>
  </div>

{langfuse_sections}

  <hr class="section-divider">

{local_sections}

</body>
</html>"""


def _render_langfuse_sections(
    trace_volume: list[dict],
    model_usage: list[dict],
    obs_latency: list[dict],
    errors: list[dict],
    cost_over_time: list[dict],
    trace_latency: list[dict],
    days: int,
) -> str:
    """Render the Langfuse-sourced sections of the dashboard."""
    parts: list[str] = []
    parts.append('<h2>Langfuse Metrics</h2>')

    # --- Trace Volume ---
    parts.append('<div class="section"><h3>Trace Volume (per day)</h3>')
    if trace_volume:
        parts.append('<div class="card">')
        max_count = max(
            (float(row.get("count_count", 0)) for row in trace_volume), default=1
        )
        if max_count == 0:
            max_count = 1
        for row in trace_volume:
            date_str = row.get("time", "")[:10]
            count = float(row.get("count_count", 0))
            pct = (count / max_count) * 100
            parts.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="width:80px;font-size:0.8rem;color:var(--text-muted);">{_safe(date_str)}</span>'
                f'<div class="bar-container" style="flex:1;">'
                f'<div class="bar bar-blue" style="width:{pct:.0f}%;">{count:.0f}</div>'
                f'</div></div>'
            )
        parts.append('</div>')
    else:
        parts.append('<div class="no-data">No trace volume data</div>')
    parts.append('</div>')

    # --- Model Usage ---
    parts.append('<div class="section"><h3>Model Usage &amp; Cost</h3>')
    if model_usage:
        parts.append('<div class="card"><table>')
        parts.append(
            '<tr><th>Model</th><th>Calls</th><th>Tokens</th><th>Cost</th></tr>'
        )
        for row in model_usage[:15]:
            model = _safe(str(row.get("providedModelName", "unknown")))
            calls = row.get("count_count", 0)
            tokens = row.get("totalTokens_sum", 0)
            cost = float(row.get("totalCost_sum", 0))
            parts.append(
                f'<tr><td>{model}</td><td>{calls}</td>'
                f'<td>{int(float(tokens)):,}</td>'
                f'<td>{_format_cost(cost)}</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No model usage data</div>')
    parts.append('</div>')

    # --- Trace Latency ---
    parts.append('<div class="section"><h3>Research Latency (per day)</h3>')
    if trace_latency:
        parts.append('<div class="card"><table>')
        parts.append('<tr><th>Date</th><th>p50</th><th>p95</th><th>Avg</th></tr>')
        for row in trace_latency:
            date_str = row.get("time", "")[:10]
            p50 = float(row.get("latency_p50", 0))
            p95 = float(row.get("latency_p95", 0))
            avg = float(row.get("latency_avg", 0))
            parts.append(
                f'<tr><td>{_safe(date_str)}</td>'
                f'<td>{_format_duration(p50 / 1000)}</td>'
                f'<td>{_format_duration(p95 / 1000)}</td>'
                f'<td>{_format_duration(avg / 1000)}</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No latency data</div>')
    parts.append('</div>')

    # --- Observation Latency by Name ---
    parts.append('<div class="section"><h3>Pipeline Step Latency</h3>')
    if obs_latency:
        parts.append('<div class="card"><table>')
        parts.append('<tr><th>Step</th><th>Calls</th><th>p50</th><th>p95</th></tr>')
        for row in obs_latency[:15]:
            name = _safe(str(row.get("name", "unknown")))
            count = row.get("count_count", 0)
            p50 = float(row.get("latency_p50", 0))
            p95 = float(row.get("latency_p95", 0))
            parts.append(
                f'<tr><td>{name}</td><td>{count}</td>'
                f'<td>{_format_duration(p50 / 1000)}</td>'
                f'<td>{_format_duration(p95 / 1000)}</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No observation latency data</div>')
    parts.append('</div>')

    # --- Cost Over Time ---
    parts.append('<div class="section"><h3>Daily Cost</h3>')
    if cost_over_time:
        parts.append('<div class="card">')
        max_cost = max(
            (float(row.get("totalCost_sum", 0)) for row in cost_over_time), default=0.01
        )
        if max_cost == 0:
            max_cost = 0.01
        for row in cost_over_time:
            date_str = row.get("time", "")[:10]
            cost = float(row.get("totalCost_sum", 0))
            pct = (cost / max_cost) * 100
            parts.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="width:80px;font-size:0.8rem;color:var(--text-muted);">{_safe(date_str)}</span>'
                f'<div class="bar-container" style="flex:1;">'
                f'<div class="bar bar-green" style="width:{pct:.0f}%;">{_format_cost(cost)}</div>'
                f'</div></div>'
            )
        parts.append('</div>')
    else:
        parts.append('<div class="no-data">No cost data</div>')
    parts.append('</div>')

    # --- Error Rates ---
    parts.append('<div class="section"><h3>Errors</h3>')
    if errors:
        parts.append('<div class="card"><table>')
        parts.append('<tr><th>Observation</th><th>Error Count</th></tr>')
        for row in errors[:10]:
            name = _safe(str(row.get("name", "unknown")))
            count = row.get("count_count", 0)
            parts.append(f'<tr><td>{name}</td><td style="color:var(--red);">{count}</td></tr>')
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No errors recorded</div>')
    parts.append('</div>')

    return "\n".join(parts)


def _render_local_sections(local: dict[str, Any]) -> str:
    """Render local metrics sections of the dashboard."""
    parts: list[str] = []
    parts.append('<h2>Local Research Metrics</h2>')

    if local["total_sessions"] == 0:
        parts.append(
            '<div class="no-data">No local research sessions recorded yet. '
            'Run a research query to see metrics here.</div>'
        )
        return "\n".join(parts)

    # --- Model Usage (local) ---
    parts.append('<div class="grid grid-2">')

    parts.append('<div class="section"><h3>Model Usage (all sessions)</h3>')
    if local["model_usage"]:
        parts.append('<div class="card"><table>')
        parts.append('<tr><th>Model</th><th>Calls</th><th>Total Time</th></tr>')
        sorted_models = sorted(
            local["model_usage"].items(),
            key=lambda x: x[1]["calls"],
            reverse=True,
        )
        for model_name, stats in sorted_models[:10]:
            parts.append(
                f'<tr><td>{_safe(model_name)}</td>'
                f'<td>{stats["calls"]}</td>'
                f'<td>{_format_duration(stats["total_duration"])}</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No model data</div>')
    parts.append('</div>')

    # --- Tool Usage (local) ---
    parts.append('<div class="section"><h3>Tool Usage (all sessions)</h3>')
    if local["tool_usage"]:
        parts.append('<div class="card"><table>')
        parts.append('<tr><th>Tool</th><th>Calls</th><th>Errors</th><th>Error Rate</th></tr>')
        sorted_tools = sorted(
            local["tool_usage"].items(),
            key=lambda x: x[1]["calls"],
            reverse=True,
        )
        for tool_name, stats in sorted_tools[:15]:
            calls = stats["calls"]
            errs = stats["errors"]
            rate = (errs / calls * 100) if calls > 0 else 0
            rate_color = "var(--green)" if rate < 5 else ("var(--yellow)" if rate < 20 else "var(--red)")
            parts.append(
                f'<tr><td>{_safe(tool_name)}</td>'
                f'<td>{calls}</td>'
                f'<td>{errs}</td>'
                f'<td style="color:{rate_color};">{rate:.1f}%</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No tool data</div>')
    parts.append('</div>')

    parts.append('</div>')  # close grid-2

    # --- Recommendations Summary ---
    if local["recommendations_summary"]:
        parts.append('<div class="section"><h3>Recurring Recommendations</h3>')
        parts.append('<div class="card">')
        sorted_recs = sorted(
            local["recommendations_summary"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        max_count = sorted_recs[0][1] if sorted_recs else 1
        for cat, count in sorted_recs[:10]:
            pct = (count / max_count) * 100
            parts.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="width:200px;font-size:0.8rem;color:var(--text-muted);">{_safe(cat)}</span>'
                f'<div class="bar-container" style="flex:1;">'
                f'<div class="bar bar-orange" style="width:{pct:.0f}%;">{count}</div>'
                f'</div></div>'
            )
        parts.append('</div></div>')

    # --- Recent Sessions Table ---
    parts.append('<div class="section"><h3>Recent Research Sessions</h3>')
    if local["sessions"]:
        parts.append('<div class="card"><table>')
        parts.append(
            '<tr><th>Session</th><th>Query</th><th>Date</th>'
            '<th>Duration</th><th>Findings</th><th>Confidence</th>'
            '<th>LLM Calls</th><th>Sources</th></tr>'
        )
        for s in local["sessions"][:20]:
            sid = _safe(s.get("session_id", "")[:12])
            query = _safe(s.get("query", "")[:60])
            date = _safe(s.get("started_at", "")[:10])
            dur = _format_duration(s.get("duration_secs", 0))
            conds = s.get("conditions", 0)
            conf = s.get("confidence", 0)
            conf_color = (
                "var(--green)" if conf >= 0.7
                else ("var(--yellow)" if conf >= 0.4 else "var(--red)")
            )
            llm = s.get("llm_calls", 0)
            domains = s.get("domains", 0)
            report_link = f'/research/report/{_safe(s.get("session_id", ""))}'
            parts.append(
                f'<tr><td><a href="{report_link}">{sid}...</a></td>'
                f'<td class="truncate" title="{_safe(s.get("query", ""))}">{query}</td>'
                f'<td>{date}</td><td>{dur}</td><td>{conds}</td>'
                f'<td style="color:{conf_color};">{conf:.0%}</td>'
                f'<td>{llm}</td><td>{domains}</td></tr>'
            )
        parts.append('</table></div>')
    else:
        parts.append('<div class="no-data">No sessions</div>')
    parts.append('</div>')

    return "\n".join(parts)
