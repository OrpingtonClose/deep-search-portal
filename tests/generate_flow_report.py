"""Generate an interactive HTML report from a live integration test.

Consumes capture data from the live SSE stream test and produces a
self-contained HTML file with:
  - Mermaid.js pipeline flow graph (node timing, status)
  - SSE event timeline (color-coded, filterable)
  - Phase breakdown with durations
  - Curated event analysis
  - Server metrics integration
  - JSONL event log summary
  - Quality issues and assertion results
  - Summary statistics

Usage (from test):
    from generate_flow_report import render_html_report
    html = render_html_report(
        capture=capture_dict,
        analysis=analysis_dict,
        assertions=assertion_list,
        phases=phase_list,
        curated_events=curated_list,
        server_metrics=metrics_dict,
        server_jsonl=jsonl_list,
    )
    Path("integration_report.html").write_text(html)
"""

from __future__ import annotations

import html as html_mod
import json
from datetime import datetime, timezone


def _esc(text):
    """HTML-escape a string."""
    return html_mod.escape(str(text)) if text else ""


def _fmt_dur(secs):
    if secs is None:
        return "n/a"
    if secs < 0.001:
        return "<1ms"
    if secs < 1:
        return f"{secs * 1000:.0f}ms"
    if secs < 60:
        return f"{secs:.2f}s"
    mins = int(secs // 60)
    remaining = secs % 60
    return f"{mins}m {remaining:.0f}s"


def _color_for_category(cat):
    return {
        "phase": "#2196F3",
        "depth": "#3F51B5",
        "spawn": "#9C27B0",
        "branch_complete": "#4CAF50",
        "all_branches_complete": "#4CAF50",
        "synthesis_start": "#FF9800",
        "status": "#607D8B",
        "finding": "#8BC34A",
        "branch": "#3F51B5",
        "verify": "#00BCD4",
        "phase_message": "#2196F3",
        "critical": "#f44336",
        "warning": "#FF9800",
        "info": "#607D8B",
        "high": "#f44336",
        "medium": "#FF9800",
        "low": "#4CAF50",
    }.get(cat, "#9E9E9E")


def _badge(text, color=None):
    if color is None:
        color = _color_for_category(text)
    return f'<span class="badge" style="background:{color}">{_esc(text)}</span>'


def _build_pipeline_mermaid(server_metrics):
    """Build a Mermaid flowchart of the 7-node pipeline with timing."""
    node_order = [
        "retrieve", "tree_research", "entities", "verify",
        "reflect", "persist", "synthesize",
    ]
    lines = ["graph LR"]

    node_timings = {}
    if isinstance(server_metrics, dict) and "error" not in server_metrics:
        node_timings = server_metrics.get("node_timings", {})

    prev = None
    for name in node_order:
        timing = node_timings.get(name, {})
        if isinstance(timing, dict):
            dur = timing.get("duration_secs", 0)
            status = timing.get("status", "unknown")
        else:
            dur = 0
            status = "unknown"
        label = f"{name}\\n{_fmt_dur(dur)}"
        lines.append(f'    {name}["{label}"]')
        if prev:
            lines.append(f"    {prev} --> {name}")
        prev = name

    # Style nodes
    for name in node_order:
        timing = node_timings.get(name, {})
        if isinstance(timing, dict):
            status = timing.get("status", "unknown")
            dur = timing.get("duration_secs", 0)
        else:
            status = "unknown"
            dur = 0
        if status == "ok" and dur > 0:
            lines.append(f"    style {name} fill:#4CAF50,color:#fff")
        elif status == "error":
            lines.append(f"    style {name} fill:#f44336,color:#fff")
        else:
            lines.append(f"    style {name} fill:#9E9E9E,color:#fff")

    return "\n".join(lines)


def _build_stats_cards(analysis, capture):
    """Build summary statistic cards."""
    cards = []
    timing = analysis.get("timing", {})
    content = analysis.get("content", {})
    events = analysis.get("events", {})
    quality = analysis.get("quality", {})
    ce = analysis.get("curated_events", {})

    stats = [
        ("Total Duration", _fmt_dur(timing.get("total_duration_secs", 0))),
        ("SSE Events", str(events.get("total_sse_events", 0))),
        ("Content", f"{content.get('total_words', 0)} words"),
        ("Think Block", f"{content.get('think_words', 0)} words"),
        ("Phases", str(analysis.get("phases", {}).get("count", 0))),
        ("Curated Events", str(ce.get("total", 0))),
        ("Quality Issues", str(quality.get("issue_count", 0))),
        ("Time to First Content", _fmt_dur(timing.get("time_to_first_content"))),
    ]
    for label, value in stats:
        is_error = (
            (label == "Quality Issues" and value != "0")
            or (label == "Content" and value == "0 words")
        )
        color = "#f44336" if is_error else "#2196F3"
        cards.append(
            f'<div class="stat-card">'
            f'<div class="stat-value" style="color:{color}">{_esc(value)}</div>'
            f'<div class="stat-label">{_esc(label)}</div>'
            f'</div>'
        )
    return "".join(cards)


def _build_phase_timeline(phases):
    """Build an HTML timeline of detected phases."""
    if not phases:
        return "<p>No phases detected.</p>"
    rows = []
    for p in phases:
        elapsed = p.get("elapsed", "")
        elapsed_str = f"+{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        rows.append(
            f'<div class="event-row">'
            f'<span class="ts">{elapsed_str}</span>'
            f'{_badge(p.get("type", "unknown"))}'
            f'<span class="detail">{_esc(p.get("match", ""))}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _build_curated_events_html(curated_events):
    """Build HTML for curated events."""
    if not curated_events:
        return "<p>No curated events detected.</p>"
    rows = []
    for c in curated_events:
        elapsed = c.get("elapsed", 0)
        rows.append(
            f'<div class="curated-evt">'
            f'<span class="ts">+{elapsed:.1f}s</span> '
            f'{_badge(c.get("type", "unknown"))}'
            f' {_esc(c.get("text", "")[:200])}'
            f'</div>'
        )
    return "\n".join(rows)


def _build_sse_timeline(capture, limit=200):
    """Build the HTML for SSE event timeline (last N events with content)."""
    events = capture.get("events", [])
    # Filter to content-bearing events for readability
    content_events = [e for e in events if e.get("content_delta")]
    display_events = content_events[:limit]

    if not display_events:
        return "<p>No content events captured.</p>"

    rows = []
    for evt in display_events:
        elapsed = evt.get("elapsed", 0)
        delta = evt.get("content_delta", "")[:150]
        rows.append(
            f'<div class="event-row">'
            f'<span class="ts">+{elapsed:.2f}s</span>'
            f'<span class="detail">{_esc(delta)}</span>'
            f'</div>'
        )
    total = len(content_events)
    if total > limit:
        rows.append(
            f'<div class="event-row">'
            f'<span class="ts">...</span>'
            f'<span class="detail">({total - limit} more events not shown)</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _build_assertions_html(assertions):
    """Build pass/fail assertion results."""
    if not assertions:
        return "<p>No assertions run.</p>"
    rows = []
    for a in assertions:
        passed = a.get("passed", False)
        icon = "PASS" if passed else "FAIL"
        color = "#4CAF50" if passed else "#f44336"
        reason = f" -- {_esc(a.get('reason', ''))}" if not passed else ""
        rows.append(
            f'<div class="assertion" style="border-left:4px solid {color}">'
            f'<span style="color:{color};font-weight:bold">[{icon}]</span> '
            f'{_esc(a.get("name", ""))}{reason}'
            f'</div>'
        )
    return "\n".join(rows)


def _build_quality_issues(analysis):
    """Build quality issues section."""
    issues = analysis.get("quality", {}).get("issues", [])
    if not issues:
        return "<p>No quality issues detected.</p>"
    rows = []
    for issue in issues:
        severity = issue.get("severity", "info")
        color = _color_for_category(severity)
        rows.append(
            f'<div class="assertion" style="border-left:4px solid {color}">'
            f'{_badge(severity, color)} '
            f'{_esc(issue.get("issue", ""))}'
            f'</div>'
        )
    return "\n".join(rows)


def _build_gaps_html(analysis):
    """Build notable gaps section."""
    gaps = analysis.get("timing", {}).get("notable_gaps", [])
    if not gaps:
        return "<p>No notable gaps (>2s) detected.</p>"
    rows = []
    for g in gaps:
        rows.append(
            f'<div class="event-row">'
            f'<span class="ts">+{g["from_elapsed"]:.1f}s</span>'
            f'{_badge("gap", "#FF9800")}'
            f'<span class="detail">'
            f'{g["gap_secs"]:.1f}s gap (events {g["event_index"]-1} to {g["event_index"]})'
            f'</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _build_server_metrics_html(server_metrics):
    """Build server metrics section."""
    if not isinstance(server_metrics, dict) or "error" in server_metrics:
        error = server_metrics.get("error", "unknown") if isinstance(server_metrics, dict) else "not available"
        return f"<p>Server metrics not available: {_esc(str(error))}</p>"

    sections = []

    # Node timings
    node_timings = server_metrics.get("node_timings", {})
    if node_timings:
        rows = []
        for node, timing in sorted(node_timings.items()):
            if isinstance(timing, dict):
                dur = timing.get("duration_secs", 0)
                status = timing.get("status", "unknown")
                rows.append(f"<tr><td>{_esc(node)}</td><td>{_fmt_dur(dur)}</td><td>{_esc(status)}</td></tr>")
        if rows:
            sections.append(
                '<table class="data-table"><thead>'
                '<tr><th>Node</th><th>Duration</th><th>Status</th></tr>'
                f'</thead><tbody>{"".join(rows)}</tbody></table>'
            )

    # Other metrics as key-value
    skip_keys = {"node_timings", "error"}
    kv_rows = []
    for k, v in server_metrics.items():
        if k in skip_keys:
            continue
        val_str = json.dumps(v, default=str) if not isinstance(v, str) else v
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        kv_rows.append(f"<tr><td><b>{_esc(k)}</b></td><td>{_esc(val_str)}</td></tr>")
    if kv_rows:
        sections.append(
            '<table class="data-table"><thead>'
            '<tr><th>Metric</th><th>Value</th></tr>'
            f'</thead><tbody>{"".join(kv_rows)}</tbody></table>'
        )

    return "\n".join(sections) if sections else "<p>No server metrics data.</p>"


def _build_jsonl_summary(server_jsonl):
    """Build JSONL event log summary."""
    if not server_jsonl:
        return "<p>No JSONL events available.</p>"
    if len(server_jsonl) == 1 and "error" in server_jsonl[0]:
        return f"<p>JSONL error: {_esc(str(server_jsonl[0].get('error', '')))}</p>"

    # Count event types
    type_counts = {}
    for evt in server_jsonl:
        t = evt.get("event", evt.get("type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1

    rows = []
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        rows.append(f"<tr><td>{_esc(t)}</td><td>{count}</td></tr>")

    return (
        f"<p>{len(server_jsonl)} total JSONL events</p>"
        '<table class="data-table"><thead>'
        '<tr><th>Event Type</th><th>Count</th></tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table>'
    )


def render_html_report(
    capture,
    analysis,
    assertions=None,
    phases=None,
    curated_events=None,
    server_metrics=None,
    server_jsonl=None,
):
    """Render the full HTML integration report from live test data.

    Parameters
    ----------
    capture : dict
        Raw SSE stream capture (events, full_content, think_content, etc.)
    analysis : dict
        Analysis results from analyse_stream()
    assertions : list[dict] | None
        List of {name, passed, reason} dicts for invariant checks.
    phases : list[dict] | None
        Detected phase transitions.
    curated_events : list[dict] | None
        Detected curated thought events.
    server_metrics : dict | None
        Server-side metrics JSON.
    server_jsonl : list[dict] | None
        Server-side JSONL event log.
    """
    assertions = assertions or []
    phases = phases or []
    curated_events = curated_events or []
    server_metrics = server_metrics or {}
    server_jsonl = server_jsonl or []

    query = _esc(capture.get("events", [{}])[0].get("raw", "")[:60] if capture.get("events") else "")
    # Use the test query from analysis if available
    timing = analysis.get("timing", {})

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build all sections
    pipeline_mermaid = _build_pipeline_mermaid(server_metrics)
    stats_html = _build_stats_cards(analysis, capture)
    phase_html = _build_phase_timeline(phases)
    curated_html = _build_curated_events_html(curated_events)
    sse_timeline = _build_sse_timeline(capture)
    assertions_html = _build_assertions_html(assertions)
    quality_html = _build_quality_issues(analysis)
    gaps_html = _build_gaps_html(analysis)
    metrics_html = _build_server_metrics_html(server_metrics)
    jsonl_html = _build_jsonl_summary(server_jsonl)

    # Assertion summary
    passed = sum(1 for a in assertions if a["passed"])
    total = len(assertions)
    pass_color = "#4CAF50" if passed == total else "#f44336"

    response_id = _esc(capture.get("response_id", "none"))
    model = _esc(capture.get("model", "unknown"))
    error = capture.get("error", "")
    error_banner = ""
    if error:
        error_banner = (
            f'<div style="background:#f44336;color:#fff;padding:12px;'
            f'border-radius:8px;margin:16px 0;font-weight:bold">'
            f'Stream Error: {_esc(error)}</div>'
        )

    content_preview = _esc(capture.get("full_content", "")[:1000])
    think_preview = _esc(capture.get("think_content", "")[:500])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Integration Test Report</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
:root {{
    --bg: #121212;
    --surface: #1e1e1e;
    --text: #e0e0e0;
    --text-dim: #999;
    --border: #333;
    --accent: #2196F3;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
}}
h1 {{ color: var(--accent); margin-bottom: 4px; }}
h2 {{
    color: var(--text);
    margin: 24px 0 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    cursor: pointer;
}}
h2:hover {{ color: var(--accent); }}
.subtitle {{ color: var(--text-dim); margin-bottom: 24px; }}
.meta {{ color: var(--text-dim); font-size: 0.85em; margin-bottom: 16px; }}
.stats-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    min-width: 120px;
    text-align: center;
}}
.stat-value {{ font-size: 1.8em; font-weight: bold; }}
.stat-label {{ font-size: 0.85em; color: var(--text-dim); }}
section {{ margin-bottom: 32px; }}
.mermaid {{
    background: #fff;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    overflow-x: auto;
}}
.event-row {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9em;
}}
.event-row:hover {{ background: rgba(255,255,255,0.03); }}
.ts {{
    color: var(--text-dim);
    min-width: 80px;
    font-family: monospace;
    font-size: 0.85em;
}}
.badge {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    color: #fff;
    min-width: 70px;
    text-align: center;
    white-space: nowrap;
}}
.detail {{ flex: 1; }}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88em;
    margin: 12px 0;
}}
.data-table th {{
    background: var(--surface);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}}
.data-table td {{
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.data-table tr:hover {{ background: rgba(255,255,255,0.03); }}
.assertion {{
    padding: 8px 16px;
    margin: 4px 0;
    background: var(--surface);
    border-radius: 4px;
}}
.curated-evt {{
    padding: 6px 12px;
    margin: 4px 0;
    background: var(--surface);
    border-left: 3px solid #00BCD4;
    border-radius: 0 4px 4px 0;
    font-size: 0.9em;
}}
.content-preview {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-size: 0.9em;
}}
.collapsible-content {{ display: none; }}
.collapsible-content.show {{ display: block; }}
</style>
</head>
<body>
<h1>Live Integration Test Report</h1>
<div class="subtitle">Generated {generated_at}</div>
<div class="meta">
    Response ID: {response_id} | Model: {model} |
    Duration: {_fmt_dur(timing.get('total_duration_secs', 0))}
</div>

{error_banner}

<section>
    <h2>Summary</h2>
    <div class="stats-row">{stats_html}</div>
</section>

<section>
    <h2>Assertions
        <span style="font-size:0.7em;color:{pass_color}">
            ({passed}/{total} passed)
        </span>
    </h2>
    {assertions_html}
</section>

<section>
    <h2>Pipeline Flow</h2>
    <p>7-node LangGraph pipeline with server-side timing.</p>
    <div class="mermaid">{pipeline_mermaid}</div>
</section>

<section>
    <h2>Quality Issues</h2>
    {quality_html}
</section>

<section>
    <h2 onclick="toggleSection('phases-content')">Phases Detected ({len(phases)})</h2>
    <div id="phases-content" class="collapsible-content show">
        {phase_html}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('curated-content')">Curated Events ({len(curated_events)})</h2>
    <div id="curated-content" class="collapsible-content show">
        {curated_html}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('gaps-content')">Notable Gaps</h2>
    <div id="gaps-content" class="collapsible-content show">
        {gaps_html}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('sse-content')">SSE Event Timeline (content events)</h2>
    <div id="sse-content" class="collapsible-content">
        {sse_timeline}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('metrics-content')">Server Metrics</h2>
    <div id="metrics-content" class="collapsible-content show">
        {metrics_html}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('jsonl-content')">Server JSONL Log</h2>
    <div id="jsonl-content" class="collapsible-content">
        {jsonl_html}
    </div>
</section>

<section>
    <h2 onclick="toggleSection('content-preview')">Final Content Preview</h2>
    <div id="content-preview" class="collapsible-content">
        <div class="content-preview">{content_preview if content_preview else '(empty)'}</div>
    </div>
</section>

<section>
    <h2 onclick="toggleSection('think-preview')">Think Block Preview</h2>
    <div id="think-preview" class="collapsible-content">
        <div class="content-preview">{think_preview if think_preview else '(empty)'}</div>
    </div>
</section>

<script>
mermaid.initialize({{ theme: 'default', securityLevel: 'loose' }});

function toggleSection(id) {{
    var el = document.getElementById(id);
    if (el) el.classList.toggle('show');
}}
</script>
</body>
</html>"""
