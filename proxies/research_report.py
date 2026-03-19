"""HTML Report Generator for research sessions.

Generates self-contained HTML reports with:
  - Executive summary (query, duration, key stats)
  - Research angles with findings per angle (collapsible)
  - Source quality breakdown (trust tiers, confidence distribution)
  - Tool usage summary
  - Censorship/filtering warnings
  - Cost breakdown
  - Novelty curves per subagent (inline SVG)
  - Full conditions list (searchable, filterable)
  - LLM performance metrics for analyst consumption
  - Timeline of pipeline phases

All CSS/JS is inlined — no external dependencies.
"""

from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("research_report")

REPORTS_DIR = os.getenv(
    "RESEARCH_REPORTS_DIR",
    "/opt/persistent_research_logs/reports",
)


def generate_report(
    metrics: dict[str, Any],
    conditions: list[dict],
    final_answer: str = "",
    subagent_results: list[dict] | None = None,
    progress_log: list[str] | None = None,
) -> str:
    """Generate a self-contained HTML report from metrics and conditions.

    Args:
        metrics: The metrics dict from ResearchMetrics.to_dict().
        conditions: List of condition dicts (fact, confidence, trust_score, angle, source_url, ...).
        final_answer: The final synthesised answer text.
        subagent_results: Optional raw subagent result data.
        progress_log: Optional progress log messages.

    Returns:
        Complete HTML string.
    """
    session_id = metrics.get("session_id", "unknown")
    query = metrics.get("query", "")
    started_at = metrics.get("started_at", "")
    finished_at = metrics.get("finished_at", "")
    total_duration = metrics.get("total_duration_secs", 0)

    pipeline = metrics.get("pipeline", {})
    node_timings = pipeline.get("node_timings", [])
    slowest_node = pipeline.get("slowest_node", "")
    slowest_dur = pipeline.get("slowest_node_duration", 0)

    llm_data = metrics.get("llm_calls", {})
    tool_data = metrics.get("tool_calls", {})
    subagent_data = metrics.get("subagents", {})
    quality = metrics.get("quality", {})
    sources = metrics.get("sources", {})
    cost = metrics.get("cost", {})
    efficiency = metrics.get("efficiency", {})
    recommendations = metrics.get("recommendations", [])

    # Build sections
    sections: list[str] = []

    # --- Executive Summary ---
    sections.append(_section_executive_summary(
        session_id, query, started_at, finished_at, total_duration,
        quality, subagent_data, llm_data, tool_data, sources,
    ))

    # --- Timeline ---
    sections.append(_section_timeline(node_timings, total_duration))

    # --- Final Answer ---
    if final_answer:
        sections.append(_section_final_answer(final_answer))

    # --- Research Angles & Findings ---
    sections.append(_section_findings_by_angle(conditions))

    # --- Source Quality ---
    sections.append(_section_source_quality(quality, sources))

    # --- Subagent Performance ---
    sections.append(_section_subagent_performance(subagent_data))

    # --- Novelty Curves ---
    sections.append(_section_novelty_curves(subagent_data))

    # --- Tool Usage ---
    sections.append(_section_tool_usage(tool_data))

    # --- LLM Performance ---
    sections.append(_section_llm_performance(llm_data))

    # --- Efficiency & Recommendations ---
    sections.append(_section_efficiency(efficiency, recommendations))

    # --- Cost Breakdown ---
    if cost:
        sections.append(_section_cost(cost))

    # --- Warnings ---
    warnings = _collect_warnings(conditions, quality, recommendations)
    if warnings:
        sections.append(_section_warnings(warnings))

    # --- Full Conditions Table ---
    sections.append(_section_conditions_table(conditions))

    # --- Progress Log ---
    if progress_log:
        sections.append(_section_progress_log(progress_log))

    # --- Metrics JSON (embedded for LLM consumption) ---
    sections.append(_section_raw_metrics(metrics))

    body = "\n".join(sections)

    return _wrap_html(session_id, query, started_at, body)


def save_report(html_content: str, session_id: str) -> str:
    """Save HTML report to disk. Returns the file path."""
    try:
        Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(f"Failed to create reports dir {REPORTS_DIR}: {e}")

    path = os.path.join(REPORTS_DIR, f"{session_id}.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        log.info(f"Saved report to {path}")
    except Exception as e:
        log.error(f"Failed to save report: {e}")
    return path


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _section_executive_summary(
    session_id, query, started_at, finished_at, total_duration,
    quality, subagent_data, llm_data, tool_data, sources,
) -> str:
    total_conditions = quality.get("total_conditions", 0)
    avg_confidence = quality.get("avg_condition_confidence", 0)
    n_subagents = subagent_data.get("count", 0)
    n_llm_calls = llm_data.get("total_calls", 0)
    n_tool_calls = tool_data.get("total_calls", 0)
    n_domains = sources.get("domain_count", 0)
    diversity = sources.get("diversity_score", 0)

    return f"""
    <section class="card" id="summary">
      <h2>Executive Summary</h2>
      <div class="stats-grid">
        <div class="stat">
          <span class="stat-value">{_esc(query[:120])}</span>
          <span class="stat-label">Query</span>
        </div>
        <div class="stat">
          <span class="stat-value">{total_duration:.1f}s</span>
          <span class="stat-label">Total Duration</span>
        </div>
        <div class="stat">
          <span class="stat-value">{total_conditions}</span>
          <span class="stat-label">Conditions Found</span>
        </div>
        <div class="stat">
          <span class="stat-value">{avg_confidence:.2f}</span>
          <span class="stat-label">Avg Confidence</span>
        </div>
        <div class="stat">
          <span class="stat-value">{n_subagents}</span>
          <span class="stat-label">Subagents</span>
        </div>
        <div class="stat">
          <span class="stat-value">{n_llm_calls}</span>
          <span class="stat-label">LLM Calls</span>
        </div>
        <div class="stat">
          <span class="stat-value">{n_tool_calls}</span>
          <span class="stat-label">Tool Calls</span>
        </div>
        <div class="stat">
          <span class="stat-value">{n_domains}</span>
          <span class="stat-label">Unique Domains</span>
        </div>
        <div class="stat">
          <span class="stat-value">{diversity:.2f}</span>
          <span class="stat-label">Diversity Score</span>
        </div>
      </div>
      <div class="meta">
        <span>Session: <code>{_esc(session_id)}</code></span>
        <span>Started: {_esc(started_at)}</span>
        <span>Finished: {_esc(finished_at)}</span>
      </div>
    </section>
    """


def _section_timeline(node_timings: list[dict], total_duration: float) -> str:
    if not node_timings:
        return ""

    bars = []
    for nt in node_timings:
        name = nt.get("node_name", "?")
        dur = nt.get("duration_secs", 0)
        pct = (dur / max(total_duration, 0.01)) * 100
        bars.append(
            f'<div class="timeline-bar" style="width:{max(pct, 1):.1f}%" '
            f'title="{_esc(name)}: {dur:.2f}s ({pct:.1f}%)">'
            f'<span class="timeline-label">{_esc(name)}</span>'
            f'<span class="timeline-dur">{dur:.1f}s</span>'
            f'</div>'
        )

    return f"""
    <section class="card" id="timeline">
      <h2>Pipeline Timeline</h2>
      <div class="timeline-container">
        {"".join(bars)}
      </div>
      <div class="timeline-legend">
        {" &rarr; ".join(f'<span>{_esc(nt.get("node_name", "?"))}</span>' for nt in node_timings)}
      </div>
    </section>
    """


def _section_final_answer(answer: str) -> str:
    # Convert markdown-ish formatting to basic HTML
    formatted = _esc(answer).replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"""
    <section class="card" id="answer">
      <h2>Final Answer</h2>
      <div class="answer-content">
        <p>{formatted}</p>
      </div>
    </section>
    """


def _section_findings_by_angle(conditions: list[dict]) -> str:
    by_angle: dict[str, list[dict]] = {}
    for c in conditions:
        angle = c.get("angle", "unknown")
        by_angle.setdefault(angle, []).append(c)

    if not by_angle:
        return ""

    angle_blocks = []
    for angle, conds in sorted(by_angle.items()):
        high = sum(1 for c in conds if c.get("confidence", 0) >= 0.7)
        rows = []
        for c in conds:
            conf = c.get("confidence", 0.5)
            conf_class = "conf-high" if conf >= 0.7 else ("conf-med" if conf >= 0.4 else "conf-low")
            trust = c.get("trust_score", 0.5)
            source = c.get("source_url", "")
            source_link = f'<a href="{_esc(source)}" target="_blank" rel="noopener">{_esc(source[:60])}</a>' if source else "N/A"
            serendip = " [S]" if c.get("is_serendipitous") else ""

            rows.append(f"""
            <tr>
              <td class="fact-cell">{_esc(c.get("fact", ""))}{serendip}</td>
              <td class="{conf_class}">{conf:.2f}</td>
              <td>{trust:.2f}</td>
              <td class="source-cell">{source_link}</td>
            </tr>""")

        angle_blocks.append(f"""
        <details class="angle-details">
          <summary>
            <strong>{_esc(angle)}</strong>
            <span class="angle-badge">{len(conds)} conditions, {high} high-confidence</span>
          </summary>
          <table class="conditions-table">
            <thead><tr><th>Finding</th><th>Confidence</th><th>Trust</th><th>Source</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </details>""")

    return f"""
    <section class="card" id="findings">
      <h2>Findings by Research Angle</h2>
      {"".join(angle_blocks)}
    </section>
    """


def _section_source_quality(quality: dict, sources: dict) -> str:
    conf = quality.get("confidence_distribution", {})
    trust = quality.get("trust_distribution", {})
    domains = sources.get("unique_domains", [])

    conf_items = "".join(
        f"<li><strong>{_esc(k)}</strong>: {v}</li>" for k, v in conf.items()
    )
    trust_items = "".join(
        f"<li><strong>{_esc(k)}</strong>: {v}</li>" for k, v in trust.items()
    )
    domain_list = ", ".join(f"<code>{_esc(d)}</code>" for d in domains[:30])
    if len(domains) > 30:
        domain_list += f" ... and {len(domains) - 30} more"

    return f"""
    <section class="card" id="source-quality">
      <h2>Source Quality</h2>
      <div class="two-col">
        <div>
          <h3>Confidence Distribution</h3>
          <ul>{conf_items}</ul>
        </div>
        <div>
          <h3>Trust Distribution</h3>
          <ul>{trust_items}</ul>
        </div>
      </div>
      <h3>Unique Domains ({len(domains)})</h3>
      <p class="domain-list">{domain_list if domains else "No domains recorded"}</p>
      <p>Serendipitous findings: <strong>{quality.get("serendipitous_findings", 0)}</strong></p>
      <p>Reflection quality: <strong>{quality.get("reflection_quality_score", 0):.2f}</strong></p>
    </section>
    """


def _section_subagent_performance(subagent_data: dict) -> str:
    records = subagent_data.get("records", [])
    summary = subagent_data.get("summary", {})

    if not records:
        return ""

    rows = []
    for r in records:
        error_class = ' class="error-row"' if r.get("error") else ""
        rows.append(f"""
        <tr{error_class}>
          <td>{r.get("index", 0) + 1}</td>
          <td>{_esc(r.get("angle", "")[:50])}</td>
          <td>{r.get("turns_used", 0)}</td>
          <td>{r.get("tool_calls_made", 0)}</td>
          <td>{r.get("conditions_found", 0)}</td>
          <td>{r.get("children_spawned", 0)}</td>
          <td>{r.get("duration_secs", 0):.1f}s</td>
          <td>{_esc(r.get("error", "")[:60]) if r.get("error") else "OK"}</td>
        </tr>""")

    summary_html = ""
    if summary:
        summary_html = f"""
        <div class="stats-grid compact">
          <div class="stat"><span class="stat-value">{summary.get("avg_turns_per_agent", 0):.1f}</span><span class="stat-label">Avg Turns</span></div>
          <div class="stat"><span class="stat-value">{summary.get("avg_conditions_per_agent", 0):.1f}</span><span class="stat-label">Avg Conditions</span></div>
          <div class="stat"><span class="stat-value">{summary.get("total_children_spawned", 0)}</span><span class="stat-label">Children Spawned</span></div>
          <div class="stat"><span class="stat-value">{summary.get("agents_with_errors", 0)}</span><span class="stat-label">Agents with Errors</span></div>
        </div>"""

    return f"""
    <section class="card" id="subagents">
      <h2>Subagent Performance</h2>
      {summary_html}
      <table class="data-table">
        <thead><tr><th>#</th><th>Angle</th><th>Turns</th><th>Tools</th><th>Conditions</th><th>Children</th><th>Duration</th><th>Status</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _section_novelty_curves(subagent_data: dict) -> str:
    records = subagent_data.get("records", [])
    curves_with_data = [r for r in records if r.get("novelty_history")]

    if not curves_with_data:
        return ""

    svg_charts = []
    for r in curves_with_data:
        history = r["novelty_history"]
        angle = r.get("angle", "?")[:40]
        svg_charts.append(_mini_svg_line_chart(history, angle, width=300, height=80))

    return f"""
    <section class="card" id="novelty">
      <h2>Novelty Curves</h2>
      <p class="hint">Novelty score per turn for each subagent. Declining curves indicate research saturation.</p>
      <div class="novelty-grid">
        {"".join(svg_charts)}
      </div>
    </section>
    """


def _mini_svg_line_chart(
    values: list[float], label: str, width: int = 300, height: int = 80,
) -> str:
    """Generate an inline SVG line chart."""
    if not values:
        return ""

    n = len(values)
    max_val = max(max(values), 0.01)
    padding = 5

    points = []
    for i, v in enumerate(values):
        x = padding + (i / max(n - 1, 1)) * (width - 2 * padding)
        y = height - padding - (v / max_val) * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Threshold line at 0.3 (expand threshold)
    thresh_y = height - padding - (0.3 / max_val) * (height - 2 * padding)

    return f"""
    <div class="novelty-chart">
      <svg width="{width}" height="{height + 20}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="{height}" fill="#1a1a2e" rx="4"/>
        <line x1="{padding}" y1="{thresh_y:.1f}" x2="{width - padding}" y2="{thresh_y:.1f}"
              stroke="#555" stroke-dasharray="4,4" stroke-width="1"/>
        <polyline points="{polyline}" fill="none" stroke="#4fc3f7" stroke-width="2"/>
        <text x="{width//2}" y="{height + 15}" text-anchor="middle" fill="#aaa" font-size="11">{_esc(label)}</text>
      </svg>
    </div>
    """


def _section_tool_usage(tool_data: dict) -> str:
    summary = tool_data.get("summary_by_tool", {})
    if not summary:
        return ""

    rows = []
    for tool_name, stats in sorted(summary.items(), key=lambda x: x[1].get("count", 0), reverse=True):
        rows.append(f"""
        <tr>
          <td><code>{_esc(tool_name)}</code></td>
          <td>{stats.get("count", 0)}</td>
          <td>{stats.get("total_duration", 0):.2f}s</td>
          <td>{stats.get("avg_duration", 0):.3f}s</td>
          <td>{stats.get("errors", 0)}</td>
          <td>{stats.get("total_result_chars", 0):,}</td>
        </tr>""")

    return f"""
    <section class="card" id="tools">
      <h2>Tool Usage</h2>
      <table class="data-table">
        <thead><tr><th>Tool</th><th>Calls</th><th>Total Time</th><th>Avg Time</th><th>Errors</th><th>Result Chars</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _section_llm_performance(llm_data: dict) -> str:
    summary = llm_data.get("summary_by_model", {})
    records = llm_data.get("records", [])

    if not summary:
        return ""

    model_rows = []
    for model, stats in sorted(summary.items()):
        model_rows.append(f"""
        <tr>
          <td><code>{_esc(model)}</code></td>
          <td>{stats.get("count", 0)}</td>
          <td>{stats.get("total_duration", 0):.2f}s</td>
          <td>{stats.get("avg_duration", 0):.3f}s</td>
          <td>{stats.get("total_tokens_est", 0):,}</td>
          <td>{stats.get("errors", 0)}</td>
        </tr>""")

    # Top 10 slowest LLM calls
    sorted_calls = sorted(records, key=lambda r: r.get("duration_secs", 0), reverse=True)[:10]
    slow_rows = []
    for r in sorted_calls:
        slow_rows.append(f"""
        <tr>
          <td><code>{_esc(r.get("model", "?"))}</code></td>
          <td>{r.get("duration_secs", 0):.2f}s</td>
          <td>{r.get("total_tokens_est", 0):,}</td>
          <td>{_esc(r.get("node_context", ""))}</td>
          <td>{_esc(r.get("error", "")) if r.get("error") else "OK"}</td>
        </tr>""")

    return f"""
    <section class="card" id="llm">
      <h2>LLM Performance</h2>
      <h3>Summary by Model</h3>
      <table class="data-table">
        <thead><tr><th>Model</th><th>Calls</th><th>Total Time</th><th>Avg Time</th><th>Est. Tokens</th><th>Errors</th></tr></thead>
        <tbody>{"".join(model_rows)}</tbody>
      </table>
      <h3>Top 10 Slowest Calls</h3>
      <table class="data-table">
        <thead><tr><th>Model</th><th>Duration</th><th>Est. Tokens</th><th>Node</th><th>Status</th></tr></thead>
        <tbody>{"".join(slow_rows)}</tbody>
      </table>
    </section>
    """


def _section_efficiency(efficiency: dict, recommendations: list[dict]) -> str:
    cpt = efficiency.get("conditions_per_tool_call", 0)
    avg_tool = efficiency.get("avg_tool_call_duration_secs", 0)
    avg_llm = efficiency.get("avg_llm_call_duration_secs", 0)

    rec_items = ""
    if recommendations:
        for rec in recommendations:
            sev = rec.get("severity", "info")
            sev_class = f"rec-{sev}"
            rec_items += f"""
            <div class="recommendation {sev_class}">
              <span class="rec-badge">{_esc(sev.upper())}</span>
              <span class="rec-cat">[{_esc(rec.get("category", ""))}]</span>
              {_esc(rec.get("message", ""))}
              <code class="rec-evidence">{_esc(rec.get("evidence", ""))}</code>
            </div>"""

    return f"""
    <section class="card" id="efficiency">
      <h2>Efficiency & Recommendations</h2>
      <div class="stats-grid compact">
        <div class="stat"><span class="stat-value">{cpt:.3f}</span><span class="stat-label">Conditions/Tool Call</span></div>
        <div class="stat"><span class="stat-value">{avg_tool:.3f}s</span><span class="stat-label">Avg Tool Duration</span></div>
        <div class="stat"><span class="stat-value">{avg_llm:.3f}s</span><span class="stat-label">Avg LLM Duration</span></div>
      </div>
      {f'<h3>Recommendations</h3>{rec_items}' if rec_items else '<p class="hint">No recommendations generated.</p>'}
    </section>
    """


def _section_cost(cost: dict) -> str:
    if not cost:
        return ""

    items = "".join(
        f"<tr><td>{_esc(str(k))}</td><td>{_esc(str(v))}</td></tr>"
        for k, v in cost.items()
    )

    return f"""
    <section class="card" id="cost">
      <h2>Cost Breakdown</h2>
      <table class="data-table">
        <thead><tr><th>Item</th><th>Value</th></tr></thead>
        <tbody>{items}</tbody>
      </table>
    </section>
    """


def _collect_warnings(conditions: list[dict], quality: dict, recommendations: list[dict]) -> list[str]:
    warnings: list[str] = []

    # Censorship warnings
    for c in conditions:
        if c.get("censorship_warning"):
            warnings.append(f"Censorship warning for source {c.get('source_url', '?')}: {c['censorship_warning']}")

    # Low quality
    if quality.get("reflection_quality_score", 1.0) < 0.4:
        warnings.append(f"Low reflection quality score: {quality['reflection_quality_score']:.2f}")

    # Error recommendations
    for rec in recommendations:
        if rec.get("severity") == "error":
            warnings.append(rec.get("message", ""))

    return warnings


def _section_warnings(warnings: list[str]) -> str:
    items = "".join(f"<li class='warning-item'>{_esc(w)}</li>" for w in warnings)
    return f"""
    <section class="card warning-card" id="warnings">
      <h2>Warnings</h2>
      <ul>{items}</ul>
    </section>
    """


def _section_conditions_table(conditions: list[dict]) -> str:
    if not conditions:
        return ""

    rows = []
    for i, c in enumerate(conditions):
        conf = c.get("confidence", 0.5)
        conf_class = "conf-high" if conf >= 0.7 else ("conf-med" if conf >= 0.4 else "conf-low")
        source = c.get("source_url", "")
        source_link = f'<a href="{_esc(source)}" target="_blank">{_esc(source[:50])}</a>' if source else ""

        rows.append(f"""
        <tr class="condition-row">
          <td>{i + 1}</td>
          <td class="fact-cell">{_esc(c.get("fact", ""))}</td>
          <td>{_esc(c.get("angle", ""))}</td>
          <td class="{conf_class}">{conf:.2f}</td>
          <td>{c.get("trust_score", 0.5):.2f}</td>
          <td>{source_link}</td>
        </tr>""")

    return f"""
    <section class="card" id="all-conditions">
      <h2>All Conditions ({len(conditions)})</h2>
      <input type="text" id="condition-search" placeholder="Search conditions..."
             onkeyup="filterConditions()" class="search-input"/>
      <table class="data-table conditions-full" id="conditions-table">
        <thead><tr><th>#</th><th>Fact</th><th>Angle</th><th>Confidence</th><th>Trust</th><th>Source</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _section_progress_log(progress_log: list[str]) -> str:
    lines = "".join(_esc(line) for line in progress_log)
    return f"""
    <details class="card" id="progress-log">
      <summary><h2 style="display:inline">Progress Log</h2></summary>
      <pre class="log-content">{lines}</pre>
    </details>
    """


def _section_raw_metrics(metrics: dict) -> str:
    metrics_json = json.dumps(metrics, indent=2, default=str)
    return f"""
    <details class="card" id="raw-metrics">
      <summary><h2 style="display:inline">Raw Metrics JSON (for LLM analysis)</h2></summary>
      <pre class="json-content">{_esc(metrics_json)}</pre>
    </details>
    """


# ---------------------------------------------------------------------------
# HTML Wrapper
# ---------------------------------------------------------------------------

def _wrap_html(session_id: str, query: str, started_at: str, body: str) -> str:
    """Wrap sections in a complete self-contained HTML document."""
    title = f"Research Report: {query[:80]}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
:root {{
  --bg: #0f0f23;
  --card-bg: #1a1a2e;
  --text: #e0e0e0;
  --text-dim: #888;
  --accent: #4fc3f7;
  --accent2: #81c784;
  --warning: #ffb74d;
  --error: #ef5350;
  --border: #2a2a4a;
  --conf-high: #66bb6a;
  --conf-med: #ffa726;
  --conf-low: #ef5350;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}}
h1 {{ color: var(--accent); margin-bottom: 10px; font-size: 1.6em; }}
h2 {{ color: var(--accent); margin-bottom: 12px; font-size: 1.3em; }}
h3 {{ color: var(--text); margin: 12px 0 8px; font-size: 1.1em; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ background: #2a2a4a; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}

.header {{
  border-bottom: 2px solid var(--accent);
  padding-bottom: 15px;
  margin-bottom: 20px;
}}
.header p {{ color: var(--text-dim); font-size: 0.9em; }}

.card {{
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}}
.warning-card {{
  border-color: var(--warning);
  background: #2a1f0f;
}}

.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  margin: 12px 0;
}}
.stats-grid.compact {{
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
}}
.stat {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  text-align: center;
}}
.stat-value {{
  display: block;
  font-size: 1.4em;
  font-weight: bold;
  color: var(--accent);
}}
.stat-label {{
  display: block;
  font-size: 0.8em;
  color: var(--text-dim);
  margin-top: 4px;
}}

.meta {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-top: 12px;
  font-size: 0.85em;
  color: var(--text-dim);
}}

/* Timeline */
.timeline-container {{
  display: flex;
  gap: 2px;
  margin: 12px 0;
  height: 40px;
  border-radius: 6px;
  overflow: hidden;
}}
.timeline-bar {{
  background: var(--accent);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-width: 30px;
  opacity: 0.8;
  transition: opacity 0.2s;
  cursor: pointer;
}}
.timeline-bar:hover {{ opacity: 1; }}
.timeline-bar:nth-child(even) {{ background: var(--accent2); }}
.timeline-label {{ font-size: 0.65em; font-weight: bold; color: #000; white-space: nowrap; overflow: hidden; }}
.timeline-dur {{ font-size: 0.6em; color: #333; }}
.timeline-legend {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 0.8em;
  color: var(--text-dim);
  margin-top: 6px;
}}

/* Answer */
.answer-content {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  max-height: 600px;
  overflow-y: auto;
}}

/* Tables */
.data-table, .conditions-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85em;
  margin-top: 8px;
}}
.data-table th, .conditions-table th {{
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 8px;
  text-align: left;
  font-weight: 600;
  position: sticky;
  top: 0;
}}
.data-table td, .conditions-table td {{
  border: 1px solid var(--border);
  padding: 6px 8px;
  vertical-align: top;
}}
.data-table tr:hover, .conditions-table tr:hover {{
  background: rgba(79, 195, 247, 0.05);
}}
.error-row {{ background: rgba(239, 83, 80, 0.1); }}
.fact-cell {{ max-width: 500px; word-wrap: break-word; }}
.source-cell {{ max-width: 200px; word-wrap: break-word; }}

.conf-high {{ color: var(--conf-high); font-weight: bold; }}
.conf-med {{ color: var(--conf-med); }}
.conf-low {{ color: var(--conf-low); }}

/* Details/Collapsible */
details {{ margin: 8px 0; }}
summary {{
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
  user-select: none;
}}
summary:hover {{ background: rgba(255,255,255,0.03); }}
.angle-details {{ margin: 6px 0; }}
.angle-badge {{
  background: var(--accent);
  color: #000;
  font-size: 0.75em;
  padding: 2px 8px;
  border-radius: 10px;
  margin-left: 8px;
}}

/* Search */
.search-input {{
  width: 100%;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font-size: 0.9em;
  margin-bottom: 10px;
}}
.search-input:focus {{
  outline: none;
  border-color: var(--accent);
}}

/* Recommendations */
.recommendation {{
  padding: 10px 14px;
  margin: 6px 0;
  border-radius: 6px;
  border-left: 4px solid var(--text-dim);
  background: var(--bg);
}}
.rec-error {{ border-left-color: var(--error); }}
.rec-warning {{ border-left-color: var(--warning); }}
.rec-info {{ border-left-color: var(--accent); }}
.rec-badge {{
  display: inline-block;
  font-size: 0.7em;
  font-weight: bold;
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 6px;
}}
.rec-error .rec-badge {{ background: var(--error); color: #fff; }}
.rec-warning .rec-badge {{ background: var(--warning); color: #000; }}
.rec-info .rec-badge {{ background: var(--accent); color: #000; }}
.rec-cat {{ color: var(--text-dim); font-size: 0.85em; margin-right: 4px; }}
.rec-evidence {{ display: block; margin-top: 4px; font-size: 0.8em; color: var(--text-dim); }}

/* Novelty */
.novelty-grid {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}}
.novelty-chart {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px;
}}

/* Log */
.log-content, .json-content {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  font-size: 0.8em;
  max-height: 500px;
  overflow: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}}

.warning-item {{
  color: var(--warning);
  margin: 6px 0;
  list-style: none;
}}
.warning-item::before {{ content: "\\26A0 "; }}

.hint {{ color: var(--text-dim); font-size: 0.85em; margin-bottom: 8px; }}

.domain-list {{ font-size: 0.85em; line-height: 1.8; }}

.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}

/* Nav */
.nav {{
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 8px 0;
  margin-bottom: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
.nav a {{
  font-size: 0.8em;
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--card-bg);
  border: 1px solid var(--border);
}}
.nav a:hover {{ background: var(--border); text-decoration: none; }}

@media (max-width: 768px) {{
  .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .two-col {{ grid-template-columns: 1fr; }}
  body {{ padding: 10px; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>Deep Research Report</h1>
  <p>Session: {_esc(session_id)} | Generated: {_esc(started_at)}</p>
</div>

<nav class="nav">
  <a href="#summary">Summary</a>
  <a href="#timeline">Timeline</a>
  <a href="#answer">Answer</a>
  <a href="#findings">Findings</a>
  <a href="#source-quality">Sources</a>
  <a href="#subagents">Subagents</a>
  <a href="#novelty">Novelty</a>
  <a href="#tools">Tools</a>
  <a href="#llm">LLM</a>
  <a href="#efficiency">Efficiency</a>
  <a href="#all-conditions">Conditions</a>
  <a href="#raw-metrics">Raw JSON</a>
</nav>

{body}

<footer style="text-align:center; color:var(--text-dim); font-size:0.8em; margin-top:30px; padding:20px;">
  Generated by Deep Search Portal &mdash; Persistent Research Proxy
</footer>

<script>
function filterConditions() {{
  const query = document.getElementById('condition-search').value.toLowerCase();
  const rows = document.querySelectorAll('#conditions-table tbody tr');
  rows.forEach(row => {{
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(query) ? '' : 'none';
  }});
}}
</script>

</body>
</html>"""
