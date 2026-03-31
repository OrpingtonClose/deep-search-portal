"""Knowledge Wikipedia Generator.

Produces a self-contained, encyclopedia-style HTML article from
accumulated ``AtomicCondition`` research findings.  The output is
designed to be emitted as a LibreChat Artifact so it renders in
a side pane alongside the chat.
"""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlparse

from tools.models import AtomicCondition, CrossRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _domain_from_url(url: str) -> str:
    """Extract domain from URL for display."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or url[:50]
    except Exception:
        return url[:50]


def _slug(text: str) -> str:
    """Convert text to a URL-safe slug for anchor IDs."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:80] or "section"


def _confidence_badge(confidence: float) -> str:
    """Return an inline HTML confidence badge (green/yellow/red)."""
    if confidence >= 0.7:
        color = "#2e7d32"
        bg = "#e8f5e9"
        label = f"High ({confidence:.0%})"
    elif confidence >= 0.4:
        color = "#f57f17"
        bg = "#fff8e1"
        label = f"Medium ({confidence:.0%})"
    else:
        color = "#c62828"
        bg = "#ffebee"
        label = f"Low ({confidence:.0%})"
    return (
        f'<span class="conf-badge" style="background:{bg};color:{color};">'
        f"{label}</span>"
    )


def _verification_badge(status: str) -> str:
    """Return an inline HTML badge for verification status."""
    badges = {
        "verified": ('<span class="ver-badge ver-verified">'
                     "Verified</span>"),
        "speculative": ('<span class="ver-badge ver-speculative">'
                        "Speculative</span>"),
        "fabricated": ('<span class="ver-badge ver-fabricated">'
                       "Fabricated</span>"),
        "overconfident": ('<span class="ver-badge ver-overconfident">'
                          "Overconfident</span>"),
    }
    return badges.get(status, "")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_by_angle(
    conditions: list[AtomicCondition],
) -> dict[str, list[AtomicCondition]]:
    """Group conditions by research angle."""
    by_angle: dict[str, list[AtomicCondition]] = {}
    for c in conditions:
        angle = c.angle or "General"
        by_angle.setdefault(angle, []).append(c)
    return by_angle


def _group_by_entity(
    conditions: list[AtomicCondition],
) -> dict[str, list[AtomicCondition]]:
    """Group conditions by entity.  Conditions with no entities go under
    'General Findings'."""
    by_entity: dict[str, list[AtomicCondition]] = {}
    for c in conditions:
        if c.entities:
            for entity in c.entities:
                by_entity.setdefault(entity, []).append(c)
        else:
            by_entity.setdefault("General Findings", []).append(c)
    return by_entity


# ---------------------------------------------------------------------------
# Source index
# ---------------------------------------------------------------------------

def _build_source_index(
    conditions: list[AtomicCondition],
) -> tuple[dict[str, int], list[AtomicCondition]]:
    """Build a mapping from source_url -> citation number (1-based).

    Returns (url_to_num, ordered_conditions_with_sources).
    """
    url_to_num: dict[str, int] = {}
    ordered: list[AtomicCondition] = []
    counter = 1
    for c in conditions:
        if c.source_url and c.source_url not in url_to_num:
            url_to_num[c.source_url] = counter
            ordered.append(c)
            counter += 1
    return url_to_num, ordered


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_toc(
    by_angle: dict[str, list[AtomicCondition]],
) -> str:
    """Build a Wikipedia-style table of contents."""
    items: list[str] = []
    for idx, angle in enumerate(sorted(by_angle.keys()), 1):
        slug = _slug(angle)
        items.append(
            f'<li><a href="#section-{slug}">{idx}. {_esc(angle)}</a></li>'
        )
    items.append('<li><a href="#sources">Sources</a></li>')
    return (
        '<nav class="wiki-toc">'
        "<h3>Contents</h3>"
        f'<ol>{"".join(items)}</ol>'
        "</nav>"
    )


def _build_cross_ref_links(
    condition: AtomicCondition,
    all_conditions: list[AtomicCondition],
) -> str:
    """Build inline cross-reference links for a condition."""
    if not condition.cross_refs:
        return ""
    links: list[str] = []
    for ref in condition.cross_refs:
        if 0 <= ref.target_idx < len(all_conditions):
            target = all_conditions[ref.target_idx]
            target_angle = target.angle or "General"
            slug = _slug(target_angle)
            rel_label = ref.relation.capitalize()
            short_fact = _esc(target.fact[:60])
            links.append(
                f'<a href="#section-{slug}" class="xref" '
                f'title="{_esc(target.fact[:120])}">'
                f"{rel_label}: {short_fact}&hellip;</a>"
            )
    if not links:
        return ""
    return (
        '<div class="cross-refs">'
        '<span class="xref-label">See also:</span> '
        + " &middot; ".join(links)
        + "</div>"
    )


def _inline_citation(url: str, url_to_num: dict[str, int]) -> str:
    """Return a superscript citation link like [1]."""
    num = url_to_num.get(url)
    if num is None:
        return ""
    return (
        f'<sup class="cite"><a href="#source-{num}" '
        f'title="{_esc(_domain_from_url(url))}">[{num}]</a></sup>'
    )


def _build_angle_section(
    angle: str,
    conditions: list[AtomicCondition],
    all_conditions: list[AtomicCondition],
    url_to_num: dict[str, int],
    section_idx: int,
) -> str:
    """Build a full Wikipedia-style section for one research angle."""
    slug = _slug(angle)
    by_entity = _group_by_entity(conditions)

    # Sort conditions within angle by confidence (best first)
    sorted_conditions = sorted(
        conditions, key=lambda c: c.confidence, reverse=True
    )

    # Opening prose paragraph — summarise the angle
    top_facts = [c.fact for c in sorted_conditions[:3]]
    summary_text = " ".join(_esc(f) for f in top_facts)

    parts: list[str] = []
    parts.append(f'<section id="section-{slug}" class="wiki-section">')
    parts.append(f"<h2>{section_idx}. {_esc(angle)}</h2>")

    # Prose summary
    if summary_text:
        parts.append(f'<p class="section-summary">{summary_text}</p>')

    # Entity sub-sections
    for entity, entity_conds in sorted(by_entity.items()):
        entity_slug = _slug(f"{angle}-{entity}")
        parts.append(f'<div class="entity-block" id="entity-{entity_slug}">')
        parts.append(f"<h3>{_esc(entity)}</h3>")
        parts.append("<ul>")
        # Sort by confidence within entity
        for c in sorted(entity_conds, key=lambda x: x.confidence, reverse=True):
            citation = _inline_citation(c.source_url, url_to_num)
            badge = _confidence_badge(c.confidence)
            ver_badge = _verification_badge(c.verification_status)
            xrefs = _build_cross_ref_links(c, all_conditions)

            parts.append("<li>")
            parts.append(f"<p>{_esc(c.fact)}{citation} {badge} {ver_badge}</p>")
            if c.author or c.publication_date:
                meta_parts: list[str] = []
                if c.author:
                    meta_parts.append(f"Author: {_esc(c.author)}")
                if c.publication_date:
                    meta_parts.append(f"Date: {_esc(c.publication_date)}")
                parts.append(
                    f'<p class="claim-meta">{" | ".join(meta_parts)}</p>'
                )
            if xrefs:
                parts.append(xrefs)
            parts.append("</li>")

        parts.append("</ul>")
        parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


def _build_sources_section(
    url_to_num: dict[str, int],
    conditions: list[AtomicCondition],
) -> str:
    """Build the Sources section at the bottom."""
    # Build a lookup for extra metadata per URL
    url_meta: dict[str, AtomicCondition] = {}
    for c in conditions:
        if c.source_url and c.source_url not in url_meta:
            url_meta[c.source_url] = c

    items: list[str] = []
    for url, num in sorted(url_to_num.items(), key=lambda x: x[1]):
        domain = _domain_from_url(url)
        meta = url_meta.get(url)
        extra = ""
        if meta:
            parts: list[str] = []
            if meta.source_type:
                parts.append(f"via {_esc(meta.source_type)}")
            if meta.content_type:
                parts.append(_esc(meta.content_type))
            if meta.author:
                parts.append(f"by {_esc(meta.author)}")
            if parts:
                extra = f' <span class="source-meta">({", ".join(parts)})</span>'

        items.append(
            f'<li id="source-{num}">'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener">'
            f"{_esc(domain)}</a>{extra}</li>"
        )

    return (
        '<section id="sources" class="wiki-section">'
        "<h2>Sources</h2>"
        f'<ol class="source-list">{"".join(items)}</ol>'
        "</section>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_WIKI_CSS = """
:root {
  --wiki-bg: #1a1a2e;
  --wiki-card: #222244;
  --wiki-text: #e0e0e0;
  --wiki-dim: #999;
  --wiki-accent: #4fc3f7;
  --wiki-border: #333366;
  --wiki-green: #2e7d32;
  --wiki-green-bg: #e8f5e9;
  --wiki-yellow: #f57f17;
  --wiki-yellow-bg: #fff8e1;
  --wiki-red: #c62828;
  --wiki-red-bg: #ffebee;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--wiki-bg);
  color: var(--wiki-text);
  line-height: 1.7;
  padding: 24px;
  max-width: 960px;
  margin: 0 auto;
}
h1 { color: var(--wiki-accent); font-size: 1.6em; margin-bottom: 4px; }
h2 {
  color: var(--wiki-accent); font-size: 1.3em;
  margin: 24px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--wiki-border);
}
h3 { color: var(--wiki-text); font-size: 1.05em; margin: 16px 0 8px; }
a { color: var(--wiki-accent); text-decoration: none; }
a:hover { text-decoration: underline; }
p { margin: 8px 0; }

/* Header */
.wiki-header { margin-bottom: 20px; }
.wiki-header .subtitle { color: var(--wiki-dim); font-size: 0.9em; }

/* Table of contents */
.wiki-toc {
  background: var(--wiki-card);
  border: 1px solid var(--wiki-border);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 24px;
}
.wiki-toc h3 { margin: 0 0 8px; font-size: 1em; color: var(--wiki-accent); }
.wiki-toc ol { padding-left: 20px; }
.wiki-toc li { margin: 4px 0; font-size: 0.9em; }

/* Sections */
.wiki-section {
  background: var(--wiki-card);
  border: 1px solid var(--wiki-border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
}
.section-summary {
  color: var(--wiki-dim);
  font-style: italic;
  font-size: 0.95em;
  margin-bottom: 12px;
  line-height: 1.6;
}

/* Entity blocks */
.entity-block { margin: 12px 0; }
.entity-block ul { list-style: none; padding: 0; }
.entity-block li {
  padding: 10px 12px;
  margin: 6px 0;
  background: rgba(255,255,255,0.03);
  border-radius: 6px;
  border-left: 3px solid var(--wiki-border);
}

/* Confidence badges */
.conf-badge {
  display: inline-block;
  font-size: 0.7em;
  font-weight: 600;
  padding: 1px 7px;
  border-radius: 10px;
  margin-left: 6px;
  vertical-align: middle;
}

/* Verification badges */
.ver-badge {
  display: inline-block;
  font-size: 0.65em;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 3px;
  margin-left: 4px;
  vertical-align: middle;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.ver-verified { background: #1b5e20; color: #a5d6a7; }
.ver-speculative { background: #4a148c; color: #ce93d8; }
.ver-fabricated { background: #b71c1c; color: #ef9a9a; }
.ver-overconfident { background: #e65100; color: #ffcc80; }

/* Citations */
.cite a {
  font-size: 0.75em;
  color: var(--wiki-accent);
  text-decoration: none;
}
.cite a:hover { text-decoration: underline; }

/* Claim metadata */
.claim-meta {
  font-size: 0.8em;
  color: var(--wiki-dim);
  margin-top: 2px;
}

/* Cross-references */
.cross-refs {
  margin-top: 6px;
  font-size: 0.82em;
  color: var(--wiki-dim);
}
.xref-label { font-weight: 600; }
.xref {
  color: var(--wiki-accent);
  font-style: italic;
}

/* Source list */
.source-list { padding-left: 20px; }
.source-list li { margin: 6px 0; font-size: 0.9em; }
.source-meta { color: var(--wiki-dim); font-size: 0.85em; }

/* Stats bar */
.stats-bar {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin: 12px 0 20px;
  font-size: 0.85em;
  color: var(--wiki-dim);
}
.stats-bar .stat-item strong { color: var(--wiki-accent); }

/* Footer */
.wiki-footer {
  text-align: center;
  color: var(--wiki-dim);
  font-size: 0.8em;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--wiki-border);
}

@media (max-width: 600px) {
  body { padding: 12px; }
  .wiki-section { padding: 14px; }
}
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_wiki_article(
    conditions: list[AtomicCondition],
    query: str,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained Wikipedia-style HTML article.

    Args:
        conditions: All accumulated AtomicCondition objects from research.
        query: The original user research query.
        metrics: Optional metrics dict for summary statistics.

    Returns:
        Self-contained HTML string.
    """
    if metrics is None:
        metrics = {}

    if not conditions:
        return (
            "<!DOCTYPE html><html><body>"
            "<p>No research findings to display.</p>"
            "</body></html>"
        )

    # Build data structures
    by_angle = _group_by_angle(conditions)
    url_to_num, _ordered = _build_source_index(conditions)

    # Stats
    total_conditions = len(conditions)
    n_sources = len(url_to_num)
    n_angles = len(by_angle)
    avg_confidence = (
        sum(c.confidence for c in conditions) / total_conditions
        if total_conditions
        else 0
    )
    n_verified = sum(
        1 for c in conditions if c.verification_status == "verified"
    )

    # Build sections
    toc = _build_toc(by_angle)

    angle_sections: list[str] = []
    for idx, angle in enumerate(sorted(by_angle.keys()), 1):
        angle_sections.append(
            _build_angle_section(
                angle,
                by_angle[angle],
                conditions,
                url_to_num,
                idx,
            )
        )

    sources_section = _build_sources_section(url_to_num, conditions)

    # Assemble page
    title = f"Knowledge Base: {_esc(query[:80])}"
    duration = metrics.get("total_duration_secs", 0)
    duration_str = (
        f"{int(duration // 60)}m {int(duration % 60)}s"
        if duration >= 60
        else f"{duration:.0f}s"
    ) if duration else ""

    stats_items: list[str] = [
        f"<span class='stat-item'><strong>{total_conditions}</strong> findings</span>",
        f"<span class='stat-item'><strong>{n_sources}</strong> sources</span>",
        f"<span class='stat-item'><strong>{n_angles}</strong> topics</span>",
        f"<span class='stat-item'>Avg confidence: <strong>{avg_confidence:.0%}</strong></span>",
    ]
    if n_verified:
        stats_items.append(
            f"<span class='stat-item'><strong>{n_verified}</strong> verified</span>"
        )
    if duration_str:
        stats_items.append(
            f"<span class='stat-item'>Research time: <strong>{duration_str}</strong></span>"
        )

    body_parts = [
        '<div class="wiki-header">',
        f"<h1>{_esc(query)}</h1>",
        f'<p class="subtitle">Encyclopedia-style research compilation</p>',
        f'<div class="stats-bar">{"".join(stats_items)}</div>',
        "</div>",
        toc,
        "\n".join(angle_sections),
        sources_section,
        '<div class="wiki-footer">',
        "Generated by Deep Search Portal &mdash; Knowledge Wikipedia",
        "</div>",
    ]

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>{title}</title>\n"
        f"<style>{_WIKI_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        + "\n".join(body_parts)
        + "\n</body>\n</html>"
    )
