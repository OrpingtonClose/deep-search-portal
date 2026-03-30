"""
Search API Gateway — unified interface for all external search sources.

Every search tool in the system routes through this gateway.  It fans
out queries to multiple backends concurrently, deduplicates by URL, and
returns merged results in a consistent format.

Architecture:

    Subagent calls ``gateway_search(query, sources=...)``
        ↓
    ┌─────────────────────────────────────────────────┐
    │                 Search Gateway                   │
    │                                                  │
    │  ┌─────────────┐  ┌──────────┐  ┌────────────┐ │
    │  │ Grok 4.20   │  │ SearXNG  │  │  Apify /   │ │
    │  │ Responses   │  │ + DDG +  │  │ Bright Data│ │
    │  │ web+X search│  │ Brave +  │  │ Reddit,    │ │
    │  │ (primary)   │  │ Mojeek   │  │ Twitter,   │ │
    │  │             │  │(fallback)│  │ forums     │ │
    │  └─────────────┘  └──────────┘  └────────────┘ │
    │                                                  │
    │  ┌─────────────┐  ┌──────────┐  ┌────────────┐ │
    │  │ Academic    │  │ Archive  │  │ Community  │ │
    │  │ PubMed,     │  │ Wayback, │  │ 4chan, HN,  │ │
    │  │ arXiv,      │  │ Archive  │  │ StackExch,  │ │
    │  │ Scholar     │  │ .org     │  │ YouTube     │ │
    │  └─────────────┘  └──────────┘  └────────────┘ │
    │                                                  │
    │        Dedup by URL → Merge → Return             │
    └─────────────────────────────────────────────────┘

Roles are strictly separated:
    - Grok Responses API = data source (never synthesises)
    - Synthesis Grok     = may use web_search as *extended reasoning*
                           during synthesis (separate config, never
                           called by gateway)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

from .config import log
from .grok_search import tool_grok_deep_search, XAI_API_KEY


# ---------------------------------------------------------------------------
# Source category definitions
# ---------------------------------------------------------------------------

# Sources in each category.  The gateway selects categories based on the
# caller's ``sources`` parameter.
SOURCE_CATEGORIES = {
    "grok": {
        "label": "Grok Deep Search (web + X/Twitter)",
        "description": "Primary search via Grok 4.20 Responses API — autonomous web + X searches",
    },
    "searxng": {
        "label": "SearXNG + DuckDuckGo + Brave + Mojeek",
        "description": "Meta-search aggregator with commercial fallbacks",
    },
    "social": {
        "label": "Social Media (Reddit, Twitter, Telegram, Discord)",
        "description": "Via Apify / Bright Data commercial scrapers",
    },
    "community": {
        "label": "Community (4chan, HN, StackExchange, forums)",
        "description": "Direct API access to community platforms",
    },
    "academic": {
        "label": "Academic (PubMed, arXiv, Scholar)",
        "description": "Academic databases and preprint servers",
    },
    "archive": {
        "label": "Archive (Wayback, Archive.org)",
        "description": "Historical web archives",
    },
    "video": {
        "label": "Video (YouTube)",
        "description": "YouTube search + transcript extraction",
    },
}


async def gateway_search(
    query: str,
    *,
    sources: str = "all",
    search_type: str = "both",
    max_results_per_source: int = 10,
    req_id: str = "",
) -> str:
    """Unified search gateway — fans out to multiple backends and merges.

    Args:
        query: The search query.
        sources: Comma-separated source categories to use.  Options:
            ``"all"`` (default), ``"grok"``, ``"searxng"``, ``"social"``,
            ``"community"``, ``"academic"``, ``"archive"``, ``"video"``.
            Example: ``"grok,social,community"``
        search_type: For Grok: ``"web"``, ``"x"``, or ``"both"``.
        max_results_per_source: Max results from each backend.
        req_id: Request ID for logging.

    Returns:
        Merged, deduplicated search results from all requested sources.
    """
    requested = _parse_sources(sources)
    tasks: list[tuple[str, asyncio.Task]] = []

    # --- Fan out to all requested sources concurrently ---
    if "grok" in requested and XAI_API_KEY:
        tasks.append((
            "grok",
            asyncio.create_task(
                _safe_call("grok", tool_grok_deep_search, query, search_type=search_type)
            ),
        ))

    if "searxng" in requested:
        tasks.append((
            "searxng",
            asyncio.create_task(
                _safe_call("searxng", _searxng_search, query)
            ),
        ))

    if "social" in requested:
        tasks.append((
            "social",
            asyncio.create_task(
                _safe_call("social", _social_search, query)
            ),
        ))

    if "community" in requested:
        tasks.append((
            "community",
            asyncio.create_task(
                _safe_call("community", _community_search, query)
            ),
        ))

    if "academic" in requested:
        tasks.append((
            "academic",
            asyncio.create_task(
                _safe_call("academic", _academic_search, query)
            ),
        ))

    if "archive" in requested:
        tasks.append((
            "archive",
            asyncio.create_task(
                _safe_call("archive", _archive_search, query)
            ),
        ))

    if "video" in requested:
        tasks.append((
            "video",
            asyncio.create_task(
                _safe_call("video", _video_search, query)
            ),
        ))

    if not tasks:
        return f"[TOOL_ERROR] No search sources available for: {query}"

    # --- Gather results ---
    results: list[str] = []
    errors: list[str] = []
    seen_urls: set[str] = set()

    for source_name, task in tasks:
        try:
            result = await task
            if not result or len(result.strip()) < 30:
                errors.append(f"[{source_name}] returned empty")
                continue
            if result.startswith("[TOOL_ERROR]"):
                errors.append(f"[{source_name}] {result}")
                continue

            # Dedup URLs across sources
            deduped = _dedup_result(result, seen_urls)
            if deduped and len(deduped.strip()) > 30:
                results.append(f"--- Source: {source_name} ---\n{deduped}")
        except Exception as e:
            errors.append(f"[{source_name}] exception: {e}")

    # --- Format output ---
    output_parts = []

    if results:
        output_parts.append(
            f"**Gateway Search: {query}**\n"
            f"({len(results)} source(s) returned results, "
            f"{len(errors)} source(s) failed/empty)\n"
        )
        output_parts.extend(results)

    if errors and not results:
        # All sources failed
        output_parts.append(
            f"[TOOL_ERROR] All {len(errors)} search sources failed for: {query}\n"
            + "\n".join(errors)
        )
    elif errors:
        # Some sources failed — append as info
        output_parts.append(
            f"\n--- Source failures ({len(errors)}) ---\n"
            + "\n".join(errors)
        )

    return "\n\n".join(output_parts) if output_parts else f"No results from any source for: {query}"


# ---------------------------------------------------------------------------
# Source category implementations
# ---------------------------------------------------------------------------

async def _searxng_search(query: str) -> str:
    """SearXNG + commercial fallbacks (DuckDuckGo, Brave, Mojeek)."""
    from .search_tools2 import tool_web_search
    return await tool_web_search(query)


async def _social_search(query: str) -> str:
    """Fan out to Reddit + Twitter + Telegram + Facebook + Discord."""
    import social_media_scrapers
    from .search_tools2 import tool_telegram_search

    sub_tasks = [
        _safe_call("reddit", social_media_scrapers.tool_reddit_search, query=query),
        _safe_call("telegram", tool_telegram_search, query),
    ]

    results = await asyncio.gather(*sub_tasks, return_exceptions=True)
    parts = []
    for r in results:
        if isinstance(r, str) and len(r.strip()) > 30 and not r.startswith("[TOOL_ERROR]"):
            parts.append(r)
    return "\n\n".join(parts) if parts else ""


async def _community_search(query: str) -> str:
    """Fan out to 4chan archives + HN + StackExchange + forums."""
    from .search_tools2 import (
        tool_hackernews_search,
        tool_stackexchange_search,
        tool_forum_search,
    )
    from .web_fetch import tool_4plebs_search

    sub_tasks = [
        _safe_call("hackernews", tool_hackernews_search, query),
        _safe_call("stackexchange", tool_stackexchange_search, query),
        _safe_call("forums", tool_forum_search, query),
        _safe_call("4plebs", tool_4plebs_search, query, "pol"),
    ]

    results = await asyncio.gather(*sub_tasks, return_exceptions=True)
    parts = []
    for r in results:
        if isinstance(r, str) and len(r.strip()) > 30 and not r.startswith("[TOOL_ERROR]"):
            parts.append(r)
    return "\n\n".join(parts) if parts else ""


async def _academic_search(query: str) -> str:
    """Fan out to PubMed + arXiv + Google Scholar."""
    from .search_tools2 import (
        tool_pubmed_search,
        tool_arxiv_search,
        tool_scholar_search,
    )

    sub_tasks = [
        _safe_call("pubmed", tool_pubmed_search, query),
        _safe_call("arxiv", tool_arxiv_search, query),
        _safe_call("scholar", tool_scholar_search, query),
    ]

    results = await asyncio.gather(*sub_tasks, return_exceptions=True)
    parts = []
    for r in results:
        if isinstance(r, str) and len(r.strip()) > 30 and not r.startswith("[TOOL_ERROR]"):
            parts.append(r)
    return "\n\n".join(parts) if parts else ""


async def _archive_search(query: str) -> str:
    """Search Archive.org + Wayback Machine."""
    from .search_tools2 import tool_archiveorg_search

    return await tool_archiveorg_search(query)


async def _video_search(query: str) -> str:
    """YouTube search + transcript extraction."""
    from .search_tools2 import tool_youtube_search as yt_search_tool

    return await yt_search_tool(query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sources(sources: str) -> set[str]:
    """Parse a comma-separated source string into a set of category names."""
    if sources.strip().lower() == "all":
        return set(SOURCE_CATEGORIES.keys())
    requested = set()
    for s in sources.split(","):
        s = s.strip().lower()
        if s in SOURCE_CATEGORIES:
            requested.add(s)
    return requested or set(SOURCE_CATEGORIES.keys())


async def _safe_call(label: str, fn, *args, **kwargs) -> str:
    """Call a search function safely, catching all exceptions."""
    try:
        result = await fn(*args, **kwargs)
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        log.debug(f"[gateway:{label}] error: {e}")
        return f"[TOOL_ERROR] {label} failed: {e}"


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    try:
        parsed = urlparse(url)
        # Strip scheme, www prefix, trailing slash
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url.lower().strip()


def _dedup_result(text: str, seen_urls: set[str]) -> str:
    """Remove lines containing already-seen URLs from a result block."""
    lines = text.split("\n")
    output_lines = []
    for line in lines:
        # Extract URLs from the line
        urls_in_line = re.findall(r'https?://[^\s\)\"\'<>]+', line)
        is_dup = False
        for url in urls_in_line:
            norm = _normalize_url(url)
            if norm in seen_urls:
                is_dup = True
                break
            seen_urls.add(norm)
        if not is_dup:
            output_lines.append(line)
    return "\n".join(output_lines)
