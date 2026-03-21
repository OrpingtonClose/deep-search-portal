"""
Search tools: SearXNG query, format results, searxng_search, news_search, fetch_webpage.
"""
from __future__ import annotations

import html
import re

import httpx

import search_providers
from shared import http_client

from .config import (
    WEBPAGE_MAX_CHARS,
    log,
)
from .scoring import trust_score_url
from .web_fetch import enhanced_web_fetch


# ============================================================================
# Tool Implementations
# ============================================================================


def _format_search_results(results: list[dict], source_label: str = "") -> str:
    """Format search results into a readable string.  Returns empty string on empty input."""
    if not results:
        return ""

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))[:300]
        trust = trust_score_url(url)
        tag = f" ({source_label})" if source_label else ""
        formatted.append(
            f"{i}. **{title}** [trust: {trust:.1f}]{tag}\n"
            f"   URL: {url}\n   {snippet}"
        )

    return "\n\n".join(formatted)


async def _searxng_query(
    query: str,
    categories: str = "general",
    time_range: str = "",
) -> list[dict]:
    """Multi-source search — delegates to search_providers module.

    Routes queries to DuckDuckGo, Brave, Mojeek, and SearXNG concurrently,
    deduplicates results by URL, and returns raw dicts in the same format
    as the old SearXNG-only path for backward compatibility.

    Raises on total failure so callers can provide descriptive error messages.
    """
    results = await search_providers.search_as_raw(
        query, categories=categories, time_range=time_range, max_results=10,
    )
    if not results:
        log.debug(f"multi-source search returned 0 results for categories={categories}")
    return results


# News-intent keywords: if a search query contains any of these, it likely
# wants recent news rather than evergreen web pages.
_NEWS_INTENT_KEYWORDS = re.compile(
    r"\b(news|latest|today|yesterday|this week|this month|breaking|recent|update|announced|announces"
    r"|just released|market|stock market|stocks today|crypto today|bitcoin today|headlines"
    r"|march 2026|april 2026|2026)\b",
    re.IGNORECASE,
)


def _has_news_intent(query: str) -> bool:
    """Detect whether a search query is looking for recent news."""
    return bool(_NEWS_INTENT_KEYWORDS.search(query))


async def tool_searxng_search(query: str) -> str:
    """Execute a SearXNG search and return formatted results.

    If the query has news-intent (mentions dates, 'news', 'today', etc.),
    automatically queries both the general AND news categories and merges
    results.
    """
    try:
        general_results = await _searxng_query(query, categories="general")
    except httpx.TimeoutException:
        return "Search error: request timed out after 20s"
    except Exception as e:
        return f"Search error: {str(e)}"

    # Auto-detect news intent and merge news-category results.
    # Wrapped separately so a news-query failure doesn't discard general results.
    if _has_news_intent(query):
        try:
            news_results = await _searxng_query(query, categories="news", time_range="week")
            seen_urls = {r.get("url", "") for r in general_results}
            for r in news_results:
                if r.get("url", "") not in seen_urls:
                    general_results.append(r)
                    seen_urls.add(r.get("url", ""))
        except Exception:
            log.warning("News-category query failed; returning general results only")

    return _format_search_results(general_results) or "No results found."


async def tool_news_search(query: str, time_range: str = "week") -> str:
    """Search for recent news using SearXNG's news category.

    Always queries news-specific search engines (Google News, Bing News, etc.)
    with an explicit time_range filter.
    """
    valid_ranges = {"day", "week", "month", "year"}
    if time_range not in valid_ranges:
        time_range = "week"

    try:
        news_results = await _searxng_query(query, categories="news", time_range=time_range)
    except httpx.TimeoutException:
        return "News search error: request timed out after 20s"
    except Exception as e:
        return f"News search error: {str(e)}"

    # Also query general as fallback — some news sites are indexed there.
    # Wrapped separately so a general-query failure doesn't discard news results.
    try:
        general_results = await _searxng_query(query, categories="general", time_range=time_range)
        seen_urls = {r.get("url", "") for r in news_results}
        for r in general_results:
            if r.get("url", "") not in seen_urls:
                news_results.append(r)
                seen_urls.add(r.get("url", ""))
    except Exception:
        log.warning("General-category fallback failed; returning news results only")

    return _format_search_results(news_results, source_label="news") or "No recent news found."




async def _tool_fetch_webpage_direct(url: str, extract_info: str = "") -> str:
    """Direct fetch — original implementation."""
    # (identical to old tool_fetch_webpage, now an internal helper)
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code} for {url}"

        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            return f"PDF document at {url} (binary content, cannot extract text directly)"
        if ("text/html" not in content_type and "text/plain" not in content_type
                and "text/xml" not in content_type and "application/json" not in content_type):
            return f"Non-text content type: {content_type} at {url}"

        raw = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return f"No readable text content found at {url}"

        if len(text) > WEBPAGE_MAX_CHARS:
            text = text[:WEBPAGE_MAX_CHARS] + "\n[...truncated...]\n"

        result = f"Content from {url}:\n{text}"
        if extract_info:
            result = f"Instructions: {extract_info}\n\n{result}"
        return result

    except Exception as e:
        return f"Fetch error for {url}: {e}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage with enhanced scraping fallback chain."""
    return await enhanced_web_fetch(url, extract_info)
