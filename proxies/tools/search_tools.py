"""All search tool implementations: SearXNG, news, 4chan archives, Twitter, etc.

Extracted from persistent_deep_research_proxy.py lines 1749-1876, 2304-3463.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import httpx

import social_media_scrapers

from shared import get_throttler

from . import _get_http_client

from .config import (
    SEARXNG_URL,
    WEBPAGE_MAX_CHARS,
    PYTHON_TIMEOUT,
    PYTHON_OUTPUT_MAX,
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_CUSTOMER_ID,
    BRIGHT_DATA_ZONE,
    OXYLABS_USERNAME,
    OXYLABS_PASSWORD,
    COMMERCIAL_SEARCH_ENABLED,
)
from .scoring import trust_score_url
from .moderation import moderate_query, _commercial_search
from .web_fetch import enhanced_web_fetch, tool_fetch_webpage, _strip_html, _is_censored_response

log = logging.getLogger("persistent-research")

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
    """Low-level SearXNG query.  Returns raw result dicts.

    Raises on HTTP errors and timeouts so callers can provide
    descriptive error messages to the subagent.
    """
    async with get_throttler("searxng").throttle():
        client = _get_http_client()
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        if time_range:
            params["time_range"] = time_range

        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params=params,
            timeout=20.0,
        )
        if resp.status_code != 200:
            log.warning(f"SearXNG returned HTTP {resp.status_code} for categories={categories}")
            raise RuntimeError(f"SearXNG HTTP {resp.status_code}")

        data = resp.json()
        return data.get("results", [])[:10]


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


async def tool_4plebs_search(query: str, board: str = "pol") -> str:
    """Search 4plebs archive (covers /pol/, /sp/, /int/, /tv/, /k/, /vg/, etc.).

    Returns formatted search results from the 4plebs full-text search API.
    """
    board = board.strip("/").lower()
    try:
        async with get_throttler("imageboard").throttle():
            client = _get_http_client()
            resp = await client.get(
                f"https://archive.4plebs.org/_/api/chan/search/",
                params={"boards": board, "text": query},
                timeout=20.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
            )
        if resp.status_code != 200:
            return f"4plebs search error: HTTP {resp.status_code}"

        data = resp.json()
        posts = data.get("0", {}).get("posts", [])
        if not posts:
            return f"No results found on /{board}/ for: {query}"

        formatted = []
        for i, post in enumerate(list(posts.values())[:10] if isinstance(posts, dict) else posts[:10], 1):
            thread_num = post.get("thread_num", "")
            num = post.get("num", "")
            comment = post.get("comment") or ""
            # Strip HTML from comment
            comment = re.sub(r'<[^>]+>', ' ', comment)
            comment = html.unescape(comment).strip()
            if len(comment) > 500:
                comment = comment[:500] + "..."
            timestamp = post.get("timestamp", 0)
            date_str = ""
            if timestamp:
                try:
                    date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, OSError):
                    pass

            url = f"https://archive.4plebs.org/{board}/thread/{thread_num}/#p{num}"
            formatted.append(
                f"{i}. **/{board}/** [{date_str}] (thread #{thread_num})\n"
                f"   URL: {url}\n"
                f"   {comment}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "4plebs search error: request timed out"
    except Exception as e:
        return f"4plebs search error: {str(e)}"


async def tool_b4k_search(query: str) -> str:
    """Search arch.b4k.co archive for /biz/ (crypto/financial discussions).

    This is the only reliable /biz/ archive, covering 2017–present.
    """
    try:
        async with get_throttler("imageboard").throttle():
            client = _get_http_client()
            resp = await client.get(
                f"https://arch.b4k.co/_/api/chan/search/",
                params={"boards": "biz", "text": query},
                timeout=20.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
            )
        if resp.status_code != 200:
            return f"b4k search error: HTTP {resp.status_code}"

        data = resp.json()
        posts = data.get("0", {}).get("posts", [])
        if not posts:
            return f"No results found on /biz/ for: {query}"

        formatted = []
        for i, post in enumerate(list(posts.values())[:10] if isinstance(posts, dict) else posts[:10], 1):
            thread_num = post.get("thread_num", "")
            num = post.get("num", "")
            comment = post.get("comment") or ""
            comment = re.sub(r'<[^>]+>', ' ', comment)
            comment = html.unescape(comment).strip()
            if len(comment) > 500:
                comment = comment[:500] + "..."
            timestamp = post.get("timestamp", 0)
            date_str = ""
            if timestamp:
                try:
                    date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, OSError):
                    pass

            url = f"https://arch.b4k.co/biz/thread/{thread_num}/#p{num}"
            formatted.append(
                f"{i}. **/biz/** [{date_str}] (thread #{thread_num})\n"
                f"   URL: {url}\n"
                f"   {comment}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "b4k search error: request timed out"
    except Exception as e:
        return f"b4k search error: {str(e)}"


async def tool_warosu_search(query: str, board: str = "g") -> str:
    """Search warosu.org archive (covers /g/, /sci/, /lit/, /jp/, /vr/, /fa/).

    Warosu archives technology, science, and literature boards.
    """
    board = board.strip("/").lower()
    try:
        async with get_throttler("imageboard").throttle():
            client = _get_http_client()
            # Warosu uses a GET search endpoint
            resp = await client.get(
                f"https://warosu.org/{board}/",
                params={"task": "search2", "search_text": query, "offset": 0},
                timeout=20.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
        if resp.status_code != 200:
            return f"Warosu search error: HTTP {resp.status_code}"

        # Warosu returns HTML, not JSON — parse results from HTML
        raw = resp.text
        # Extract post blocks: <td class="reply" id="pNNNNN">
        post_blocks = re.findall(
            r'<td[^>]*class="reply"[^>]*id="p(\d+)"[^>]*>(.*?)</td>',
            raw, re.DOTALL,
        )
        if not post_blocks:
            return f"No results found on /{board}/ for: {query}"

        formatted = []
        for i, (post_id, block) in enumerate(post_blocks[:10], 1):
            # Extract comment text
            comment_m = re.search(r'<blockquote>(.*?)</blockquote>', block, re.DOTALL)
            comment = ""
            if comment_m:
                comment = re.sub(r'<[^>]+>', ' ', comment_m.group(1))
                comment = html.unescape(comment).strip()
                if len(comment) > 500:
                    comment = comment[:500] + "..."

            # Extract thread number from reply link
            thread_m = re.search(r'href="/\w+/thread/(\d+)', block)
            thread_num = thread_m.group(1) if thread_m else post_id

            url = f"https://warosu.org/{board}/thread/{thread_num}#p{post_id}"
            formatted.append(
                f"{i}. **/{board}/** (post #{post_id})\n"
                f"   URL: {url}\n"
                f"   {comment}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Warosu search error: request timed out"
    except Exception as e:
        return f"Warosu search error: {str(e)}"


# ============================================================================
# Twitter/X Search Tool
# ============================================================================

async def tool_twitter_search(query: str) -> str:
    """Search Twitter/X for tweets matching the query.

    Uses a tiered approach:
      1. Bright Data Twitter Scraper (if configured) — most reliable
      2. Oxylabs Web Scraper (if configured) — fallback
      3. Nitter instances (degraded, sporadic) — last resort

    Accepts Twitter search operators: from:handle, since:YYYY-MM-DD,
    until:YYYY-MM-DD, "exact phrase", etc.
    """
    # Tier 1: Bright Data Web Unlocker for Twitter search
    if BRIGHT_DATA_API_KEY:
        result = await _twitter_via_bright_data(query)
        if result:
            return result

    # Tier 2: Oxylabs for Twitter search
    if OXYLABS_USERNAME:
        result = await _twitter_via_oxylabs(query)
        if result:
            return result

    # Tier 3: Nitter instances (degraded fallback)
    result = await _twitter_via_nitter(query)
    if result:
        return result

    return (
        f"Twitter search failed for: {query}\n\n"
        "All access tiers exhausted. Twitter requires commercial proxy access "
        "(Bright Data or Oxylabs) for reliable results. Nitter instances are "
        "frequently blocked by X Corp."
    )


async def _twitter_via_bright_data(query: str) -> Optional[str]:
    """Scrape Twitter search results via Bright Data Web Unlocker."""
    try:
        encoded_query = quote(query, safe="")
        search_url = f"https://x.com/search?q={encoded_query}&src=typed_query&f=live"
        proxy_url = (
            f"https://brd-customer-{BRIGHT_DATA_CUSTOMER_ID}-zone-{BRIGHT_DATA_ZONE}"
            f":{BRIGHT_DATA_API_KEY}@brd.superproxy.io:33335"
        )
        async with get_throttler("bright_data").throttle():
            async with httpx.AsyncClient(
                proxy=proxy_url,
                verify=False,
                timeout=httpx.Timeout(45.0, connect=15.0),
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    search_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                if resp.status_code != 200:
                    return None

                text = _strip_html(resp.text)
                if not text or len(text.strip()) < 100:
                    return None
                if _is_censored_response(text):
                    return None

                return f"**Twitter/X search results for: {query}**\n\n{text[:WEBPAGE_MAX_CHARS]}"
    except Exception as e:
        log.debug(f"Bright Data Twitter fetch failed: {e}")
        return None


async def _twitter_via_oxylabs(query: str) -> Optional[str]:
    """Scrape Twitter search results via Oxylabs Web Scraper."""
    if not OXYLABS_USERNAME or not OXYLABS_PASSWORD:
        return None
    try:
        encoded_query = quote(query, safe="")
        search_url = f"https://x.com/search?q={encoded_query}&src=typed_query&f=live"
        async with get_throttler("oxylabs").throttle():
            async with httpx.AsyncClient(
                proxy=f"https://{OXYLABS_USERNAME}:{OXYLABS_PASSWORD}@unblock.oxylabs.io:60000",
                verify=False,
                timeout=httpx.Timeout(45.0, connect=15.0),
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    search_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
                if resp.status_code != 200:
                    return None

                text = _strip_html(resp.text)
                if not text or len(text.strip()) < 100:
                    return None
                if _is_censored_response(text):
                    return None

                return f"**Twitter/X search results for: {query}**\n\n{text[:WEBPAGE_MAX_CHARS]}"
    except Exception as e:
        log.debug(f"Oxylabs Twitter fetch failed: {e}")
        return None


_NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
]


async def _twitter_via_nitter(query: str) -> Optional[str]:
    """Search Twitter via Nitter instances (degraded, sporadic availability)."""
    client = _get_http_client()
    for instance in _NITTER_INSTANCES:
        try:
            async with get_throttler("nitter").throttle():
                resp = await client.get(
                    f"{instance}/search",
                    params={"f": "tweets", "q": query},
                    timeout=15.0,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
            if resp.status_code != 200:
                continue

            raw = resp.text

            # Parse Nitter HTML for tweet content
            tweet_blocks = re.findall(
                r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>',
                raw, re.DOTALL,
            )
            if not tweet_blocks:
                continue

            # Extract usernames
            usernames = re.findall(
                r'<a class="username"[^>]*>@([^<]+)</a>',
                raw,
            )

            formatted = []
            for i, block in enumerate(tweet_blocks[:10], 1):
                text = re.sub(r'<[^>]+>', ' ', block)
                text = html.unescape(text).strip()
                if len(text) > 400:
                    text = text[:400] + "..."
                user = f"@{usernames[i-1]}" if i <= len(usernames) else "@unknown"
                formatted.append(f"{i}. **{user}**: {text}")

            if formatted:
                return (
                    f"**Twitter/X search results for: {query}**\n"
                    f"(via Nitter — may be incomplete)\n\n"
                    + "\n\n".join(formatted)
                )
        except Exception:
            continue

    return None


def tool_python_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > PYTHON_OUTPUT_MAX:
            output = output[:PYTHON_OUTPUT_MAX] + "\n[... output truncated ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


async def tool_arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers using the arXiv API."""
    try:
        max_results = min(max_results, 10)
        async with get_throttler("arxiv").throttle():
            client = _get_http_client()
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=20.0,
            )
        if resp.status_code != 200:
            return f"arXiv search error: HTTP {resp.status_code}"

        text = resp.text
        entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
        if not entries:
            return "No arXiv papers found."

        formatted = []
        for i, entry in enumerate(entries, 1):
            title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_m.group(1).strip().replace('\n', ' ') if title_m else "Unknown"
            summary_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            summary = summary_m.group(1).strip()[:300] if summary_m else ""
            id_m = re.search(r'<id>(.*?)</id>', entry)
            arxiv_url = id_m.group(1).strip() if id_m else ""
            authors = re.findall(r'<name>(.*?)</name>', entry)
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"
            published_m = re.search(r'<published>(.*?)</published>', entry)
            published = published_m.group(1)[:10] if published_m else ""

            formatted.append(
                f"{i}. **{title}**\n"
                f"   Authors: {author_str}\n"
                f"   Published: {published}\n"
                f"   URL: {arxiv_url}\n"
                f"   Abstract: {summary}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "arXiv search error: request timed out"
    except Exception as e:
        return f"arXiv search error: {str(e)}"


async def tool_wayback_fetch(url: str) -> str:
    """Fetch an archived version of a URL from the Wayback Machine."""
    try:
        client = _get_http_client()
        avail_resp = await client.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=15.0,
        )
        if avail_resp.status_code != 200:
            return f"Wayback Machine error: HTTP {avail_resp.status_code}"

        avail_data = avail_resp.json()
        snapshots = avail_data.get("archived_snapshots", {})
        closest = snapshots.get("closest", {})
        if not closest or not closest.get("available"):
            return f"No archived version found for {url}"

        archive_url = closest.get("url", "")
        timestamp = closest.get("timestamp", "")

        content = await tool_fetch_webpage(archive_url)
        return (
            f"**Wayback Machine archive** (captured: {timestamp})\n"
            f"Original URL: {url}\n"
            f"Archive URL: {archive_url}\n\n{content}"
        )

    except httpx.TimeoutException:
        return "Wayback Machine error: request timed out"
    except Exception as e:
        return f"Wayback Machine error: {str(e)}"


async def tool_wikidata_query(entity: str) -> str:
    """Query Wikidata for structured facts about an entity."""
    try:
        _wikidata_throttle = get_throttler("wikidata")
        async with _wikidata_throttle.throttle():
            client = _get_http_client()
            search_resp = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": entity,
                    "language": "en",
                    "format": "json",
                    "limit": 3,
                },
                timeout=15.0,
            )
        if search_resp.status_code != 200:
            return f"Wikidata search error: HTTP {search_resp.status_code}"

        search_data = search_resp.json()
        results = search_data.get("search", [])
        if not results:
            return f"No Wikidata entity found for '{entity}'"

        formatted = []
        for r in results:
            qid = r.get("id", "")
            label = r.get("label", "")
            desc = r.get("description", "")
            url = f"https://www.wikidata.org/wiki/{qid}"
            formatted.append(
                f"- **{label}** ({qid}): {desc}\n  URL: {url}"
            )

        top_qid = results[0].get("id", "")
        if top_qid:
            async with _wikidata_throttle.throttle():
                entity_resp = await client.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": top_qid,
                        "languages": "en",
                        "format": "json",
                        "props": "labels|descriptions|claims",
                    },
                    timeout=15.0,
                )
            if entity_resp.status_code == 200:
                entity_data = entity_resp.json()
                ent_info = entity_data.get("entities", {}).get(top_qid, {})
                claims = ent_info.get("claims", {})
                claim_strs = []
                for prop_id, claim_list in list(claims.items())[:10]:
                    for claim in claim_list[:1]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        value = datavalue.get("value", "")
                        if isinstance(value, dict):
                            value = value.get("id", str(value))
                        claim_strs.append(f"  {prop_id}: {value}")
                if claim_strs:
                    formatted.append(f"\nTop claims for {top_qid}:\n" + "\n".join(claim_strs[:8]))

        return "\n".join(formatted)

    except httpx.TimeoutException:
        return "Wikidata query error: request timed out"
    except Exception as e:
        return f"Wikidata query error: {str(e)}"


async def tool_web_search(query: str) -> str:
    """Unified web search: SearXNG + commercial APIs (if moderation passes).

    Always runs SearXNG.  If COMMERCIAL_SEARCH_ENABLED and the query passes
    Mistral moderation, also queries Bright Data / Oxylabs SERP and merges
    results (deduped by URL).
    """
    # Always run SearXNG as baseline.
    searxng_result = await tool_searxng_search(query)

    if not COMMERCIAL_SEARCH_ENABLED:
        return searxng_result

    # Gate commercial APIs behind Mistral moderation.
    is_safe, mod_details = await moderate_query(query)
    if not is_safe:
        flagged = mod_details.get("flagged", [])
        if flagged:
            log.info(
                f"Commercial search skipped (moderation): {flagged}"
            )
        return searxng_result

    # Fetch commercial results.
    commercial_results = await _commercial_search(query)
    if not commercial_results:
        return searxng_result

    # Merge: extract URLs already in SearXNG results to dedup.
    seen_urls: set[str] = set()
    for line in searxng_result.split("\n"):
        stripped = line.strip()
        if stripped.startswith("URL: "):
            seen_urls.add(stripped[5:].strip())

    extra_formatted = []
    next_idx = searxng_result.count("**") // 2 + 1  # rough count of existing results
    for r in commercial_results:
        url = r.get("url", "")
        if url in seen_urls or not url:
            continue
        seen_urls.add(url)
        title = r.get("title", "No title")
        snippet = r.get("snippet", "")[:300]
        trust = trust_score_url(url)
        source_tag = r.get("source", "commercial")
        extra_formatted.append(
            f"{next_idx}. **{title}** [trust: {trust:.1f}] ({source_tag})\n"
            f"   URL: {url}\n   {snippet}"
        )
        next_idx += 1

    if extra_formatted:
        if searxng_result in ("No results found.", ""):
            return "\n\n".join(extra_formatted)
        return searxng_result + "\n\n" + "\n\n".join(extra_formatted)
    return searxng_result


# ============================================================================
# Hacker News Search (Algolia API — free, no auth)
# ============================================================================

async def tool_hackernews_search(query: str, sort_by: str = "relevance", time_range: str = "") -> str:
    """Search Hacker News via the Algolia API.

    Covers stories, comments, and Ask HN / Show HN posts.
    Free API, no authentication required.
    """
    try:
        async with get_throttler("hackernews").throttle():
            client = _get_http_client()
            endpoint = "search" if sort_by == "relevance" else "search_by_date"
            params: dict[str, str | int] = {
                "query": query,
                "hitsPerPage": 15,
                "tags": "(story,comment)",
            }
            # Time range filtering via numericFilters
            if time_range:
                now = int(time.time())
                range_map = {
                    "day": 86400,
                    "week": 604800,
                    "month": 2592000,
                    "year": 31536000,
                }
                seconds = range_map.get(time_range, 0)
                if seconds:
                    params["numericFilters"] = f"created_at_i>{now - seconds}"

            resp = await client.get(
                f"https://hn.algolia.com/api/v1/{endpoint}",
                params=params,
                timeout=15.0,
            )
        if resp.status_code != 200:
            return f"Hacker News search error: HTTP {resp.status_code}"

        data = resp.json()
        hits = data.get("hits", [])
        if not hits:
            return f"No Hacker News results for: {query}"

        formatted = []
        for i, hit in enumerate(hits[:15], 1):
            title = hit.get("title") or hit.get("story_title") or ""
            comment_text = hit.get("comment_text") or ""
            author = hit.get("author", "unknown")
            points = hit.get("points") if hit.get("points") is not None else 0
            created = hit.get("created_at", "")[:10]
            obj_id = hit.get("objectID", "")
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={obj_id}"
            hn_url = f"https://news.ycombinator.com/item?id={obj_id}"

            if comment_text:
                # Strip HTML from comments
                comment_text = re.sub(r'<[^>]+>', ' ', comment_text)
                comment_text = html.unescape(comment_text).strip()
                if len(comment_text) > 400:
                    comment_text = comment_text[:400] + "..."
                formatted.append(
                    f"{i}. **Comment by {author}** [{created}] (on: {title or 'thread'})\n"
                    f"   HN: {hn_url}\n"
                    f"   {comment_text}"
                )
            else:
                formatted.append(
                    f"{i}. **{title}** by {author} [{created}] ({points} points)\n"
                    f"   URL: {url}\n"
                    f"   HN: {hn_url}"
                )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Hacker News search error: request timed out"
    except Exception as e:
        return f"Hacker News search error: {str(e)}"


# ============================================================================
# Stack Exchange Search (SE API v2.3 — free, no auth for read)
# ============================================================================

async def tool_stackexchange_search(query: str, site: str = "stackoverflow", sort: str = "relevance") -> str:
    """Search Stack Exchange sites for Q&A content.

    Covers hundreds of niche communities: stackoverflow, superuser, serverfault,
    askubuntu, math, physics, chemistry, biology, electronics, diy, cooking,
    gaming, rpg, worldbuilding, etc.

    Free API, no authentication required for read access.
    """
    try:
        async with get_throttler("stackexchange").throttle():
            client = _get_http_client()
            resp = await client.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "q": query,
                    "site": site,
                    "sort": sort,
                    "order": "desc",
                    "pagesize": 10,
                    "filter": "withbody",
                },
                timeout=15.0,
                headers={"Accept-Encoding": "gzip"},
            )
        if resp.status_code != 200:
            return f"Stack Exchange search error: HTTP {resp.status_code}"

        data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No results on {site} for: {query}"

        formatted = []
        for i, item in enumerate(items[:10], 1):
            title = html.unescape(item.get("title", ""))
            score = item.get("score", 0)
            answers = item.get("answer_count", 0)
            is_answered = item.get("is_answered", False)
            link = item.get("link", "")
            tags = ", ".join(item.get("tags", [])[:5])
            creation = datetime.fromtimestamp(
                item.get("creation_date", 0), tz=timezone.utc
            ).strftime("%Y-%m-%d") if item.get("creation_date") else ""

            # Extract body text (HTML -> plain text)
            body = item.get("body", "")
            if body:
                body = re.sub(r'<[^>]+>', ' ', body)
                body = html.unescape(body).strip()
                if len(body) > 400:
                    body = body[:400] + "..."

            status = "ANSWERED" if is_answered else f"{answers} answers"
            formatted.append(
                f"{i}. **{title}** [score: {score}, {status}] [{creation}]\n"
                f"   Tags: {tags}\n"
                f"   URL: {link}\n"
                f"   {body}"
            )

        quota_remaining = data.get("quota_remaining", "?")
        return "\n\n".join(formatted) + f"\n\n[API quota remaining: {quota_remaining}]"

    except httpx.TimeoutException:
        return "Stack Exchange search error: request timed out"
    except Exception as e:
        return f"Stack Exchange search error: {str(e)}"


# ============================================================================
# PubMed / Biomedical Search (NCBI E-utilities — free, no auth)
# ============================================================================

async def tool_pubmed_search(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical and life science literature.

    Uses NCBI E-utilities (esearch + efetch). Covers medical journals,
    clinical trials, pharmacology, biochemistry, genetics, epidemiology,
    public health, and more. Free API, no authentication required.
    """
    max_results = min(max_results, 15)
    try:
        client = _get_http_client()

        # Step 1: Search for PMIDs
        async with get_throttler("pubmed").throttle():
            search_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
                timeout=15.0,
            )
        if search_resp.status_code != 200:
            return f"PubMed search error: HTTP {search_resp.status_code}"

        search_data = search_resp.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return f"No PubMed results for: {query}"

        # Step 2: Fetch article summaries
        async with get_throttler("pubmed").throttle():
            fetch_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                },
                timeout=15.0,
            )
        if fetch_resp.status_code != 200:
            return f"PubMed fetch error: HTTP {fetch_resp.status_code}"

        fetch_data = fetch_resp.json()
        results = fetch_data.get("result", {})

        formatted = []
        for i, pmid in enumerate(id_list, 1):
            article = results.get(pmid, {})
            if not article or isinstance(article, str):
                continue

            title = article.get("title", "No title")
            authors_list = article.get("authors", [])
            authors = ", ".join(
                a.get("name", "") for a in authors_list[:3]
            )
            if len(authors_list) > 3:
                authors += f" et al. ({len(authors_list)} authors)"

            journal = article.get("fulljournalname") or article.get("source", "")
            pub_date = article.get("pubdate", "")
            doi = ""
            for aid in article.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
                    break

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            doi_line = f"\n   DOI: https://doi.org/{doi}" if doi else ""

            formatted.append(
                f"{i}. **{title}**\n"
                f"   Authors: {authors}\n"
                f"   Journal: {journal} [{pub_date}]\n"
                f"   PMID: {pmid} | URL: {url}{doi_line}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "PubMed search error: request timed out"
    except Exception as e:
        return f"PubMed search error: {str(e)}"


# ============================================================================
# Wikipedia Full-Text Search (MediaWiki API — free, no auth)
# ============================================================================

async def tool_wikipedia_search(query: str, limit: int = 8) -> str:
    """Search Wikipedia for article content via the MediaWiki API.

    Returns article extracts with full text snippets, not just titles.
    Covers the entire English Wikipedia. Free API, no authentication required.
    """
    limit = min(limit, 15)
    try:
        async with get_throttler("wikipedia").throttle():
            client = _get_http_client()
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "srprop": "snippet|timestamp|wordcount",
                    "format": "json",
                },
                timeout=15.0,
                headers={"User-Agent": "DeepSearchPortal/1.0 (research tool)"},
            )
        if resp.status_code != 200:
            return f"Wikipedia search error: HTTP {resp.status_code}"

        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results for: {query}"

        formatted = []
        for i, result in enumerate(results[:limit], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            # Strip HTML from snippet
            snippet = re.sub(r'<[^>]+>', '', snippet)
            snippet = html.unescape(snippet).strip()
            wordcount = result.get("wordcount", 0)
            timestamp = result.get("timestamp", "")[:10]
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            formatted.append(
                f"{i}. **{title}** [{timestamp}] ({wordcount:,} words)\n"
                f"   URL: {url}\n"
                f"   {snippet}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Wikipedia search error: request timed out"
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


# ============================================================================
# Archive.org Full-Text Search (Internet Archive — free, no auth)
# ============================================================================

async def tool_archiveorg_search(query: str, media_type: str = "", max_results: int = 10) -> str:
    """Search the Internet Archive's full-text index across all collections.

    Covers books, magazines, government documents, academic papers, audio,
    video, software, and web archives. This is NOT the Wayback Machine URL
    lookup — this searches the actual content of archived materials.

    Free API, no authentication required.
    """
    max_results = min(max_results, 15)
    try:
        async with get_throttler("archiveorg").throttle():
            client = _get_http_client()
            params: dict[str, str | int] = {
                "q": query,
                "rows": max_results,
                "output": "json",
                "fl[]": "identifier,title,creator,date,description,mediatype,downloads",
            }
            if media_type:
                params["q"] = f"{query} AND mediatype:{media_type}"

            resp = await client.get(
                "https://archive.org/advancedsearch.php",
                params=params,
                timeout=20.0,
                headers={"User-Agent": "DeepSearchPortal/1.0 (research tool)"},
            )
        if resp.status_code != 200:
            return f"Archive.org search error: HTTP {resp.status_code}"

        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return f"No Archive.org results for: {query}"

        formatted = []
        for i, doc in enumerate(docs[:max_results], 1):
            title = doc.get("title", "No title")
            if isinstance(title, list):
                title = title[0] if title else "No title"
            creator = doc.get("creator", "Unknown")
            if isinstance(creator, list):
                creator = ", ".join(creator[:3])
            date = doc.get("date", "")[:10] if doc.get("date") else ""
            media = doc.get("mediatype", "")
            identifier = doc.get("identifier", "")
            downloads = doc.get("downloads", 0)
            description = doc.get("description", "")
            if isinstance(description, list):
                description = " ".join(description)
            if description:
                description = re.sub(r'<[^>]+>', ' ', description)
                description = html.unescape(description).strip()
                if len(description) > 300:
                    description = description[:300] + "..."

            url = f"https://archive.org/details/{identifier}"
            formatted.append(
                f"{i}. **{title}** by {creator} [{date}] ({media}, {downloads} downloads)\n"
                f"   URL: {url}\n"
                f"   {description}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Archive.org search error: request timed out"
    except Exception as e:
        return f"Archive.org search error: {str(e)}"


# ============================================================================
# Niche Forum Search (SearXNG site-targeted)
# ============================================================================

# Common forum platforms and niche communities to target
_FORUM_SITE_TARGETS = [
    "site:forums.somethingawful.com",
    "site:forum.bodybuilding.com",
    "site:boards.straightdope.com",
    "site:discourse.org",
    "site:community.cloudflare.com",
    "site:forum.xda-developers.com",
    "site:forums.anandtech.com",
    "site:arstechnica.com/civis",
    "site:forums.hardwarezone.com.sg",
    "site:kiwifarms.net",
    "site:resetera.com",
    "site:neogaf.com",
    "site:overclock.net",
    "site:head-fi.org",
    "site:avsforum.com",
    "site:forum.lowyat.net",
]


async def tool_forum_search(query: str, forum_url: str = "") -> str:
    """Search niche internet forums via SearXNG with site-targeting.

    If forum_url is provided, searches that specific forum.
    Otherwise, searches across a curated list of popular niche forums
    (SomethingAwful, Bodybuilding.com, XDA, Head-Fi, AVSForum, etc.)
    plus general forum-targeted queries.

    Use this for niche expertise, hobbyist knowledge, first-hand experiences,
    and discussions not found on mainstream platforms.
    """
    try:
        results_all: list[dict] = []

        if forum_url:
            # Search specific forum
            site_query = f"site:{forum_url.replace('https://', '').replace('http://', '').rstrip('/')} {query}"
            try:
                results = await _searxng_query(site_query, categories="general")
                results_all.extend(results)
            except Exception:
                pass
        else:
            # Search across curated forum list in batches
            # Use 3 batches of forum site targets for breadth
            batch_size = 5
            for batch_start in range(0, min(len(_FORUM_SITE_TARGETS), 15), batch_size):
                batch = _FORUM_SITE_TARGETS[batch_start:batch_start + batch_size]
                site_clause = " OR ".join(batch)
                forum_query = f"({site_clause}) {query}"
                try:
                    batch_results = await _searxng_query(forum_query, categories="general")
                    results_all.extend(batch_results)
                except Exception:
                    continue

            # Also try a generic forum query
            try:
                generic_results = await _searxng_query(
                    f"{query} forum discussion thread",
                    categories="general",
                )
                seen_urls = {r.get("url", "") for r in results_all}
                for r in generic_results:
                    if r.get("url", "") not in seen_urls:
                        results_all.append(r)
                        seen_urls.add(r.get("url", ""))
            except Exception:
                pass

        if not results_all:
            return f"No forum results for: {query}"

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[dict] = []
        for r in results_all:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        return _format_search_results(unique[:15], source_label="forum") or f"No forum results for: {query}"

    except Exception as e:
        return f"Forum search error: {str(e)}"


# ============================================================================
# Google Scholar Search (via SearXNG scholar engine)
# ============================================================================

async def tool_scholar_search(query: str) -> str:
    """Search Google Scholar for academic papers, citations, and theses.

    Uses SearXNG's scholar category which aggregates results from
    Google Scholar, Semantic Scholar, and other academic search engines.
    Broader coverage than arXiv alone — includes journals, conference
    proceedings, theses, patents, and court opinions.
    """
    try:
        results = await _searxng_query(query, categories="science")
    except httpx.TimeoutException:
        return "Scholar search error: request timed out"
    except Exception as e:
        return f"Scholar search error: {str(e)}"

    if not results:
        # Fallback: try general search with academic site targeting
        try:
            fallback_query = (
                f"({query}) (site:scholar.google.com OR site:semanticscholar.org "
                f"OR site:researchgate.net OR site:academia.edu OR site:ssrn.com "
                f"OR site:jstor.org OR site:ncbi.nlm.nih.gov)"
            )
            results = await _searxng_query(fallback_query, categories="general")
        except Exception:
            pass

    return _format_search_results(results[:15], source_label="scholar") or f"No scholar results for: {query}"


# ============================================================================
# Substack Search (SearXNG site-targeted)
# ============================================================================

async def tool_substack_search(query: str) -> str:
    """Search Substack newsletters for long-form analysis and independent journalism.

    Uses SearXNG with site:substack.com targeting. Covers investigative
    journalism, expert commentary, niche analysis, and independent reporting
    that doesn't appear in mainstream media. Use fetch_webpage on URLs
    found here to get full article text.
    """
    try:
        # Primary: site-targeted search
        results = await _searxng_query(
            f"site:substack.com {query}", categories="general"
        )

        # Supplement with site:*.substack.com for custom domains
        try:
            extra = await _searxng_query(
                f"site:*.substack.com {query}", categories="general"
            )
            seen_urls = {r.get("url", "") for r in results}
            for r in extra:
                if r.get("url", "") not in seen_urls:
                    results.append(r)
                    seen_urls.add(r.get("url", ""))
        except Exception:
            pass

        if not results:
            return f"No Substack results for: {query}"

        return _format_search_results(results[:15], source_label="substack") or f"No Substack results for: {query}"

    except httpx.TimeoutException:
        return "Substack search error: request timed out"
    except Exception as e:
        return f"Substack search error: {str(e)}"


# ============================================================================
# Knowledge Graph Search (via knowledge engine microservice)
# ============================================================================

async def tool_knowledge_graph_search(arguments: dict) -> str:
    """Search the knowledge graph via the knowledge engine microservice."""
    try:
        import knowledge_client
        result = await knowledge_client.search(
            namespace=arguments.get("namespace", "default"),
            query=arguments.get("query", ""),
            mode=arguments.get("mode", "hybrid"),
            limit=min(arguments.get("limit", 10), 50),
        )
        results = result.get("results", [])
        if not results:
            return "No matching knowledge found in the graph."

        formatted = []
        for i, r in enumerate(results, 1):
            node_type = r.get("node_type", "")
            name = r.get("name", "")
            content = r.get("content", "")
            score = r.get("score", 0)
            props = r.get("properties", {})
            source_doc = r.get("source_doc", "")

            header = f"{i}. [{node_type}]"
            if name:
                header += f" **{name}**"
            if source_doc:
                header += f" (from: {source_doc})"
            header += f" [score: {score:.3f}]"

            body = content[:2000] if content else ""
            if props:
                prop_strs = []
                for k, v in props.items():
                    if k not in ("id",) and v is not None:
                        prop_strs.append(f"{k}: {v}")
                if prop_strs:
                    body += "\n  Properties: " + ", ".join(prop_strs[:5])

            formatted.append(f"{header}\n{body}" if body else header)

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Knowledge graph search error: {e}"


async def tool_knowledge_discover(arguments: dict) -> str:
    """Run graph discovery algorithms via the knowledge engine microservice."""
    try:
        import knowledge_client
        algorithm = arguments.get("algorithm", "")
        namespace = arguments.get("namespace", "default")
        seed_concepts = arguments.get("seed_concepts", [])
        limit = arguments.get("limit", 15)

        if algorithm == "spreading_activation":
            if not seed_concepts:
                return "Error: seed_concepts required for spreading_activation"
            result = await knowledge_client.spreading_activation(
                namespace=namespace,
                seed_concepts=seed_concepts,
                limit=limit,
            )
        elif algorithm == "swanson_abc":
            if not seed_concepts:
                return "Error: seed_concepts required for swanson_abc"
            result = await knowledge_client.swanson_abc(
                namespace=namespace,
                seed_concept=seed_concepts[0],
                limit=limit,
            )
        elif algorithm == "information_gaps":
            result = await knowledge_client.information_gaps(
                namespace=namespace,
                limit=limit,
            )
        else:
            return f"Unknown algorithm: {algorithm}. Use: spreading_activation, swanson_abc, information_gaps"

        discoveries = result.get("results", [])
        if not discoveries:
            return f"No discoveries from {algorithm}."

        formatted = [f"**{algorithm.replace('_', ' ').title()} Results:**\n"]
        for i, d in enumerate(discoveries, 1):
            parts = [f"{i}."]
            if "name" in d:
                parts.append(f"**{d['name']}**")
            elif "target_concept" in d:
                parts.append(f"**{d['target_concept']}**")
            if "activation" in d:
                parts.append(f"(activation: {d['activation']:.3f})")
            if "discovery_score" in d:
                parts.append(f"(discovery score: {d['discovery_score']:.3f})")
            if "gap_score" in d:
                parts.append(f"(gap score: {d['gap_score']:.3f})")
            if "bridge_count" in d:
                parts.append(f"via {d['bridge_count']} bridge concepts")
            if "top_bridges" in d:
                bridge_names = [b.get("name", "?") for b in d["top_bridges"][:3]]
                parts.append(f"[bridges: {', '.join(bridge_names)}]")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    except Exception as e:
        return f"Knowledge discovery error: {e}"

