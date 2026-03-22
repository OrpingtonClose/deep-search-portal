"""
Additional search tools: Twitter/X, Python exec, arXiv, Wayback, Wikidata,
Hacker News, Stack Exchange, PubMed, Wikipedia, Archive.org, forums,
Google Scholar, Substack, YouTube deep extraction pipeline.
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from shared import get_throttler, http_client

from .config import (
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_CUSTOMER_ID,
    BRIGHT_DATA_ZONE,
    COMMERCIAL_SEARCH_ENABLED,
    OXYLABS_PASSWORD,
    OXYLABS_USERNAME,
    PYTHON_OUTPUT_MAX,
    PYTHON_TIMEOUT,
    WEBPAGE_MAX_CHARS,
    log,
)
from .moderation import _commercial_search
from .scoring import trust_score_url
from .search_tools import _format_search_results, _searxng_query, tool_fetch_webpage, tool_searxng_search
from .web_fetch import _is_censored_response, _strip_html


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
        from urllib.parse import quote
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
        from urllib.parse import quote
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
    client = http_client()
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
            client = http_client()
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
        client = http_client()
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
            client = http_client()
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
    """Unified web search: SearXNG + commercial APIs.

    Always runs SearXNG.  If COMMERCIAL_SEARCH_ENABLED, also queries
    Bright Data / Oxylabs SERP and merges results (deduped by URL).
    No moderation gate — the research proxy must be able to search any topic.
    """
    # Always run SearXNG as baseline.
    searxng_result = await tool_searxng_search(query)

    if not COMMERCIAL_SEARCH_ENABLED:
        return searxng_result

    # Fetch commercial results — no moderation gate.
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
            client = http_client()
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
            client = http_client()
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
        client = http_client()

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
            client = http_client()
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
            client = http_client()
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


async def tool_youtube_search(query: str) -> str:
    """Search YouTube for video content — tutorials, discussions, practitioner knowledge.

    YouTube is a severely underutilized source of deep knowledge: practitioner
    teardowns, community discussions, conference talks, investigative videos,
    and how-to content that rarely appears in text sources.  Uses SearXNG with
    site:youtube.com targeting.  Returns video titles, URLs, and descriptions.
    Use youtube_transcript to get full spoken content from any video.
    Use youtube_video_analyze to have a vision model evaluate video visuals.
    """
    try:
        results = await _searxng_query(
            f"site:youtube.com {query}", categories="general"
        )

        # Also search via the videos category for broader coverage
        try:
            video_results = await _searxng_query(query, categories="videos")
            seen_urls = {r.get("url", "") for r in results}
            for r in video_results:
                url = r.get("url", "")
                if url not in seen_urls and "youtube.com" in url:
                    results.append(r)
                    seen_urls.add(url)
        except Exception:
            pass

        if not results:
            return f"No YouTube results for: {query}"

        return _format_search_results(results[:15], source_label="youtube") or f"No YouTube results for: {query}"

    except httpx.TimeoutException:
        return "YouTube search error: request timed out"
    except Exception as e:
        return f"YouTube search error: {str(e)}"


# ============================================================================
# YouTube Deep Extraction Pipeline
# ============================================================================
# Three layers of video content extraction:
# 1. youtube_transcript — full spoken content via youtube-transcript-api
# 2. youtube_video_metadata — title, description, chapters, comments via yt-dlp
# 3. youtube_video_analyze — visual analysis of video frames via Qwen Omni


def _extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract YouTube video ID from a URL or return it if already an ID."""
    url_or_id = url_or_id.strip()
    # Already a bare video ID (11 chars, alphanumeric + - _)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url_or_id):
        return url_or_id
    # Standard youtube.com/watch?v=ID
    m = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', url_or_id)
    if m:
        return m.group(1)
    # Short youtu.be/ID
    m = re.search(r'youtu\.be/([A-Za-z0-9_-]{11})', url_or_id)
    if m:
        return m.group(1)
    # Embed youtube.com/embed/ID
    m = re.search(r'youtube\.com/embed/([A-Za-z0-9_-]{11})', url_or_id)
    if m:
        return m.group(1)
    # Shorts youtube.com/shorts/ID
    m = re.search(r'youtube\.com/shorts/([A-Za-z0-9_-]{11})', url_or_id)
    if m:
        return m.group(1)
    return None


async def tool_youtube_transcript(url: str, lang: str = "en") -> str:
    """Extract the full transcript/subtitles from a YouTube video.

    Primary path uses LangChain's YoutubeLoader (wraps youtube-transcript-api).
    Falls back to raw youtube-transcript-api with timestamped output if the
    LangChain loader fails.  No API key needed, no browser needed.

    This is the PRIMARY way to extract spoken content from YouTube videos.
    Contains the actual knowledge — practitioner explanations, lecture content,
    interview dialogue, tutorial steps.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return f"Could not extract video ID from: {url}"

    loop = asyncio.get_running_loop()
    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    def _fetch_transcript() -> str:
        # --- Primary: LangChain YoutubeLoader ---
        try:
            from langchain_community.document_loaders import YoutubeLoader

            loader = YoutubeLoader.from_youtube_url(
                yt_url,
                add_video_info=True,
                language=[lang, "en"],
                continue_on_failure=True,
            )
            docs = loader.load()
            if docs:
                meta = docs[0].metadata
                header_parts = []
                if meta.get("title"):
                    header_parts.append(f"TITLE: {meta['title']}")
                if meta.get("author"):
                    header_parts.append(f"CHANNEL: {meta['author']}")
                if meta.get("publish_date"):
                    header_parts.append(f"DATE: {meta['publish_date']}")
                if meta.get("length"):
                    m, s = divmod(int(meta["length"]), 60)
                    header_parts.append(f"DURATION: {m}:{s:02d}")
                if meta.get("view_count"):
                    header_parts.append(f"VIEWS: {meta['view_count']:,}")

                header = "\n".join(header_parts)
                content = docs[0].page_content

                # Cap output
                max_chars = 30000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"

                return f"YOUTUBE TRANSCRIPT for {video_id}:\n{header}\n\n{content}"
        except Exception as e:
            log.debug(f"YoutubeLoader failed for {video_id}, falling back: {e}")

        # --- Fallback: raw youtube-transcript-api with timestamps ---
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return "youtube-transcript-api not installed"

        ytt = YouTubeTranscriptApi()

        try:
            snippets = ytt.fetch(video_id, languages=[lang, "en"])
        except Exception:
            try:
                transcript_list = ytt.list(video_id)
                available = list(transcript_list)
                if not available:
                    return f"No transcripts available for video {video_id}"
                snippets = available[0].fetch()
            except Exception as e:
                return f"Transcript fetch failed for {video_id}: {e}"

        lines = []
        for s in snippets:
            start = s.start if hasattr(s, 'start') else s.get("start", 0)
            text = s.text if hasattr(s, 'text') else s.get("text", "")
            mins, secs = divmod(int(start), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                ts = f"[{hours}:{mins:02d}:{secs:02d}]"
            else:
                ts = f"[{mins}:{secs:02d}]"
            lines.append(f"{ts} {text}")

        full_text = "\n".join(lines)
        max_chars = 30000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + f"\n\n[TRANSCRIPT TRUNCATED — {len(lines)} segments total]"

        return f"YOUTUBE TRANSCRIPT for {video_id}:\n{full_text}"

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_transcript),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        return f"Transcript extraction timed out for {video_id}"
    except Exception as e:
        return f"Transcript extraction error for {video_id}: {e}"


async def tool_youtube_video_metadata(url: str) -> str:
    """Extract rich metadata from a YouTube video using yt-dlp.

    Returns: title, channel, upload date, view count, like count,
    full description, chapter markers, tags, categories, duration,
    and top comments (if available).

    This complements youtube_transcript — the description often contains
    links, timestamps, and context that the spoken content doesn't.
    Comments contain corrections, additional knowledge, and community
    reactions.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return f"Could not extract video ID from: {url}"

    loop = asyncio.get_running_loop()

    def _fetch_metadata() -> str:
        try:
            import yt_dlp
        except ImportError:
            return "yt-dlp not installed"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": False,
            "getcomments": True,
            "extractor_args": {"youtube": {"max_comments": ["30", "0", "0", "0"]}},
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False,
                )
        except Exception as e:
            return f"yt-dlp extraction failed for {video_id}: {e}"

        if not info:
            return f"No metadata returned for {video_id}"

        parts = []
        parts.append(f"TITLE: {info.get('title', 'Unknown')}")
        parts.append(f"CHANNEL: {info.get('channel', info.get('uploader', 'Unknown'))}")
        parts.append(f"UPLOAD DATE: {info.get('upload_date', 'Unknown')}")

        duration = info.get("duration")
        if duration:
            m, s = divmod(int(duration), 60)
            h, m = divmod(m, 60)
            parts.append(f"DURATION: {h}:{m:02d}:{s:02d}" if h else f"DURATION: {m}:{s:02d}")

        view_count = info.get("view_count")
        if view_count is not None:
            parts.append(f"VIEWS: {view_count:,}")

        like_count = info.get("like_count")
        if like_count is not None:
            parts.append(f"LIKES: {like_count:,}")

        tags = info.get("tags") or []
        if tags:
            parts.append(f"TAGS: {', '.join(tags[:20])}")

        categories = info.get("categories") or []
        if categories:
            parts.append(f"CATEGORIES: {', '.join(categories)}")

        description = info.get("description", "")
        if description:
            # Cap description at 3000 chars
            desc_text = description[:3000]
            if len(description) > 3000:
                desc_text += "... [truncated]"
            parts.append(f"\nDESCRIPTION:\n{desc_text}")

        # Chapter markers — these are gold for navigating long videos
        chapters = info.get("chapters") or []
        if chapters:
            parts.append("\nCHAPTERS:")
            for ch in chapters:
                start = ch.get("start_time", 0)
                m, s = divmod(int(start), 60)
                h, m = divmod(m, 60)
                ts = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                parts.append(f"  {ts} - {ch.get('title', 'Untitled')}")

        # Comments — community knowledge, corrections, additional context
        comments = info.get("comments") or []
        if comments:
            parts.append(f"\nTOP COMMENTS ({len(comments)}):")
            for c in comments[:20]:
                author = c.get("author", "Anonymous")
                text = (c.get("text") or "")[:500]
                likes = c.get("like_count", 0)
                prefix = f"  [{likes} likes]" if likes else "  "
                parts.append(f"{prefix} @{author}: {text}")

        return "\n".join(parts)

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_metadata),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        return f"Metadata extraction timed out for {video_id}"
    except Exception as e:
        return f"Metadata extraction error for {video_id}: {e}"


# --- Qwen Omni vision model endpoint for video analysis ---
_QWEN_OMNI_BASE = os.getenv("QWEN_OMNI_BASE_URL", "")
_QWEN_OMNI_KEY = os.getenv("QWEN_OMNI_API_KEY", "")
_QWEN_OMNI_MODEL = os.getenv("QWEN_OMNI_MODEL", "qwen3-omni-30b-a3b-instruct")


async def tool_youtube_video_analyze(
    url: str,
    question: str = "",
) -> str:
    """Analyze a YouTube video's visual content using a vision-capable model.

    Downloads the video (or key frames) and sends to Qwen Omni or another
    vision-language model for in-depth visual analysis.  Use this when:
    - The video contains diagrams, charts, code on screen, product teardowns
    - You need to understand what is SHOWN, not just what is SAID
    - The transcript alone doesn't capture the visual information
    - You want to evaluate the credibility of visual evidence

    Parameters:
        url: YouTube video URL or ID
        question: Specific question about the video visuals (optional).
                  If empty, does a general visual analysis.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return f"Could not extract video ID from: {url}"

    if not _QWEN_OMNI_BASE:
        return (
            "Video visual analysis not available: QWEN_OMNI_BASE_URL not configured. "
            "Set QWEN_OMNI_BASE_URL and QWEN_OMNI_API_KEY environment variables to "
            "point at a Qwen Omni endpoint (vLLM, Ollama, or Alibaba Cloud API). "
            "Use youtube_transcript for spoken content instead."
        )

    loop = asyncio.get_running_loop()
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Step 1: Extract key frames from the video using yt-dlp
    def _extract_frames() -> list[str]:
        """Download video and extract key frames as base64-encoded images."""
        import base64

        try:
            import yt_dlp
        except ImportError:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "worst[ext=mp4]/worst",  # Smallest quality for frame extraction
                "outtmpl": video_path,
                "max_filesize": 100 * 1024 * 1024,  # 100MB cap
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as e:
                log.warning(f"Video download failed for {video_id}: {e}")
                return []

            if not os.path.exists(video_path):
                return []

            # Extract frames using ffmpeg at key intervals
            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            try:
                # Extract 1 frame every 30 seconds, max 20 frames
                subprocess.run(
                    [
                        "ffmpeg", "-i", video_path,
                        "-vf", "fps=1/30,scale=768:-1",
                        "-frames:v", "20",
                        "-q:v", "3",
                        os.path.join(frames_dir, "frame_%03d.jpg"),
                    ],
                    capture_output=True,
                    timeout=60,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log.warning(f"Frame extraction failed for {video_id}: {e}")
                return []

            # Read frames as base64
            frame_files = sorted(
                f for f in os.listdir(frames_dir) if f.endswith(".jpg")
            )
            frames_b64 = []
            for fname in frame_files[:20]:
                fpath = os.path.join(frames_dir, fname)
                with open(fpath, "rb") as f:
                    frames_b64.append(base64.b64encode(f.read()).decode())
            return frames_b64

    try:
        frames_b64 = await asyncio.wait_for(
            loop.run_in_executor(None, _extract_frames),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        return f"Video frame extraction timed out for {video_id}"
    except Exception as e:
        return f"Video frame extraction error for {video_id}: {e}"

    if not frames_b64:
        return (
            f"Could not extract frames from video {video_id}. "
            "Use youtube_transcript for spoken content instead."
        )

    # Step 2: Send frames to Qwen Omni for visual analysis
    prompt = question or (
        "Analyze this video's visual content in detail. Describe what is shown: "
        "diagrams, text on screen, code, products, demonstrations, charts, "
        "environments, people, and any visual evidence relevant to understanding "
        "the video's subject matter. Be specific and factual."
    )

    # Build multimodal message content
    content_parts: list[dict] = []
    content_parts.append({
        "type": "text",
        "text": (
            f"These are {len(frames_b64)} key frames extracted from a YouTube video "
            f"(ID: {video_id}), sampled every 30 seconds. {prompt}"
        ),
    })
    for i, b64 in enumerate(frames_b64):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        client = http_client()
        resp = await client.post(
            f"{_QWEN_OMNI_BASE.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {_QWEN_OMNI_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _QWEN_OMNI_MODEL,
                "messages": [
                    {"role": "user", "content": content_parts},
                ],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            return (
                f"Qwen Omni API error ({resp.status_code}): {resp.text[:500]}. "
                "Use youtube_transcript for spoken content instead."
            )
        data = resp.json()
        analysis = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not analysis:
            return f"Qwen Omni returned empty analysis for {video_id}"

        return (
            f"VISUAL ANALYSIS of YouTube video {video_id} "
            f"({len(frames_b64)} frames analyzed):\n\n{analysis}"
        )
    except httpx.TimeoutException:
        return f"Qwen Omni API timed out analyzing video {video_id}"
    except Exception as e:
        return f"Qwen Omni analysis error for {video_id}: {e}"


