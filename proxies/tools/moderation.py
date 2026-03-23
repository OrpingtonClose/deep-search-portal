"""
Content classification (advisory only), moderation, and commercial SERP APIs.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus, unquote

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from shared import get_throttler, http_client

from .config import (
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_SERP_ZONE,
    MODERATION_MODEL,
    OXYLABS_PASSWORD,
    OXYLABS_USERNAME,
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    log,
)


# ============================================================================
# Content Classifier (advisory only — does NOT block searches)
# ============================================================================

# Categories for content classification.  These are ADVISORY only — they
# inform model routing (e.g. use uncensored model for flagged queries) but
# do NOT block any search or tool execution.  The whole point of the research
# proxy is to investigate topics that mainstream APIs refuse to touch.
_MODERATION_CATEGORIES = frozenset({
    "sexual",
    "hate_and_discrimination",
    "violence_and_threats",
    "dangerous_and_criminal_content",
    "selfharm",
})


_MODERATION_PROMPT = """You are a content classifier. Categorise the user query below.

Categories:
- sexual
- hate_and_discrimination
- violence_and_threats
- dangerous_and_criminal_content
- selfharm

Respond with ONLY valid JSON (no markdown, no explanation):
{{"categories": ["category1", ...]}}

If no category applies, return {{"categories": []}}.
Be precise — only tag categories that clearly apply to the query's primary topic.
Research or informational queries about sensitive topics should still be categorised accurately."""


def _get_moderation_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for the moderation model."""
    return ChatOpenAI(
        model=MODERATION_MODEL,
        api_key=UPSTREAM_KEY,
        base_url=UPSTREAM_BASE,
        temperature=0.0,
        extra_body={"max_tokens": 100},
    )


async def classify_query(query: str) -> list[str]:
    """Classify a query into content categories (advisory only).

    This is used for MODEL ROUTING — choosing uncensored vs censored LLMs.
    It does NOT gate or block any search, tool, or API call.  The research
    proxy must be able to investigate any topic the user asks about.

    Returns:
        List of category strings that apply (empty list = no flags).
    """
    if not UPSTREAM_KEY:
        return []

    try:
        llm = _get_moderation_llm()
        messages = [
            SystemMessage(content=_MODERATION_PROMPT),
            HumanMessage(content=query),
        ]
        async with get_throttler("mistral").throttle():
            ai_msg = await llm.ainvoke(messages)
        content = ai_msg.content.strip()

        # Parse the JSON response
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                data = json.loads(m.group())
            else:
                log.warning(f"Classifier returned unparseable response: {content[:200]}")
                return []

        categories = [
            cat for cat in data.get("categories", data.get("flagged_categories", []))
            if cat in _MODERATION_CATEGORIES
        ]

        if categories:
            log.info(
                f"Query classified: query='{query[:60]}' "
                f"categories={categories}"
            )
        return categories

    except Exception as e:
        log.warning(f"Classification failed: {e}")
        return []


async def moderate_query(query: str) -> tuple[bool, dict]:
    """Legacy wrapper — always returns is_safe=True.

    Kept for backward compatibility.  The moderation gate no longer blocks
    any search or tool.  Use classify_query() for advisory classification.
    """
    categories = await classify_query(query)
    return True, {"categories": categories}


# ============================================================================
# Commercial SERP APIs
# ============================================================================


def _parse_google_html(html_text: str, source: str = "bright_data") -> list[dict]:
    """Extract search results from raw Google HTML."""
    results: list[dict] = []
    # Pattern 1: Standard organic results with <h3> inside <a href="/url?q=...">
    for m in re.finditer(
        r'<a[^>]+href="/url\?q=([^&"]+)[^"]*"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
        html_text, re.DOTALL,
    ):
        url = unquote(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if url and title and not url.startswith("/"):
            results.append({"title": title, "url": url, "snippet": "", "source": source})
    # Pattern 2: Fallback — <a href="https://..." with <h3>
    if not results:
        for m in re.finditer(
            r'<a[^>]+href="(https?://(?!google\.com|accounts\.google)[^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
            html_text, re.DOTALL,
        ):
            url = unquote(m.group(1))
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if url and title:
                results.append({"title": title, "url": url, "snippet": "", "source": source})
    # Extract snippets from nearby <span> tags (best-effort)
    for r in results[:10]:
        title_esc = re.escape(r["title"][:30])
        snippet_m = re.search(
            title_esc + r'.*?<span[^>]*>(.*?)</span>',
            html_text, re.DOTALL,
        )
        if snippet_m:
            r["snippet"] = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip()[:300]
    return results[:10]


async def _search_bright_data_serp(query: str) -> list[dict]:
    """Search via Bright Data Web Unlocker API.  Returns list of {title, url, snippet}.

    Uses the Web Unlocker zone (mcp_unlocker) to fetch Google search results
    and parses the raw HTML to extract organic results.
    """
    if not BRIGHT_DATA_API_KEY:
        return []
    try:
        async with get_throttler("bright_data").throttle():
            client = http_client()
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&gl=us"
            resp = await client.post(
                "https://api.brightdata.com/request",
                headers={
                    "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "zone": BRIGHT_DATA_SERP_ZONE,
                    "url": search_url,
                    "format": "raw",
                },
                timeout=30.0,
            )
        if resp.status_code != 200:
            body = resp.text[:300] if resp.text else "(empty)"
            log.warning(f"Bright Data SERP: HTTP {resp.status_code} — {body}")
            return []

        # Try JSON first (in case a dedicated SERP zone is configured)
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            organic = data.get("organic", [])
            results = []
            for item in organic[:10]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", item.get("url", "")),
                    "snippet": item.get("description", item.get("snippet", ""))[:300],
                    "source": "bright_data",
                })
            return results

        # Web Unlocker returns raw HTML — parse it
        return _parse_google_html(resp.text)

    except Exception as e:
        log.warning(f"Bright Data SERP error: {e}")
        return []


async def _search_oxylabs_serp(query: str) -> list[dict]:
    """Search via Oxylabs Web Scraper.  Returns list of {title, url, snippet}.

    Tries the Realtime SERP API first (source=google_search with parse=True).
    Falls back to Web Scraper proxy if SERP API returns 401 (credentials
    may be for the proxy product rather than the SERP API).
    """
    if not OXYLABS_USERNAME or not OXYLABS_PASSWORD:
        return []
    try:
        # Try SERP API first
        async with get_throttler("oxylabs").throttle():
            client = http_client()
            resp = await client.post(
                "https://realtime.oxylabs.io/v1/queries",
                auth=(OXYLABS_USERNAME, OXYLABS_PASSWORD),
                json={
                    "source": "google_search",
                    "query": query,
                    "parse": True,
                },
                timeout=30.0,
            )

        if resp.status_code == 200:
            data = resp.json()
            oxy_results = data.get("results", [])
            if oxy_results:
                content = oxy_results[0].get("content", {})
                organic = content.get("results", {}).get("organic", [])
                results = []
                for item in organic[:10]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("desc", item.get("description", ""))[:300],
                        "source": "oxylabs",
                    })
                return results
            return []

        # SERP API failed — try Web Scraper proxy as fallback
        if resp.status_code in (401, 403):
            log.info("Oxylabs SERP API auth failed, trying Web Scraper proxy fallback")
            import httpx as _httpx
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&gl=us"
            async with get_throttler("oxylabs").throttle():
                async with _httpx.AsyncClient(
                    proxy=f"http://{OXYLABS_USERNAME}:{OXYLABS_PASSWORD}@realtime.oxylabs.io:60000",
                    verify=False,
                    timeout=_httpx.Timeout(30.0, connect=10.0),
                    follow_redirects=True,
                ) as proxy_client:
                    proxy_resp = await proxy_client.get(
                        search_url,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            ),
                        },
                    )
                if proxy_resp.status_code == 200:
                    return _parse_google_html(proxy_resp.text, source="oxylabs")
                log.warning(f"Oxylabs proxy fallback: HTTP {proxy_resp.status_code}")
                return []

        body = resp.text[:300] if resp.text else "(empty)"
        log.warning(f"Oxylabs SERP: HTTP {resp.status_code} — {body}")
        return []

    except Exception as e:
        log.warning(f"Oxylabs SERP error: {e}")
        return []


async def _commercial_search(query: str) -> list[dict]:
    """Try Bright Data SERP first, fall back to Oxylabs."""
    results = await _search_bright_data_serp(query)
    if results:
        return results
    return await _search_oxylabs_serp(query)
