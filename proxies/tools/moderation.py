"""Mistral moderation gate and commercial SERP APIs (Bright Data, Oxylabs).

Extracted from persistent_deep_research_proxy.py lines 1547-1741.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote_plus

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from shared import get_throttler

from . import _get_http_client

from .config import (
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    MODERATION_MODEL,
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_SERP_ZONE,
    OXYLABS_USERNAME,
    OXYLABS_PASSWORD,
    COMMERCIAL_SEARCH_ENABLED,
)

log = logging.getLogger("persistent-research")

# ============================================================================
# Mistral Moderation Gate
# ============================================================================

# Categories that indicate content too risky for commercial APIs.
_MODERATION_BLOCK_CATEGORIES = frozenset({
    "sexual",
    "hate_and_discrimination",
    "violence_and_threats",
    "dangerous_and_criminal_content",
    "selfharm",
})


_MODERATION_PROMPT = """You are a content safety classifier. Evaluate the user query below and determine if it falls into any of these blocked categories:
- sexual
- hate_and_discrimination
- violence_and_threats
- dangerous_and_criminal_content
- selfharm

Respond with ONLY valid JSON (no markdown, no explanation):
{"safe": true/false, "flagged_categories": ["category1", ...]}

If the query is safe for commercial search APIs, set safe=true and flagged_categories=[].
If any blocked category applies, set safe=false and list the matching categories.
Be permissive — only flag content that clearly and primarily promotes the listed harms.
Research queries about sensitive topics (drugs, weapons, conflicts) for informational purposes are SAFE."""


def _get_moderation_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for the moderation model."""
    return ChatOpenAI(
        model=MODERATION_MODEL,
        api_key=UPSTREAM_KEY,
        base_url=UPSTREAM_BASE,
        temperature=0.0,
        extra_body={"max_tokens": 100},
    )


async def moderate_query(query: str) -> tuple[bool, dict]:
    """Check a query with Mistral moderation via LangChain before sending to commercial APIs.

    Uses ChatOpenAI with the moderation model so the call is tracked
    by LangChain callbacks (metrics, tracing).

    Returns:
        (is_safe, details)  —  is_safe=True means commercial APIs can be used.
        details contains the flagged categories for logging.
    """
    if not UPSTREAM_KEY:
        return False, {"error": "no API key"}

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
                log.warning(f"Moderation returned unparseable response: {content[:200]}")
                return False, {"error": "unparseable response"}

        is_safe = data.get("safe", False)
        flagged_cats = [
            cat for cat in data.get("flagged_categories", [])
            if cat in _MODERATION_BLOCK_CATEGORIES
        ]

        if not is_safe or flagged_cats:
            log.info(
                f"Moderation blocked commercial search: query='{query[:60]}' "
                f"flagged={flagged_cats}"
            )
            return False, {"flagged": flagged_cats}

        return True, {"flagged_categories": []}

    except Exception as e:
        log.warning(f"Moderation check failed: {e}")
        # Fail closed — don't use commercial APIs if we can't moderate.
        return False, {"error": str(e)}


# ============================================================================
# Commercial SERP APIs
# ============================================================================


async def _search_bright_data_serp(query: str) -> list[dict]:
    """Search via Bright Data SERP API.  Returns list of {title, url, snippet}."""
    if not BRIGHT_DATA_API_KEY:
        return []
    try:
        async with get_throttler("bright_data").throttle():
            client = _get_http_client()
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
                    "format": "json",
                },
                timeout=30.0,
            )
        if resp.status_code != 200:
            log.warning(f"Bright Data SERP: HTTP {resp.status_code}")
            return []

        data = resp.json()
        # Bright Data SERP returns organic results under "organic" key.
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

    except Exception as e:
        log.warning(f"Bright Data SERP error: {e}")
        return []


async def _search_oxylabs_serp(query: str) -> list[dict]:
    """Search via Oxylabs Web Scraper SERP API.  Returns list of {title, url, snippet}."""
    if not OXYLABS_USERNAME or not OXYLABS_PASSWORD:
        return []
    try:
        async with get_throttler("oxylabs").throttle():
            client = _get_http_client()
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
        if resp.status_code != 200:
            log.warning(f"Oxylabs SERP: HTTP {resp.status_code}")
            return []

        data = resp.json()
        # Oxylabs nests results under results[0].content.results.organic
        oxy_results = data.get("results", [])
        if not oxy_results:
            return []

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

    except Exception as e:
        log.warning(f"Oxylabs SERP error: {e}")
        return []


async def _commercial_search(query: str) -> list[dict]:
    """Try Bright Data SERP first, fall back to Oxylabs."""
    results = await _search_bright_data_serp(query)
    if results:
        return results
    return await _search_oxylabs_serp(query)

