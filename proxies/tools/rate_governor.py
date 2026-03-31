"""
Rate Governor — centralised outbound request management for all internet-accessing tools.

Provides:
  1. Per-provider token-bucket rate limiting (reuses shared.TokenBucketThrottler)
  2. Global concurrency cap — prevents more than N simultaneous outbound requests
  3. Request staggering with jitter — avoids thundering-herd when many subagents
     fire in parallel

Every tool that touches the network MUST go through ``governed_request()`` or
use the ``@governed`` decorator.  This ensures harmonious resource usage for a
permanently running research engine.

Configuration (environment variables):
  GOVERNOR_GLOBAL_CONCURRENCY   — max simultaneous outbound requests (default: 25)
  GOVERNOR_JITTER_MAX_MS        — max random jitter added per request in ms (default: 500)
  THROTTLE_{PROVIDER}_RPS       — per-provider requests/sec (overrides defaults)
  THROTTLE_{PROVIDER}_BURST     — per-provider burst capacity
  THROTTLE_{PROVIDER}_MAX_CONCURRENT — per-provider concurrency cap
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from shared import get_throttler

log = logging.getLogger("rate_governor")

# ---------------------------------------------------------------------------
# Global concurrency limiter
# ---------------------------------------------------------------------------

_GLOBAL_MAX = int(os.getenv("GOVERNOR_GLOBAL_CONCURRENCY", "25"))
_JITTER_MAX_MS = int(os.getenv("GOVERNOR_JITTER_MAX_MS", "500"))

_global_sem: asyncio.Semaphore | None = None
_active_count: int = 0
_total_governed: int = 0
_total_waited: float = 0.0
_total_jitter: float = 0.0


def _get_global_sem() -> asyncio.Semaphore:
    """Lazy-init the global semaphore (must be created inside an event loop)."""
    global _global_sem
    if _global_sem is None:
        _global_sem = asyncio.Semaphore(_GLOBAL_MAX)
    return _global_sem


# ---------------------------------------------------------------------------
# Provider-to-tool mapping
# ---------------------------------------------------------------------------

# Tools that already do their own per-provider throttling internally
# (e.g. search_providers._search_searxng calls get_throttler("searxng") itself).
# For these, the governor applies ONLY global concurrency + jitter, NOT the
# per-provider token bucket — otherwise requests pass through the bucket twice
# and effective throughput is halved.
_SELF_THROTTLED_TOOLS: set[str] = {
    # SearXNG-based tools throttle via search_providers._search_searxng
    "searxng_search", "web_search", "news_search",
    "telegram_search", "darknet_market_search", "facebook_search",
    "discord_search", "signal_search", "whatsapp_search",
    "crunchbase_search", "trustpilot_search",
    "forum_search", "scholar_search", "substack_search",
    # Twitter search throttles via bright_data/oxylabs/nitter internally
    "twitter_search",
    # Social media scrapers throttle via bright_data internally
    "social_media_search", "reddit_search", "instagram_search",
    "tiktok_search", "linkedin_search", "youtube_search",
    # Direct API tools that call get_throttler(provider).throttle() internally
    # — double-acquiring would deadlock at max_concurrent requests
    "arxiv_search", "wikidata_query", "hackernews_search",
    "stackexchange_search", "pubmed_search", "wikipedia_search",
    "archiveorg_search",
    # Imageboard archive tools throttle via get_throttler("imageboard")
    "chan_4plebs_search", "chan_b4k_search", "chan_warosu_search",
    # Wayback fetch throttles via get_throttler("wayback")
    "wayback_fetch",
    # Onion fetch throttles via get_throttler("tor") internally
    "onion_fetch",
    # Grok deep search throttles via get_throttler("xai") internally
    "grok_deep_search",
    # Search gateway delegates to sub-tools that all self-throttle
    "search_gateway",
}

# Maps tool names to provider keys for throttler lookup.
# Tools sharing the same provider share the same rate-limit bucket.
TOOL_PROVIDER_MAP: dict[str, str] = {
    # SearXNG-based tools
    "searxng_search": "searxng",
    "web_search": "searxng",
    "news_search": "searxng",
    "telegram_search": "searxng",
    "darknet_market_search": "searxng",
    "facebook_search": "searxng",
    "discord_search": "searxng",
    "signal_search": "searxng",
    "whatsapp_search": "searxng",
    "crunchbase_search": "searxng",
    "trustpilot_search": "searxng",
    "forum_search": "searxng",
    "scholar_search": "searxng",
    "substack_search": "searxng",
    # Direct API tools
    "arxiv_search": "arxiv",
    "wayback_fetch": "wayback",
    "wikidata_query": "wikidata",
    "hackernews_search": "hackernews",
    "stackexchange_search": "stackexchange",
    "pubmed_search": "pubmed",
    "wikipedia_search": "wikipedia",
    "archiveorg_search": "archiveorg",
    "whois_lookup": "rdap",
    # Social media
    "twitter_search": "nitter",
    "social_media_search": "bright_data",
    "reddit_search": "bright_data",
    "instagram_search": "bright_data",
    "tiktok_search": "bright_data",
    "linkedin_search": "bright_data",
    # YouTube
    "youtube_search": "youtube",
    "youtube_transcript": "youtube",
    "youtube_video_metadata": "youtube",
    "youtube_video_analyze": "youtube",
    # 4chan archives
    "chan_4plebs_search": "imageboard",
    "chan_b4k_search": "imageboard",
    "chan_warosu_search": "imageboard",
    # Grok deep search (xAI Responses API)
    "grok_deep_search": "xai",
    # Search gateway (composite — delegates to sub-tools)
    "search_gateway": "gateway",
    # Web fetch
    "fetch_webpage": "web_fetch",
    # Tor / onion fetch
    "onion_fetch": "tor",
    # Knowledge engine (local, but still throttled)
    "knowledge_graph_search": "knowledge_engine",
    "knowledge_discover": "knowledge_engine",
}

# Additional provider defaults not already in shared._PROVIDER_DEFAULTS.
# These are registered lazily when first accessed via get_throttler().
_EXTRA_PROVIDER_DEFAULTS: dict[str, tuple[float, int, int]] = {
    # (rps, burst_capacity, max_concurrent)
    "hackernews": (3.0, 5, 5),
    "stackexchange": (3.0, 5, 5),
    "pubmed": (3.0, 5, 5),
    "wikipedia": (5.0, 10, 8),
    "archiveorg": (2.0, 5, 5),
    "rdap": (2.0, 5, 5),
    "youtube": (3.0, 5, 5),
}


def _ensure_extra_defaults() -> None:
    """Register extra provider defaults in the shared module's registry."""
    from shared import _PROVIDER_DEFAULTS
    for provider, defaults in _EXTRA_PROVIDER_DEFAULTS.items():
        if provider not in _PROVIDER_DEFAULTS:
            _PROVIDER_DEFAULTS[provider] = defaults


# Call once at import time
_ensure_extra_defaults()


# ---------------------------------------------------------------------------
# Core governed request context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def governed_request(
    tool_name: str,
    *,
    skip_jitter: bool = False,
) -> AsyncGenerator[None, None]:
    """Context manager that enforces rate limiting and concurrency control.

    Usage::

        async with governed_request("arxiv_search"):
            result = await _actual_arxiv_call(query)

    This acquires:
      1. A global concurrency slot (limits total outbound requests)
      2. A provider-specific token-bucket token (limits per-provider rate)
      3. Optional random jitter (avoids thundering-herd)
    """
    global _active_count, _total_governed, _total_waited, _total_jitter

    t0 = time.monotonic()

    # Step 1: Global concurrency gate
    sem = _get_global_sem()
    await sem.acquire()
    _active_count += 1

    try:
        # Step 2: Per-provider rate limit (skip for self-throttled tools)
        throttler = None
        waited = 0.0
        if tool_name not in _SELF_THROTTLED_TOOLS:
            provider = TOOL_PROVIDER_MAP.get(tool_name, "default")
            throttler = get_throttler(provider)
            waited = await throttler.acquire()

        try:
            # Step 3: Jitter to stagger requests
            jitter = 0.0
            if not skip_jitter and _JITTER_MAX_MS > 0:
                jitter = random.uniform(0, _JITTER_MAX_MS / 1000.0)
                await asyncio.sleep(jitter)

            _total_governed += 1
            _total_waited += waited
            _total_jitter += jitter

            yield
        finally:
            if throttler is not None:
                throttler.release()
    finally:
        _active_count -= 1
        sem.release()


# ---------------------------------------------------------------------------
# Stats / monitoring
# ---------------------------------------------------------------------------

def governor_stats() -> dict:
    """Return current governor statistics for health endpoints."""
    return {
        "global_max_concurrent": _GLOBAL_MAX,
        "global_active": _active_count,
        "jitter_max_ms": _JITTER_MAX_MS,
        "total_governed_requests": _total_governed,
        "total_rate_limit_waited_sec": round(_total_waited, 2),
        "total_jitter_sec": round(_total_jitter, 2),
    }
