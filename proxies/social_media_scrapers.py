"""Social Media Scrapers — Bright Data + Apify with censorship-aware filtering and cost tracking.

This module provides structured social media search tools that route through
commercial scraping services (Bright Data Web Scraper API as primary,
Apify actors as fallback).  Because these are **censored** services that may
silently filter, truncate, or drop results, every response is annotated with
a censorship-confidence flag and the LLM is warned when results look
suspiciously thin.

Cost tracking is built in: every API call is logged to a JSONL cost ledger
and checked against configurable session/monthly budgets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

import search_providers
from shared import get_throttler

log = logging.getLogger("social_media_scrapers")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY", "")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# Bright Data Web Scraper dataset IDs per platform
# These are the standard Bright Data marketplace dataset identifiers.
# Override via env vars if your account uses custom datasets.
BD_DATASET_TWITTER = os.getenv("BD_DATASET_TWITTER", "gd_lwdb4vjm1ehb499uxs")
BD_DATASET_REDDIT = os.getenv("BD_DATASET_REDDIT", "gd_lwe7v5up1cbtqgm16s")
BD_DATASET_INSTAGRAM = os.getenv("BD_DATASET_INSTAGRAM", "gd_l1villgoiiidt09ci")
BD_DATASET_TIKTOK = os.getenv("BD_DATASET_TIKTOK", "gd_lu702kvil5l8fz8cx0")
BD_DATASET_LINKEDIN = os.getenv("BD_DATASET_LINKEDIN", "gd_l1viktl72ow3t9dh8l")
BD_DATASET_YOUTUBE = os.getenv("BD_DATASET_YOUTUBE", "gd_lk538t2k2p1k3oos71")

# Apify actor IDs per platform (community / official actors)
APIFY_ACTOR_TWITTER = os.getenv("APIFY_ACTOR_TWITTER", "quacker/twitter-scraper")
APIFY_ACTOR_REDDIT = os.getenv("APIFY_ACTOR_REDDIT", "trudax/reddit-scraper-lite")
APIFY_ACTOR_INSTAGRAM = os.getenv("APIFY_ACTOR_INSTAGRAM", "apify/instagram-scraper")
APIFY_ACTOR_TIKTOK = os.getenv("APIFY_ACTOR_TIKTOK", "clockworks/tiktok-scraper")
APIFY_ACTOR_YOUTUBE = os.getenv("APIFY_ACTOR_YOUTUBE", "bernardo/youtube-scraper")

# Cost tracking
COST_LOG_DIR = os.getenv(
    "SOCIAL_SCRAPER_COST_LOG_DIR",
    "/opt/persistent_research_logs/costs",
)
SESSION_BUDGET = float(os.getenv("SOCIAL_SCRAPER_SESSION_BUDGET", "5.00"))
MONTHLY_BUDGET = float(os.getenv("SOCIAL_SCRAPER_MONTHLY_BUDGET", "100.00"))

# Bright Data pricing: ~$1.50 per 1K records (Web Scraper API)
BD_COST_PER_RECORD = 0.0015
# Apify pricing: ~$0.25 per actor run (rough average for social scrapers)
APIFY_COST_PER_RUN = 0.25

# Polling config for async Bright Data collections
BD_POLL_INTERVAL = 5  # seconds
BD_POLL_TIMEOUT = 120  # seconds max wait

# Maximum results to request per query (cost control)
MAX_RESULTS_PER_QUERY = 25

# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Track API costs per session and monthly, with budget enforcement."""

    def __init__(self) -> None:
        self._session_total: float = 0.0
        self._entries: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        # Ensure log directory exists
        try:
            Path(COST_LOG_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning(f"Failed to create cost log directory {COST_LOG_DIR}: {e}")

    async def record(
        self,
        provider: str,
        platform: str,
        estimated_cost: float,
        result_count: int,
        query: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record an API call and its estimated cost."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "platform": platform,
            "query": query,
            "estimated_cost_usd": round(estimated_cost, 6),
            "result_count": result_count,
            "session_total_usd": 0.0,  # filled below
            "metadata": metadata or {},
        }
        async with self._lock:
            self._session_total += estimated_cost
            entry["session_total_usd"] = round(self._session_total, 4)
            self._entries.append(entry)

        # Append to daily JSONL log
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_path = Path(COST_LOG_DIR) / f"social_scraper_costs_{day}.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.warning(f"Failed to write cost log: {e}")

    async def check_budget(self, provider: str, platform: str) -> Optional[str]:
        """Check if budget allows another API call.  Returns warning string if over budget."""
        async with self._lock:
            if self._session_total >= SESSION_BUDGET:
                return (
                    f"[BUDGET EXCEEDED] Session budget of ${SESSION_BUDGET:.2f} reached "
                    f"(current: ${self._session_total:.2f}). "
                    f"Social media scraper calls are disabled for this session. "
                    f"Use free tools (searxng_search, chan_* tools) instead."
                )

        # Check monthly budget from JSONL logs
        try:
            monthly_total = await self._get_monthly_total()
            if monthly_total >= MONTHLY_BUDGET:
                return (
                    f"[BUDGET EXCEEDED] Monthly budget of ${MONTHLY_BUDGET:.2f} reached "
                    f"(current: ${monthly_total:.2f}). "
                    f"Social media scraper calls are disabled until next month."
                )
        except Exception:
            pass  # If we can't read logs, allow the call

        return None

    async def _get_monthly_total(self) -> float:
        """Sum costs from this month's JSONL logs."""
        total = 0.0
        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        cost_dir = Path(COST_LOG_DIR)
        if not cost_dir.exists():
            return 0.0
        for log_file in cost_dir.glob(f"social_scraper_costs_{month_prefix}-*.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            total += entry.get("estimated_cost_usd", 0.0)
            except Exception:
                continue
        return total

    def get_session_stats(self) -> dict[str, Any]:
        """Return session cost stats."""
        return {
            "session_total_usd": round(self._session_total, 4),
            "session_budget_usd": SESSION_BUDGET,
            "session_budget_remaining_usd": round(
                max(0, SESSION_BUDGET - self._session_total), 4
            ),
            "call_count": len(self._entries),
            "entries": self._entries[-10:],  # last 10 entries
        }


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Return the global cost tracker instance."""
    return _cost_tracker


# ---------------------------------------------------------------------------
# Censorship-Aware Content Filter
# ---------------------------------------------------------------------------

# Sensitive query terms that commercial scrapers are known to filter
_SENSITIVE_TERMS = [
    "nsfw", "porn", "nude", "sex", "drugs", "suicide", "self-harm",
    "terrorism", "bomb", "weapon", "kill", "murder", "gore",
    "hack", "exploit", "leak", "doxx", "swat",
    "scam", "fraud", "pump and dump", "insider trading",
]

# Minimum expected results for broad queries on active platforms
_MIN_EXPECTED_RESULTS = {
    "twitter": 3,
    "reddit": 3,
    "instagram": 2,
    "tiktok": 2,
    "linkedin": 1,
    "youtube": 2,
}


def _has_sensitive_terms(query: str) -> bool:
    """Check if query contains terms known to trigger commercial scraper filtering."""
    lower = query.lower()
    return any(term in lower for term in _SENSITIVE_TERMS)


def _censorship_warning(
    platform: str,
    query: str,
    result_count: int,
    provider: str,
) -> str:
    """Generate censorship/filtering warning metadata for tool output."""
    warnings = []

    if _has_sensitive_terms(query):
        warnings.append(
            f"Query contains terms that {provider} may filter or suppress. "
            f"Results may be incomplete."
        )

    min_expected = _MIN_EXPECTED_RESULTS.get(platform, 2)
    if result_count == 0:
        warnings.append(
            f"Zero results returned for a {platform} search. This is suspicious "
            f"for an active platform — the query may have been silently filtered "
            f"by {provider}, or the platform may be blocking this type of search."
        )
    elif result_count < min_expected:
        warnings.append(
            f"Only {result_count} result(s) returned (expected at least "
            f"{min_expected} for {platform}). Results may be filtered."
        )

    if not warnings:
        return ""

    header = (
        f"\n\n[CENSORSHIP WARNING — source: {provider} ({platform})]\n"
        f"This data comes from a commercial scraping service that applies "
        f"content moderation. Treat 'no results' or thin results with "
        f"skepticism. Cross-validate with uncensored sources "
        f"(chan archives, Wayback Machine, direct web search).\n"
    )
    for w in warnings:
        header += f"  - {w}\n"

    return header


# ---------------------------------------------------------------------------
# Bright Data Web Scraper API
# ---------------------------------------------------------------------------


async def _bd_trigger_collection(
    dataset_id: str,
    inputs: list[dict[str, Any]],
) -> Optional[str]:
    """Trigger a Bright Data Web Scraper collection, return snapshot_id."""
    if not BRIGHT_DATA_API_KEY:
        return None
    try:
        async with get_throttler("bright_data").throttle(), httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.brightdata.com/datasets/v3/trigger",
                params={"dataset_id": dataset_id, "limit_multiple_results": MAX_RESULTS_PER_QUERY},
                headers={
                    "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=inputs,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("snapshot_id")
            log.debug(f"BD trigger failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
    except Exception as e:
        log.debug(f"BD trigger error: {e}")
        return None


async def _bd_poll_snapshot(snapshot_id: str) -> Optional[list[dict]]:
    """Poll Bright Data for snapshot results until ready or timeout."""
    if not BRIGHT_DATA_API_KEY or not snapshot_id:
        return None
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() - start < BD_POLL_TIMEOUT:
                resp = await client.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    params={"format": "json"},
                    headers={"Authorization": f"Bearer {BRIGHT_DATA_API_KEY}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                    # Sometimes returns {"status": "running"}
                    status = data.get("status", "")
                    if status in ("ready", "complete"):
                        return data.get("data", [])
                elif resp.status_code == 202:
                    pass  # Still processing
                else:
                    log.debug(f"BD poll error: HTTP {resp.status_code}")
                    return None
                await asyncio.sleep(BD_POLL_INTERVAL)
        log.debug(f"BD poll timeout for snapshot {snapshot_id}")
        return None
    except Exception as e:
        log.debug(f"BD poll error: {e}")
        return None


async def _bd_search(
    platform: str,
    dataset_id: str,
    inputs: list[dict[str, Any]],
    query: str,
) -> Optional[list[dict]]:
    """Full Bright Data search: trigger → poll → return results."""
    budget_warning = await _cost_tracker.check_budget("bright_data", platform)
    if budget_warning:
        return None  # Caller will handle budget warning separately

    snapshot_id = await _bd_trigger_collection(dataset_id, inputs)
    if not snapshot_id:
        return None

    results = await _bd_poll_snapshot(snapshot_id)
    if results is not None:
        cost = len(results) * BD_COST_PER_RECORD
        await _cost_tracker.record(
            provider="bright_data",
            platform=platform,
            estimated_cost=cost,
            result_count=len(results),
            query=query,
            metadata={"snapshot_id": snapshot_id, "dataset_id": dataset_id},
        )
    return results


# ---------------------------------------------------------------------------
# Apify Actor API
# ---------------------------------------------------------------------------


async def _apify_run_actor(
    actor_id: str,
    run_input: dict[str, Any],
    platform: str,
    query: str,
    timeout_secs: int = 120,
) -> Optional[list[dict]]:
    """Run an Apify actor and retrieve results."""
    if not APIFY_API_TOKEN:
        return None

    budget_warning = await _cost_tracker.check_budget("apify", platform)
    if budget_warning:
        return None

    try:
        # Use REST API directly to avoid requiring apify-client dependency
        # Apify REST API uses ~ as separator in actor IDs (e.g., "trudax~reddit-scraper-lite")
        api_actor_id = actor_id.replace("/", "~")
        async with httpx.AsyncClient(timeout=httpx.Timeout(float(timeout_secs + 30), connect=15.0)) as client:
            # Throttle only the trigger POST, not the polling loop
            async with get_throttler("apify").throttle():
                resp = await client.post(
                    f"https://api.apify.com/v2/acts/{api_actor_id}/runs",
                    params={"token": APIFY_API_TOKEN, "timeout": timeout_secs},
                    json=run_input,
                    headers={"Content-Type": "application/json"},
                )
            if resp.status_code not in (200, 201):
                log.debug(f"Apify run failed: HTTP {resp.status_code} — {resp.text[:200]}")
                return None

            run_data = resp.json().get("data", {})
            run_id = run_data.get("id")
            if not run_id:
                return None

            # Poll for completion (unthrottled — uses its own sleep interval)
            start = time.monotonic()
            while time.monotonic() - start < timeout_secs + 10:
                status_resp = await client.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}",
                    params={"token": APIFY_API_TOKEN},
                )
                if status_resp.status_code != 200:
                    break
                run_status = status_resp.json().get("data", {}).get("status", "")
                if run_status == "SUCCEEDED":
                    dataset_id = status_resp.json().get("data", {}).get("defaultDatasetId")
                    if not dataset_id:
                        return []
                    # Fetch results
                    items_resp = await client.get(
                        f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                        params={"token": APIFY_API_TOKEN, "limit": MAX_RESULTS_PER_QUERY},
                    )
                    if items_resp.status_code == 200:
                        results = items_resp.json()
                        if isinstance(results, list):
                            await _cost_tracker.record(
                                provider="apify",
                                platform=platform,
                                estimated_cost=APIFY_COST_PER_RUN,
                                result_count=len(results),
                                query=query,
                                metadata={"actor_id": actor_id, "run_id": run_id},
                            )
                            return results
                    return []
                elif run_status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    log.debug(f"Apify run {run_id} {run_status}")
                    return None
                await asyncio.sleep(5)

            log.debug(f"Apify run {run_id} polling timeout")
            return None
    except Exception as e:
        log.debug(f"Apify error: {e}")
        return None


# ---------------------------------------------------------------------------
# Platform-Specific Formatters
# ---------------------------------------------------------------------------


def _format_twitter_results(records: list[dict], provider: str) -> str:
    """Format Twitter/X records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        user = r.get("user_posted") or r.get("author") or r.get("username") or "unknown"
        text = r.get("description") or r.get("text") or r.get("content") or ""
        date = r.get("date_posted") or r.get("date") or r.get("created_at") or ""
        likes = r.get("likes") or r.get("favorite_count") or 0
        retweets = r.get("retweets") or r.get("retweet_count") or 0
        url = r.get("url") or r.get("tweet_url") or ""
        lines.append(
            f"{i}. @{user} [{date}] (likes: {likes}, RT: {retweets})\n"
            f"   {text[:500]}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


def _format_reddit_results(records: list[dict], provider: str) -> str:
    """Format Reddit records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        title = r.get("title") or r.get("post_title") or ""
        author = r.get("author") or r.get("user") or "unknown"
        subreddit = r.get("subreddit") or r.get("subreddit_name") or ""
        score = r.get("score") or r.get("upvotes") or 0
        comments = r.get("num_comments") or r.get("comments_count") or 0
        text = r.get("selftext") or r.get("body") or r.get("content") or ""
        url = r.get("url") or r.get("post_url") or ""
        date = r.get("created_utc") or r.get("date") or ""
        lines.append(
            f"{i}. r/{subreddit} — **{title}** by u/{author} [{date}] "
            f"(score: {score}, comments: {comments})\n"
            f"   {text[:500]}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


def _format_instagram_results(records: list[dict], provider: str) -> str:
    """Format Instagram records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        user = r.get("user_posted") or r.get("owner_username") or r.get("username") or "unknown"
        caption = r.get("description") or r.get("caption") or r.get("text") or ""
        likes = r.get("likes") or r.get("like_count") or 0
        comments = r.get("num_comments") or r.get("comment_count") or 0
        date = r.get("date_posted") or r.get("taken_at") or ""
        url = r.get("url") or r.get("post_url") or ""
        hashtags = r.get("hashtags") or []
        if isinstance(hashtags, list):
            hashtags = " ".join(f"#{t}" for t in hashtags[:5])
        lines.append(
            f"{i}. @{user} [{date}] (likes: {likes}, comments: {comments})\n"
            f"   {caption[:500]}\n"
            f"   {hashtags}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


def _format_tiktok_results(records: list[dict], provider: str) -> str:
    """Format TikTok records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        user = r.get("author") or r.get("username") or r.get("user_posted") or "unknown"
        text = r.get("description") or r.get("text") or r.get("caption") or ""
        likes = r.get("likes") or r.get("digg_count") or 0
        views = r.get("views") or r.get("play_count") or 0
        date = r.get("date_posted") or r.get("create_time") or ""
        url = r.get("url") or r.get("video_url") or ""
        lines.append(
            f"{i}. @{user} [{date}] (views: {views}, likes: {likes})\n"
            f"   {text[:500]}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


def _format_linkedin_results(records: list[dict], provider: str) -> str:
    """Format LinkedIn records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        author = r.get("author") or r.get("user_posted") or r.get("name") or "unknown"
        title = r.get("title") or r.get("headline") or ""
        text = r.get("post_text") or r.get("text") or r.get("content") or ""
        likes = r.get("likes") or r.get("num_likes") or 0
        date = r.get("date_posted") or r.get("date") or ""
        url = r.get("url") or r.get("post_url") or ""
        lines.append(
            f"{i}. {author} — {title} [{date}] (likes: {likes})\n"
            f"   {text[:500]}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


def _format_youtube_results(records: list[dict], provider: str) -> str:
    """Format YouTube records into readable text."""
    if not records:
        return ""
    lines = []
    for i, r in enumerate(records[:MAX_RESULTS_PER_QUERY], 1):
        title = r.get("title") or ""
        channel = r.get("youtuber") or r.get("channel_name") or r.get("author") or "unknown"
        views = r.get("views") or r.get("view_count") or 0
        likes = r.get("likes") or r.get("like_count") or 0
        date = r.get("date") or r.get("upload_date") or r.get("published_at") or ""
        url = r.get("url") or r.get("video_url") or ""
        description = r.get("description") or r.get("text") or ""
        lines.append(
            f"{i}. **{title}** by {channel} [{date}] "
            f"(views: {views}, likes: {likes})\n"
            f"   {description[:300]}\n"
            f"   {url}"
        )
    return "\n\n".join(lines)


_FORMATTERS = {
    "twitter": _format_twitter_results,
    "reddit": _format_reddit_results,
    "instagram": _format_instagram_results,
    "tiktok": _format_tiktok_results,
    "linkedin": _format_linkedin_results,
    "youtube": _format_youtube_results,
}


# ---------------------------------------------------------------------------
# Bright Data Platform-Specific Input Builders
# ---------------------------------------------------------------------------


def _bd_twitter_input(query: str) -> list[dict]:
    from urllib.parse import quote
    return [{"url": f"https://x.com/search?q={quote(query, safe='')}&src=typed_query&f=live"}]


def _bd_reddit_input(query: str, subreddit: str = "", sort: str = "relevance") -> list[dict]:
    from urllib.parse import quote
    if subreddit:
        return [{"url": f"https://www.reddit.com/r/{subreddit}/search/?q={quote(query, safe='')}&sort={sort}&t=all"}]
    return [{"url": f"https://www.reddit.com/search/?q={quote(query, safe='')}&sort={sort}&t=all"}]


def _bd_instagram_input(query: str) -> list[dict]:
    from urllib.parse import quote
    return [{"url": f"https://www.instagram.com/explore/tags/{quote(query.replace(' ', '').lower(), safe='')}/"}]


def _bd_tiktok_input(query: str) -> list[dict]:
    from urllib.parse import quote
    return [{"url": f"https://www.tiktok.com/search?q={quote(query, safe='')}"}]


def _bd_linkedin_input(query: str) -> list[dict]:
    from urllib.parse import quote
    return [{"url": f"https://www.linkedin.com/search/results/content/?keywords={quote(query, safe='')}"}]


def _bd_youtube_input(query: str) -> list[dict]:
    from urllib.parse import quote
    return [{"url": f"https://www.youtube.com/results?search_query={quote(query, safe='')}"}]


_BD_DATASETS = {
    "twitter": BD_DATASET_TWITTER,
    "reddit": BD_DATASET_REDDIT,
    "instagram": BD_DATASET_INSTAGRAM,
    "tiktok": BD_DATASET_TIKTOK,
    "linkedin": BD_DATASET_LINKEDIN,
    "youtube": BD_DATASET_YOUTUBE,
}

_BD_INPUT_BUILDERS = {
    "twitter": lambda q, **kw: _bd_twitter_input(q),
    "reddit": lambda q, **kw: _bd_reddit_input(q, kw.get("subreddit", ""), kw.get("sort", "relevance")),
    "instagram": lambda q, **kw: _bd_instagram_input(q),
    "tiktok": lambda q, **kw: _bd_tiktok_input(q),
    "linkedin": lambda q, **kw: _bd_linkedin_input(q),
    "youtube": lambda q, **kw: _bd_youtube_input(q),
}

# ---------------------------------------------------------------------------
# Apify Platform-Specific Input Builders
# ---------------------------------------------------------------------------

_APIFY_ACTORS = {
    "twitter": APIFY_ACTOR_TWITTER,
    "reddit": APIFY_ACTOR_REDDIT,
    "instagram": APIFY_ACTOR_INSTAGRAM,
    "tiktok": APIFY_ACTOR_TIKTOK,
    "youtube": APIFY_ACTOR_YOUTUBE,
    # LinkedIn not supported via Apify free actors
}


def _apify_twitter_input(query: str) -> dict:
    return {"searchTerms": [query], "maxTweets": MAX_RESULTS_PER_QUERY, "sort": "Latest"}


def _apify_reddit_input(query: str, subreddit: str = "", sort: str = "relevance") -> dict:
    from urllib.parse import quote
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search/?q={quote(query, safe='')}&sort={sort}&t=all"
    else:
        url = f"https://www.reddit.com/search/?q={quote(query, safe='')}&sort={sort}&t=all"
    return {"startUrls": [{"url": url}], "maxItems": MAX_RESULTS_PER_QUERY}


def _apify_instagram_input(query: str) -> dict:
    return {"hashtags": [query.replace(" ", "").lower()], "resultsLimit": MAX_RESULTS_PER_QUERY}


def _apify_tiktok_input(query: str) -> dict:
    return {"searchQueries": [query], "maxResults": MAX_RESULTS_PER_QUERY}


def _apify_youtube_input(query: str) -> dict:
    return {"searchKeywords": [query], "maxResults": MAX_RESULTS_PER_QUERY}


_APIFY_INPUT_BUILDERS = {
    "twitter": lambda q, **kw: _apify_twitter_input(q),
    "reddit": lambda q, **kw: _apify_reddit_input(q, kw.get("subreddit", ""), kw.get("sort", "relevance")),
    "instagram": lambda q, **kw: _apify_instagram_input(q),
    "tiktok": lambda q, **kw: _apify_tiktok_input(q),
    "youtube": lambda q, **kw: _apify_youtube_input(q),
}


# ---------------------------------------------------------------------------
# Tier 3: SearXNG site-scoped fallback (free, no credentials needed)
# ---------------------------------------------------------------------------

# Map platform → site domain for SearXNG site: operator
_SEARXNG_SITE_DOMAINS: dict[str, str] = {
    "reddit": "reddit.com",
    "twitter": "x.com OR site:twitter.com",
    "youtube": "youtube.com",
}


async def _searxng_site_fallback(
    platform: str, query: str, **kwargs: Any
) -> Optional[list[dict]]:
    """Search via SearXNG using site: operator as a free fallback.

    Only supports platforms with publicly indexable web content
    (reddit, twitter, youtube).  Returns raw dicts compatible with
    the formatter pipeline, or None on failure.
    """
    site_domain = _SEARXNG_SITE_DOMAINS.get(platform)
    if not site_domain:
        return None

    # Build site-scoped query
    subreddit = kwargs.get("subreddit", "")
    if platform == "reddit" and subreddit:
        scoped_query = f"site:reddit.com/r/{subreddit} {query}"
    elif "OR" in (site_domain or ""):
        # Multi-domain (twitter): use parenthesised OR
        scoped_query = f"(site:{site_domain}) {query}"
    else:
        scoped_query = f"site:{site_domain} {query}"

    try:
        raw_results = await search_providers.search_as_raw(
            scoped_query, categories="general", max_results=15,
        )
        if not raw_results:
            return None

        # Normalise into the dict shape that formatters expect
        normalised: list[dict] = []
        for r in raw_results:
            url = r.get("url", "")
            title = r.get("title", "")
            content = r.get("content", r.get("snippet", ""))

            if platform == "reddit":
                # Extract subreddit from URL heuristics
                sub_match = re.search(r"reddit\.com/r/(\w+)", url)
                normalised.append({
                    "title": title,
                    "url": url,
                    "body": content,
                    "subreddit": sub_match.group(1) if sub_match else "",
                    "author": "",
                    "created_utc": "",
                    "score": 0,
                    "num_comments": 0,
                })
            else:
                normalised.append({
                    "title": title,
                    "url": url,
                    "description": content,
                    "content": content,
                })

        return normalised if normalised else None
    except Exception as e:
        log.debug(f"SearXNG site-scoped fallback failed for {platform}: {e}")
        return None


# ---------------------------------------------------------------------------
# Unified Social Media Search
# ---------------------------------------------------------------------------


async def social_media_search(
    platform: str,
    query: str,
    **kwargs: Any,
) -> str:
    """Unified social media search with Bright Data → Apify fallback.

    Args:
        platform: One of "twitter", "reddit", "instagram", "tiktok", "linkedin", "youtube"
        query: Search query string
        **kwargs: Platform-specific options (e.g., subreddit="wallstreetbets")

    Returns:
        Formatted search results with censorship warnings and cost metadata.
    """
    platform = platform.lower().strip()
    if platform not in _FORMATTERS:
        return f"Unsupported platform: {platform}. Supported: {', '.join(_FORMATTERS.keys())}"

    # Check budget before doing anything
    budget_warning = await _cost_tracker.check_budget("any", platform)
    if budget_warning:
        return budget_warning

    formatter = _FORMATTERS[platform]
    results: Optional[list[dict]] = None
    provider_used = "none"

    # Tier 1: Bright Data Web Scraper API
    if BRIGHT_DATA_API_KEY and platform in _BD_DATASETS:
        dataset_id = _BD_DATASETS[platform]
        input_builder = _BD_INPUT_BUILDERS.get(platform)
        if input_builder:
            inputs = input_builder(query, **kwargs)
            results = await _bd_search(platform, dataset_id, inputs, query)
            if results is not None:
                provider_used = "bright_data"

    # Tier 2: Apify Actor API (also triggered when BD returns empty list)
    if not results and APIFY_API_TOKEN and platform in _APIFY_ACTORS:
        actor_id = _APIFY_ACTORS[platform]
        input_builder = _APIFY_INPUT_BUILDERS.get(platform)
        if input_builder:
            actor_input = input_builder(query, **kwargs)
            results = await _apify_run_actor(actor_id, actor_input, platform, query)
            if results is not None:
                provider_used = "apify"

    # Tier 3: SearXNG site-scoped fallback for platforms with web-searchable content
    if not results:
        searxng_fallback = await _searxng_site_fallback(platform, query, **kwargs)
        if searxng_fallback:
            results = searxng_fallback
            provider_used = "searxng_fallback"

    # Format results
    if results is None or len(results) == 0:
        result_count = 0
        if not BRIGHT_DATA_API_KEY and not APIFY_API_TOKEN:
            formatted = (
                f"[TOOL_ERROR] {platform} search failed for: {query}. "
                "No social media scraper credentials configured "
                "(BRIGHT_DATA_API_KEY and APIFY_API_TOKEN both missing). "
                "SearXNG site-scoped fallback was also attempted but returned nothing. "
                "This is a technical/configuration failure, NOT 'no results found'."
            )
        else:
            provider_label = provider_used if provider_used != "none" else "bright_data, apify, searxng_fallback (all failed)"
            formatted = (
                f"[TOOL_ERROR] {platform} search returned 0 results for: {query}. "
                f"Tried providers: {provider_label}. All tiers returned empty. "
                "Results may be filtered or the search may have failed silently."
            )
        provider_used = provider_used if provider_used != "none" else ("none (no credentials)" if not BRIGHT_DATA_API_KEY and not APIFY_API_TOKEN else "none (search failed)")
    else:
        result_count = len(results)
        formatted = formatter(results, provider_used)

    # Build response with metadata
    header = f"**{platform.title()} search results for: {query}** (via {provider_used}, {result_count} results)\n\n"

    # Add censorship warning if applicable
    warning = _censorship_warning(platform, query, result_count, provider_used)

    # Cost info
    stats = _cost_tracker.get_session_stats()
    cost_line = (
        f"\n\n[Cost: session total ${stats['session_total_usd']:.4f} "
        f"/ ${stats['session_budget_usd']:.2f} budget, "
        f"{stats['call_count']} API calls this session]"
    )

    return header + formatted + warning + cost_line


# ---------------------------------------------------------------------------
# Individual Platform Tool Functions
# ---------------------------------------------------------------------------


async def tool_reddit_search(
    query: str,
    subreddit: str = "",
    sort: str = "relevance",
) -> str:
    """Search Reddit posts and comments.

    Args:
        query: Search terms
        subreddit: Optional subreddit to search within (e.g., "wallstreetbets")
        sort: Sort order — "relevance", "hot", "top", "new"
    """
    return await social_media_search("reddit", query, subreddit=subreddit, sort=sort)


async def tool_instagram_search(
    query: str,
    result_type: str = "posts",
) -> str:
    """Search Instagram posts by hashtag or keyword.

    Args:
        query: Hashtag or keyword to search
        result_type: "posts" (default) or "profiles"
    """
    return await social_media_search("instagram", query, result_type=result_type)


async def tool_tiktok_search(
    query: str,
    result_type: str = "posts",
) -> str:
    """Search TikTok videos by keyword.

    Args:
        query: Search terms
        result_type: "posts" (default)
    """
    return await social_media_search("tiktok", query, result_type=result_type)


async def tool_linkedin_search(
    query: str,
    result_type: str = "posts",
) -> str:
    """Search LinkedIn posts by keyword.

    Note: LinkedIn scraping is only available via Bright Data (not Apify).

    Args:
        query: Search terms
        result_type: "posts" (default)
    """
    return await social_media_search("linkedin", query, result_type=result_type)


async def tool_youtube_search(
    query: str,
    result_type: str = "videos",
) -> str:
    """Search YouTube videos by keyword.

    Args:
        query: Search terms
        result_type: "videos" (default)
    """
    return await social_media_search("youtube", query, result_type=result_type)


async def tool_social_media_search(
    platform: str,
    query: str,
    subreddit: str = "",
    result_type: str = "posts",
) -> str:
    """Unified social media search across all supported platforms.

    Args:
        platform: "twitter", "reddit", "instagram", "tiktok", "linkedin", "youtube"
        query: Search terms
        subreddit: Reddit-specific — subreddit to search within
        result_type: Platform-specific result type
    """
    return await social_media_search(platform, query, subreddit=subreddit, result_type=result_type)
