"""Live integration tests for all data source tools.

These tests call REAL APIs with real queries — they are NOT mocked.
They validate that each tool:
  1. Returns a non-empty string
  2. Does NOT return a [TOOL_ERROR] prefix (meaning the tool worked)
  3. Does NOT return a generic "Unknown tool" error
  4. Returns actual content (not just "No results" for well-known queries)

Usage:
  # Free tools only (no credentials needed, but SearXNG must be running):
  pytest tests/test_live_data_sources.py -m live_free -v --timeout=60

  # All tools (requires BD/Apify/Oxylabs credentials):
  pytest tests/test_live_data_sources.py -m live -v --timeout=60

  # Generate JSON report:
  pytest tests/test_live_data_sources.py -m live -v --json-report --json-report-file=live_results.json

Environment:
  Run from the production VM where SearXNG is running on localhost:8888
  and credentials (BRIGHT_DATA_API_KEY, APIFY_API_TOKEN, etc.) are set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure proxies/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxies"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def live_results():
    """Collect pass/fail results across all live tests for summary report."""
    results: list[dict] = []
    yield results
    # Print summary at the end of the session
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a pass/fail matrix and write JSON report."""
    if not results:
        return

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    errored = [r for r in results if r["status"] == "ERROR"]

    print("\n" + "=" * 72)
    print("LIVE DATA SOURCE TEST SUMMARY")
    print("=" * 72)
    print(f"  PASSED:  {len(passed)}/{len(results)}")
    print(f"  FAILED:  {len(failed)}/{len(results)}")
    print(f"  ERRORED: {len(errored)}/{len(results)}")
    print("-" * 72)

    for r in results:
        icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}[r["status"]]
        duration = f"{r.get('duration_s', 0):.1f}s"
        print(f"  [{icon:>4}] {r['tool']:<30} {duration:>8}  {r.get('note', '')[:40]}")

    print("=" * 72)

    # Write JSON report
    report_dir = os.getenv("PERSISTENT_RESEARCH_LOG_DIR", "/opt/persistent_research_logs")
    report_path = os.path.join(report_dir, "live_tool_test_results.json")
    try:
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total": len(results),
                "passed": len(passed),
                "failed": len(failed),
                "errored": len(errored),
                "results": results,
            }, f, indent=2)
        print(f"\nReport written to: {report_path}")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(live_results, tool: str, status: str, duration: float, note: str = ""):
    live_results.append({
        "tool": tool,
        "status": status,
        "duration_s": round(duration, 2),
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _assert_tool_success(result: str, tool_name: str) -> None:
    """Common assertions for a successful tool call."""
    assert isinstance(result, str), f"{tool_name} returned non-string: {type(result)}"
    assert len(result.strip()) > 0, f"{tool_name} returned empty string"
    assert not result.startswith("[TOOL_ERROR]"), (
        f"{tool_name} returned a tool error: {result[:200]}"
    )
    assert "Unknown tool:" not in result, (
        f"{tool_name} is not registered in tool_executor: {result[:200]}"
    )


def _assert_has_content(result: str, tool_name: str) -> None:
    """Assert the result contains actual content, not just 'No results'."""
    lower = result.lower()
    # These patterns indicate the tool ran but found nothing — for well-known
    # queries this should not happen if the tool is working properly.
    empty_patterns = [
        "no results found",
        "no results for:",
        "no forum results",
        "no substack results",
        "no telegram results",
        "no hacker news results",
        "no arxiv papers found",
        "no pubmed results",
        "no wikipedia results",
        "no archive.org results",
        "no scholar results",
        "no wikidata entity found",
        "no archived version found",
        "no darknet market osint results",
    ]
    for pattern in empty_patterns:
        if pattern in lower:
            pytest.fail(
                f"{tool_name} returned 'no results' for a well-known query "
                f"(this likely means the tool is broken): {result[:300]}"
            )


# ---------------------------------------------------------------------------
# Import tool functions
# ---------------------------------------------------------------------------

# We import lazily inside tests to avoid import-time crashes on machines
# without the full environment configured.


def _import_search_tools2():
    from tools.search_tools2 import (
        tool_arxiv_search,
        tool_hackernews_search,
        tool_stackexchange_search,
        tool_pubmed_search,
        tool_wikipedia_search,
        tool_archiveorg_search,
        tool_forum_search,
        tool_scholar_search,
        tool_substack_search,
        tool_telegram_search,
        tool_darknet_market_search,
        tool_youtube_search,
        tool_wikidata_query,
        tool_wayback_fetch,
        tool_twitter_search,
    )
    return {
        "arxiv_search": tool_arxiv_search,
        "hackernews_search": tool_hackernews_search,
        "stackexchange_search": tool_stackexchange_search,
        "pubmed_search": tool_pubmed_search,
        "wikipedia_search": tool_wikipedia_search,
        "archiveorg_search": tool_archiveorg_search,
        "forum_search": tool_forum_search,
        "scholar_search": tool_scholar_search,
        "substack_search": tool_substack_search,
        "telegram_search": tool_telegram_search,
        "darknet_market_search": tool_darknet_market_search,
        "youtube_search": tool_youtube_search,
        "wikidata_query": tool_wikidata_query,
        "wayback_fetch": tool_wayback_fetch,
        "twitter_search": tool_twitter_search,
    }


def _import_web_fetch():
    from tools.web_fetch import (
        enhanced_web_fetch,
        tool_4plebs_search,
        tool_b4k_search,
        tool_warosu_search,
    )
    return {
        "fetch_webpage": enhanced_web_fetch,
        "4plebs_search": tool_4plebs_search,
        "b4k_search": tool_b4k_search,
        "warosu_search": tool_warosu_search,
    }


def _import_search_tools():
    from tools.search_tools import (
        tool_searxng_search,
        tool_news_search,
    )
    return {
        "searxng_search": tool_searxng_search,
        "news_search": tool_news_search,
    }


def _import_social_media():
    import social_media_scrapers
    return {
        "reddit_search": social_media_scrapers.tool_reddit_search,
        "instagram_search": social_media_scrapers.tool_instagram_search,
        "tiktok_search": social_media_scrapers.tool_tiktok_search,
        "social_media_search": social_media_scrapers.tool_social_media_search,
    }


def _import_tool_executor():
    from tools.tool_executor import execute_tool
    return execute_tool


# ============================================================================
# FREE TIER TESTS — no credentials needed (except SearXNG running)
# ============================================================================


@pytest.mark.live
@pytest.mark.live_free
class TestFreeAPIs:
    """Tools that use free, unauthenticated APIs."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_hackernews_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["hackernews_search"]("startup funding", "relevance", "")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "hackernews_search")
            _assert_has_content(result, "hackernews_search")
            _record(live_results, "hackernews_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "hackernews_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_stackexchange_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["stackexchange_search"]("python asyncio", "stackoverflow", "relevance")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "stackexchange_search")
            _assert_has_content(result, "stackexchange_search")
            _record(live_results, "stackexchange_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "stackexchange_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_pubmed_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["pubmed_search"]("insulin diabetes", 5)
            duration = time.monotonic() - t0
            _assert_tool_success(result, "pubmed_search")
            _assert_has_content(result, "pubmed_search")
            assert "pubmed" in result.lower() or "PMID" in result, "Expected PubMed article references"
            _record(live_results, "pubmed_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "pubmed_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_wikipedia_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["wikipedia_search"]("insulin", 5)
            duration = time.monotonic() - t0
            _assert_tool_success(result, "wikipedia_search")
            _assert_has_content(result, "wikipedia_search")
            assert "wikipedia.org" in result.lower(), "Expected Wikipedia URLs in results"
            _record(live_results, "wikipedia_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "wikipedia_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_arxiv_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["arxiv_search"]("machine learning", 3)
            duration = time.monotonic() - t0
            _assert_tool_success(result, "arxiv_search")
            _assert_has_content(result, "arxiv_search")
            assert "arxiv.org" in result.lower(), "Expected arXiv URLs in results"
            _record(live_results, "arxiv_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "arxiv_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_archiveorg_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["archiveorg_search"]("world war history", "", 5)
            duration = time.monotonic() - t0
            _assert_tool_success(result, "archiveorg_search")
            _assert_has_content(result, "archiveorg_search")
            _record(live_results, "archiveorg_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "archiveorg_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_wikidata_query(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["wikidata_query"]("insulin")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "wikidata_query")
            _assert_has_content(result, "wikidata_query")
            assert "wikidata.org" in result.lower() or "Q" in result, "Expected Wikidata entity IDs"
            _record(live_results, "wikidata_query", "PASS", duration)
        except Exception as e:
            _record(live_results, "wikidata_query", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_wayback_fetch(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["wayback_fetch"]("https://example.com")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "wayback_fetch")
            # example.com is always archived
            assert "wayback" in result.lower() or "archive" in result.lower(), \
                "Expected Wayback Machine reference in result"
            _record(live_results, "wayback_fetch", "PASS", duration)
        except Exception as e:
            _record(live_results, "wayback_fetch", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_4plebs_search(self, live_results):
        tools = _import_web_fetch()
        t0 = time.monotonic()
        try:
            result = await tools["4plebs_search"]("bitcoin", "pol")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "4plebs_search")
            _assert_has_content(result, "4plebs_search")
            _record(live_results, "4plebs_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "4plebs_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_b4k_search(self, live_results):
        tools = _import_web_fetch()
        t0 = time.monotonic()
        try:
            result = await tools["b4k_search"]("ethereum")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "b4k_search")
            _assert_has_content(result, "b4k_search")
            _record(live_results, "b4k_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "b4k_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_warosu_search(self, live_results):
        tools = _import_web_fetch()
        t0 = time.monotonic()
        try:
            result = await tools["warosu_search"]("linux kernel", "g")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "warosu_search")
            _assert_has_content(result, "warosu_search")
            _record(live_results, "warosu_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "warosu_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_fetch_webpage(self, live_results):
        tools = _import_web_fetch()
        t0 = time.monotonic()
        try:
            result = await tools["fetch_webpage"]("https://en.wikipedia.org/wiki/Insulin", "")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "fetch_webpage")
            assert len(result) > 100, "Expected substantial content from Wikipedia page"
            _record(live_results, "fetch_webpage", "PASS", duration)
        except Exception as e:
            _record(live_results, "fetch_webpage", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise


# ============================================================================
# SEARXNG-DEPENDENT TESTS — requires SearXNG running on localhost:8888
# ============================================================================


@pytest.mark.live
@pytest.mark.live_free
class TestSearXNGDependent:
    """Tools that require a running SearXNG instance."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_searxng_search(self, live_results):
        tools = _import_search_tools()
        t0 = time.monotonic()
        try:
            result = await tools["searxng_search"]("python programming")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "searxng_search")
            assert len(result) > 50, "Expected search results from SearXNG"
            _record(live_results, "searxng_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "searxng_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_news_search(self, live_results):
        tools = _import_search_tools()
        t0 = time.monotonic()
        try:
            result = await tools["news_search"]("technology news", "week")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "news_search")
            _record(live_results, "news_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "news_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(45)
    async def test_forum_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["forum_search"]("mechanical keyboard review", "")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "forum_search")
            _assert_has_content(result, "forum_search")
            _record(live_results, "forum_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "forum_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scholar_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["scholar_search"]("insulin resistance")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "scholar_search")
            _assert_has_content(result, "scholar_search")
            _record(live_results, "scholar_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "scholar_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_substack_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["substack_search"]("cryptocurrency analysis")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "substack_search")
            _assert_has_content(result, "substack_search")
            _record(live_results, "substack_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "substack_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_youtube_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["youtube_search"]("python tutorial")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "youtube_search")
            _assert_has_content(result, "youtube_search")
            _record(live_results, "youtube_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "youtube_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(45)
    async def test_telegram_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["telegram_search"]("cryptocurrency trading")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "telegram_search")
            _assert_has_content(result, "telegram_search")
            _record(live_results, "telegram_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "telegram_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(45)
    async def test_darknet_market_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["darknet_market_search"]("drug market review")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "darknet_market_search")
            _assert_has_content(result, "darknet_market_search")
            _record(live_results, "darknet_market_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "darknet_market_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise


# ============================================================================
# CREDENTIALED TESTS — requires commercial API keys
# ============================================================================


@pytest.mark.live
@pytest.mark.live_credentialed
class TestCredentialedAPIs:
    """Tools that require Bright Data, Apify, or Oxylabs credentials."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_twitter_search(self, live_results):
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["twitter_search"]("insulin pricing")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "twitter_search")
            _record(live_results, "twitter_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "twitter_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_reddit_search(self, live_results):
        sm = _import_social_media()
        t0 = time.monotonic()
        try:
            result = await sm["reddit_search"]("insulin without prescription", "", "relevance")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "reddit_search")
            _record(live_results, "reddit_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "reddit_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_instagram_search(self, live_results):
        sm = _import_social_media()
        t0 = time.monotonic()
        try:
            result = await sm["instagram_search"]("diabetes awareness", "posts")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "instagram_search")
            _record(live_results, "instagram_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "instagram_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_tiktok_search(self, live_results):
        sm = _import_social_media()
        t0 = time.monotonic()
        try:
            result = await sm["tiktok_search"]("insulin pump", "posts")
            duration = time.monotonic() - t0
            _assert_tool_success(result, "tiktok_search")
            _record(live_results, "tiktok_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "tiktok_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise


# ============================================================================
# TOOL EXECUTOR ROUTING TESTS — validates phantom tools are now registered
# ============================================================================


@pytest.mark.live
@pytest.mark.live_free
class TestToolExecutorRouting:
    """Verify the tool executor routes all tools correctly (no 'Unknown tool')."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_telegram_search_routed(self, live_results):
        """telegram_search was previously a phantom tool — verify it's now routed."""
        execute_tool = _import_tool_executor()
        t0 = time.monotonic()
        try:
            result = await execute_tool("telegram_search", {"query": "test"})
            duration = time.monotonic() - t0
            assert "Unknown tool" not in result, (
                f"telegram_search is still a phantom tool! Got: {result[:200]}"
            )
            _record(live_results, "executor:telegram_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "executor:telegram_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_darknet_market_search_routed(self, live_results):
        """darknet_market_search was previously a phantom tool — verify it's now routed."""
        execute_tool = _import_tool_executor()
        t0 = time.monotonic()
        try:
            result = await execute_tool("darknet_market_search", {"query": "test"})
            duration = time.monotonic() - t0
            assert "Unknown tool" not in result, (
                f"darknet_market_search is still a phantom tool! Got: {result[:200]}"
            )
            _record(live_results, "executor:darknet_market_search", "PASS", duration)
        except Exception as e:
            _record(live_results, "executor:darknet_market_search", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_unknown_tool_returns_error_prefix(self, live_results):
        """Unknown tools must return [TOOL_ERROR] prefix, not a plain string."""
        execute_tool = _import_tool_executor()
        t0 = time.monotonic()
        try:
            result = await execute_tool("nonexistent_tool_xyz", {"query": "test"})
            duration = time.monotonic() - t0
            assert result.startswith("[TOOL_ERROR]"), (
                f"Unknown tool should return [TOOL_ERROR] prefix, got: {result[:200]}"
            )
            _record(live_results, "executor:unknown_tool_prefix", "PASS", duration)
        except Exception as e:
            _record(live_results, "executor:unknown_tool_prefix", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise


# ============================================================================
# [TOOL_ERROR] PREFIX VALIDATION — verify error returns use the convention
# ============================================================================


@pytest.mark.live
@pytest.mark.live_free
class TestToolErrorPrefix:
    """Verify that tool failures use the [TOOL_ERROR] prefix convention.

    These tests intentionally provoke failures (bad URLs, nonexistent entities)
    to verify the error prefix is present.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_wayback_bad_url_returns_gracefully(self, live_results):
        """Wayback with a non-archived URL should return gracefully (not crash)."""
        tools = _import_search_tools2()
        t0 = time.monotonic()
        try:
            result = await tools["wayback_fetch"]("https://this-domain-definitely-does-not-exist-xyz123.com/page")
            duration = time.monotonic() - t0
            # Should either return "No archived version" or [TOOL_ERROR] — not crash
            assert isinstance(result, str) and len(result) > 0
            _record(live_results, "error_prefix:wayback_bad_url", "PASS", duration)
        except Exception as e:
            _record(live_results, "error_prefix:wayback_bad_url", "ERROR", time.monotonic() - t0, str(e)[:100])
            raise
