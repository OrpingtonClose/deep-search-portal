"""Tests for social_media_scrapers module.

Covers:
- CostTracker: recording, budget enforcement, session stats
- Censorship filter: sensitive term detection, empty result warnings
- Bright Data trigger/poll pattern (mocked HTTP)
- Apify actor run/poll pattern (mocked HTTP)
- Platform-specific formatters
- Unified social_media_search with fallback
- Individual tool functions
- execute_tool routing
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure proxies/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxies"))

import social_media_scrapers
from social_media_scrapers import (
    CostTracker,
    _censorship_warning,
    _has_sensitive_terms,
    _format_twitter_results,
    _format_reddit_results,
    _format_instagram_results,
    _format_tiktok_results,
    _format_linkedin_results,
    _format_youtube_results,
    _bd_trigger_collection,
    _bd_poll_snapshot,
    _bd_search,
    _apify_run_actor,
    social_media_search,
    tool_reddit_search,
    tool_instagram_search,
    tool_tiktok_search,
    tool_linkedin_search,
    tool_youtube_search,
    tool_social_media_search,
    get_cost_tracker,
    BD_COST_PER_RECORD,
    APIFY_COST_PER_RUN,
    MAX_RESULTS_PER_QUERY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_twitter_records(n: int = 3) -> list[dict]:
    return [
        {
            "user_posted": f"user{i}",
            "description": f"Tweet about topic {i}",
            "date_posted": f"2024-01-0{i}",
            "likes": i * 10,
            "retweets": i * 5,
            "url": f"https://x.com/user{i}/status/{i}",
        }
        for i in range(1, n + 1)
    ]


def _make_reddit_records(n: int = 3) -> list[dict]:
    return [
        {
            "title": f"Reddit post {i}",
            "author": f"redditor{i}",
            "subreddit": "testsubreddit",
            "score": i * 100,
            "num_comments": i * 20,
            "selftext": f"Post body content {i}",
            "url": f"https://reddit.com/r/test/{i}",
            "created_utc": f"2024-01-0{i}",
        }
        for i in range(1, n + 1)
    ]


def _make_instagram_records(n: int = 2) -> list[dict]:
    return [
        {
            "owner_username": f"iguser{i}",
            "caption": f"IG post caption {i}",
            "like_count": i * 50,
            "comment_count": i * 10,
            "taken_at": f"2024-01-0{i}",
            "post_url": f"https://instagram.com/p/{i}",
            "hashtags": ["test", "research"],
        }
        for i in range(1, n + 1)
    ]


def _make_tiktok_records(n: int = 2) -> list[dict]:
    return [
        {
            "author": f"tiktoker{i}",
            "description": f"TikTok video {i}",
            "digg_count": i * 1000,
            "play_count": i * 50000,
            "create_time": f"2024-01-0{i}",
            "video_url": f"https://tiktok.com/@user/{i}",
        }
        for i in range(1, n + 1)
    ]


def _make_linkedin_records(n: int = 2) -> list[dict]:
    return [
        {
            "author": f"Professional {i}",
            "title": f"LinkedIn post {i}",
            "post_text": f"Professional content {i}",
            "num_likes": i * 30,
            "date_posted": f"2024-01-0{i}",
            "post_url": f"https://linkedin.com/posts/{i}",
        }
        for i in range(1, n + 1)
    ]


def _make_youtube_records(n: int = 2) -> list[dict]:
    return [
        {
            "title": f"YouTube video {i}",
            "channel_name": f"Channel{i}",
            "view_count": i * 100000,
            "like_count": i * 5000,
            "published_at": f"2024-01-0{i}",
            "video_url": f"https://youtube.com/watch?v={i}",
            "description": f"Video description {i}",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------


class TestCostTracker:
    @pytest.fixture(autouse=True)
    def _setup_tmpdir(self, tmp_path):
        """Use a temp directory for cost logs."""
        self._orig_dir = social_media_scrapers.COST_LOG_DIR
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        self.tracker = CostTracker()
        yield
        social_media_scrapers.COST_LOG_DIR = self._orig_dir

    @pytest.mark.asyncio
    async def test_record_and_stats(self):
        await self.tracker.record("bright_data", "twitter", 0.015, 10, "test query")
        stats = self.tracker.get_session_stats()
        assert stats["session_total_usd"] == 0.015
        assert stats["call_count"] == 1
        assert len(stats["entries"]) == 1
        assert stats["entries"][0]["provider"] == "bright_data"
        assert stats["entries"][0]["platform"] == "twitter"

    @pytest.mark.asyncio
    async def test_multiple_records_accumulate(self):
        await self.tracker.record("bright_data", "twitter", 0.01, 5, "q1")
        await self.tracker.record("apify", "reddit", 0.25, 10, "q2")
        stats = self.tracker.get_session_stats()
        assert stats["session_total_usd"] == pytest.approx(0.26)
        assert stats["call_count"] == 2

    @pytest.mark.asyncio
    async def test_session_budget_enforcement(self):
        orig_budget = social_media_scrapers.SESSION_BUDGET
        social_media_scrapers.SESSION_BUDGET = 0.10
        try:
            await self.tracker.record("bright_data", "twitter", 0.12, 80, "expensive query")
            warning = await self.tracker.check_budget("bright_data", "twitter")
            assert warning is not None
            assert "BUDGET EXCEEDED" in warning
            assert "Session budget" in warning
        finally:
            social_media_scrapers.SESSION_BUDGET = orig_budget

    @pytest.mark.asyncio
    async def test_under_budget_returns_none(self):
        orig_budget = social_media_scrapers.SESSION_BUDGET
        social_media_scrapers.SESSION_BUDGET = 100.0
        try:
            await self.tracker.record("bright_data", "twitter", 0.01, 5, "cheap query")
            warning = await self.tracker.check_budget("bright_data", "twitter")
            assert warning is None
        finally:
            social_media_scrapers.SESSION_BUDGET = orig_budget

    @pytest.mark.asyncio
    async def test_cost_log_written_to_disk(self, tmp_path):
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        tracker = CostTracker()
        await tracker.record("apify", "reddit", 0.25, 10, "test")
        log_files = list(tmp_path.glob("social_scraper_costs_*.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())
        assert entry["provider"] == "apify"
        assert entry["estimated_cost_usd"] == 0.25

    @pytest.mark.asyncio
    async def test_budget_remaining_calculation(self):
        orig_budget = social_media_scrapers.SESSION_BUDGET
        social_media_scrapers.SESSION_BUDGET = 5.0
        try:
            await self.tracker.record("bright_data", "twitter", 1.5, 1000, "q")
            stats = self.tracker.get_session_stats()
            assert stats["session_budget_remaining_usd"] == pytest.approx(3.5)
        finally:
            social_media_scrapers.SESSION_BUDGET = orig_budget

    @pytest.mark.asyncio
    async def test_monthly_budget_enforcement(self, tmp_path):
        """Monthly budget reads from JSONL logs on disk."""
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        orig_monthly = social_media_scrapers.MONTHLY_BUDGET
        social_media_scrapers.MONTHLY_BUDGET = 0.50
        try:
            tracker = CostTracker()
            # Write a fake cost log for this month
            from datetime import datetime, timezone
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_path = tmp_path / f"social_scraper_costs_{day}.jsonl"
            with open(log_path, "w") as f:
                f.write(json.dumps({"estimated_cost_usd": 0.60}) + "\n")

            warning = await tracker.check_budget("bright_data", "twitter")
            assert warning is not None
            assert "Monthly budget" in warning
        finally:
            social_media_scrapers.MONTHLY_BUDGET = orig_monthly


# ---------------------------------------------------------------------------
# Censorship filter tests
# ---------------------------------------------------------------------------


class TestCensorshipFilter:
    def test_sensitive_terms_detected(self):
        assert _has_sensitive_terms("nsfw content search") is True
        assert _has_sensitive_terms("terrorism news") is True
        assert _has_sensitive_terms("pump and dump crypto") is True

    def test_non_sensitive_terms(self):
        assert _has_sensitive_terms("python programming tutorial") is False
        assert _has_sensitive_terms("weather forecast new york") is False

    def test_censorship_warning_zero_results(self):
        warning = _censorship_warning("twitter", "test query", 0, "bright_data")
        assert "CENSORSHIP WARNING" in warning
        assert "Zero results" in warning

    def test_censorship_warning_low_results(self):
        warning = _censorship_warning("twitter", "test query", 1, "bright_data")
        assert "CENSORSHIP WARNING" in warning
        assert "Only 1 result" in warning

    def test_censorship_warning_sufficient_results(self):
        warning = _censorship_warning("twitter", "test query", 10, "bright_data")
        assert warning == ""

    def test_censorship_warning_sensitive_query(self):
        warning = _censorship_warning("reddit", "nsfw content", 5, "apify")
        assert "CENSORSHIP WARNING" in warning
        assert "may filter" in warning

    def test_censorship_warning_sensitive_and_zero(self):
        warning = _censorship_warning("instagram", "nsfw art", 0, "bright_data")
        assert "CENSORSHIP WARNING" in warning
        assert "may filter" in warning
        assert "Zero results" in warning


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_twitter_formatter(self):
        records = _make_twitter_records(2)
        result = _format_twitter_results(records, "bright_data")
        assert "@user1" in result
        assert "Tweet about topic 1" in result
        assert "likes: 10" in result

    def test_reddit_formatter(self):
        records = _make_reddit_records(2)
        result = _format_reddit_results(records, "apify")
        assert "r/testsubreddit" in result
        assert "Reddit post 1" in result
        assert "u/redditor1" in result

    def test_instagram_formatter(self):
        records = _make_instagram_records(2)
        result = _format_instagram_results(records, "bright_data")
        assert "@iguser1" in result
        assert "#test" in result

    def test_tiktok_formatter(self):
        records = _make_tiktok_records(2)
        result = _format_tiktok_results(records, "apify")
        assert "@tiktoker1" in result
        assert "views:" in result

    def test_linkedin_formatter(self):
        records = _make_linkedin_records(2)
        result = _format_linkedin_results(records, "bright_data")
        assert "Professional 1" in result

    def test_youtube_formatter(self):
        records = _make_youtube_records(2)
        result = _format_youtube_results(records, "bright_data")
        assert "YouTube video 1" in result
        assert "Channel1" in result

    def test_empty_records(self):
        assert _format_twitter_results([], "test") == ""
        assert _format_reddit_results([], "test") == ""
        assert _format_instagram_results([], "test") == ""


# ---------------------------------------------------------------------------
# Bright Data API tests (mocked)
# ---------------------------------------------------------------------------


class TestBrightDataAPI:
    @pytest.mark.asyncio
    async def test_trigger_collection_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"snapshot_id": "snap_123"}

        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers.httpx.AsyncClient") as MockClient:
                client_instance = AsyncMock()
                client_instance.post.return_value = mock_resp
                client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = client_instance

                result = await _bd_trigger_collection("dataset_123", [{"url": "https://test.com"}])
                assert result == "snap_123"

    @pytest.mark.asyncio
    async def test_trigger_collection_no_api_key(self):
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", ""):
            result = await _bd_trigger_collection("dataset_123", [{"url": "https://test.com"}])
            assert result is None

    @pytest.mark.asyncio
    async def test_trigger_collection_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers.httpx.AsyncClient") as MockClient:
                client_instance = AsyncMock()
                client_instance.post.return_value = mock_resp
                client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = client_instance

                result = await _bd_trigger_collection("dataset_123", [{"url": "https://test.com"}])
                assert result is None

    @pytest.mark.asyncio
    async def test_poll_snapshot_returns_list(self):
        records = _make_twitter_records(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = records

        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers.httpx.AsyncClient") as MockClient:
                client_instance = AsyncMock()
                client_instance.get.return_value = mock_resp
                client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = client_instance

                result = await _bd_poll_snapshot("snap_123")
                assert result == records

    @pytest.mark.asyncio
    async def test_poll_snapshot_no_key(self):
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", ""):
            result = await _bd_poll_snapshot("snap_123")
            assert result is None

    @pytest.mark.asyncio
    async def test_bd_search_full_flow(self, tmp_path):
        """Full Bright Data search: trigger → poll → results with cost tracking."""
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        records = _make_reddit_records(5)

        # Reset global tracker for this test
        tracker = CostTracker()
        orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = tracker

        try:
            with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
                with patch("social_media_scrapers._bd_trigger_collection", return_value="snap_456"):
                    with patch("social_media_scrapers._bd_poll_snapshot", return_value=records):
                        result = await _bd_search(
                            "reddit", "dataset_reddit", [{"url": "https://reddit.com/search?q=test"}], "test query"
                        )
                        assert result == records
                        stats = tracker.get_session_stats()
                        assert stats["call_count"] == 1
                        assert stats["session_total_usd"] == pytest.approx(5 * BD_COST_PER_RECORD)
        finally:
            social_media_scrapers._cost_tracker = orig_tracker

    @pytest.mark.asyncio
    async def test_bd_search_budget_exceeded(self, tmp_path):
        """Budget exceeded should return None without making API calls."""
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        orig_budget = social_media_scrapers.SESSION_BUDGET
        social_media_scrapers.SESSION_BUDGET = 0.001

        tracker = CostTracker()
        await tracker.record("bright_data", "twitter", 0.01, 5, "prev")
        orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = tracker

        try:
            with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
                with patch("social_media_scrapers._bd_trigger_collection") as mock_trigger:
                    result = await _bd_search("twitter", "dataset_tw", [{}], "test")
                    assert result is None
                    mock_trigger.assert_not_called()
        finally:
            social_media_scrapers._cost_tracker = orig_tracker
            social_media_scrapers.SESSION_BUDGET = orig_budget


# ---------------------------------------------------------------------------
# Apify API tests (mocked)
# ---------------------------------------------------------------------------


class TestApifyAPI:
    @pytest.mark.asyncio
    async def test_apify_run_actor_no_token(self):
        with patch.object(social_media_scrapers, "APIFY_API_TOKEN", ""):
            result = await _apify_run_actor("test/actor", {}, "twitter", "query")
            assert result is None

    @pytest.mark.asyncio
    async def test_apify_run_actor_success(self, tmp_path):
        """Apify full flow: start run → poll → get dataset items."""
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        records = _make_twitter_records(3)

        # Reset tracker
        tracker = CostTracker()
        orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = tracker

        # Mock responses for: start run, poll status, get items
        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {"data": {"id": "run_123"}}

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {
            "data": {"status": "SUCCEEDED", "defaultDatasetId": "ds_456"}
        }

        items_resp = MagicMock()
        items_resp.status_code = 200
        items_resp.json.return_value = records

        try:
            with patch.object(social_media_scrapers, "APIFY_API_TOKEN", "test-token"):
                with patch("social_media_scrapers.httpx.AsyncClient") as MockClient:
                    client_instance = AsyncMock()
                    client_instance.post.return_value = start_resp
                    client_instance.get.side_effect = [status_resp, items_resp]
                    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                    client_instance.__aexit__ = AsyncMock(return_value=False)
                    MockClient.return_value = client_instance

                    result = await _apify_run_actor(
                        "test/actor", {"query": "test"}, "twitter", "test query"
                    )
                    assert result == records
                    stats = tracker.get_session_stats()
                    assert stats["call_count"] == 1
                    assert stats["session_total_usd"] == pytest.approx(APIFY_COST_PER_RUN)
        finally:
            social_media_scrapers._cost_tracker = orig_tracker

    @pytest.mark.asyncio
    async def test_apify_run_actor_failed_status(self, tmp_path):
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)

        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {"data": {"id": "run_fail"}}

        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"data": {"status": "FAILED"}}

        tracker = CostTracker()
        orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = tracker

        try:
            with patch.object(social_media_scrapers, "APIFY_API_TOKEN", "test-token"):
                with patch("social_media_scrapers.httpx.AsyncClient") as MockClient:
                    client_instance = AsyncMock()
                    client_instance.post.return_value = start_resp
                    client_instance.get.return_value = status_resp
                    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                    client_instance.__aexit__ = AsyncMock(return_value=False)
                    MockClient.return_value = client_instance

                    result = await _apify_run_actor(
                        "test/actor", {"query": "test"}, "reddit", "test query"
                    )
                    assert result is None
        finally:
            social_media_scrapers._cost_tracker = orig_tracker


# ---------------------------------------------------------------------------
# Unified social_media_search tests
# ---------------------------------------------------------------------------


class TestSocialMediaSearch:
    @pytest.fixture(autouse=True)
    def _reset_tracker(self, tmp_path):
        self._orig_dir = social_media_scrapers.COST_LOG_DIR
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        self._orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = CostTracker()
        yield
        social_media_scrapers._cost_tracker = self._orig_tracker
        social_media_scrapers.COST_LOG_DIR = self._orig_dir

    @pytest.mark.asyncio
    async def test_unsupported_platform(self):
        result = await social_media_search("myspace", "query")
        assert "Unsupported platform" in result

    @pytest.mark.asyncio
    async def test_no_credentials(self):
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", ""):
            with patch.object(social_media_scrapers, "APIFY_API_TOKEN", ""):
                result = await social_media_search("twitter", "test")
                assert "No results" in result
                assert "No social media scraper credentials" in result

    @pytest.mark.asyncio
    async def test_bright_data_primary_success(self):
        records = _make_twitter_records(5)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await social_media_search("twitter", "AI news")
                assert "Twitter search results" in result
                assert "bright_data" in result
                assert "@user1" in result
                assert "Cost:" in result

    @pytest.mark.asyncio
    async def test_apify_fallback_on_bd_failure(self):
        records = _make_reddit_records(3)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch.object(social_media_scrapers, "APIFY_API_TOKEN", "test-token"):
                with patch("social_media_scrapers._bd_search", return_value=None):
                    with patch("social_media_scrapers._apify_run_actor", return_value=records):
                        result = await social_media_search("reddit", "crypto news")
                        assert "apify" in result
                        assert "Reddit search results" in result

    @pytest.mark.asyncio
    async def test_budget_exceeded_blocks_search(self):
        orig_budget = social_media_scrapers.SESSION_BUDGET
        social_media_scrapers.SESSION_BUDGET = 0.001
        try:
            await social_media_scrapers._cost_tracker.record("bd", "tw", 0.01, 5, "prev")
            result = await social_media_search("twitter", "test")
            assert "BUDGET EXCEEDED" in result
        finally:
            social_media_scrapers.SESSION_BUDGET = orig_budget

    @pytest.mark.asyncio
    async def test_censorship_warning_on_empty_results(self):
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers._bd_search", return_value=[]):
                with patch.object(social_media_scrapers, "APIFY_API_TOKEN", ""):
                    result = await social_media_search("instagram", "nsfw art")
                    assert "CENSORSHIP WARNING" in result

    @pytest.mark.asyncio
    async def test_reddit_with_subreddit(self):
        records = _make_reddit_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "test-key"):
            with patch("social_media_scrapers._bd_search", return_value=records) as mock_bd:
                result = await social_media_search("reddit", "GME", subreddit="wallstreetbets")
                assert "Reddit search results" in result
                # Verify the BD search was called (which means input_builder was invoked)
                mock_bd.assert_called_once()


# ---------------------------------------------------------------------------
# Individual tool function tests
# ---------------------------------------------------------------------------


class TestToolFunctions:
    @pytest.fixture(autouse=True)
    def _reset_tracker(self, tmp_path):
        self._orig_dir = social_media_scrapers.COST_LOG_DIR
        social_media_scrapers.COST_LOG_DIR = str(tmp_path)
        self._orig_tracker = social_media_scrapers._cost_tracker
        social_media_scrapers._cost_tracker = CostTracker()
        yield
        social_media_scrapers._cost_tracker = self._orig_tracker
        social_media_scrapers.COST_LOG_DIR = self._orig_dir

    @pytest.mark.asyncio
    async def test_tool_reddit_search(self):
        records = _make_reddit_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_reddit_search("crypto", subreddit="bitcoin")
                assert "Reddit search results" in result

    @pytest.mark.asyncio
    async def test_tool_instagram_search(self):
        records = _make_instagram_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_instagram_search("travel")
                assert "Instagram search results" in result

    @pytest.mark.asyncio
    async def test_tool_tiktok_search(self):
        records = _make_tiktok_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_tiktok_search("cooking")
                assert "Tiktok search results" in result

    @pytest.mark.asyncio
    async def test_tool_linkedin_search(self):
        records = _make_linkedin_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_linkedin_search("AI hiring")
                assert "Linkedin search results" in result

    @pytest.mark.asyncio
    async def test_tool_youtube_search(self):
        records = _make_youtube_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_youtube_search("machine learning")
                assert "Youtube search results" in result

    @pytest.mark.asyncio
    async def test_tool_social_media_search_unified(self):
        records = _make_twitter_records(2)
        with patch.object(social_media_scrapers, "BRIGHT_DATA_API_KEY", "k"):
            with patch("social_media_scrapers._bd_search", return_value=records):
                result = await tool_social_media_search("twitter", "breaking news")
                assert "Twitter search results" in result


# ---------------------------------------------------------------------------
# execute_tool integration tests
# ---------------------------------------------------------------------------


class TestExecuteToolRouting:
    """Verify that the new tools are routed correctly in execute_tool.

    The persistent proxy requires UPSTREAM_KEY env var at import time,
    so we set a dummy value before importing.
    """

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("UPSTREAM_KEY", "test-dummy-key")

    @pytest.mark.asyncio
    async def test_social_media_search_routing(self):
        with patch("social_media_scrapers.tool_social_media_search", return_value="mocked") as mock:
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("social_media_search", {
                "platform": "twitter", "query": "test"
            })
            assert result == "mocked"

    @pytest.mark.asyncio
    async def test_reddit_search_routing(self):
        with patch("social_media_scrapers.tool_reddit_search", return_value="reddit result") as mock:
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("reddit_search", {"query": "test"})
            assert result == "reddit result"

    @pytest.mark.asyncio
    async def test_instagram_search_routing(self):
        with patch("social_media_scrapers.tool_instagram_search", return_value="ig result"):
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("instagram_search", {"query": "test"})
            assert result == "ig result"

    @pytest.mark.asyncio
    async def test_tiktok_search_routing(self):
        with patch("social_media_scrapers.tool_tiktok_search", return_value="tt result"):
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("tiktok_search", {"query": "test"})
            assert result == "tt result"

    @pytest.mark.asyncio
    async def test_linkedin_search_routing(self):
        with patch("social_media_scrapers.tool_linkedin_search", return_value="li result"):
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("linkedin_search", {"query": "test"})
            assert result == "li result"

    @pytest.mark.asyncio
    async def test_youtube_search_routing(self):
        with patch("social_media_scrapers.tool_youtube_search", return_value="yt result"):
            from persistent_deep_research_proxy import execute_tool
            result = await execute_tool("youtube_search", {"query": "test"})
            assert result == "yt result"


# ---------------------------------------------------------------------------
# Input builder tests
# ---------------------------------------------------------------------------


class TestInputBuilders:
    def test_bd_twitter_input(self):
        from social_media_scrapers import _bd_twitter_input
        result = _bd_twitter_input("AI news")
        assert len(result) == 1
        assert "x.com/search" in result[0]["url"]
        assert "AI%20news" in result[0]["url"]

    def test_bd_reddit_input_global(self):
        from social_media_scrapers import _bd_reddit_input
        result = _bd_reddit_input("crypto")
        assert "reddit.com/search" in result[0]["url"]

    def test_bd_reddit_input_subreddit(self):
        from social_media_scrapers import _bd_reddit_input
        result = _bd_reddit_input("GME", subreddit="wallstreetbets")
        assert "r/wallstreetbets" in result[0]["url"]

    def test_bd_url_encoding_special_chars(self):
        from social_media_scrapers import _bd_twitter_input
        result = _bd_twitter_input("test&query=bad#fragment")
        url = result[0]["url"]
        assert "&query=bad" not in url  # & should be encoded
        assert "#fragment" not in url  # # should be encoded
        assert "test%26query%3Dbad%23fragment" in url

    def test_bd_instagram_input(self):
        from social_media_scrapers import _bd_instagram_input
        result = _bd_instagram_input("travel photos")
        assert "instagram.com/explore/tags/travelphotos" in result[0]["url"]

    def test_bd_tiktok_input(self):
        from social_media_scrapers import _bd_tiktok_input
        result = _bd_tiktok_input("cooking tips")
        assert "tiktok.com/search" in result[0]["url"]

    def test_bd_linkedin_input(self):
        from social_media_scrapers import _bd_linkedin_input
        result = _bd_linkedin_input("AI hiring")
        assert "linkedin.com/search" in result[0]["url"]

    def test_bd_youtube_input(self):
        from social_media_scrapers import _bd_youtube_input
        result = _bd_youtube_input("machine learning")
        assert "youtube.com/results" in result[0]["url"]

    def test_apify_twitter_input(self):
        from social_media_scrapers import _apify_twitter_input
        result = _apify_twitter_input("AI news")
        assert result["searchTerms"] == ["AI news"]
        assert result["maxTweets"] == MAX_RESULTS_PER_QUERY

    def test_apify_reddit_input_with_subreddit(self):
        from social_media_scrapers import _apify_reddit_input
        result = _apify_reddit_input("GME", subreddit="wallstreetbets")
        assert "startUrls" in result
        assert "wallstreetbets" in result["startUrls"][0]["url"]
        assert "GME" in result["startUrls"][0]["url"]

    def test_apify_instagram_input(self):
        from social_media_scrapers import _apify_instagram_input
        result = _apify_instagram_input("food art")
        assert result["hashtags"] == ["foodart"]


# ---------------------------------------------------------------------------
# Global cost tracker singleton
# ---------------------------------------------------------------------------


class TestGlobalTracker:
    def test_get_cost_tracker_returns_singleton(self):
        t1 = get_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is t2
