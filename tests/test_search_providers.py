"""Tests for the multi-source search providers module.

Covers all provider backends (DuckDuckGo, Brave, Mojeek, SearXNG),
deduplication, category routing, and the backward-compatible search_as_raw API.

All external calls are mocked — no services need to be running.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock shared module before importing search_providers
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

_mock_shared = MagicMock()
_mock_shared.setup_logging.return_value = MagicMock()
_mock_shared.require_env.return_value = "test-key"
_mock_shared.env_int.side_effect = lambda name, default, **kw: default
_mock_shared.http_client.return_value = MagicMock()
_mock_shared.create_app.return_value = MagicMock()
_mock_shared.register_standard_routes = MagicMock()

# Throttler mock — must return an async context manager
_throttler_instance = MagicMock()
_throttler_instance.__aenter__ = AsyncMock(return_value=None)
_throttler_instance.__aexit__ = AsyncMock(return_value=False)
_mock_shared.get_throttler.return_value = _throttler_instance

if "shared" not in sys.modules or isinstance(sys.modules["shared"], MagicMock):
    sys.modules["shared"] = _mock_shared

import search_providers as sp


# ===================================================================
# SearchResult dataclass
# ===================================================================

class TestSearchResult:
    def test_basic_construction(self):
        r = sp.SearchResult(
            title="Example", url="https://example.com",
            snippet="A test snippet", source="test",
        )
        assert r.title == "Example"
        assert r.url == "https://example.com"
        assert r.snippet == "A test snippet"
        assert r.source == "test"
        assert r.score == 0.0
        assert r.published_date == ""

    def test_to_dict(self):
        r = sp.SearchResult(
            title="T", url="https://t.com", snippet="S",
            source="brave", score=0.9, published_date="2026-01-01",
        )
        d = r.to_dict()
        assert d["title"] == "T"
        assert d["url"] == "https://t.com"
        assert d["snippet"] == "S"
        assert d["source"] == "brave"
        assert d["score"] == 0.9
        assert d["publishedDate"] == "2026-01-01"


# ===================================================================
# URL normalisation and deduplication
# ===================================================================

class TestNormaliseUrl:
    def test_strips_protocol(self):
        assert sp._normalise_url("https://example.com") == "example.com"
        assert sp._normalise_url("http://example.com") == "example.com"

    def test_strips_www(self):
        assert sp._normalise_url("https://www.example.com") == "example.com"

    def test_strips_trailing_slash(self):
        assert sp._normalise_url("https://example.com/") == "example.com"

    def test_combined(self):
        assert sp._normalise_url("http://www.example.com/path/") == "example.com/path"


class TestDeduplicate:
    def test_removes_duplicates(self):
        results = [
            sp.SearchResult(title="A", url="https://example.com", snippet="", source="ddg"),
            sp.SearchResult(title="B", url="http://example.com", snippet="", source="brave"),
            sp.SearchResult(title="C", url="https://other.com", snippet="", source="mojeek"),
        ]
        deduped = sp._deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0].title == "A"  # first occurrence wins
        assert deduped[1].title == "C"

    def test_removes_www_duplicates(self):
        results = [
            sp.SearchResult(title="A", url="https://www.example.com", snippet="", source="ddg"),
            sp.SearchResult(title="B", url="https://example.com", snippet="", source="brave"),
        ]
        deduped = sp._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].title == "A"

    def test_empty_url_skipped(self):
        results = [
            sp.SearchResult(title="A", url="", snippet="", source="ddg"),
            sp.SearchResult(title="B", url="https://example.com", snippet="", source="brave"),
        ]
        deduped = sp._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].title == "B"

    def test_empty_input(self):
        assert sp._deduplicate([]) == []


# ===================================================================
# DuckDuckGo provider
# ===================================================================

class TestDuckDuckGoSearch:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_ddg = MagicMock()
        mock_ddg.results.return_value = [
            {"title": "Duck Result 1", "link": "https://duck1.com", "snippet": "First result"},
            {"title": "Duck Result 2", "link": "https://duck2.com", "snippet": "Second result"},
        ]

        with patch("search_providers.DuckDuckGoSearchAPIWrapper", create=True) as mock_cls:
            # Patch the import inside the function
            with patch.dict("sys.modules", {
                "langchain_community.utilities": MagicMock(DuckDuckGoSearchAPIWrapper=lambda **kw: mock_ddg),
                "langchain_community": MagicMock(),
            }):
                # Re-import to pick up the mock
                import importlib
                importlib.reload(sp)
                results = await sp._search_duckduckgo("test query", max_results=5)

        assert len(results) == 2
        assert results[0].title == "Duck Result 1"
        assert results[0].url == "https://duck1.com"
        assert results[0].source == "duckduckgo"

    @pytest.mark.asyncio
    async def test_import_error_returns_empty(self):
        with patch.dict("sys.modules", {"langchain_community.utilities": None}):
            import importlib
            importlib.reload(sp)
            results = await sp._search_duckduckgo("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        mock_ddg = MagicMock()
        mock_ddg.results.side_effect = RuntimeError("connection failed")

        with patch.dict("sys.modules", {
            "langchain_community.utilities": MagicMock(DuckDuckGoSearchAPIWrapper=lambda **kw: mock_ddg),
            "langchain_community": MagicMock(),
        }):
            import importlib
            importlib.reload(sp)
            results = await sp._search_duckduckgo("test query")
        assert results == []


# ===================================================================
# DuckDuckGo News provider
# ===================================================================

class TestDuckDuckGoNews:
    @pytest.mark.asyncio
    async def test_successful_news_search(self):
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.news.return_value = [
            {"title": "News 1", "url": "https://news1.com", "body": "Breaking news", "date": "2026-03-20"},
        ]

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)

        with patch.dict("sys.modules", {
            "duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls),
        }):
            import importlib
            importlib.reload(sp)
            results = await sp._search_duckduckgo_news("breaking news")

        assert len(results) == 1
        assert results[0].title == "News 1"
        assert results[0].source == "duckduckgo_news"
        assert results[0].published_date == "2026-03-20"

    @pytest.mark.asyncio
    async def test_import_error_returns_empty(self):
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            import importlib
            importlib.reload(sp)
            results = await sp._search_duckduckgo_news("test")
        assert results == []


# ===================================================================
# Brave Search provider
# ===================================================================

class TestBraveSearch:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        with patch.object(sp, "BRAVE_SEARCH_API_KEY", ""):
            results = await sp._search_brave("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_brave = MagicMock()
        mock_brave.run.return_value = json.dumps([
            {"title": "Brave Result", "link": "https://brave1.com", "snippet": "Found via Brave"},
        ])

        with patch.object(sp, "BRAVE_SEARCH_API_KEY", "test-key"):
            with patch.dict("sys.modules", {
                "langchain_community.utilities": MagicMock(BraveSearchWrapper=lambda **kw: mock_brave),
                "langchain_community": MagicMock(),
            }):
                import importlib
                importlib.reload(sp)
                # Re-set the API key after reload
                sp.BRAVE_SEARCH_API_KEY = "test-key"
                results = await sp._search_brave("test query")

        assert len(results) == 1
        assert results[0].title == "Brave Result"
        assert results[0].source == "brave"

    @pytest.mark.asyncio
    async def test_plain_text_response(self):
        mock_brave = MagicMock()
        mock_brave.run.return_value = "Plain text results here"

        with patch.object(sp, "BRAVE_SEARCH_API_KEY", "test-key"):
            with patch.dict("sys.modules", {
                "langchain_community.utilities": MagicMock(BraveSearchWrapper=lambda **kw: mock_brave),
                "langchain_community": MagicMock(),
            }):
                import importlib
                importlib.reload(sp)
                sp.BRAVE_SEARCH_API_KEY = "test-key"
                results = await sp._search_brave("test query")

        assert len(results) == 1
        assert results[0].snippet == "Plain text results here"


# ===================================================================
# Mojeek Search provider
# ===================================================================

class TestMojeekSearch:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        with patch.object(sp, "MOJEEK_API_KEY", ""):
            results = await sp._search_mojeek("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_mojeek = MagicMock()
        mock_mojeek.results.return_value = [
            {"title": "Mojeek Result", "url": "https://mojeek1.com", "desc": "Independent crawler"},
        ]

        with patch.object(sp, "MOJEEK_API_KEY", "test-key"):
            with patch.dict("sys.modules", {
                "langchain_community.utilities": MagicMock(MojeekSearchAPIWrapper=lambda **kw: mock_mojeek),
                "langchain_community": MagicMock(),
            }):
                import importlib
                importlib.reload(sp)
                sp.MOJEEK_API_KEY = "test-key"
                results = await sp._search_mojeek("test query")

        assert len(results) == 1
        assert results[0].title == "Mojeek Result"
        assert results[0].source == "mojeek"


# ===================================================================
# SearXNG provider
# ===================================================================

class TestSearXNGSearch:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        with patch.object(sp, "SEARXNG_ENABLED", False):
            results = await sp._search_searxng("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "SearXNG Result", "url": "https://sx1.com", "content": "Meta-search result"},
            ]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("search_providers.http_client", return_value=mock_client):
            results = await sp._search_searxng("test query", categories="general")

        assert len(results) == 1
        assert results[0].title == "SearXNG Result"
        assert results[0].source == "searxng:general"

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("search_providers.http_client", return_value=mock_client):
            results = await sp._search_searxng("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        mock_client = AsyncMock()
        import httpx
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with patch("search_providers.http_client", return_value=mock_client):
            results = await sp._search_searxng("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_time_range_passed(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("search_providers.http_client", return_value=mock_client):
            await sp._search_searxng("test", categories="news", time_range="week")

        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["params"]["time_range"] == "week"
        assert call_kwargs.kwargs["params"]["categories"] == "news"


# ===================================================================
# Multi-source search (integration of all providers)
# ===================================================================

class TestMultiSearch:
    @pytest.mark.asyncio
    async def test_merges_and_deduplicates(self):
        ddg_results = [
            sp.SearchResult(title="DDG 1", url="https://example.com", snippet="", source="duckduckgo"),
            sp.SearchResult(title="DDG 2", url="https://unique-ddg.com", snippet="", source="duckduckgo"),
        ]
        brave_results = [
            sp.SearchResult(title="Brave 1", url="https://example.com", snippet="", source="brave"),  # dup
            sp.SearchResult(title="Brave 2", url="https://unique-brave.com", snippet="", source="brave"),
        ]
        searxng_results = [
            sp.SearchResult(title="SX 1", url="https://unique-sx.com", snippet="", source="searxng:general"),
        ]

        with patch("search_providers._search_duckduckgo", AsyncMock(return_value=ddg_results)), \
             patch("search_providers._search_brave", AsyncMock(return_value=brave_results)), \
             patch("search_providers._search_mojeek", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=searxng_results)):
            results = await sp.multi_search("test query", max_results=10)

        # example.com should appear only once (DDG wins as first)
        urls = [r.url for r in results]
        assert urls.count("https://example.com") == 1
        assert len(results) == 4  # DDG1, DDG2, Brave2, SX1

    @pytest.mark.asyncio
    async def test_handles_provider_exceptions(self):
        with patch("search_providers._search_duckduckgo", AsyncMock(side_effect=RuntimeError("fail"))), \
             patch("search_providers._search_brave", AsyncMock(return_value=[])), \
             patch("search_providers._search_mojeek", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[
                 sp.SearchResult(title="Fallback", url="https://fb.com", snippet="", source="searxng:general"),
             ])):
            results = await sp.multi_search("test query")

        assert len(results) == 1
        assert results[0].title == "Fallback"

    @pytest.mark.asyncio
    async def test_max_results_respected(self):
        many_results = [
            sp.SearchResult(title=f"R{i}", url=f"https://r{i}.com", snippet="", source="ddg")
            for i in range(20)
        ]
        with patch("search_providers._search_duckduckgo", AsyncMock(return_value=many_results)), \
             patch("search_providers._search_brave", AsyncMock(return_value=[])), \
             patch("search_providers._search_mojeek", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[])):
            results = await sp.multi_search("test", max_results=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_time_range_passed_to_searxng(self):
        """time_range must propagate to the SearXNG sub-call inside multi_search."""
        with patch("search_providers._search_duckduckgo", AsyncMock(return_value=[])), \
             patch("search_providers._search_brave", AsyncMock(return_value=[])), \
             patch("search_providers._search_mojeek", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[])) as mock_sx:
            await sp.multi_search("test", time_range="month")

        mock_sx.assert_called_once_with(
            "test", categories="general", time_range="month", max_results=10,
        )

    @pytest.mark.asyncio
    async def test_include_news_flag(self):
        ddg_news_called = False

        async def mock_ddg_news(query, max_results=10):
            nonlocal ddg_news_called
            ddg_news_called = True
            return [sp.SearchResult(title="News", url="https://news.com", snippet="", source="ddg_news")]

        with patch("search_providers._search_duckduckgo", AsyncMock(return_value=[])), \
             patch("search_providers._search_brave", AsyncMock(return_value=[])), \
             patch("search_providers._search_mojeek", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[])), \
             patch("search_providers._search_duckduckgo_news", mock_ddg_news):
            results = await sp.multi_search("test", include_news=True)

        assert ddg_news_called
        assert len(results) == 1


# ===================================================================
# News search
# ===================================================================

class TestMultiSearchNews:
    @pytest.mark.asyncio
    async def test_merges_news_sources(self):
        ddg_news = [
            sp.SearchResult(title="DDG News", url="https://ddg-news.com", snippet="", source="ddg_news"),
        ]
        sx_news = [
            sp.SearchResult(title="SX News", url="https://sx-news.com", snippet="", source="searxng:news"),
        ]

        with patch("search_providers._search_duckduckgo_news", AsyncMock(return_value=ddg_news)), \
             patch("search_providers._search_searxng", AsyncMock(return_value=sx_news)):
            results = await sp.multi_search_news("breaking news")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_supplements_when_few_results(self):
        supplement_called = False

        async def mock_searxng(query, categories="general", time_range="", max_results=10):
            nonlocal supplement_called
            if categories == "general":
                supplement_called = True
            return []

        with patch("search_providers._search_duckduckgo_news", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", mock_searxng):
            await sp.multi_search_news("test")

        assert supplement_called


# ===================================================================
# Science search
# ===================================================================

class TestMultiSearchScience:
    @pytest.mark.asyncio
    async def test_uses_science_category(self):
        sci_results = [
            sp.SearchResult(title="Paper 1", url="https://arxiv.org/abs/123", snippet="", source="searxng:science"),
        ]

        with patch("search_providers._search_searxng", AsyncMock(return_value=sci_results)):
            results = await sp.multi_search_science("quantum computing")

        assert len(results) == 1
        assert "arxiv" in results[0].url

    @pytest.mark.asyncio
    async def test_falls_back_to_ddg_academic(self):
        ddg_results = [
            sp.SearchResult(title="Academic DDG", url="https://scholar.google.com/x", snippet="", source="ddg"),
        ]

        with patch("search_providers._search_searxng", AsyncMock(return_value=[])), \
             patch("search_providers._search_duckduckgo", AsyncMock(return_value=ddg_results)):
            results = await sp.multi_search_science("quantum computing")

        assert len(results) == 1
        assert results[0].title == "Academic DDG"


# ===================================================================
# Site-targeted search
# ===================================================================

class TestMultiSearchSite:
    @pytest.mark.asyncio
    async def test_prepends_site_prefix(self):
        captured_queries = []

        async def mock_ddg(query, max_results=10):
            captured_queries.append(query)
            return []

        with patch("search_providers._search_duckduckgo", mock_ddg), \
             patch("search_providers._search_brave", AsyncMock(return_value=[])), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[])):
            await sp.multi_search_site("test", site="substack.com")

        assert any("site:substack.com" in q for q in captured_queries)


# ===================================================================
# Forum search
# ===================================================================

class TestMultiSearchForums:
    @pytest.mark.asyncio
    async def test_searches_forum_sites(self):
        ddg_results = [
            sp.SearchResult(title="Forum Post", url="https://forums.example.com/t/1", snippet="", source="ddg"),
        ]

        with patch("search_providers._search_duckduckgo", AsyncMock(return_value=ddg_results)), \
             patch("search_providers._search_searxng", AsyncMock(return_value=[])):
            results = await sp.multi_search_forums(
                "GPU overclocking",
                forum_sites=["forums.overclock.net", "forums.tomshardware.com"],
            )

        assert len(results) >= 1


# ===================================================================
# Backward-compatible API
# ===================================================================

class TestResultsToRawDicts:
    def test_conversion(self):
        results = [
            sp.SearchResult(
                title="T", url="https://t.com", snippet="S",
                source="brave", published_date="2026-01-01",
            ),
        ]
        raw = sp.results_to_raw_dicts(results)
        assert len(raw) == 1
        assert raw[0]["title"] == "T"
        assert raw[0]["url"] == "https://t.com"
        assert raw[0]["content"] == "S"  # mapped from snippet
        assert raw[0]["source"] == "brave"
        assert raw[0]["publishedDate"] == "2026-01-01"


class TestSearchAsRaw:
    @pytest.mark.asyncio
    async def test_general_routes_to_multi_search(self):
        with patch("search_providers.multi_search", AsyncMock(return_value=[
            sp.SearchResult(title="R1", url="https://r1.com", snippet="S", source="ddg"),
        ])) as mock_ms:
            raw = await sp.search_as_raw("test query", categories="general")

        mock_ms.assert_called_once()
        assert len(raw) == 1
        assert raw[0]["title"] == "R1"

    @pytest.mark.asyncio
    async def test_general_passes_time_range(self):
        """time_range must propagate through multi_search to SearXNG."""
        with patch("search_providers.multi_search", AsyncMock(return_value=[])) as mock_ms:
            await sp.search_as_raw("test", categories="general", time_range="week")

        mock_ms.assert_called_once_with("test", max_results=10, time_range="week")

    @pytest.mark.asyncio
    async def test_news_routes_to_multi_search_news(self):
        with patch("search_providers.multi_search_news", AsyncMock(return_value=[
            sp.SearchResult(title="N1", url="https://n1.com", snippet="S", source="ddg_news"),
        ])) as mock_mn:
            raw = await sp.search_as_raw("breaking news", categories="news", time_range="week")

        mock_mn.assert_called_once()
        assert raw[0]["title"] == "N1"

    @pytest.mark.asyncio
    async def test_science_routes_to_multi_search_science(self):
        with patch("search_providers.multi_search_science", AsyncMock(return_value=[
            sp.SearchResult(title="P1", url="https://arxiv.org/abs/1", snippet="S", source="searxng:science"),
        ])) as mock_sci:
            raw = await sp.search_as_raw("quantum", categories="science")

        mock_sci.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_category_falls_to_searxng(self):
        with patch("search_providers._search_searxng", AsyncMock(return_value=[
            sp.SearchResult(title="V1", url="https://v1.com", snippet="S", source="searxng:videos"),
        ])) as mock_sx:
            raw = await sp.search_as_raw("funny cats", categories="videos")

        mock_sx.assert_called_once()
        assert raw[0]["source"] == "searxng:videos"


# ===================================================================
# Provider availability
# ===================================================================

class TestAvailableProviders:
    def test_returns_dict_with_all_providers(self):
        avail = sp.available_providers()
        assert "duckduckgo" in avail
        assert "brave" in avail
        assert "mojeek" in avail
        assert "searxng" in avail

    def test_brave_requires_api_key(self):
        with patch.object(sp, "BRAVE_SEARCH_API_KEY", ""):
            avail = sp.available_providers()
            assert avail["brave"] is False

        with patch.object(sp, "BRAVE_SEARCH_API_KEY", "sk-123"):
            avail = sp.available_providers()
            assert avail["brave"] is True

    def test_mojeek_requires_api_key(self):
        with patch.object(sp, "MOJEEK_API_KEY", ""):
            avail = sp.available_providers()
            assert avail["mojeek"] is False

    def test_searxng_can_be_disabled(self):
        with patch.object(sp, "SEARXNG_ENABLED", False):
            avail = sp.available_providers()
            assert avail["searxng"] is False
