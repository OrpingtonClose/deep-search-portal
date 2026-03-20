"""Tests for knowledge gathering tools: HackerNews, StackExchange, PubMed,
Wikipedia, Archive.org, forum search, Google Scholar, Substack, retry wrapper,
PDF extraction, and enhanced_web_fetch PDF integration.

All external HTTP calls are mocked -- no services need to be running.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the shared module before importing the proxy -- it calls require_env()
# and setup_logging() at import time which would fail in CI.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

_mock_shared = MagicMock()
_mock_shared.setup_logging.return_value = MagicMock()
_mock_shared.require_env.return_value = "test-key"
_mock_shared.env_int.side_effect = lambda name, default, **kw: default
_mock_shared.http_client.return_value = MagicMock()
_mock_shared.create_app.return_value = MagicMock()
_mock_shared.register_standard_routes = MagicMock()
_mock_shared.make_sse_chunk = MagicMock(side_effect=lambda *a, **kw: "data: {}\n\n")
_mock_shared.ConcurrencyLimiter = MagicMock
_mock_shared.RequestTracker = MagicMock
_mock_shared.is_utility_request = MagicMock(return_value=False)
_mock_shared.stream_passthrough = MagicMock()
_mock_shared.all_throttler_stats = MagicMock(return_value={})

# Throttler mock -- must return an async context manager
_throttler_instance = MagicMock()
_throttler_instance.__aenter__ = AsyncMock(return_value=None)
_throttler_instance.__aexit__ = AsyncMock(return_value=False)
_mock_shared.get_throttler.return_value = _throttler_instance

if "shared" not in sys.modules or isinstance(sys.modules["shared"], MagicMock):
    sys.modules["shared"] = _mock_shared

if "knowledge_client" not in sys.modules:
    _mock_kc = MagicMock()
    _mock_kc.store_research_conditions = AsyncMock()
    _mock_kc.search_research_conditions = AsyncMock(return_value=[])
    _mock_kc.get_graph_neighbors = AsyncMock(return_value=[])
    _mock_kc.store_entities = AsyncMock()
    _mock_kc.get_stats = AsyncMock(return_value={})
    sys.modules["knowledge_client"] = _mock_kc

import persistent_deep_research_proxy as proxy


# ===================================================================
# AtomicCondition metadata fields
# ===================================================================

class TestAtomicConditionMetadata:
    def test_default_metadata_empty(self):
        c = proxy.AtomicCondition(fact="test fact")
        assert c.publication_date == ""
        assert c.author == ""
        assert c.content_type == ""
        assert c.source_type == ""

    def test_metadata_in_to_text(self):
        c = proxy.AtomicCondition(
            fact="test fact",
            source_type="pubmed",
            author="Dr. Smith",
            publication_date="2026-01-15",
        )
        text = c.to_text()
        assert "[via: pubmed]" in text
        assert "[author: Dr. Smith]" in text
        assert "[date: 2026-01-15]" in text

    def test_empty_metadata_not_in_to_text(self):
        c = proxy.AtomicCondition(fact="test fact")
        text = c.to_text()
        assert "[via:" not in text
        assert "[author:" not in text
        assert "[date:" not in text


# ===================================================================
# tool_hackernews_search
# ===================================================================

class TestHackerNewsSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "hits": [
                {
                    "title": "Show HN: My New Project",
                    "author": "testuser",
                    "points": 150,
                    "num_comments": 42,
                    "created_at_i": 1710000000,
                    "objectID": "12345",
                    "story_text": None,
                    "comment_text": None,
                    "_tags": ["story"],
                },
                {
                    "title": None,
                    "author": "commenter",
                    "points": 5,
                    "num_comments": 0,
                    "created_at_i": 1710000100,
                    "objectID": "12346",
                    "story_text": None,
                    "comment_text": "Great discussion about the topic here",
                    "_tags": ["comment"],
                },
            ]
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_hackernews_search("my new project")
        assert "Show HN: My New Project" in result
        assert "testuser" in result
        assert "150" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": []}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_hackernews_search("xyznonexistent12345")
        assert "No Hacker News results" in result

    @pytest.mark.asyncio
    async def test_time_range_filter(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": []}
        self.mock_client.get = AsyncMock(return_value=resp)

        await proxy.tool_hackernews_search("AI", time_range="week")
        call_args = self.mock_client.get.call_args
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        # Should have numericFilters for time range
        if isinstance(params, dict):
            assert "numericFilters" in params or "tags" in params

    @pytest.mark.asyncio
    async def test_sort_by_date(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": []}
        self.mock_client.get = AsyncMock(return_value=resp)

        await proxy.tool_hackernews_search("test", sort_by="date")
        call_url = self.mock_client.get.call_args[0][0]
        assert "search_by_date" in call_url

    @pytest.mark.asyncio
    async def test_error_handling(self):
        self.mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        result = await proxy.tool_hackernews_search("test")
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_stackexchange_search
# ===================================================================

class TestStackExchangeSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "items": [
                {
                    "title": "How to parse JSON in Python?",
                    "link": "https://stackoverflow.com/q/123",
                    "score": 450,
                    "answer_count": 12,
                    "tags": ["python", "json", "parsing"],
                    "body_markdown": "I need to parse a JSON string...",
                },
            ]
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_stackexchange_search("parse JSON python")
        assert "How to parse JSON" in result
        assert "450" in result or "score" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_stackexchange_search("xyznonexistent12345")
        assert "No" in result and "results" in result.lower()

    @pytest.mark.asyncio
    async def test_different_site(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        self.mock_client.get = AsyncMock(return_value=resp)

        await proxy.tool_stackexchange_search("quantum entanglement", site="physics")
        call_args = self.mock_client.get.call_args
        params = call_args[1].get("params", {})
        assert params.get("site") == "physics"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        self.mock_client.get = AsyncMock(side_effect=Exception("API error"))
        result = await proxy.tool_stackexchange_search("test")
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_pubmed_search
# ===================================================================

class TestPubMedSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        # First call: esearch returns PMIDs
        esearch_resp = MagicMock()
        esearch_resp.status_code = 200
        esearch_resp.json.return_value = {
            "esearchresult": {
                "idlist": ["12345678", "87654321"],
                "count": "2",
            }
        }

        # Second call: esummary returns article details
        esummary_resp = MagicMock()
        esummary_resp.status_code = 200
        esummary_resp.json.return_value = {
            "result": {
                "uids": ["12345678", "87654321"],
                "12345678": {
                    "title": "Effects of caffeine on cognitive performance",
                    "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                    "source": "J Neuroscience",
                    "pubdate": "2024 Jan",
                    "elocationid": "doi: 10.1234/jn.2024.001",
                },
                "87654321": {
                    "title": "Meta-analysis of sleep interventions",
                    "authors": [{"name": "Johnson B"}],
                    "source": "Sleep Medicine Reviews",
                    "pubdate": "2023 Dec",
                    "elocationid": "",
                },
            }
        }

        self.mock_client.get = AsyncMock(side_effect=[esearch_resp, esummary_resp])

        result = await proxy.tool_pubmed_search("caffeine cognitive performance")
        assert "caffeine" in result.lower() or "Effects of caffeine" in result
        assert "Smith" in result
        assert "Neuroscience" in result or "J Neuroscience" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        esearch_resp = MagicMock()
        esearch_resp.status_code = 200
        esearch_resp.json.return_value = {
            "esearchresult": {"idlist": [], "count": "0"}
        }
        self.mock_client.get = AsyncMock(return_value=esearch_resp)

        result = await proxy.tool_pubmed_search("xyznonexistent12345")
        assert "No PubMed results" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        self.mock_client.get = AsyncMock(side_effect=Exception("NCBI API error"))
        result = await proxy.tool_pubmed_search("test")
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_wikipedia_search
# ===================================================================

class TestWikipediaSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Quantum computing",
                        "snippet": "A <span class=\"searchmatch\">quantum</span> computer uses quantum-mechanical phenomena...",
                        "wordcount": 15000,
                        "timestamp": "2024-01-15T10:30:00Z",
                    },
                ]
            }
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_wikipedia_search("quantum computing")
        assert "Quantum computing" in result
        assert "quantum" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"query": {"search": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_wikipedia_search("xyznonexistent12345")
        assert "No Wikipedia results" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        self.mock_client.get = AsyncMock(side_effect=Exception("MediaWiki API error"))
        result = await proxy.tool_wikipedia_search("test")
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_archiveorg_search
# ===================================================================

class TestArchiveOrgSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "response": {
                "docs": [
                    {
                        "identifier": "gov-report-2023",
                        "title": "US Government Report on AI Safety",
                        "creator": "National Science Foundation",
                        "date": "2023-06-15",
                        "mediatype": "texts",
                        "downloads": 5000,
                        "description": "A comprehensive report on artificial intelligence safety measures.",
                    },
                ]
            }
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_archiveorg_search("government AI safety report")
        assert "AI Safety" in result or "gov-report" in result
        assert "National Science Foundation" in result or "texts" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"response": {"docs": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_archiveorg_search("xyznonexistent12345")
        assert "No Archive.org results" in result

    @pytest.mark.asyncio
    async def test_media_type_filter(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"response": {"docs": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        await proxy.tool_archiveorg_search("old recordings", media_type="audio")
        call_args = self.mock_client.get.call_args
        params = call_args[1].get("params", {})
        q_value = params.get("q", "")
        assert "mediatype:audio" in q_value or "audio" in str(call_args)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        self.mock_client.get = AsyncMock(side_effect=Exception("Archive.org API error"))
        result = await proxy.tool_archiveorg_search("test")
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_forum_search
# ===================================================================

class TestForumSearch:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_results = [
            {"title": "XDA thread about rooting", "url": "https://forum.xda-developers.com/t/123", "content": "How to root..."},
        ]
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=mock_results)):
            result = await proxy.tool_forum_search("android rooting guide")
            assert "XDA" in result or "rooting" in result.lower()

    @pytest.mark.asyncio
    async def test_specific_forum_url(self):
        mock_results = [
            {"title": "Head-Fi review", "url": "https://head-fi.org/t/456", "content": "Great headphones..."},
        ]
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=mock_results)):
            result = await proxy.tool_forum_search("best headphones", forum_url="head-fi.org")
            assert "Head-Fi" in result or "headphones" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=[])):
            result = await proxy.tool_forum_search("xyznonexistent12345")
            assert "No forum results" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(side_effect=Exception("SearXNG error"))):
            result = await proxy.tool_forum_search("test")
            # Forum search catches errors gracefully and returns "No forum results"
            assert "No forum results" in result or "error" in result.lower()


# ===================================================================
# tool_scholar_search
# ===================================================================

class TestScholarSearch:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_results = [
            {"title": "Deep Learning for NLP", "url": "https://arxiv.org/abs/1234", "content": "A survey of deep learning..."},
        ]
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=mock_results)):
            result = await proxy.tool_scholar_search("deep learning NLP")
            assert "Deep Learning" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=[])):
            result = await proxy.tool_scholar_search("xyznonexistent12345")
            assert "No scholar results" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(side_effect=Exception("SearXNG error"))):
            result = await proxy.tool_scholar_search("test")
            assert "error" in result.lower() or "Error" in result


# ===================================================================
# tool_substack_search
# ===================================================================

class TestSubstackSearch:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_results = [
            {"title": "The Great AI Debate", "url": "https://example.substack.com/p/ai-debate", "content": "Analysis of AI policy..."},
        ]
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=mock_results)):
            result = await proxy.tool_substack_search("AI policy analysis")
            assert "AI" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(return_value=[])):
            result = await proxy.tool_substack_search("xyznonexistent12345")
            assert "No Substack results" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with patch.object(proxy, "_searxng_query", AsyncMock(side_effect=Exception("SearXNG error"))):
            result = await proxy.tool_substack_search("test")
            assert "error" in result.lower() or "Error" in result


# ===================================================================
# _retry_tool_call
# ===================================================================

class TestRetryToolCall:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await proxy._retry_tool_call(factory, max_retries=3)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return "success after retries"

        result = await proxy._retry_tool_call(factory, max_retries=3, backoff_base=0.01)
        assert result == "success after retries"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        async def factory():
            raise Exception("permanent error")

        result = await proxy._retry_tool_call(factory, max_retries=2, backoff_base=0.01)
        assert "error" in result.lower() or "Error" in result


# ===================================================================
# _extract_pdf_text
# ===================================================================

class TestExtractPdfText:
    @pytest.mark.asyncio
    async def test_returns_none_on_non_pdf(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.content = b"<html>not a pdf</html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        with patch.object(proxy, "http_client", return_value=mock_client):
            result = await proxy._extract_pdf_text("https://example.com/page.html")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_download_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Download failed"))
        with patch.object(proxy, "http_client", return_value=mock_client):
            result = await proxy._extract_pdf_text("https://example.com/paper.pdf")
            assert result is None


# ===================================================================
# enhanced_web_fetch PDF integration
# ===================================================================

class TestEnhancedWebFetchPdf:
    @pytest.mark.asyncio
    async def test_pdf_url_tries_extraction_first(self):
        """PDF URLs should try _extract_pdf_text before falling through to httpx."""
        pdf_text = "This is extracted PDF content about machine learning. " * 20

        with patch.object(proxy, "_extract_pdf_text", AsyncMock(return_value=pdf_text)):
            result = await proxy.enhanced_web_fetch("https://example.com/paper.pdf")
            assert "PDF content" in result
            assert "machine learning" in result

    @pytest.mark.asyncio
    async def test_pdf_url_falls_through_on_extraction_failure(self):
        """If PDF extraction fails, fall through to normal fetch chain."""
        html_content = "Fallback HTML content from the page. " * 20

        with patch.object(proxy, "_extract_pdf_text", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=html_content)), \
             patch.object(proxy, "_fetch_via_playwright", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com/paper.pdf")
            assert "Fallback HTML content" in result

    @pytest.mark.asyncio
    async def test_non_pdf_url_skips_extraction(self):
        """Non-PDF URLs should not try PDF extraction."""
        html_content = "Normal HTML page content here. " * 20
        pdf_mock = AsyncMock(return_value=None)

        with patch.object(proxy, "_extract_pdf_text", pdf_mock), \
             patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=html_content)):
            result = await proxy.enhanced_web_fetch("https://example.com/page.html")
            pdf_mock.assert_not_awaited()
            assert "Normal HTML page content" in result


# ===================================================================
# _execute_tool_inner routing for new tools
# ===================================================================

class TestExecuteToolInnerRouting:
    @pytest.mark.asyncio
    async def test_hackernews_search_route(self):
        with patch.object(proxy, "tool_hackernews_search", AsyncMock(return_value="HN results")) as mock:
            result = await proxy._execute_tool_inner("hackernews_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "HN results"

    @pytest.mark.asyncio
    async def test_stackexchange_search_route(self):
        with patch.object(proxy, "tool_stackexchange_search", AsyncMock(return_value="SE results")) as mock:
            result = await proxy._execute_tool_inner("stackexchange_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "SE results"

    @pytest.mark.asyncio
    async def test_pubmed_search_route(self):
        with patch.object(proxy, "tool_pubmed_search", AsyncMock(return_value="PubMed results")) as mock:
            result = await proxy._execute_tool_inner("pubmed_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "PubMed results"

    @pytest.mark.asyncio
    async def test_wikipedia_search_route(self):
        with patch.object(proxy, "tool_wikipedia_search", AsyncMock(return_value="Wiki results")) as mock:
            result = await proxy._execute_tool_inner("wikipedia_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "Wiki results"

    @pytest.mark.asyncio
    async def test_archiveorg_search_route(self):
        with patch.object(proxy, "tool_archiveorg_search", AsyncMock(return_value="Archive results")) as mock:
            result = await proxy._execute_tool_inner("archiveorg_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "Archive results"

    @pytest.mark.asyncio
    async def test_forum_search_route(self):
        with patch.object(proxy, "tool_forum_search", AsyncMock(return_value="Forum results")) as mock:
            result = await proxy._execute_tool_inner("forum_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "Forum results"

    @pytest.mark.asyncio
    async def test_scholar_search_route(self):
        with patch.object(proxy, "tool_scholar_search", AsyncMock(return_value="Scholar results")) as mock:
            result = await proxy._execute_tool_inner("scholar_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "Scholar results"

    @pytest.mark.asyncio
    async def test_substack_search_route(self):
        with patch.object(proxy, "tool_substack_search", AsyncMock(return_value="Substack results")) as mock:
            result = await proxy._execute_tool_inner("substack_search", {"query": "test"})
            mock.assert_awaited_once()
            assert result == "Substack results"


# ===================================================================
# NATIVE_TOOLS registration
# ===================================================================

class TestNativeToolsRegistration:
    """Verify all new tools are registered in NATIVE_TOOLS."""

    def _tool_names(self):
        return [t["function"]["name"] for t in proxy.NATIVE_TOOLS]

    def test_hackernews_registered(self):
        assert "hackernews_search" in self._tool_names()

    def test_stackexchange_registered(self):
        assert "stackexchange_search" in self._tool_names()

    def test_pubmed_registered(self):
        assert "pubmed_search" in self._tool_names()

    def test_wikipedia_registered(self):
        assert "wikipedia_search" in self._tool_names()

    def test_archiveorg_registered(self):
        assert "archiveorg_search" in self._tool_names()

    def test_forum_search_registered(self):
        assert "forum_search" in self._tool_names()

    def test_scholar_registered(self):
        assert "scholar_search" in self._tool_names()

    def test_substack_registered(self):
        assert "substack_search" in self._tool_names()

    def test_all_tools_have_required_fields(self):
        for tool in proxy.NATIVE_TOOLS:
            assert "type" in tool
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "required" in func["parameters"]
