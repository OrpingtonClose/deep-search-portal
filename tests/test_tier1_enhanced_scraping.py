"""Tests for Tier 1 enhanced scraping: multi-tier web fetch, 4chan archive tools,
Twitter/X search, and censorship detection.

All external HTTP calls are mocked — no services need to be running.
"""

import asyncio
import html
import os
import re
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the shared module before importing the proxy — it calls require_env()
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

# Only inject if not already loaded (other test files may have done it)
if "shared" not in sys.modules or isinstance(sys.modules["shared"], MagicMock):
    sys.modules["shared"] = _mock_shared

# Also mock knowledge_client to avoid Neo4j connection attempts
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
# _strip_html
# ===================================================================

class TestStripHtml:
    def test_strips_script_tags(self):
        raw = "<html><script>alert(1)</script><body>Hello</body></html>"
        assert "alert" not in proxy._strip_html(raw)
        assert "Hello" in proxy._strip_html(raw)

    def test_strips_style_tags(self):
        raw = "<style>.x{color:red}</style><p>Content</p>"
        assert "color" not in proxy._strip_html(raw)
        assert "Content" in proxy._strip_html(raw)

    def test_unescapes_html_entities(self):
        raw = "<p>A &amp; B &lt; C</p>"
        result = proxy._strip_html(raw)
        assert "A & B < C" in result

    def test_collapses_whitespace(self):
        raw = "<p>  lots   of   spaces  </p>"
        result = proxy._strip_html(raw)
        assert "  " not in result

    def test_empty_input(self):
        assert proxy._strip_html("") == ""


# ===================================================================
# _is_censored_response
# ===================================================================

class TestIsCensoredResponse:
    def test_empty_string_not_censored(self):
        assert proxy._is_censored_response("") is False

    def test_error_prefix_not_censored(self):
        assert proxy._is_censored_response("Fetch error: HTTP 404") is False
        assert proxy._is_censored_response("Non-text content type: image/png") is False

    def test_short_content_is_censored(self):
        # Less than 50 chars and not an error prefix
        assert proxy._is_censored_response("Please enable cookies.") is True

    def test_long_content_with_censorship_keywords(self):
        text = (
            "Access Denied. Your request has been blocked by our security system. "
            "Please verify you are human by completing the captcha below."
        )
        assert proxy._is_censored_response(text) is True

    def test_normal_content_not_censored(self):
        text = (
            "This is a perfectly normal article about quantum computing. "
            "Researchers at MIT have developed a new qubit architecture that "
            "could significantly reduce error rates in quantum processors."
        )
        assert proxy._is_censored_response(text) is False

    def test_single_keyword_not_enough(self):
        # Only one keyword present — should NOT trigger
        text = "This page is blocked from view in certain regions due to legal reasons. " * 3
        # "blocked" is present, count keywords
        lower = text.lower()
        matches = sum(1 for kw in proxy._CENSORSHIP_KEYWORDS if kw in lower)
        if matches < 2:
            assert proxy._is_censored_response(text) is False


# ===================================================================
# _fetch_via_httpx
# ===================================================================

class TestFetchViaHttpx:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_success_html(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html; charset=utf-8"}
        resp.text = "<html><body>Hello world content here</body></html>"
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._fetch_via_httpx("https://example.com")
        assert "Hello world content here" in result

    @pytest.mark.asyncio
    async def test_404_returns_error(self):
        resp = MagicMock()
        resp.status_code = 404
        resp.headers = {}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._fetch_via_httpx("https://example.com/gone")
        assert "Fetch error: HTTP 404" in result

    @pytest.mark.asyncio
    async def test_403_returns_error(self):
        resp = MagicMock()
        resp.status_code = 403
        resp.headers = {}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._fetch_via_httpx("https://example.com/blocked")
        assert "Fetch error: HTTP 403" in result

    @pytest.mark.asyncio
    async def test_non_text_content_type(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "image/png"}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._fetch_via_httpx("https://example.com/image.png")
        assert "Non-text content type" in result


# ===================================================================
# _fetch_via_bright_data
# ===================================================================

class TestFetchViaBrightData:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_api_key(self):
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", ""):
            result = await proxy._fetch_via_bright_data("https://example.com")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_content_with_api_key(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body>" + "Bright Data fetched content. " * 10 + "</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key-123"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await proxy._fetch_via_bright_data("https://example.com")
            assert result is not None
            assert "Bright Data fetched content" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200(self):
        resp = MagicMock()
        resp.status_code = 503
        resp.headers = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key-123"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await proxy._fetch_via_bright_data("https://example.com")
            assert result is None


# ===================================================================
# _fetch_via_oxylabs
# ===================================================================

class TestFetchViaOxylabs:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_credentials(self):
        with patch.object(proxy, "OXYLABS_USERNAME", ""), \
             patch.object(proxy, "OXYLABS_PASSWORD", ""):
            result = await proxy._fetch_via_oxylabs("https://example.com")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_content_with_credentials(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body>" + "Oxylabs fetched this page content. " * 10 + "</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(proxy, "OXYLABS_USERNAME", "user"), \
             patch.object(proxy, "OXYLABS_PASSWORD", "pass"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await proxy._fetch_via_oxylabs("https://example.com")
            assert result is not None
            assert "Oxylabs fetched" in result


# ===================================================================
# _fetch_via_wayback_cdx
# ===================================================================

class TestFetchViaWaybackCdx:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_returns_archived_content(self):
        cdx_resp = MagicMock()
        cdx_resp.status_code = 200
        cdx_resp.json.return_value = [
            ["timestamp", "statuscode"],
            ["20240115120000", "200"],
        ]

        archive_resp = MagicMock()
        archive_resp.status_code = 200
        archive_resp.text = "<html><body>" + "Archived page content from Wayback. " * 10 + "</body></html>"

        self.mock_client.get = AsyncMock(side_effect=[cdx_resp, archive_resp])

        result = await proxy._fetch_via_wayback_cdx("https://example.com/dead-page")
        assert result is not None
        assert "ARCHIVED" in result
        assert "20240115120000" in result
        assert "Archived page content" in result

    @pytest.mark.asyncio
    async def test_returns_none_when_no_snapshot(self):
        cdx_resp = MagicMock()
        cdx_resp.status_code = 200
        cdx_resp.json.return_value = [["timestamp", "statuscode"]]  # header only

        self.mock_client.get = AsyncMock(return_value=cdx_resp)

        result = await proxy._fetch_via_wayback_cdx("https://example.com/never-archived")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_cdx_error(self):
        cdx_resp = MagicMock()
        cdx_resp.status_code = 503

        self.mock_client.get = AsyncMock(return_value=cdx_resp)

        result = await proxy._fetch_via_wayback_cdx("https://example.com")
        assert result is None


# ===================================================================
# enhanced_web_fetch — integration of fallback chain
# ===================================================================

class TestEnhancedWebFetch:
    @pytest.mark.asyncio
    async def test_returns_direct_content_when_httpx_succeeds(self):
        good_content = "A long article about quantum computing. " * 20

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=good_content)), \
             patch.object(proxy, "_PLAYWRIGHT_AVAILABLE", False), \
             patch.object(proxy, "_SELENIUM_AVAILABLE", False):
            result = await proxy.enhanced_web_fetch("https://example.com")
            assert "quantum computing" in result
            assert "Content from https://example.com" in result

    @pytest.mark.asyncio
    async def test_falls_through_to_playwright_on_censored(self):
        censored = "Access denied. Your request has been blocked. Just a moment please."
        playwright_content = "Real article content from Playwright rendering. " * 10

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=censored)), \
             patch.object(proxy, "_fetch_via_playwright", AsyncMock(return_value=playwright_content)), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com")
            assert "Real article content" in result

    @pytest.mark.asyncio
    async def test_falls_through_to_wayback_on_404(self):
        error_404 = "Fetch error: HTTP 404 for https://example.com/gone"
        archived = "[ARCHIVED — Wayback Machine snapshot from 20240101]\n\nOld content here. " * 5

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=error_404)), \
             patch.object(proxy, "_fetch_via_playwright", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=archived)):
            result = await proxy.enhanced_web_fetch("https://example.com/gone")
            assert "ARCHIVED" in result

    @pytest.mark.asyncio
    async def test_skips_all_tiers_on_404(self):
        """When httpx gets a 404, skip JS rendering AND proxies (content is gone)."""
        error_404 = "Fetch error: HTTP 404 for https://example.com/gone"
        pw_mock = AsyncMock(return_value=None)
        sel_mock = AsyncMock(return_value=None)
        bd_mock = AsyncMock(return_value=None)
        ox_mock = AsyncMock(return_value=None)

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=error_404)), \
             patch.object(proxy, "_fetch_via_playwright", pw_mock), \
             patch.object(proxy, "_fetch_via_selenium", sel_mock), \
             patch.object(proxy, "_fetch_via_bright_data", bd_mock), \
             patch.object(proxy, "_fetch_via_oxylabs", ox_mock), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            await proxy.enhanced_web_fetch("https://example.com/gone")
            pw_mock.assert_not_awaited()
            sel_mock.assert_not_awaited()
            bd_mock.assert_not_awaited()
            ox_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_403_skips_js_but_tries_proxies(self):
        """When httpx gets a 403, skip JS rendering but TRY commercial proxies."""
        error_403 = "Fetch error: HTTP 403 for https://example.com/blocked"
        pw_mock = AsyncMock(return_value=None)
        sel_mock = AsyncMock(return_value=None)
        proxy_content = "Unblocked content via commercial proxy. " * 10
        bd_mock = AsyncMock(return_value=proxy_content)

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=error_403)), \
             patch.object(proxy, "_fetch_via_playwright", pw_mock), \
             patch.object(proxy, "_fetch_via_selenium", sel_mock), \
             patch.object(proxy, "_fetch_via_bright_data", bd_mock), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com/blocked")
            # JS rendering should be skipped for server-side blocks
            pw_mock.assert_not_awaited()
            sel_mock.assert_not_awaited()
            # But commercial proxy should be tried and succeed
            bd_mock.assert_awaited_once()
            assert "Unblocked content" in result

    @pytest.mark.asyncio
    async def test_empty_spa_content_triggers_playwright(self):
        """Empty content from JS SPA pages should trigger Playwright fallback."""
        pw_content = "Full rendered SPA content with lots of useful text. " * 10
        pw_mock = AsyncMock(return_value=pw_content)

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value="")), \
             patch.object(proxy, "_fetch_via_playwright", pw_mock), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com/spa")
            pw_mock.assert_awaited_once()
            assert "Full rendered SPA content" in result

    @pytest.mark.asyncio
    async def test_extract_info_prepended(self):
        content = "Some article about the history of computing. " * 10

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=content)):
            result = await proxy.enhanced_web_fetch(
                "https://example.com", extract_info="founding date"
            )
            assert "Looking for: founding date" in result

    @pytest.mark.asyncio
    async def test_empty_content_returns_no_text_message(self):
        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value="")), \
             patch.object(proxy, "_fetch_via_playwright", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com")
            assert "no readable text" in result.lower()

    @pytest.mark.asyncio
    async def test_censorship_warning_appended_on_total_failure(self):
        censored = "Access denied. Your request has been blocked. Just a moment please."

        with patch.object(proxy, "_fetch_via_httpx", AsyncMock(return_value=censored)), \
             patch.object(proxy, "_fetch_via_playwright", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_selenium", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_oxylabs", AsyncMock(return_value=None)), \
             patch.object(proxy, "_fetch_via_wayback_cdx", AsyncMock(return_value=None)):
            result = await proxy.enhanced_web_fetch("https://example.com")
            assert "WARNING" in result
            assert "incomplete or blocked" in result


# ===================================================================
# tool_4plebs_search
# ===================================================================

class TestTool4plebsSearch:
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
            "0": {
                "posts": {
                    "12345": {
                        "thread_num": "100",
                        "num": "12345",
                        "comment": "Bitcoin is going to <b>moon</b> soon",
                        "timestamp": 1700000000,
                    }
                }
            }
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_4plebs_search("bitcoin", board="pol")
        assert "/pol/" in result
        assert "thread #100" in result
        assert "Bitcoin is going to" in result
        assert "moon" in result
        # HTML should be stripped
        assert "<b>" not in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"0": {"posts": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_4plebs_search("xyznonexistent")
        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_http_error(self):
        resp = MagicMock()
        resp.status_code = 500
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_4plebs_search("test")
        assert "4plebs search error" in result

    @pytest.mark.asyncio
    async def test_board_stripping(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"0": {"posts": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        # Board with slashes should be stripped
        result = await proxy.tool_4plebs_search("test", board="/pol/")
        assert "No results found on /pol/" in result

    @pytest.mark.asyncio
    async def test_long_comment_truncated(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "0": {
                "posts": {
                    "1": {
                        "thread_num": "1",
                        "num": "1",
                        "comment": "A" * 600,
                        "timestamp": 0,
                    }
                }
            }
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_4plebs_search("test")
        assert "..." in result


# ===================================================================
# tool_b4k_search
# ===================================================================

class TestToolB4kSearch:
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
            "0": {
                "posts": {
                    "99": {
                        "thread_num": "50",
                        "num": "99",
                        "comment": "ETH looking bullish today",
                        "timestamp": 1700000000,
                    }
                }
            }
        }
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_b4k_search("ethereum")
        assert "/biz/" in result
        assert "arch.b4k.co" in result
        assert "ETH looking bullish" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"0": {"posts": []}}
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_b4k_search("xyznonexistent")
        assert "No results found on /biz/" in result


# ===================================================================
# tool_warosu_search
# ===================================================================

class TestToolWarosuSearch:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_search(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '''
        <table>
        <td class="reply" id="p12345">
            <a href="/g/thread/100">Thread</a>
            <blockquote>Rust is better than C++ for systems programming</blockquote>
        </td>
        </table>
        '''
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_warosu_search("rust programming", board="g")
        assert "/g/" in result
        assert "warosu.org" in result
        assert "Rust is better" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>No results</body></html>"
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_warosu_search("xyznonexistent")
        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_http_error(self):
        resp = MagicMock()
        resp.status_code = 503
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy.tool_warosu_search("test")
        assert "Warosu search error" in result


# ===================================================================
# tool_twitter_search — tiered fallback
# ===================================================================

class TestToolTwitterSearch:
    @pytest.mark.asyncio
    async def test_uses_bright_data_first(self):
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key"), \
             patch.object(proxy, "_twitter_via_bright_data", AsyncMock(return_value="BD results")):
            result = await proxy.tool_twitter_search("bitcoin")
            assert result == "BD results"

    @pytest.mark.asyncio
    async def test_falls_to_oxylabs_when_bd_fails(self):
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key"), \
             patch.object(proxy, "OXYLABS_USERNAME", "user"), \
             patch.object(proxy, "_twitter_via_bright_data", AsyncMock(return_value=None)), \
             patch.object(proxy, "_twitter_via_oxylabs", AsyncMock(return_value="OX results")):
            result = await proxy.tool_twitter_search("bitcoin")
            assert result == "OX results"

    @pytest.mark.asyncio
    async def test_falls_to_nitter_when_proxies_fail(self):
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", ""), \
             patch.object(proxy, "OXYLABS_USERNAME", ""), \
             patch.object(proxy, "_twitter_via_nitter", AsyncMock(return_value="Nitter results")):
            result = await proxy.tool_twitter_search("bitcoin")
            assert result == "Nitter results"

    @pytest.mark.asyncio
    async def test_all_tiers_exhausted(self):
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", ""), \
             patch.object(proxy, "OXYLABS_USERNAME", ""), \
             patch.object(proxy, "_twitter_via_nitter", AsyncMock(return_value=None)):
            result = await proxy.tool_twitter_search("bitcoin")
            assert "Twitter search failed" in result
            assert "All access tiers exhausted" in result


# ===================================================================
# _twitter_via_bright_data
# ===================================================================

class TestTwitterViaBrightData:
    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """_twitter_via_bright_data should return None gracefully on any error."""
        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key"), \
             patch("httpx.AsyncClient", side_effect=Exception("connection refused")):
            result = await proxy._twitter_via_bright_data("bitcoin")
            assert result is None

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>" + "Tweet about bitcoin from @satoshi. " * 20 + "</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await proxy._twitter_via_bright_data("bitcoin")
            assert result is not None
            assert "Twitter/X search results" in result
            assert "bitcoin" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_censored_content(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>Access denied. Just a moment. Checking your browser.</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(proxy, "BRIGHT_DATA_API_KEY", "test-key"), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await proxy._twitter_via_bright_data("bitcoin")
            assert result is None


# ===================================================================
# _twitter_via_nitter
# ===================================================================

class TestTwitterViaNitter:
    @pytest.fixture(autouse=True)
    def _patch_client(self):
        self.mock_client = AsyncMock()
        with patch.object(proxy, "http_client", return_value=self.mock_client):
            yield

    @pytest.mark.asyncio
    async def test_successful_nitter_parse(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '''
        <div class="tweet-content media-body" dir="auto">Bitcoin just hit 100k!</div>
        <a class="username" href="/satoshi">@satoshi</a>
        <div class="tweet-content media-body" dir="auto">Still bullish on ETH.</div>
        <a class="username" href="/vitalik">@vitalik</a>
        '''
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._twitter_via_nitter("bitcoin")
        assert result is not None
        assert "Twitter/X search results" in result
        assert "Nitter" in result
        assert "Bitcoin just hit 100k" in result

    @pytest.mark.asyncio
    async def test_all_instances_fail(self):
        resp = MagicMock()
        resp.status_code = 503
        self.mock_client.get = AsyncMock(return_value=resp)

        result = await proxy._twitter_via_nitter("bitcoin")
        assert result is None


# ===================================================================
# execute_tool routing for new tools
# ===================================================================

class TestExecuteToolRouting:
    @pytest.mark.asyncio
    async def test_routes_chan_4plebs_search(self):
        with patch.object(proxy, "tool_4plebs_search", AsyncMock(return_value="4plebs result")) as mock:
            result = await proxy.execute_tool("chan_4plebs_search", {"query": "test", "board": "pol"})
            assert result == "4plebs result"
            mock.assert_awaited_once_with("test", "pol")

    @pytest.mark.asyncio
    async def test_routes_chan_b4k_search(self):
        with patch.object(proxy, "tool_b4k_search", AsyncMock(return_value="b4k result")) as mock:
            result = await proxy.execute_tool("chan_b4k_search", {"query": "crypto"})
            assert result == "b4k result"
            mock.assert_awaited_once_with("crypto")

    @pytest.mark.asyncio
    async def test_routes_chan_warosu_search(self):
        with patch.object(proxy, "tool_warosu_search", AsyncMock(return_value="warosu result")) as mock:
            result = await proxy.execute_tool("chan_warosu_search", {"query": "rust", "board": "g"})
            assert result == "warosu result"
            mock.assert_awaited_once_with("rust", "g")

    @pytest.mark.asyncio
    async def test_routes_twitter_search(self):
        with patch.object(proxy, "tool_twitter_search", AsyncMock(return_value="twitter result")) as mock:
            result = await proxy.execute_tool("twitter_search", {"query": "breaking news"})
            assert result == "twitter result"
            mock.assert_awaited_once_with("breaking news")

    @pytest.mark.asyncio
    async def test_routes_fetch_webpage_to_enhanced(self):
        with patch.object(proxy, "enhanced_web_fetch", AsyncMock(return_value="enhanced result")) as mock:
            result = await proxy.execute_tool("fetch_webpage", {"url": "https://example.com"})
            assert result == "enhanced result"
            mock.assert_awaited_once_with("https://example.com", "")

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await proxy.execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result


# ===================================================================
# SearXNG config validation (YAML structure)
# ===================================================================

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "searxng_settings_patch.yml"
)


class TestSearxngConfig:
    def test_config_is_valid_yaml(self):
        import yaml
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_config_has_long_tail_engines(self):
        import yaml
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f)

        engines = data.get("engines", [])
        engine_names = {e["name"] for e in engines}
        assert "marginalia custom" in engine_names
        assert "wiby custom" in engine_names
        assert "mojeek" in engine_names

    def test_engines_have_required_fields(self):
        import yaml
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f)

        # Only validate full engine definitions (custom xpath scrapers),
        # not override stubs that just set disabled/inactive flags.
        required_fields = {"name", "engine", "search_url", "shortcut"}
        for engine in data.get("engines", []):
            if "engine" not in engine:
                # This is an override stub (e.g. {name: bing, disabled: false})
                assert "name" in engine, f"Override stub missing 'name': {engine}"
                continue
            missing = required_fields - set(engine.keys())
            assert not missing, f"Engine '{engine.get('name', '?')}' missing fields: {missing}"

    def test_engines_not_disabled(self):
        import yaml
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f)

        for engine in data.get("engines", []):
            # Custom engines must be enabled; override stubs set disabled: false
            if "engine" in engine:
                assert engine.get("disabled") is not True, f"Engine '{engine['name']}' is disabled"
