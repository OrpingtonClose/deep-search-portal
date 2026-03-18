"""Tests for b2_publisher.py — Backblaze B2 report publishing."""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

import b2_publisher


def _reset():
    b2_publisher._b2_api = None
    b2_publisher._b2_bucket = None


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------

class TestIsConfigured:
    def teardown_method(self):
        _reset()

    def test_configured_when_both_set(self):
        b2_publisher.B2_KEY_ID = "key-id"
        b2_publisher.B2_APP_KEY = "key"
        assert b2_publisher.is_configured() is True

    def test_not_configured_when_missing_key_id(self):
        b2_publisher.B2_KEY_ID = ""
        b2_publisher.B2_APP_KEY = "key"
        assert b2_publisher.is_configured() is False

    def test_not_configured_when_missing_key(self):
        b2_publisher.B2_KEY_ID = "key-id"
        b2_publisher.B2_APP_KEY = ""
        assert b2_publisher.is_configured() is False

    def test_not_configured_when_both_missing(self):
        b2_publisher.B2_KEY_ID = ""
        b2_publisher.B2_APP_KEY = ""
        assert b2_publisher.is_configured() is False


# ---------------------------------------------------------------------------
# _get_b2_bucket — idempotent bucket creation
# ---------------------------------------------------------------------------

class TestGetBucket:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_raises_without_credentials(self):
        b2_publisher.B2_KEY_ID = ""
        b2_publisher.B2_APP_KEY = ""
        with pytest.raises(RuntimeError, match="B2_APPLICATION_KEY_ID"):
            b2_publisher._get_b2_bucket()

    def test_creates_bucket_if_not_exists(self):
        b2_publisher.B2_KEY_ID = "test-key-id"
        b2_publisher.B2_APP_KEY = "test-app-key"

        mock_api = MagicMock()
        mock_bucket = MagicMock()
        mock_api.get_bucket_by_name.side_effect = Exception("not found")
        mock_api.create_bucket.return_value = mock_bucket

        mock_b2sdk_v2 = ModuleType("b2sdk.v2")
        mock_b2sdk_v2.B2Api = MagicMock(return_value=mock_api)
        mock_b2sdk_v2.InMemoryAccountInfo = MagicMock()

        with patch.dict("sys.modules", {"b2sdk": MagicMock(), "b2sdk.v2": mock_b2sdk_v2}):
            bucket = b2_publisher._get_b2_bucket()

        assert bucket is mock_bucket
        mock_api.authorize_account.assert_called_once_with("production", "test-key-id", "test-app-key")
        mock_api.create_bucket.assert_called_once()

    def test_uses_existing_bucket(self):
        b2_publisher.B2_KEY_ID = "test-key-id"
        b2_publisher.B2_APP_KEY = "test-app-key"

        mock_api = MagicMock()
        mock_bucket = MagicMock()
        mock_api.get_bucket_by_name.return_value = mock_bucket

        mock_b2sdk_v2 = ModuleType("b2sdk.v2")
        mock_b2sdk_v2.B2Api = MagicMock(return_value=mock_api)
        mock_b2sdk_v2.InMemoryAccountInfo = MagicMock()

        with patch.dict("sys.modules", {"b2sdk": MagicMock(), "b2sdk.v2": mock_b2sdk_v2}):
            bucket = b2_publisher._get_b2_bucket()

        assert bucket is mock_bucket
        mock_api.create_bucket.assert_not_called()

    def test_cached_bucket_returned(self):
        mock_bucket = MagicMock()
        b2_publisher._b2_bucket = mock_bucket
        assert b2_publisher._get_b2_bucket() is mock_bucket


# ---------------------------------------------------------------------------
# publish_report
# ---------------------------------------------------------------------------

class TestPublishReport:
    def setup_method(self):
        self.mock_bucket = MagicMock()
        self.mock_file = MagicMock()
        self.mock_file.id_ = "file-id-123"
        self.mock_bucket.upload.return_value = self.mock_file

        self.mock_api = MagicMock()
        self.mock_api.get_download_url_for_fileid.return_value = (
            "https://f001.backblazeb2.com/file/deep-search-reports/reports/sess-1.html"
        )

        b2_publisher._b2_bucket = self.mock_bucket
        b2_publisher._b2_api = self.mock_api

    def teardown_method(self):
        _reset()

    def _mock_b2sdk(self):
        mock_mod = ModuleType("b2sdk.v2")
        mock_mod.UploadSourceBytes = MagicMock(side_effect=lambda data: data)
        return patch.dict("sys.modules", {"b2sdk": MagicMock(), "b2sdk.v2": mock_mod})

    def test_publishes_html(self):
        with self._mock_b2sdk():
            url = b2_publisher.publish_report("sess-1", "<html>Test</html>")
        assert "sess-1.html" in url
        self.mock_bucket.upload.assert_called_once()

    def test_correct_file_name(self):
        with self._mock_b2sdk():
            b2_publisher.publish_report("my-session", "<html/>")
        call_kwargs = self.mock_bucket.upload.call_args
        assert call_kwargs[1]["file_name"] == "reports/my-session.html"

    def test_correct_content_type(self):
        with self._mock_b2sdk():
            b2_publisher.publish_report("sess-2", "<html/>")
        call_kwargs = self.mock_bucket.upload.call_args
        assert call_kwargs[1]["content_type"] == "text/html"

    def test_returns_download_url(self):
        with self._mock_b2sdk():
            url = b2_publisher.publish_report("sess-1", "<html/>")
        self.mock_api.get_download_url_for_fileid.assert_called_once_with("file-id-123")
        assert url.startswith("https://")


# ---------------------------------------------------------------------------
# publish_metrics
# ---------------------------------------------------------------------------

class TestPublishMetrics:
    def setup_method(self):
        self.mock_bucket = MagicMock()
        self.mock_file = MagicMock()
        self.mock_file.id_ = "file-id-456"
        self.mock_bucket.upload.return_value = self.mock_file

        self.mock_api = MagicMock()
        self.mock_api.get_download_url_for_fileid.return_value = (
            "https://f001.backblazeb2.com/file/deep-search-reports/reports/sess-1_metrics.json"
        )

        b2_publisher._b2_bucket = self.mock_bucket
        b2_publisher._b2_api = self.mock_api

    def teardown_method(self):
        _reset()

    def _mock_b2sdk(self):
        mock_mod = ModuleType("b2sdk.v2")
        mock_mod.UploadSourceBytes = MagicMock(side_effect=lambda data: data)
        return patch.dict("sys.modules", {"b2sdk": MagicMock(), "b2sdk.v2": mock_mod})

    def test_publishes_json(self):
        with self._mock_b2sdk():
            url = b2_publisher.publish_metrics("sess-1", '{"session_id": "sess-1"}')
        assert "metrics.json" in url
        self.mock_bucket.upload.assert_called_once()

    def test_correct_file_name(self):
        with self._mock_b2sdk():
            b2_publisher.publish_metrics("my-session", "{}")
        call_kwargs = self.mock_bucket.upload.call_args
        assert call_kwargs[1]["file_name"] == "reports/my-session_metrics.json"

    def test_correct_content_type(self):
        with self._mock_b2sdk():
            b2_publisher.publish_metrics("sess-2", "{}")
        call_kwargs = self.mock_bucket.upload.call_args
        assert call_kwargs[1]["content_type"] == "application/json"
