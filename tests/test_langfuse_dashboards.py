"""Tests for langfuse_dashboards module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the proxies directory is importable
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

from langfuse_dashboards import (
    _format_cost,
    _format_duration,
    _langfuse_configured,
    _langfuse_query,
    _load_all_local_metrics,
    _safe,
    _time_range,
    aggregate_local_metrics,
    render_dashboard_html,
)


# ---------------------------------------------------------------------------
# Configuration detection
# ---------------------------------------------------------------------------


class TestLangfuseConfigured:
    def test_not_configured_when_no_keys(self):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", ""), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""):
            assert _langfuse_configured() is False

    def test_not_configured_when_only_public_key(self):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", "pk-123"), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""):
            assert _langfuse_configured() is False

    def test_configured_when_both_keys(self):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", "pk-123"), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", "sk-456"):
            assert _langfuse_configured() is True


# ---------------------------------------------------------------------------
# Time range helper
# ---------------------------------------------------------------------------


class TestTimeRange:
    def test_returns_two_iso_strings(self):
        from_ts, to_ts = _time_range(7)
        assert from_ts.endswith("Z")
        assert to_ts.endswith("Z")
        assert from_ts < to_ts

    def test_different_days(self):
        from7, _ = _time_range(7)
        from30, _ = _time_range(30)
        assert from30 < from7  # 30-day window starts earlier


# ---------------------------------------------------------------------------
# Langfuse query (mocked)
# ---------------------------------------------------------------------------


class TestLangfuseQuery:
    def test_returns_empty_when_not_configured(self):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", ""), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""):
            result = _langfuse_query({"view": "traces", "metrics": []})
            assert result == []

    @patch("langfuse_dashboards.httpx.get")
    def test_returns_data_on_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"name": "test", "count": 5}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", "pk-123"), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", "sk-456"):
            result = _langfuse_query({"view": "traces", "metrics": []})
            assert len(result) == 1
            assert result[0]["name"] == "test"

    @patch("langfuse_dashboards.httpx.get")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", "pk-123"), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", "sk-456"):
            result = _langfuse_query({"view": "traces", "metrics": []})
            assert result == []


# ---------------------------------------------------------------------------
# Local metrics loading
# ---------------------------------------------------------------------------


class TestLocalMetrics:
    def test_load_empty_dir(self, tmp_path):
        with patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            result = _load_all_local_metrics()
            assert result == []

    def test_load_nonexistent_dir(self):
        with patch("langfuse_dashboards.METRICS_DIR", "/nonexistent/path"):
            result = _load_all_local_metrics()
            assert result == []

    def test_load_metrics_files(self, tmp_path):
        # Create sample metrics files
        for i in range(3):
            data = {
                "session_id": f"session-{i}",
                "query": f"test query {i}",
                "started_at": f"2026-03-{15+i}T10:00:00Z",
                "total_duration_secs": 100 + i * 10,
                "quality": {
                    "total_conditions": 5 + i,
                    "avg_condition_confidence": 0.6 + i * 0.05,
                },
                "llm_calls": {"total_calls": 10 + i, "summary_by_model": {}},
                "tool_calls": {"total_calls": 20 + i, "summary_by_tool": {}},
                "sources": {"unique_domains": ["a.com", "b.org"], "domain_count": 2},
                "recommendations": [],
            }
            (tmp_path / f"session-{i}.json").write_text(json.dumps(data))

        with patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            result = _load_all_local_metrics()
            assert len(result) == 3

    def test_aggregate_empty(self, tmp_path):
        with patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            agg = aggregate_local_metrics()
            assert agg["total_sessions"] == 0
            assert agg["sessions"] == []

    def test_aggregate_with_data(self, tmp_path):
        data = {
            "session_id": "sess-1",
            "query": "how to test",
            "started_at": "2026-03-15T10:00:00Z",
            "total_duration_secs": 120,
            "quality": {
                "total_conditions": 8,
                "avg_condition_confidence": 0.72,
            },
            "llm_calls": {
                "total_calls": 15,
                "summary_by_model": {
                    "mistral-small": {"count": 10, "total_duration_secs": 30},
                    "mistral-large": {"count": 5, "total_duration_secs": 20},
                },
            },
            "tool_calls": {
                "total_calls": 25,
                "summary_by_tool": {
                    "searxng_search": {"count": 15, "errors": 1, "total_duration_secs": 10},
                    "fetch_webpage": {"count": 10, "errors": 3, "total_duration_secs": 8},
                },
            },
            "sources": {"unique_domains": ["a.com", "b.org"], "domain_count": 2},
            "recommendations": [
                {"category": "latency", "severity": "medium", "message": "slow"},
            ],
        }
        (tmp_path / "sess-1.json").write_text(json.dumps(data))

        with patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            agg = aggregate_local_metrics()
            assert agg["total_sessions"] == 1
            assert agg["avg_duration_secs"] == 120.0
            assert agg["avg_conditions"] == 8.0
            assert agg["avg_confidence"] == 0.72
            assert agg["total_llm_calls"] == 15
            assert agg["total_tool_calls"] == 25
            assert "mistral-small" in agg["model_usage"]
            assert agg["model_usage"]["mistral-small"]["calls"] == 10
            assert "searxng_search" in agg["tool_usage"]
            assert agg["tool_usage"]["searxng_search"]["errors"] == 1
            assert agg["recommendations_summary"]["latency"] == 1
            assert len(agg["sessions"]) == 1
            assert agg["sessions"][0]["session_id"] == "sess-1"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_safe_escapes_html(self):
        assert _safe("<script>alert('xss')</script>") == (
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
            if "&#x27;" in _safe("'")
            else "&lt;script&gt;alert('xss')&lt;/script&gt;"
        )
        assert "&lt;" in _safe("<b>")
        assert "&amp;" in _safe("&")

    def test_format_duration_seconds(self):
        assert _format_duration(45) == "45s"

    def test_format_duration_minutes(self):
        assert _format_duration(150) == "2.5m"

    def test_format_duration_hours(self):
        assert _format_duration(7200) == "2.0h"

    def test_format_cost_small(self):
        assert _format_cost(0.001) == "$0.0010"

    def test_format_cost_normal(self):
        assert _format_cost(1.50) == "$1.50"


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------


class TestRenderDashboard:
    def test_renders_html_without_langfuse(self, tmp_path):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", ""), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""), \
             patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            html = render_dashboard_html(days=7)
            assert "<!DOCTYPE html>" in html
            assert "Deep Search Portal" in html
            assert "Not configured" in html
            assert "Observability" in html

    def test_renders_html_with_local_data(self, tmp_path):
        data = {
            "session_id": "sess-abc",
            "query": "test query for dashboard",
            "started_at": "2026-03-15T10:00:00Z",
            "total_duration_secs": 90,
            "quality": {
                "total_conditions": 5,
                "avg_condition_confidence": 0.65,
            },
            "llm_calls": {"total_calls": 8, "summary_by_model": {}},
            "tool_calls": {"total_calls": 12, "summary_by_tool": {}},
            "sources": {"unique_domains": ["test.com"], "domain_count": 1},
            "recommendations": [],
        }
        (tmp_path / "sess-abc.json").write_text(json.dumps(data))

        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", ""), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""), \
             patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            html = render_dashboard_html(days=7)
            assert "sess-abc" in html
            assert "test query" in html
            assert "1" in html  # total sessions

    @patch("langfuse_dashboards._langfuse_query")
    def test_renders_langfuse_sections_when_configured(self, mock_query, tmp_path):
        mock_query.return_value = [
            {"providedModelName": "gpt-4", "count_count": "10",
             "totalTokens_sum": "5000", "totalCost_sum": "0.15"},
        ]

        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", "pk-123"), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", "sk-456"), \
             patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            html = render_dashboard_html(days=7)
            assert "Connected" in html
            assert "Langfuse Metrics" in html

    def test_days_parameter_affects_output(self, tmp_path):
        with patch("langfuse_dashboards.LANGFUSE_PUBLIC_KEY", ""), \
             patch("langfuse_dashboards.LANGFUSE_SECRET_KEY", ""), \
             patch("langfuse_dashboards.METRICS_DIR", str(tmp_path)):
            html_7 = render_dashboard_html(days=7)
            html_30 = render_dashboard_html(days=30)
            assert "Last 7 days" in html_7
            assert "Last 30 days" in html_30
