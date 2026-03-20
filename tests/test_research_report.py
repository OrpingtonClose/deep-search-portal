"""Tests for research_report.py — HTML report generation."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

from research_report import (
    generate_report,
    save_report,
    save_metrics_json,
    _esc,
    _mini_svg_line_chart,
    _section_executive_summary,
    _section_timeline,
    _section_final_answer,
    _section_findings_by_angle,
    _section_source_quality,
    _section_subagent_performance,
    _section_novelty_curves,
    _section_tool_usage,
    _section_llm_performance,
    _section_efficiency,
    _section_cost,
    _section_warnings,
    _section_conditions_table,
    _section_progress_log,
    _section_raw_metrics,
)


def _sample_metrics():
    """Minimal valid metrics dict matching ResearchMetrics.to_dict() shape."""
    return {
        "session_id": "test-sess-001",
        "query": "What is the capital of France?",
        "started_at": "2026-03-18T10:00:00+00:00",
        "finished_at": "2026-03-18T10:02:30+00:00",
        "total_duration_secs": 150.0,
        "pipeline": {
            "node_timings": [
                {"node_name": "retrieve", "duration_secs": 5.2},
                {"node_name": "plan", "duration_secs": 3.1},
                {"node_name": "subagents", "duration_secs": 120.0},
                {"node_name": "synthesize", "duration_secs": 8.5},
            ],
            "slowest_node": "subagents",
            "slowest_node_duration": 120.0,
        },
        "llm_calls": {
            "total_calls": 15,
            "records": [
                {"call_id": "c1", "model": "mistral-small", "duration_secs": 1.2, "total_tokens_est": 500, "error": "", "node_context": "plan"},
            ],
            "summary_by_model": {
                "mistral-small": {"count": 10, "total_duration": 12.5, "avg_duration": 1.25, "total_tokens_est": 5000, "errors": 0},
                "mistral-large": {"count": 5, "total_duration": 8.0, "avg_duration": 1.6, "total_tokens_est": 3000, "errors": 1},
            },
        },
        "tool_calls": {
            "total_calls": 25,
            "records": [
                {"tool_name": "searxng_search", "duration_secs": 0.8, "error": "", "node_context": "subagents"},
            ],
            "summary_by_tool": {
                "searxng_search": {"count": 15, "total_duration": 12.0, "avg_duration": 0.8, "errors": 2, "total_result_chars": 50000},
                "fetch_webpage": {"count": 10, "total_duration": 20.0, "avg_duration": 2.0, "errors": 3, "total_result_chars": 200000},
            },
        },
        "subagents": {
            "count": 3,
            "records": [
                {"index": 0, "angle": "Historical context", "turns_used": 5, "tool_calls_made": 8, "conditions_found": 6, "novelty_history": [1.0, 0.8, 0.5, 0.2], "children_spawned": 1, "error": "", "duration_secs": 40.0},
                {"index": 1, "angle": "Geographic details", "turns_used": 3, "tool_calls_made": 5, "conditions_found": 4, "novelty_history": [1.0, 0.6], "children_spawned": 0, "error": "", "duration_secs": 30.0},
                {"index": 2, "angle": "Economic impact", "turns_used": 2, "tool_calls_made": 3, "conditions_found": 2, "novelty_history": [1.0], "children_spawned": 0, "error": "timeout", "duration_secs": 50.0},
            ],
            "summary": {
                "count": 3, "total_turns": 10, "total_tool_calls": 16,
                "total_conditions": 12, "avg_turns_per_agent": 3.3,
                "avg_conditions_per_agent": 4.0, "total_children_spawned": 1,
                "agents_with_errors": 1,
            },
        },
        "quality": {
            "total_conditions": 12,
            "conditions_by_angle": {"Historical context": 6, "Geographic details": 4, "Economic impact": 2},
            "confidence_distribution": {"high_0.7_plus": 5, "medium_0.4_to_0.7": 4, "low_below_0.4": 3},
            "trust_distribution": {"academic_gov_0.9": 2, "news_wiki_0.6_0.8": 5, "forum_social_0.3_0.5": 3, "default_0.5": 2},
            "serendipitous_findings": 2,
            "reflection_quality_score": 0.75,
            "reflection_issues": ["Some sources may be outdated"],
            "avg_condition_confidence": 0.62,
        },
        "sources": {
            "unique_domains": ["wikipedia.org", "bbc.co.uk", "arxiv.org", "reddit.com"],
            "domain_count": 4,
            "diversity_score": 0.85,
        },
        "cost": {
            "session_total_usd": 0.15,
            "monthly_total_usd": 2.50,
            "monthly_budget_usd": 100.0,
            "breakdown": {"bright_data": 0.10, "apify": 0.05},
        },
        "efficiency": {
            "conditions_per_tool_call": 0.48,
            "avg_tool_call_duration_secs": 1.28,
            "avg_llm_call_duration_secs": 1.37,
            "saturation_curve": [1.0, 0.8, 0.5, 0.2, 1.0, 0.6, 1.0],
        },
        "recommendations": [
            {"category": "quality", "severity": "warning", "message": "30% of conditions have low confidence", "evidence": "3/12 below 0.4"},
            {"category": "reliability", "severity": "info", "message": "Tool error rate is 20%", "evidence": "5/25 calls failed"},
        ],
    }


def _sample_conditions():
    return [
        {"fact": "Paris is the capital of France", "confidence": 0.95, "trust_score": 0.9, "angle": "Historical context", "source_url": "https://wikipedia.org/wiki/Paris", "is_serendipitous": False},
        {"fact": "Paris has been the capital since 987 AD", "confidence": 0.85, "trust_score": 0.8, "angle": "Historical context", "source_url": "https://bbc.co.uk/history", "is_serendipitous": False},
        {"fact": "The GDP of the Paris region is 700 billion EUR", "confidence": 0.6, "trust_score": 0.7, "angle": "Economic impact", "source_url": "https://arxiv.org/abs/econ", "is_serendipitous": True},
    ]


# ---------------------------------------------------------------------------
# Escape
# ---------------------------------------------------------------------------

class TestEscape:
    def test_basic_escape(self):
        assert _esc("<script>") == "&lt;script&gt;"

    def test_ampersand(self):
        assert _esc("A & B") == "A &amp; B"

    def test_quotes(self):
        assert _esc('"hello"') == "&quot;hello&quot;"


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------

class TestSections:
    def test_executive_summary_contains_query(self):
        html = _section_executive_summary(
            "sess-1", "test query", "2026-01-01", "2026-01-01",
            150.0, {"total_conditions": 5, "avg_condition_confidence": 0.7},
            {"count": 3}, {"total_calls": 10}, {"total_calls": 20},
            {"domain_count": 4, "diversity_score": 0.8},
        )
        assert "test query" in html
        assert "sess-1" in html
        assert "150.0s" in html

    def test_timeline_with_nodes(self):
        timings = [
            {"node_name": "retrieve", "duration_secs": 5.0},
            {"node_name": "plan", "duration_secs": 3.0},
        ]
        html = _section_timeline(timings, 8.0)
        assert "retrieve" in html
        assert "plan" in html
        assert "Pipeline Timeline" in html

    def test_timeline_empty(self):
        assert _section_timeline([], 0) == ""

    def test_final_answer(self):
        html = _section_final_answer("This is the answer.\n\nWith paragraphs.")
        assert "This is the answer." in html
        assert "Final Answer" in html

    def test_findings_by_angle(self):
        conditions = _sample_conditions()
        html = _section_findings_by_angle(conditions)
        assert "Historical context" in html
        assert "Economic impact" in html
        assert "Paris is the capital" in html

    def test_findings_empty(self):
        assert _section_findings_by_angle([]) == ""

    def test_source_quality(self):
        metrics = _sample_metrics()
        html = _section_source_quality(metrics["quality"], metrics["sources"])
        assert "Source Quality" in html
        assert "wikipedia.org" in html

    def test_subagent_performance(self):
        html = _section_subagent_performance(_sample_metrics()["subagents"])
        assert "Subagent Performance" in html
        assert "Historical context" in html
        assert "timeout" in html

    def test_subagent_performance_empty(self):
        assert _section_subagent_performance({"records": [], "summary": {}}) == ""

    def test_novelty_curves(self):
        html = _section_novelty_curves(_sample_metrics()["subagents"])
        assert "Novelty Curves" in html
        assert "<svg" in html

    def test_novelty_curves_no_data(self):
        assert _section_novelty_curves({"records": [{"novelty_history": []}]}) == ""

    def test_tool_usage(self):
        html = _section_tool_usage(_sample_metrics()["tool_calls"])
        assert "Tool Usage" in html
        assert "searxng_search" in html

    def test_llm_performance(self):
        html = _section_llm_performance(_sample_metrics()["llm_calls"])
        assert "LLM Performance" in html
        assert "mistral-small" in html

    def test_efficiency(self):
        metrics = _sample_metrics()
        html = _section_efficiency(metrics["efficiency"], metrics["recommendations"])
        assert "Efficiency" in html

    def test_cost(self):
        html = _section_cost(_sample_metrics()["cost"])
        assert "Cost" in html

    def test_warnings(self):
        html = _section_warnings(["Censorship detected", "Low confidence"])
        assert "Censorship detected" in html

    def test_conditions_table(self):
        html = _section_conditions_table(_sample_conditions())
        assert "Paris is the capital" in html
        assert "<table" in html

    def test_progress_log(self):
        html = _section_progress_log(["Step 1 done", "Step 2 done"])
        assert "Step 1 done" in html

    def test_raw_metrics(self):
        html = _section_raw_metrics(_sample_metrics())
        assert "Raw Metrics" in html


# ---------------------------------------------------------------------------
# SVG Chart
# ---------------------------------------------------------------------------

class TestSvgChart:
    def test_generates_svg(self):
        svg = _mini_svg_line_chart([1.0, 0.8, 0.5, 0.2], "Test")
        assert "<svg" in svg
        assert "polyline" in svg
        assert "Test" in svg

    def test_empty_values(self):
        assert _mini_svg_line_chart([], "Empty") == ""

    def test_single_value(self):
        svg = _mini_svg_line_chart([0.5], "Single")
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Full Report Generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_full_report_is_valid_html(self):
        metrics = _sample_metrics()
        conditions = _sample_conditions()
        html = generate_report(
            metrics, conditions,
            final_answer="Paris is indeed the capital of France.",
            progress_log=["Started research", "Completed synthesis"],
        )
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "test-sess-001" in html
        assert "Paris is indeed the capital" in html

    def test_report_contains_all_sections(self):
        metrics = _sample_metrics()
        conditions = _sample_conditions()
        html = generate_report(metrics, conditions)
        assert "Executive Summary" in html
        assert "Pipeline Timeline" in html
        assert "Findings by Research Angle" in html
        assert "Source Quality" in html
        assert "Subagent Performance" in html
        assert "Novelty Curves" in html
        assert "Tool Usage" in html
        assert "LLM Performance" in html
        assert "Raw Metrics" in html

    def test_report_self_contained(self):
        """Report should have no external CSS/JS references."""
        metrics = _sample_metrics()
        html = generate_report(metrics, _sample_conditions())
        assert "<style>" in html
        # No external stylesheet links
        assert 'rel="stylesheet"' not in html
        assert "https://cdn" not in html

    def test_report_empty_metrics(self):
        """Should not crash with minimal/empty metrics."""
        metrics = {
            "session_id": "empty",
            "query": "test",
            "started_at": "",
            "finished_at": "",
            "total_duration_secs": 0,
            "pipeline": {"node_timings": [], "slowest_node": "", "slowest_node_duration": 0},
            "llm_calls": {"total_calls": 0, "records": [], "summary_by_model": {}},
            "tool_calls": {"total_calls": 0, "records": [], "summary_by_tool": {}},
            "subagents": {"count": 0, "records": [], "summary": {}},
            "quality": {"total_conditions": 0, "conditions_by_angle": {}, "confidence_distribution": {}, "trust_distribution": {}, "serendipitous_findings": 0, "reflection_quality_score": 0, "reflection_issues": [], "avg_condition_confidence": 0},
            "sources": {"unique_domains": [], "domain_count": 0, "diversity_score": 0},
            "cost": {},
            "efficiency": {"conditions_per_tool_call": 0, "avg_tool_call_duration_secs": 0, "avg_llm_call_duration_secs": 0, "saturation_curve": []},
            "recommendations": [],
        }
        html = generate_report(metrics, [])
        assert "<!DOCTYPE html>" in html

    def test_xss_protection(self):
        """Malicious input should be escaped."""
        metrics = _sample_metrics()
        metrics["query"] = '<script>alert("xss")</script>'
        html = generate_report(metrics, [])
        assert '<script>alert("xss")</script>' not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Save Report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_save_and_read(self, tmp_path):
        with patch("research_report.REPORTS_DIR", str(tmp_path)):
            html = "<html><body>Test</body></html>"
            path = save_report(html, "save-test-1")
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == html

    def test_save_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "reports")
        with patch("research_report.REPORTS_DIR", new_dir):
            path = save_report("<html/>", "save-test-2")
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Save Metrics JSON
# ---------------------------------------------------------------------------

class TestSaveMetricsJson:
    def test_save_and_read(self, tmp_path):
        with patch("research_report.REPORTS_DIR", str(tmp_path)):
            json_str = '{"session_id": "test-1", "query": "hello"}'
            path = save_metrics_json(json_str, "test-1")
            assert os.path.exists(path)
            assert path.endswith("_metrics.json")
            with open(path) as f:
                assert f.read() == json_str

    def test_save_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "metrics")
        with patch("research_report.REPORTS_DIR", new_dir):
            path = save_metrics_json("{}", "test-2")
            assert os.path.exists(path)
