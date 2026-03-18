"""Tests for research_metrics.py — metrics collector, callback handler, persistence."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Ensure proxies/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

from research_metrics import (
    LLMCallRecord,
    MetricsCollector,
    NodeTiming,
    ResearchMetrics,
    ResearchMetricsCallback,
    SubagentMetrics,
    ToolCallRecord,
    list_available_reports,
    load_metrics,
    save_metrics,
)


# ---------------------------------------------------------------------------
# NodeTiming
# ---------------------------------------------------------------------------

class TestNodeTiming:
    def test_finish_sets_duration(self):
        nt = NodeTiming(node_name="retrieve", start_time=time.monotonic() - 1.5)
        nt.finish()
        assert nt.duration_secs > 0
        assert nt.end_time > nt.start_time

    def test_finish_duration_is_positive(self):
        nt = NodeTiming(node_name="plan", start_time=time.monotonic())
        time.sleep(0.01)
        nt.finish()
        assert nt.duration_secs >= 0.01


# ---------------------------------------------------------------------------
# LLMCallRecord
# ---------------------------------------------------------------------------

class TestLLMCallRecord:
    def test_finish_with_response(self):
        rec = LLMCallRecord(call_id="c1", model="mistral-small", start_time=time.monotonic() - 0.5)
        rec.finish(response_text="Hello world, this is a response from the model.")
        assert rec.duration_secs > 0
        assert rec.completion_tokens_est > 0
        assert rec.total_tokens_est == rec.prompt_tokens_est + rec.completion_tokens_est

    def test_finish_empty_response(self):
        rec = LLMCallRecord(call_id="c2", start_time=time.monotonic())
        rec.finish(response_text="")
        assert rec.completion_tokens_est == 0


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    def test_finish_with_result(self):
        rec = ToolCallRecord(tool_name="searxng_search", start_time=time.monotonic() - 0.2)
        rec.finish(result="Some search results here with content")
        assert rec.duration_secs > 0
        assert rec.result_size_chars == len("Some search results here with content")

    def test_finish_no_result(self):
        rec = ToolCallRecord(tool_name="fetch_webpage", start_time=time.monotonic())
        rec.finish()
        assert rec.result_size_chars == 0


# ---------------------------------------------------------------------------
# MetricsCollector — Core Operations
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_init(self):
        mc = MetricsCollector(session_id="sess-1", query="test query")
        assert mc.session_id == "sess-1"
        assert mc.query == "test query"

    def test_node_timing(self):
        mc = MetricsCollector(session_id="sess-2", query="q")
        mc.start_node("retrieve")
        time.sleep(0.01)
        mc.end_node("retrieve")
        metrics = mc.finalise()
        assert len(metrics.node_timings) == 1
        assert metrics.node_timings[0]["node_name"] == "retrieve"
        assert metrics.node_timings[0]["duration_secs"] >= 0.01

    def test_multiple_nodes(self):
        mc = MetricsCollector(session_id="sess-3", query="q")
        for node in ["retrieve", "plan", "subagents", "synthesize"]:
            mc.start_node(node)
            mc.end_node(node)
        metrics = mc.finalise()
        assert len(metrics.node_timings) == 4
        names = [nt["node_name"] for nt in metrics.node_timings]
        assert names == ["retrieve", "plan", "subagents", "synthesize"]

    def test_llm_call_tracking(self):
        mc = MetricsCollector(session_id="sess-4", query="q")
        mc.start_llm_call("call-1", model="mistral-small", prompt_tokens_est=100)
        mc.end_llm_call("call-1", response_text="A" * 400)  # ~100 tokens
        metrics = mc.finalise()
        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["model"] == "mistral-small"
        assert metrics.llm_calls[0]["total_tokens_est"] > 0
        assert "mistral-small" in metrics.llm_call_summary
        assert metrics.llm_call_summary["mistral-small"]["count"] == 1

    def test_tool_call_tracking(self):
        mc = MetricsCollector(session_id="sess-5", query="q")
        mc.start_tool_call("tc-1", "searxng_search", arguments='{"query": "test"}')
        mc.end_tool_call("tc-1", result="results here")
        mc.start_tool_call("tc-2", "searxng_search", arguments='{"query": "test2"}')
        mc.end_tool_call("tc-2", result="more results")
        mc.start_tool_call("tc-3", "fetch_webpage", arguments='{"url": "http://x.com"}')
        mc.end_tool_call("tc-3", error="HTTP 404")
        metrics = mc.finalise()
        assert len(metrics.tool_calls) == 3
        assert metrics.tool_call_summary["searxng_search"]["count"] == 2
        assert metrics.tool_call_summary["fetch_webpage"]["errors"] == 1

    def test_subagent_metrics(self):
        mc = MetricsCollector(session_id="sess-6", query="q")
        mc.add_subagent_metrics(SubagentMetrics(
            index=0, angle="Angle 1", turns_used=5, tool_calls_made=10,
            conditions_found=8, novelty_history=[1.0, 0.8, 0.4, 0.1],
            children_spawned=1, error="", duration_secs=30.0,
        ))
        mc.add_subagent_metrics(SubagentMetrics(
            index=1, angle="Angle 2", turns_used=3, tool_calls_made=6,
            conditions_found=4, novelty_history=[1.0, 0.5],
            children_spawned=0, error="timeout", duration_secs=15.0,
        ))
        metrics = mc.finalise()
        assert len(metrics.subagent_metrics) == 2
        assert metrics.subagent_summary["count"] == 2
        assert metrics.subagent_summary["total_turns"] == 8
        assert metrics.subagent_summary["agents_with_errors"] == 1

    def test_node_context_propagation(self):
        mc = MetricsCollector(session_id="sess-7", query="q")
        mc.start_node("subagents")
        mc.start_llm_call("lc-1", model="m")
        mc.end_llm_call("lc-1")
        mc.start_tool_call("tc-1", "search")
        mc.end_tool_call("tc-1")
        mc.end_node("subagents")
        metrics = mc.finalise()
        assert metrics.llm_calls[0]["node_context"] == "subagents"
        assert metrics.tool_calls[0]["node_context"] == "subagents"


# ---------------------------------------------------------------------------
# MetricsCollector — Quality Metrics
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def _make_conditions(self):
        return [
            {"fact": "Fact A", "confidence": 0.9, "trust_score": 0.9, "angle": "medical", "source_url": "https://pubmed.ncbi.nlm.nih.gov/123", "is_serendipitous": False},
            {"fact": "Fact B", "confidence": 0.8, "trust_score": 0.7, "angle": "medical", "source_url": "https://bbc.co.uk/news/article", "is_serendipitous": False},
            {"fact": "Fact C", "confidence": 0.3, "trust_score": 0.3, "angle": "forums", "source_url": "https://reddit.com/r/test", "is_serendipitous": False},
            {"fact": "Fact D", "confidence": 0.5, "trust_score": 0.5, "angle": "forums", "source_url": "https://example.com/page", "is_serendipitous": True},
            {"fact": "Fact E", "confidence": 0.6, "trust_score": 0.9, "angle": "academic", "source_url": "https://arxiv.org/abs/2301.00001", "is_serendipitous": False},
        ]

    def test_condition_count(self):
        mc = MetricsCollector(session_id="q-1", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert metrics.total_conditions == 5

    def test_confidence_distribution(self):
        mc = MetricsCollector(session_id="q-2", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert metrics.confidence_distribution["high_0.7_plus"] == 2
        assert metrics.confidence_distribution["medium_0.4_to_0.7"] == 2
        assert metrics.confidence_distribution["low_below_0.4"] == 1

    def test_trust_distribution(self):
        mc = MetricsCollector(session_id="q-3", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert metrics.trust_distribution["academic_gov_0.9"] == 2
        assert metrics.trust_distribution["news_wiki_0.6_0.8"] == 1

    def test_serendipitous_count(self):
        mc = MetricsCollector(session_id="q-4", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert metrics.serendipitous_findings == 1

    def test_source_diversity(self):
        mc = MetricsCollector(session_id="q-5", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert len(metrics.unique_source_domains) == 5
        assert "reddit.com" in metrics.unique_source_domains
        assert metrics.source_diversity_score > 0

    def test_avg_confidence(self):
        mc = MetricsCollector(session_id="q-6", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        expected = (0.9 + 0.8 + 0.3 + 0.5 + 0.6) / 5
        assert abs(metrics.avg_condition_confidence - expected) < 0.01

    def test_conditions_by_angle(self):
        mc = MetricsCollector(session_id="q-7", query="q")
        mc.set_conditions(self._make_conditions())
        metrics = mc.finalise()
        assert metrics.conditions_by_angle["medical"] == 2
        assert metrics.conditions_by_angle["forums"] == 2
        assert metrics.conditions_by_angle["academic"] == 1


# ---------------------------------------------------------------------------
# MetricsCollector — Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_low_confidence_warning(self):
        mc = MetricsCollector(session_id="r-1", query="q")
        # 4 out of 5 conditions are low confidence
        conditions = [{"fact": f"F{i}", "confidence": 0.2, "trust_score": 0.5, "angle": "a", "source_url": ""} for i in range(4)]
        conditions.append({"fact": "F5", "confidence": 0.9, "trust_score": 0.5, "angle": "a", "source_url": ""})
        mc.set_conditions(conditions)
        metrics = mc.finalise()
        quality_recs = [r for r in metrics.recommendations if r["category"] == "quality"]
        assert any("low confidence" in r["message"].lower() for r in quality_recs)

    def test_high_tool_error_rate_warning(self):
        mc = MetricsCollector(session_id="r-2", query="q")
        for i in range(10):
            mc.start_tool_call(f"tc-{i}", "search")
            if i < 3:
                mc.end_tool_call(f"tc-{i}", result="ok")
            else:
                mc.end_tool_call(f"tc-{i}", error="failed")
        metrics = mc.finalise()
        reliability_recs = [r for r in metrics.recommendations if r["category"] == "reliability"]
        assert any("error rate" in r["message"].lower() for r in reliability_recs)

    def test_no_serendipitous_findings_info(self):
        mc = MetricsCollector(session_id="r-3", query="q")
        conditions = [{"fact": f"F{i}", "confidence": 0.7, "trust_score": 0.5, "angle": "a", "source_url": "", "is_serendipitous": False} for i in range(15)]
        mc.set_conditions(conditions)
        metrics = mc.finalise()
        exploration_recs = [r for r in metrics.recommendations if r["category"] == "exploration"]
        assert len(exploration_recs) > 0


# ---------------------------------------------------------------------------
# MetricsCollector — Efficiency
# ---------------------------------------------------------------------------

class TestEfficiency:
    def test_research_efficiency(self):
        mc = MetricsCollector(session_id="e-1", query="q")
        for i in range(5):
            mc.start_tool_call(f"tc-{i}", "search")
            mc.end_tool_call(f"tc-{i}", result="ok")
        conditions = [{"fact": f"F{i}", "confidence": 0.5, "trust_score": 0.5, "angle": "a", "source_url": ""} for i in range(10)]
        mc.set_conditions(conditions)
        metrics = mc.finalise()
        assert metrics.research_efficiency == 10 / 5  # 2.0

    def test_saturation_curve(self):
        mc = MetricsCollector(session_id="e-2", query="q")
        mc.add_subagent_metrics(SubagentMetrics(
            index=0, angle="A", novelty_history=[1.0, 0.8, 0.5, 0.2],
        ))
        mc.add_subagent_metrics(SubagentMetrics(
            index=1, angle="B", novelty_history=[1.0, 0.6],
        ))
        metrics = mc.finalise()
        assert metrics.saturation_curve == [1.0, 0.8, 0.5, 0.2, 1.0, 0.6]


# ---------------------------------------------------------------------------
# ResearchMetrics.to_dict
# ---------------------------------------------------------------------------

class TestMetricsToDict:
    def test_to_dict_structure(self):
        mc = MetricsCollector(session_id="td-1", query="test query")
        mc.start_node("retrieve")
        mc.end_node("retrieve")
        mc.set_conditions([{"fact": "X", "confidence": 0.8, "trust_score": 0.7, "angle": "a", "source_url": "http://example.com"}])
        metrics = mc.finalise()
        d = metrics.to_dict()

        # Top-level keys
        assert "session_id" in d
        assert "query" in d
        assert "pipeline" in d
        assert "llm_calls" in d
        assert "tool_calls" in d
        assert "subagents" in d
        assert "quality" in d
        assert "sources" in d
        assert "efficiency" in d
        assert "recommendations" in d

        # Pipeline sub-keys
        assert "node_timings" in d["pipeline"]
        assert "slowest_node" in d["pipeline"]

    def test_to_dict_is_json_serialisable(self):
        mc = MetricsCollector(session_id="td-2", query="q")
        mc.start_node("plan")
        mc.end_node("plan")
        mc.start_llm_call("lc-1", model="m", prompt_tokens_est=50)
        mc.end_llm_call("lc-1", response_text="response")
        metrics = mc.finalise()
        d = metrics.to_dict()
        serialised = json.dumps(d, default=str)
        assert isinstance(serialised, str)
        assert len(serialised) > 100


# ---------------------------------------------------------------------------
# Persistence — save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        with patch("research_metrics.METRICS_DIR", str(tmp_path)):
            mc = MetricsCollector(session_id="persist-1", query="save test")
            mc.start_node("retrieve")
            mc.end_node("retrieve")
            metrics = mc.finalise()
            path = save_metrics(metrics)
            assert os.path.exists(path)

            loaded = load_metrics("persist-1")
            assert loaded is not None
            assert loaded["session_id"] == "persist-1"
            assert loaded["query"] == "save test"

    def test_load_missing(self, tmp_path):
        with patch("research_metrics.METRICS_DIR", str(tmp_path)):
            loaded = load_metrics("nonexistent")
            assert loaded is None

    def test_list_reports(self, tmp_path):
        with patch("research_metrics.METRICS_DIR", str(tmp_path)):
            # Create two metrics files
            for sid in ["list-1", "list-2"]:
                mc = MetricsCollector(session_id=sid, query=f"query {sid}")
                metrics = mc.finalise()
                save_metrics(metrics)

            reports = list_available_reports()
            assert len(reports) == 2
            session_ids = {r["session_id"] for r in reports}
            assert "list-1" in session_ids
            assert "list-2" in session_ids


# ---------------------------------------------------------------------------
# ResearchMetricsCallback (LangGraph integration)
# ---------------------------------------------------------------------------

class TestResearchMetricsCallback:
    def test_on_llm_start_end(self):
        mc = MetricsCollector(session_id="cb-1", query="q")
        cb = ResearchMetricsCallback(mc)
        run_id = uuid4()

        cb.on_llm_start(
            serialized={"kwargs": {"model": "mistral-small"}, "id": ["ChatMistral"]},
            prompts=["Hello, how are you?"],
            run_id=run_id,
        )
        assert str(run_id) in mc._active_llm_calls

        # Simulate response
        response = MagicMock()
        gen = MagicMock()
        gen.text = "I'm fine, thanks!"
        response.generations = [[gen]]
        cb.on_llm_end(response=response, run_id=run_id)
        assert str(run_id) not in mc._active_llm_calls
        assert len(mc._completed_llm_calls) == 1
        assert mc._completed_llm_calls[0].model == "mistral-small"

    def test_on_llm_error(self):
        mc = MetricsCollector(session_id="cb-2", query="q")
        cb = ResearchMetricsCallback(mc)
        run_id = uuid4()

        cb.on_llm_start(
            serialized={"kwargs": {}, "id": ["ChatMistral"]},
            prompts=["test"],
            run_id=run_id,
        )
        cb.on_llm_error(error=RuntimeError("API timeout"), run_id=run_id)
        assert len(mc._completed_llm_calls) == 1
        assert "API timeout" in mc._completed_llm_calls[0].error

    def test_on_tool_start_end(self):
        mc = MetricsCollector(session_id="cb-3", query="q")
        cb = ResearchMetricsCallback(mc)
        run_id = uuid4()

        cb.on_tool_start(
            serialized={"name": "searxng_search", "id": ["Tool"]},
            input_str='{"query": "test"}',
            run_id=run_id,
        )
        assert str(run_id) in mc._active_tool_calls

        cb.on_tool_end(output="search results", run_id=run_id)
        assert str(run_id) not in mc._active_tool_calls
        assert len(mc._completed_tool_calls) == 1
        assert mc._completed_tool_calls[0].tool_name == "searxng_search"

    def test_on_tool_error(self):
        mc = MetricsCollector(session_id="cb-4", query="q")
        cb = ResearchMetricsCallback(mc)
        run_id = uuid4()

        cb.on_tool_start(
            serialized={"name": "fetch_webpage"},
            input_str="",
            run_id=run_id,
        )
        cb.on_tool_error(error=RuntimeError("HTTP 500"), run_id=run_id)
        assert len(mc._completed_tool_calls) == 1
        assert "HTTP 500" in mc._completed_tool_calls[0].error


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_session(self):
        mc = MetricsCollector(session_id="edge-1", query="q")
        metrics = mc.finalise()
        assert metrics.total_conditions == 0
        assert metrics.research_efficiency == 0
        assert metrics.to_dict()["session_id"] == "edge-1"

    def test_end_node_without_start(self):
        mc = MetricsCollector(session_id="edge-2", query="q")
        mc.end_node("never_started")  # Should not crash
        metrics = mc.finalise()
        assert len(metrics.node_timings) == 0

    def test_end_llm_call_without_start(self):
        mc = MetricsCollector(session_id="edge-3", query="q")
        mc.end_llm_call("nonexistent", response_text="x")
        metrics = mc.finalise()
        assert len(metrics.llm_calls) == 0

    def test_conditions_with_no_urls(self):
        mc = MetricsCollector(session_id="edge-4", query="q")
        mc.set_conditions([
            {"fact": "X", "confidence": 0.5, "trust_score": 0.5, "angle": "a", "source_url": ""},
        ])
        metrics = mc.finalise()
        assert len(metrics.unique_source_domains) == 0
        assert metrics.source_diversity_score == 0.0

    def test_www_prefix_normalisation(self):
        mc = MetricsCollector(session_id="edge-5", query="q")
        mc.set_conditions([
            {"fact": "X", "confidence": 0.5, "trust_score": 0.5, "angle": "a", "source_url": "https://www.example.com/page"},
            {"fact": "Y", "confidence": 0.5, "trust_score": 0.5, "angle": "a", "source_url": "https://example.com/other"},
        ])
        metrics = mc.finalise()
        assert len(metrics.unique_source_domains) == 1
        assert "example.com" in metrics.unique_source_domains
