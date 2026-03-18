"""Research Metrics Collector — LangGraph-native instrumentation for performance analysis.

Uses LangGraph's callback ecosystem (langchain_core.callbacks.BaseCallbackHandler)
to capture comprehensive metrics from every node execution, LLM call, and tool
invocation in the persistent research pipeline.  All metrics are structured for
LLM consumption — an AI performance analyst can read the JSON output and
recommend concrete improvements to the research pipeline.

Collected data:
  - Per-node wall-clock timing (start/end/duration)
  - Per-LLM-call latency, token estimates, model used
  - Per-tool-call timing, success/failure, result size
  - Per-subagent statistics (turns, conditions, novelty curve)
  - Quality signals (reflection score, confidence distribution, trust tiers)
  - Source diversity metrics (unique domains, trust distribution)
  - Cost data (if social media scrapers used)
  - Computed aggregates and heuristic recommendations
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("research_metrics")

METRICS_DIR = os.getenv(
    "RESEARCH_METRICS_DIR",
    "/opt/persistent_research_logs/metrics",
)
REPORTS_DIR = os.getenv(
    "RESEARCH_REPORTS_DIR",
    "/opt/persistent_research_logs/reports",
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NodeTiming:
    """Timing for a single LangGraph node execution."""
    node_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0

    def finish(self) -> None:
        self.end_time = time.monotonic()
        self.duration_secs = round(self.end_time - self.start_time, 4)


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    call_id: str = ""
    model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    total_tokens_est: int = 0
    error: str = ""
    node_context: str = ""  # which pipeline node triggered this call

    def finish(self, response_text: str = "") -> None:
        self.end_time = time.monotonic()
        self.duration_secs = round(self.end_time - self.start_time, 4)
        # Rough token estimate: ~4 chars per token
        if response_text:
            self.completion_tokens_est = max(1, len(response_text) // 4)
        self.total_tokens_est = self.prompt_tokens_est + self.completion_tokens_est


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    tool_name: str = ""
    call_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    result_size_chars: int = 0
    error: str = ""
    arguments_summary: str = ""  # first 200 chars of serialised args
    node_context: str = ""

    def finish(self, result: str = "") -> None:
        self.end_time = time.monotonic()
        self.duration_secs = round(self.end_time - self.start_time, 4)
        self.result_size_chars = len(result) if result else 0


@dataclass
class SubagentMetrics:
    """Metrics for a single subagent's research run."""
    index: int = 0
    angle: str = ""
    turns_used: int = 0
    tool_calls_made: int = 0
    conditions_found: int = 0
    novelty_history: list[float] = field(default_factory=list)
    children_spawned: int = 0
    error: str = ""
    duration_secs: float = 0.0


@dataclass
class ResearchMetrics:
    """Complete metrics for a single research session.

    Designed to be serialised to JSON and consumed by an LLM analyst.
    """
    # --- Identity ---
    session_id: str = ""
    query: str = ""
    started_at: str = ""  # ISO 8601
    finished_at: str = ""
    total_duration_secs: float = 0.0

    # --- Pipeline Node Timings ---
    node_timings: list[dict] = field(default_factory=list)  # serialised NodeTiming

    # --- LLM Calls ---
    llm_calls: list[dict] = field(default_factory=list)
    llm_call_summary: dict = field(default_factory=dict)
    # {model: {count, total_duration, avg_duration, total_tokens_est}}

    # --- Tool Calls ---
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_summary: dict = field(default_factory=dict)
    # {tool_name: {count, total_duration, avg_duration, errors, total_result_chars}}

    # --- Subagent Stats ---
    subagent_metrics: list[dict] = field(default_factory=list)
    subagent_summary: dict = field(default_factory=dict)

    # --- Quality Metrics ---
    total_conditions: int = 0
    conditions_by_angle: dict = field(default_factory=dict)
    confidence_distribution: dict = field(default_factory=dict)
    # {"high_0.7_plus": N, "medium_0.4_to_0.7": N, "low_below_0.4": N}
    trust_distribution: dict = field(default_factory=dict)
    # {"academic_0.9": N, "news_0.6_0.7": N, "forum_0.3": N, "default_0.5": N}
    serendipitous_findings: int = 0
    reflection_quality_score: float = 0.0
    reflection_issues: list[str] = field(default_factory=list)

    # --- Source Diversity ---
    unique_source_domains: list[str] = field(default_factory=list)
    source_diversity_score: float = 0.0  # 0-1, based on domain variety

    # --- Cost ---
    cost_data: dict = field(default_factory=dict)

    # --- Computed Aggregates ---
    research_efficiency: float = 0.0  # conditions per tool call
    avg_condition_confidence: float = 0.0
    avg_tool_call_duration: float = 0.0
    avg_llm_call_duration: float = 0.0
    saturation_curve: list[float] = field(default_factory=list)  # novelty over time

    # --- Heuristic Recommendations ---
    recommendations: list[dict] = field(default_factory=list)
    # [{category, severity, message, evidence}]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialisation."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_secs": self.total_duration_secs,
            "pipeline": {
                "node_timings": self.node_timings,
                "slowest_node": max(self.node_timings, key=lambda n: n.get("duration_secs", 0), default={}).get("node_name", ""),
                "slowest_node_duration": max((n.get("duration_secs", 0) for n in self.node_timings), default=0),
            },
            "llm_calls": {
                "total_calls": len(self.llm_calls),
                "records": self.llm_calls,
                "summary_by_model": self.llm_call_summary,
            },
            "tool_calls": {
                "total_calls": len(self.tool_calls),
                "records": self.tool_calls,
                "summary_by_tool": self.tool_call_summary,
            },
            "subagents": {
                "count": len(self.subagent_metrics),
                "records": self.subagent_metrics,
                "summary": self.subagent_summary,
            },
            "quality": {
                "total_conditions": self.total_conditions,
                "conditions_by_angle": self.conditions_by_angle,
                "confidence_distribution": self.confidence_distribution,
                "trust_distribution": self.trust_distribution,
                "serendipitous_findings": self.serendipitous_findings,
                "reflection_quality_score": self.reflection_quality_score,
                "reflection_issues": self.reflection_issues,
                "avg_condition_confidence": self.avg_condition_confidence,
            },
            "sources": {
                "unique_domains": self.unique_source_domains,
                "domain_count": len(self.unique_source_domains),
                "diversity_score": self.source_diversity_score,
            },
            "cost": self.cost_data,
            "efficiency": {
                "conditions_per_tool_call": self.research_efficiency,
                "avg_tool_call_duration_secs": self.avg_tool_call_duration,
                "avg_llm_call_duration_secs": self.avg_llm_call_duration,
                "saturation_curve": self.saturation_curve,
            },
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Metrics Collector (session-scoped, mutable)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates metrics during a research session.

    Usage:
        collector = MetricsCollector(session_id, query)
        collector.start_node("retrieve")
        ... do work ...
        collector.end_node("retrieve")
        collector.record_llm_call(...)
        collector.record_tool_call(...)
        ...
        metrics = collector.finalise()
    """

    def __init__(self, session_id: str, query: str) -> None:
        self.session_id = session_id
        self.query = query
        self._start_time = time.monotonic()
        self._started_at = datetime.now(timezone.utc).isoformat()

        # Active node timings
        self._active_nodes: dict[str, NodeTiming] = {}
        self._completed_nodes: list[NodeTiming] = []

        # LLM call tracking
        self._active_llm_calls: dict[str, LLMCallRecord] = {}
        self._completed_llm_calls: list[LLMCallRecord] = []

        # Tool call tracking
        self._active_tool_calls: dict[str, ToolCallRecord] = {}
        self._completed_tool_calls: list[ToolCallRecord] = []

        # Subagent metrics (populated externally)
        self._subagent_metrics: list[SubagentMetrics] = []

        # Quality data (populated during pipeline)
        self._conditions: list[dict] = []  # {fact, confidence, trust_score, angle, source_url, is_serendipitous}
        self._reflection: dict = {}
        self._cost_data: dict = {}

        # Current node context for associating LLM/tool calls
        self._current_node: str = ""

    # --- Node Timing ---

    def start_node(self, node_name: str) -> None:
        """Record the start of a LangGraph node execution."""
        timing = NodeTiming(node_name=node_name, start_time=time.monotonic())
        self._active_nodes[node_name] = timing
        self._current_node = node_name
        log.debug(f"[{self.session_id}] Node started: {node_name}")

    def end_node(self, node_name: str) -> None:
        """Record the end of a LangGraph node execution."""
        timing = self._active_nodes.pop(node_name, None)
        if timing:
            timing.finish()
            self._completed_nodes.append(timing)
            log.debug(
                f"[{self.session_id}] Node finished: {node_name} "
                f"({timing.duration_secs:.2f}s)"
            )
        if self._current_node == node_name:
            self._current_node = ""

    # --- LLM Call Tracking ---

    def start_llm_call(
        self,
        call_id: str,
        model: str = "",
        prompt_tokens_est: int = 0,
    ) -> None:
        """Record the start of an LLM API call."""
        record = LLMCallRecord(
            call_id=call_id,
            model=model,
            start_time=time.monotonic(),
            prompt_tokens_est=prompt_tokens_est,
            node_context=self._current_node,
        )
        self._active_llm_calls[call_id] = record

    def end_llm_call(
        self,
        call_id: str,
        response_text: str = "",
        error: str = "",
    ) -> None:
        """Record the end of an LLM API call."""
        record = self._active_llm_calls.pop(call_id, None)
        if record:
            record.finish(response_text)
            record.error = error
            self._completed_llm_calls.append(record)

    # --- Tool Call Tracking ---

    def start_tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: str = "",
    ) -> None:
        """Record the start of a tool invocation."""
        record = ToolCallRecord(
            tool_name=tool_name,
            call_id=call_id,
            start_time=time.monotonic(),
            arguments_summary=arguments[:200] if arguments else "",
            node_context=self._current_node,
        )
        self._active_tool_calls[call_id] = record

    def end_tool_call(
        self,
        call_id: str,
        result: str = "",
        error: str = "",
    ) -> None:
        """Record the end of a tool invocation."""
        record = self._active_tool_calls.pop(call_id, None)
        if record:
            record.finish(result)
            record.error = error
            self._completed_tool_calls.append(record)

    # --- Subagent Metrics ---

    def add_subagent_metrics(self, metrics: SubagentMetrics) -> None:
        """Add metrics for a completed subagent."""
        self._subagent_metrics.append(metrics)

    # --- Quality Data ---

    def set_conditions(self, conditions: list[dict]) -> None:
        """Set the final list of research conditions for quality analysis."""
        self._conditions = conditions

    def set_reflection(self, reflection: dict) -> None:
        """Set the reflection result for quality analysis."""
        self._reflection = reflection

    def set_cost_data(self, cost_data: dict) -> None:
        """Set cost tracking data."""
        self._cost_data = cost_data

    # --- Finalisation ---

    def finalise(self) -> ResearchMetrics:
        """Compute all aggregates and return the final ResearchMetrics."""
        end_time = time.monotonic()
        finished_at = datetime.now(timezone.utc).isoformat()
        total_duration = round(end_time - self._start_time, 4)

        metrics = ResearchMetrics(
            session_id=self.session_id,
            query=self.query,
            started_at=self._started_at,
            finished_at=finished_at,
            total_duration_secs=total_duration,
        )

        # Node timings
        metrics.node_timings = [
            {"node_name": nt.node_name, "duration_secs": nt.duration_secs}
            for nt in self._completed_nodes
        ]

        # LLM call records + summary
        metrics.llm_calls = [
            {
                "call_id": r.call_id,
                "model": r.model,
                "duration_secs": r.duration_secs,
                "prompt_tokens_est": r.prompt_tokens_est,
                "completion_tokens_est": r.completion_tokens_est,
                "total_tokens_est": r.total_tokens_est,
                "error": r.error,
                "node_context": r.node_context,
            }
            for r in self._completed_llm_calls
        ]
        llm_by_model: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "total_duration": 0.0, "total_tokens_est": 0, "errors": 0}
        )
        for r in self._completed_llm_calls:
            model_key = r.model or "unknown"
            llm_by_model[model_key]["count"] += 1
            llm_by_model[model_key]["total_duration"] += r.duration_secs
            llm_by_model[model_key]["total_tokens_est"] += r.total_tokens_est
            if r.error:
                llm_by_model[model_key]["errors"] += 1
        for model_key, stats in llm_by_model.items():
            stats["avg_duration"] = round(stats["total_duration"] / max(stats["count"], 1), 4)
            stats["total_duration"] = round(stats["total_duration"], 4)
        metrics.llm_call_summary = dict(llm_by_model)

        # Tool call records + summary
        metrics.tool_calls = [
            {
                "tool_name": r.tool_name,
                "call_id": r.call_id,
                "duration_secs": r.duration_secs,
                "result_size_chars": r.result_size_chars,
                "error": r.error,
                "arguments_summary": r.arguments_summary,
                "node_context": r.node_context,
            }
            for r in self._completed_tool_calls
        ]
        tool_by_name: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "total_duration": 0.0, "errors": 0, "total_result_chars": 0}
        )
        for r in self._completed_tool_calls:
            key = r.tool_name or "unknown"
            tool_by_name[key]["count"] += 1
            tool_by_name[key]["total_duration"] += r.duration_secs
            tool_by_name[key]["total_result_chars"] += r.result_size_chars
            if r.error:
                tool_by_name[key]["errors"] += 1
        for key, stats in tool_by_name.items():
            stats["avg_duration"] = round(stats["total_duration"] / max(stats["count"], 1), 4)
            stats["total_duration"] = round(stats["total_duration"], 4)
        metrics.tool_call_summary = dict(tool_by_name)

        # Subagent metrics
        metrics.subagent_metrics = [
            {
                "index": sm.index,
                "angle": sm.angle,
                "turns_used": sm.turns_used,
                "tool_calls_made": sm.tool_calls_made,
                "conditions_found": sm.conditions_found,
                "novelty_history": sm.novelty_history,
                "children_spawned": sm.children_spawned,
                "error": sm.error,
                "duration_secs": sm.duration_secs,
            }
            for sm in self._subagent_metrics
        ]
        if self._subagent_metrics:
            total_sa_turns = sum(sm.turns_used for sm in self._subagent_metrics)
            total_sa_tools = sum(sm.tool_calls_made for sm in self._subagent_metrics)
            total_sa_conditions = sum(sm.conditions_found for sm in self._subagent_metrics)
            total_sa_children = sum(sm.children_spawned for sm in self._subagent_metrics)
            metrics.subagent_summary = {
                "count": len(self._subagent_metrics),
                "total_turns": total_sa_turns,
                "total_tool_calls": total_sa_tools,
                "total_conditions": total_sa_conditions,
                "total_children_spawned": total_sa_children,
                "avg_turns_per_agent": round(total_sa_turns / len(self._subagent_metrics), 2),
                "avg_conditions_per_agent": round(total_sa_conditions / len(self._subagent_metrics), 2),
                "agents_with_errors": sum(1 for sm in self._subagent_metrics if sm.error),
            }

        # Quality metrics from conditions
        self._compute_quality_metrics(metrics)

        # Source diversity
        self._compute_source_diversity(metrics)

        # Cost data
        metrics.cost_data = self._cost_data

        # Computed aggregates
        total_tool_calls = len(self._completed_tool_calls)
        metrics.research_efficiency = round(
            metrics.total_conditions / max(total_tool_calls, 1), 4
        )
        if self._completed_tool_calls:
            metrics.avg_tool_call_duration = round(
                sum(r.duration_secs for r in self._completed_tool_calls) / len(self._completed_tool_calls), 4
            )
        if self._completed_llm_calls:
            metrics.avg_llm_call_duration = round(
                sum(r.duration_secs for r in self._completed_llm_calls) / len(self._completed_llm_calls), 4
            )

        # Saturation curve from subagent novelty histories
        all_novelty: list[float] = []
        for sm in self._subagent_metrics:
            all_novelty.extend(sm.novelty_history)
        metrics.saturation_curve = all_novelty

        # Heuristic recommendations
        metrics.recommendations = self._generate_recommendations(metrics)

        return metrics

    def _compute_quality_metrics(self, metrics: ResearchMetrics) -> None:
        """Compute quality metrics from conditions."""
        conditions = self._conditions
        metrics.total_conditions = len(conditions)

        if not conditions:
            return

        # By angle
        by_angle: dict[str, int] = defaultdict(int)
        for c in conditions:
            angle = c.get("angle", "unknown")
            by_angle[angle] += 1
        metrics.conditions_by_angle = dict(by_angle)

        # Confidence distribution
        high = sum(1 for c in conditions if c.get("confidence", 0) >= 0.7)
        medium = sum(1 for c in conditions if 0.4 <= c.get("confidence", 0) < 0.7)
        low = sum(1 for c in conditions if c.get("confidence", 0) < 0.4)
        metrics.confidence_distribution = {
            "high_0.7_plus": high,
            "medium_0.4_to_0.7": medium,
            "low_below_0.4": low,
        }
        metrics.avg_condition_confidence = round(
            sum(c.get("confidence", 0.5) for c in conditions) / len(conditions), 4
        )

        # Trust distribution
        academic = sum(1 for c in conditions if c.get("trust_score", 0.5) >= 0.9)
        news = sum(1 for c in conditions if 0.6 <= c.get("trust_score", 0.5) < 0.9)
        forum = sum(1 for c in conditions if c.get("trust_score", 0.5) < 0.4)
        default = len(conditions) - academic - news - forum
        metrics.trust_distribution = {
            "academic_gov_0.9": academic,
            "news_wiki_0.6_0.8": news,
            "forum_social_below_0.4": forum,
            "default_0.4_0.6": default,
        }

        # Serendipitous findings
        metrics.serendipitous_findings = sum(
            1 for c in conditions if c.get("is_serendipitous", False)
        )

        # Reflection
        metrics.reflection_quality_score = self._reflection.get("quality_score", 0.0)
        metrics.reflection_issues = [
            f"[{issue.get('type', '?')}] {issue.get('description', '')}"
            for issue in self._reflection.get("issues", [])
        ]

    def _compute_source_diversity(self, metrics: ResearchMetrics) -> None:
        """Compute source diversity from condition URLs."""
        from urllib.parse import urlparse

        domains: set[str] = set()
        for c in self._conditions:
            url = c.get("source_url", "")
            if url:
                try:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        # Normalise: remove www. prefix
                        domain = parsed.netloc.lower()
                        if domain.startswith("www."):
                            domain = domain[4:]
                        domains.add(domain)
                except Exception:
                    pass

        metrics.unique_source_domains = sorted(domains)
        # Diversity score: log-scaled domain count relative to conditions
        n_domains = len(domains)
        n_conditions = max(len(self._conditions), 1)
        if n_domains == 0:
            metrics.source_diversity_score = 0.0
        else:
            import math
            # Score: ratio of domains to conditions, log-smoothed
            raw = n_domains / n_conditions
            metrics.source_diversity_score = round(min(1.0, raw * math.log2(n_domains + 1)), 4)

    def _generate_recommendations(self, metrics: ResearchMetrics) -> list[dict]:
        """Generate heuristic recommendations based on metrics."""
        recs: list[dict] = []

        # Low confidence warning
        low_pct = metrics.confidence_distribution.get("low_below_0.4", 0)
        total = max(metrics.total_conditions, 1)
        if low_pct / total > 0.3:
            recs.append({
                "category": "quality",
                "severity": "warning",
                "message": (
                    f"{low_pct}/{total} conditions ({low_pct/total*100:.0f}%) have low confidence (<0.4). "
                    f"Consider increasing MAX_SUBAGENT_TURNS or adding more verification passes."
                ),
                "evidence": f"low_confidence_count={low_pct}, total={total}",
            })

        # Low source diversity
        if metrics.source_diversity_score < 0.2 and metrics.total_conditions > 5:
            recs.append({
                "category": "diversity",
                "severity": "warning",
                "message": (
                    f"Source diversity is low ({metrics.source_diversity_score:.2f}). "
                    f"Only {len(metrics.unique_source_domains)} unique domains for {metrics.total_conditions} conditions. "
                    f"Consider adding more search engines or broadening query terms."
                ),
                "evidence": f"diversity_score={metrics.source_diversity_score}, domains={len(metrics.unique_source_domains)}",
            })

        # Saturation not reached
        if metrics.saturation_curve and all(n > 0.3 for n in metrics.saturation_curve[-3:]):
            recs.append({
                "category": "coverage",
                "severity": "info",
                "message": (
                    "Research did not saturate — novelty remained high through the end. "
                    "There may be more information available. Consider increasing MAX_SUBAGENT_TURNS."
                ),
                "evidence": f"last_3_novelty={metrics.saturation_curve[-3:]}",
            })

        # Excessive tool errors
        total_errors = sum(
            stats.get("errors", 0) for stats in metrics.tool_call_summary.values()
        )
        total_calls = len(metrics.tool_calls)
        if total_calls > 0 and total_errors / total_calls > 0.2:
            recs.append({
                "category": "reliability",
                "severity": "error",
                "message": (
                    f"High tool error rate: {total_errors}/{total_calls} calls failed "
                    f"({total_errors/total_calls*100:.0f}%). Check SearXNG connectivity "
                    f"and webpage fetch reliability."
                ),
                "evidence": f"tool_errors={total_errors}, total_calls={total_calls}",
            })

        # Slow LLM calls
        for model, stats in metrics.llm_call_summary.items():
            avg_dur = stats.get("avg_duration", 0)
            if avg_dur > 15.0:
                recs.append({
                    "category": "performance",
                    "severity": "warning",
                    "message": (
                        f"LLM calls to {model} are slow (avg {avg_dur:.1f}s). "
                        f"Consider using a faster model for subagent work or reducing max_tokens."
                    ),
                    "evidence": f"model={model}, avg_duration={avg_dur}",
                })

        # Low research efficiency
        if metrics.research_efficiency < 0.1 and total_calls > 10:
            recs.append({
                "category": "efficiency",
                "severity": "warning",
                "message": (
                    f"Low research efficiency: {metrics.research_efficiency:.3f} conditions per tool call. "
                    f"Many tool calls may be returning irrelevant results. "
                    f"Consider improving query decomposition or search query quality."
                ),
                "evidence": f"efficiency={metrics.research_efficiency}, tool_calls={total_calls}",
            })

        # Reflection quality
        if metrics.reflection_quality_score > 0 and metrics.reflection_quality_score < 0.5:
            recs.append({
                "category": "quality",
                "severity": "warning",
                "message": (
                    f"Reflection quality score is low ({metrics.reflection_quality_score:.2f}). "
                    f"Issues: {'; '.join(metrics.reflection_issues[:3])}. "
                    f"The research decomposition may need improvement."
                ),
                "evidence": f"quality_score={metrics.reflection_quality_score}",
            })

        # Subagent failures
        failed_agents = sum(1 for sm in self._subagent_metrics if sm.error)
        if failed_agents > 0:
            recs.append({
                "category": "reliability",
                "severity": "warning",
                "message": (
                    f"{failed_agents}/{len(self._subagent_metrics)} subagents encountered errors. "
                    f"Check logs for details."
                ),
                "evidence": f"failed_agents={failed_agents}",
            })

        # No serendipitous findings (if bridge queries were used)
        if metrics.serendipitous_findings == 0 and metrics.total_conditions > 10:
            recs.append({
                "category": "exploration",
                "severity": "info",
                "message": (
                    "No serendipitous/cross-domain findings. The research may be too "
                    "narrowly focused. Consider adding bridge queries in the planning phase."
                ),
                "evidence": "serendipitous_findings=0",
            })

        return recs


# ---------------------------------------------------------------------------
# LangGraph Callback Handler
# ---------------------------------------------------------------------------

class ResearchMetricsCallback(BaseCallbackHandler):
    """LangChain callback handler that feeds events into a MetricsCollector.

    Attach this to the LangGraph config to automatically capture LLM and tool
    call metrics:

        config = {"callbacks": [ResearchMetricsCallback(collector)]}
    """

    def __init__(self, collector: MetricsCollector) -> None:
        super().__init__()
        self.collector = collector

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = serialized.get("kwargs", {}).get("model", "") or serialized.get("id", [""])[-1]
        # Estimate prompt tokens from prompt text
        total_chars = sum(len(p) for p in prompts)
        prompt_tokens_est = max(1, total_chars // 4)
        self.collector.start_llm_call(
            call_id=str(run_id),
            model=model,
            prompt_tokens_est=prompt_tokens_est,
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        text = ""
        try:
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text += getattr(gen, "text", "")
        except Exception:
            pass
        self.collector.end_llm_call(call_id=str(run_id), response_text=text)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.collector.end_llm_call(call_id=str(run_id), error=str(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "") or serialized.get("id", [""])[-1]
        self.collector.start_tool_call(
            call_id=str(run_id),
            tool_name=tool_name,
            arguments=input_str,
        )

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        self.collector.end_tool_call(call_id=str(run_id), result=str(output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.collector.end_tool_call(call_id=str(run_id), error=str(error))


# ---------------------------------------------------------------------------
# Persistence (save/load metrics JSON)
# ---------------------------------------------------------------------------

def save_metrics(metrics: ResearchMetrics) -> str:
    """Save metrics JSON to disk. Returns the file path."""
    try:
        Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(f"Failed to create metrics dir {METRICS_DIR}: {e}")

    path = os.path.join(METRICS_DIR, f"{metrics.session_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)
        log.info(f"Saved metrics to {path}")
    except Exception as e:
        log.error(f"Failed to save metrics: {e}")
    return path


def load_metrics(session_id: str) -> Optional[dict]:
    """Load metrics JSON from disk. Returns None if not found."""
    path = os.path.join(METRICS_DIR, f"{session_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log.error(f"Failed to load metrics for {session_id}: {e}")
        return None


def list_available_reports() -> list[dict]:
    """List available report/metrics sessions."""
    results: list[dict] = []
    try:
        metrics_dir = Path(METRICS_DIR)
        if not metrics_dir.exists():
            return results
        for f in sorted(metrics_dir.glob("*.json"), reverse=True):
            session_id = f.stem
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results.append({
                    "session_id": session_id,
                    "query": data.get("query", ""),
                    "started_at": data.get("started_at", ""),
                    "total_duration_secs": data.get("total_duration_secs", 0),
                    "total_conditions": data.get("quality", {}).get("total_conditions", 0),
                })
            except Exception:
                results.append({"session_id": session_id})
    except Exception as e:
        log.error(f"Failed to list reports: {e}")
    return results
