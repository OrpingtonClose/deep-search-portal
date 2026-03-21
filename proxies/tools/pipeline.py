"""LangGraph state definition, pipeline nodes, and graph builder.

Extracted from persistent_deep_research_proxy.py lines 6200-6693.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from research_metrics import MetricsCollector, SubagentMetrics, save_metrics
import research_report
import social_media_scrapers

from .config import (
    UPSTREAM_MODEL,
    MAX_PRIOR_CONDITIONS,
    PORTAL_PUBLIC_URL,
    VERITAS_VERIFY_ENABLED,
    VERITAS_MIN_CONDITIONS,
    RESEARCH_NAMESPACE,
)
from .models import AtomicCondition, SubagentResult
from .heartbeat import LiveFindingsCollector
from .persistence import (
    _log_conditions_jsonl,
    _log_entities_jsonl,
    _retrieve_related,
    _retrieve_graph_neighbors,
    _store_conditions_neo4j,
    _store_entities_neo4j,
)
from .verification import (
    extract_entities_from_conditions,
    verify_conditions,
    verify_conditions_with_veritas,
)
from .planning import reflect_on_conditions
from .tree_reactor import tree_research_reactor
from .synthesis import synthesize_with_revision, relevance_gate
from .search_tools import tool_searxng_search

log = logging.getLogger("persistent-research")

# Per-request state registries (shared with main proxy)
_live_collectors: dict[str, LiveFindingsCollector] = {}
_curated_queues: dict[str, asyncio.Queue] = {}
_metrics_collectors: dict[str, MetricsCollector] = {}

# ============================================================================
# LangGraph State & Pipeline Graph
# ============================================================================


def _pdr_append_log(left: list[str], right: list[str]) -> list[str]:
    """Reducer: append new progress messages to the log."""
    return left + right


class PersistentResearchState(TypedDict):
    """LangGraph state for the persistent deep research pipeline."""
    req_id: str
    user_query: str
    start_time: float
    # Phase outputs
    prior_conditions: list[dict]
    graph_neighbors: list[dict]
    subagent_results: list  # list[SubagentResult] (not TypedDict-serialisable)
    all_conditions: list  # list[AtomicCondition]
    total_turns: int
    total_tools: int
    total_children: int
    nodes_explored: int  # tree reactor: how many nodes were explored
    reflection: dict
    final_answer: str
    # Progress
    progress_log: Annotated[list[str], _pdr_append_log]
    phase: str  # current phase name or "done"
    # Report URLs (populated at end of pipeline)
    report_url: str
    metrics_url: str


async def pdr_node_retrieve(state: PersistentResearchState) -> dict:
    """Phase 1: Retrieve prior knowledge from Neo4j."""
    user_query = state["user_query"]
    req_id = state["req_id"]
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("retrieve")
    progress: list[str] = ["**[Phase 1: Retrieving Prior Knowledge]**\n"]

    query_entities = [w for w in user_query.split() if len(w) > 3][:5]

    prior_conditions, graph_neighbors = await asyncio.gather(
        _retrieve_related(user_query, MAX_PRIOR_CONDITIONS),
        _retrieve_graph_neighbors(query_entities, max_hops=2, limit=10),
    )

    if prior_conditions:
        progress.append(f"Found {len(prior_conditions)} relevant prior findings:\n")
        for pc in prior_conditions[:5]:
            progress.append(f"  - {pc['fact'][:100]}...\n")
        if len(prior_conditions) > 5:
            progress.append(f"  ... and {len(prior_conditions) - 5} more\n")
    else:
        progress.append("No prior knowledge found via text search.\n")

    if graph_neighbors:
        progress.append(f"Found {len(graph_neighbors)} related findings via knowledge graph:\n")
        for gn in graph_neighbors[:3]:
            progress.append(f"  - {gn['fact'][:80]}... (via entity: {gn.get('via_entity', '?')})\n")
    else:
        progress.append("No graph neighbors found.\n")

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("retrieve")

    return {
        "prior_conditions": prior_conditions,
        "graph_neighbors": graph_neighbors,
        "progress_log": progress,
        "phase": "tree_research",
    }


async def pdr_node_tree_research(state: PersistentResearchState) -> dict:
    """Phase 2: Tree-based research reactor.

    Replaces the old plan-angles + parallel-subagents phases with a
    tree exploration that starts from the user query, researches it,
    and spawns focused sub-questions from each finding.
    """
    req_id = state["req_id"]
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("tree_research")

    # Get or create the live findings collector
    collector = _live_collectors.get(req_id)
    if collector is None:
        collector = LiveFindingsCollector(user_query=state["user_query"])
        _live_collectors[req_id] = collector

    # Get or create the curated queue
    curated_queue = _curated_queues.get(req_id)
    if curated_queue is None:
        curated_queue = asyncio.Queue()
        _curated_queues[req_id] = curated_queue

    result = await tree_research_reactor(
        user_query=state["user_query"],
        prior_conditions=state["prior_conditions"],
        graph_neighbors=state["graph_neighbors"],
        req_id=req_id,
        collector=collector,
        curated_queue=curated_queue,
    )

    # Record subagent metrics
    subagent_results = result["subagent_results"]
    mc = _metrics_collectors.get(req_id)
    if mc:
        for i, sr in enumerate(subagent_results):
            mc.add_subagent_metrics(SubagentMetrics(
                index=i,
                angle=sr.angle,
                turns_used=sr.turns_used,
                tool_calls_made=sr.tool_calls_made,
                conditions_found=len(sr.conditions),
                novelty_history=sr.novelty_history,
                children_spawned=sr.spawned_children,
                error=sr.error,
            ))
        mc.end_node("tree_research")

    return {
        "subagent_results": result["subagent_results"],
        "all_conditions": result["all_conditions"],
        "total_turns": result["total_turns"],
        "total_tools": result["total_tools"],
        "total_children": result["total_children"],
        "nodes_explored": len(result["subagent_results"]),
        "progress_log": result["progress_log"],
        "phase": "entities",
    }


async def pdr_node_entities(state: PersistentResearchState) -> dict:
    """Phase 4: Entity extraction + knowledge graph update."""
    req_id = state["req_id"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("entities")
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("entities")
    all_conditions = state["all_conditions"]
    progress: list[str] = []

    if all_conditions:
        progress.append("\n**[Phase 4: Knowledge Graph Update]**\n")
        progress.append("Extracting entities and relationships...\n")

        entities, relationships = await extract_entities_from_conditions(all_conditions, req_id)

        if entities or relationships:
            _log_entities_jsonl(req_id, entities, relationships)
            ent_stored, rel_stored = await _store_entities_neo4j(req_id, entities, relationships)
            progress.append(
                f"Extracted {len(entities)} entities, {len(relationships)} relationships. "
                f"Stored {ent_stored} new entities, {rel_stored} new edges.\n"
            )
        else:
            progress.append("No entities extracted.\n")

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("entities")

    return {"progress_log": progress, "phase": "verify"}


async def pdr_node_verify(state: PersistentResearchState) -> dict:
    """Phase 5: Citation verification.

    Two-stage verification (anti-hallucination, pro-speculation):
      1. Self-evaluation (fast, LLM-only): cross-checks conditions against each
         other for contradictions, source quality, and fabricated entities.
      2. Veritas Inquisitor (thorough, web-search-backed): runs the full 5-agent
         reactor to decompose claims, gather external evidence, debate, and
         produce verdicts.  Only fabricated conditions are removed;
         speculative findings are kept and labeled.
    """
    req_id = state["req_id"]
    user_query = state["user_query"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("verify")
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("verify")
    all_conditions = list(state["all_conditions"])
    progress: list[str] = []
    pre_count = len(all_conditions)

    if all_conditions and len(all_conditions) >= 2:
        # Stage 1: fast self-evaluation
        progress.append("\n**[Phase 5a: Citation Cross-Check]**\n")
        progress.append("Cross-checking claims for contradictions...\n")

        all_conditions = await verify_conditions(all_conditions, req_id)

        stage1_removed = pre_count - len(all_conditions)
        high_conf = sum(1 for c in all_conditions if c.confidence >= 0.7)
        low_conf = sum(1 for c in all_conditions if c.confidence < 0.4)
        summary = (f"Cross-check complete: {high_conf} high-confidence, "
                   f"{low_conf} low-confidence conditions.")
        if stage1_removed > 0:
            summary += f" {stage1_removed} fabricated removed."
        progress.append(summary + "\n")

    # Stage 2: Veritas Inquisitor — external evidence-based verification
    veritas_report: dict = {}
    if VERITAS_VERIFY_ENABLED and len(all_conditions) >= VERITAS_MIN_CONDITIONS:
        progress.append("\n**[Phase 5b: Veritas Fact-Check]**\n")
        progress.append(
            f"Running Veritas Inquisitor on {len(all_conditions)} conditions "
            f"(5-agent swarm with web search)...\n"
        )

        pre_veritas_count = len(all_conditions)
        all_conditions, veritas_report = await verify_conditions_with_veritas(
            all_conditions, user_query, req_id,
        )

        removed = pre_veritas_count - len(all_conditions)
        speculative_count = sum(
            1 for c in all_conditions if c.verification_status == "speculative"
        )
        verified_count = sum(
            1 for c in all_conditions if c.verification_status == "verified"
        )
        overall_score = veritas_report.get("overall_score", -1)
        halluc_prob = veritas_report.get("overall_hallucination_probability", -1)

        summary_parts = []
        if removed > 0:
            summary_parts.append(f"{removed} fabricated claim{'s' if removed != 1 else ''} removed")
        if speculative_count > 0:
            summary_parts.append(f"{speculative_count} speculative (kept)")
        if verified_count > 0:
            summary_parts.append(f"{verified_count} verified")
        if overall_score >= 0:
            summary_parts.append(f"truthfulness {overall_score:.0%}")
        if halluc_prob >= 0:
            summary_parts.append(f"fabrication probability {halluc_prob:.0%}")

        if summary_parts:
            progress.append(f"Veritas: {', '.join(summary_parts)}.\n")
        else:
            progress.append("Veritas verification complete.\n")

        progress.append(
            f"{len(all_conditions)} conditions retained out of {pre_veritas_count}.\n"
        )

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("verify")

    return {"all_conditions": all_conditions, "progress_log": progress, "phase": "reflect"}


async def pdr_node_reflect(state: PersistentResearchState) -> dict:
    """Phase 6: AoT Reflection."""
    req_id = state["req_id"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("reflect")
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("reflect")
    all_conditions = list(state["all_conditions"])
    user_query = state["user_query"]
    progress: list[str] = []
    reflection: dict = {}

    if all_conditions:
        progress.append("\n**[Phase 6: AoT Reflection]**\n")
        reflection = await reflect_on_conditions(all_conditions, user_query, req_id)
        quality = reflection.get("quality_score", 0.5)
        issues = reflection.get("issues", [])
        progress.append(f"Decomposition quality: {quality:.1f}/1.0\n")
        if issues:
            progress.append(f"Issues found: {len(issues)}\n")
            for issue in issues[:3]:
                progress.append(f"  - [{issue.get('type', '?')}] {issue.get('description', '')[:100]}\n")

        suggested = reflection.get("suggested_queries", [])
        if quality < 0.5 and suggested:
            progress.append("Quality below threshold -- running targeted additional research...\n")
            extra_results = await asyncio.gather(
                *[tool_searxng_search(q) for q in suggested[:2]],
                return_exceptions=True,
            )
            for q, sr in zip(suggested[:2], extra_results):
                if isinstance(sr, str) and not sr.startswith("Search error"):
                    all_conditions.append(AtomicCondition(
                        fact=f"[Reflection gap fill] {sr[:300]}",
                        angle="reflection",
                        confidence=0.4,
                    ))

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.set_reflection(reflection)
        mc.end_node("reflect")

    return {
        "all_conditions": all_conditions,
        "reflection": reflection,
        "progress_log": progress,
        "phase": "persist",
    }


async def pdr_node_persist(state: PersistentResearchState) -> dict:
    """Phase 7: Persist findings to Neo4j + JSONL."""
    req_id = state["req_id"]
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("persist")
    user_query = state["user_query"]
    all_conditions = state["all_conditions"]
    progress: list[str] = []

    if all_conditions:
        progress.append("\n**[Phase 7: Persisting Knowledge]**\n")
        _log_conditions_jsonl(req_id, user_query, all_conditions)
        stored = await _store_conditions_neo4j(req_id, user_query, all_conditions)
        progress.append(f"Stored {stored} conditions to persistent knowledge base.\n")

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("persist")

    return {"progress_log": progress, "phase": "synthesize"}


async def pdr_node_synthesize(state: PersistentResearchState) -> dict:
    """Final phase: Draft-Synthesis-Revision loop."""
    req_id = state["req_id"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("synthesize")
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("synthesize")
    progress: list[str] = [
        f"\n**[Synthesis Phase]** (model: {UPSTREAM_MODEL})\n",
        "Generating draft synthesis...\n",
    ]

    final_answer = await synthesize_with_revision(
        state["user_query"], state["subagent_results"], state["prior_conditions"], req_id,
    )

    # Relevance gate: check if the final answer actually addresses the query
    is_relevant = await relevance_gate(final_answer, state["user_query"], req_id)
    if not is_relevant:
        log.warning(f"[{req_id}] Final answer failed relevance gate — re-running synthesis")
        final_answer = await synthesize_with_revision(
            state["user_query"], state["subagent_results"], state["prior_conditions"], req_id,
        )

    progress.append("Critic review complete.\n")
    progress.append("Final revision complete.\n")

    elapsed = time.monotonic() - state["start_time"]
    nodes_explored = state.get("nodes_explored", 0)
    all_conditions = state["all_conditions"]
    total_children = state["total_children"]

    progress.append(
        f"\nResearch complete in {elapsed:.1f}s "
        f"({len(all_conditions)} conditions from {nodes_explored} tree nodes"
    )
    if total_children > 0:
        progress.append(f" + {total_children} recursive sub-explorations")
    progress.append(")\n")

    # Generate report + metrics
    report_url = ""
    metrics_url = ""
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("synthesize")

        # Feed conditions into metrics collector
        condition_dicts = [
            {
                "fact": c.fact,
                "source_url": c.source_url,
                "confidence": c.confidence,
                "angle": c.angle,
                "trust_score": c.trust_score,
                "is_serendipitous": c.is_serendipitous,
                "serendipity_score_val": c.serendipity_score_val,
            }
            for c in all_conditions
        ]
        mc.set_conditions(condition_dicts)

        # Try to get cost data from social media scrapers
        try:
            from social_media_scrapers import cost_tracker
            if cost_tracker:
                mc.set_cost_data({
                    "session_total": cost_tracker.session_total(req_id),
                    "monthly_total": cost_tracker.monthly_total(),
                })
        except Exception:
            pass

        # Finalise metrics
        metrics_obj = mc.finalise()
        metrics_dict = metrics_obj.to_dict()
        save_metrics(metrics_obj)

        # Generate Markdown report (user-readable)
        try:
            md_report = research_report.generate_report(
                metrics=metrics_dict,
                conditions=condition_dicts,
                final_answer=final_answer,
                progress_log=list(state.get("progress_log", [])),
            )
            research_report.save_report(md_report, req_id)

            # Save metrics JSON alongside report
            metrics_json = json.dumps(metrics_dict, indent=2, default=str)
            research_report.save_metrics_json(metrics_json, req_id)

            # Build portal URLs for the report and metrics
            base = PORTAL_PUBLIC_URL
            if not base:
                log.warning(
                    "[%s] PORTAL_PUBLIC_URL not set — report links will be relative",
                    req_id,
                )
                base = ""
            report_url = f"{base}/research/report/{req_id}"
            metrics_url = f"{base}/research/metrics/{req_id}"
            log.info(f"[{req_id}] Report available at: {report_url}")
        except Exception as e:
            log.error(f"[{req_id}] Failed to generate report: {e}")

    # Append report link to progress if available
    if report_url:
        progress.append(f"\n**Report published:** {report_url}\n")
    if metrics_url:
        progress.append(f"**Metrics published:** {metrics_url}\n")

    return {
        "final_answer": final_answer,
        "progress_log": progress,
        "phase": "done",
        "report_url": report_url,
        "metrics_url": metrics_url,
    }


def build_persistent_research_graph() -> Any:
    """Build the persistent research LangGraph.

    Graph topology (tree reactor pipeline)::

        START -> retrieve -> tree_research -> entities -> verify
              -> reflect -> persist -> synthesize -> END
    """
    graph = StateGraph(PersistentResearchState)

    graph.add_node("retrieve", pdr_node_retrieve)
    graph.add_node("tree_research", pdr_node_tree_research)
    graph.add_node("entities", pdr_node_entities)
    graph.add_node("verify", pdr_node_verify)
    graph.add_node("reflect", pdr_node_reflect)
    graph.add_node("persist", pdr_node_persist)
    graph.add_node("synthesize", pdr_node_synthesize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "tree_research")
    graph.add_edge("tree_research", "entities")
    graph.add_edge("entities", "verify")
    graph.add_edge("verify", "reflect")
    graph.add_edge("reflect", "persist")
    graph.add_edge("persist", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


