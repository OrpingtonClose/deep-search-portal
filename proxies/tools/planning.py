"""Research planning: decompose queries into angles, AoT reflection.

Extracted from persistent_deep_research_proxy.py lines 4358-4526.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .config import SUBAGENT_MODEL, MAX_SUBAGENTS
from .models import AtomicCondition
from .llm import call_llm

log = logging.getLogger("persistent-research")

# ============================================================================
# Planning Agent
# ============================================================================

PLANNING_PROMPT = """You are a research planning agent. Your job is to decompose a user's question into distinct research angles that can be investigated independently and in parallel.

Given the user's query, produce a JSON object with exactly this structure:
{
  "angles": [
    {"title": "short angle title", "query": "specific search query for this angle", "description": "what this angle investigates"},
    ...
  ],
  "bridge_queries": [
    {"query": "cross-domain search query", "domains": ["domain1", "domain2"], "rationale": "why this unexpected connection might be useful"}
  ]
}

Rules:
1. Generate 3-7 angles covering: factual/technical, historical/context, contrarian/alternative views, practical/applied, and recent developments.
2. Generate 0-2 bridge queries ONLY if a genuinely useful cross-domain insight exists. Do NOT force connections — if none are natural, output an empty array. Bridge queries must still directly help answer the user's original question.
3. Each angle should be independent enough to research separately.
4. Make search queries specific and actionable — they must be queries a human would type to answer the original question.
5. STAY ON TOPIC: Every angle and bridge query must serve the user's actual intent. If the user asks about buying X, research buying X — do not research side effects, alternative uses, or tangential associations of X.
6. Output ONLY valid JSON, no markdown fences or commentary."""


async def plan_research(
    user_query: str,
    prior_conditions: list[dict],
    graph_neighbors: list[dict],
    req_id: str,
) -> dict:
    """Use the small model to decompose the query into research angles."""
    messages = [{"role": "system", "content": PLANNING_PROMPT}]

    user_content = f"User query: {user_query}"
    if prior_conditions:
        prior_text = "\n".join(
            f"- {c['fact']} [from prior research on: {c['original_query']}]"
            for c in prior_conditions[:10]
        )
        user_content += f"\n\nPrior knowledge from previous research sessions:\n{prior_text}"
        user_content += "\n\nConsider these prior findings when planning angles. Avoid redundant research."

    if graph_neighbors:
        graph_text = "\n".join(
            f"- {g['fact']} (via entity: {g.get('via_entity', '?')})"
            for g in graph_neighbors[:5]
        )
        user_content += f"\n\nRelated entities from knowledge graph:\n{graph_text}"

    messages.append({"role": "user", "content": user_content})

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=2048, temperature=0.4)

    if "error" in result:
        log.error(f"[{req_id}] Planning agent error: {result['error']}")
        return {
            "angles": [{"title": "General research", "query": user_query, "description": "Direct research"}],
            "bridge_queries": [],
        }

    content = result.get("content", "")

    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        plan = json.loads(cleaned)

        angles = plan.get("angles", [])
        bridge_queries = plan.get("bridge_queries", [])

        if not angles:
            raise ValueError("No angles in plan")

        angles = angles[:MAX_SUBAGENTS]

        for bq in bridge_queries[:3]:
            if len(angles) < MAX_SUBAGENTS + 3:
                domains = bq.get("domains", ["?", "?"])
                d1 = domains[0] if len(domains) > 0 else "?"
                d2 = domains[1] if len(domains) > 1 else "?"
                angles.append({
                    "title": f"Bridge: {d1} x {d2}",
                    "query": bq.get("query", ""),
                    "description": bq.get("rationale", "Cross-domain exploration"),
                    "is_bridge": True,
                })

        return {"angles": angles, "bridge_queries": bridge_queries}

    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[{req_id}] Planning agent returned invalid JSON: {e}, content={content[:200]}")
        return {
            "angles": [
                {"title": "General research", "query": user_query, "description": "Direct research on the topic"},
                {"title": "Recent developments", "query": f"{user_query} recent news 2024 2025", "description": "Latest developments"},
                {"title": "Expert analysis", "query": f"{user_query} expert analysis review", "description": "Expert perspectives"},
                {"title": "Academic research", "query": f"{user_query} research paper study", "description": "Academic sources"},
            ],
            "bridge_queries": [],
        }


# ============================================================================
# AoT Reflection Mechanism
# ============================================================================

AOT_REFLECTION_PROMPT = """You are an AoT (Atom of Thoughts) reflection agent. Evaluate the quality of the following research decomposition and conditions.

Check for:
1. Missing parallel relationships (false dependencies between conditions)
2. Unnecessary complexity (conditions that overlap significantly)
3. Non-atomic conditions (statements that need further decomposition)
4. Poor contraction quality (conditions that don't reduce complexity)

Output ONLY a JSON object:
{
  "quality_score": 0.8,
  "issues": [
    {"type": "overlap", "indices": [0, 2], "description": "These conditions say essentially the same thing"},
    {"type": "non_atomic", "index": 4, "description": "This condition contains multiple claims"},
    {"type": "missing_angle", "description": "No conditions cover X perspective"}
  ],
  "should_redecompose": false,
  "suggested_queries": ["additional search query if gaps found"]
}

Output ONLY valid JSON, no markdown fences."""


async def reflect_on_conditions(
    conditions: list[AtomicCondition],
    user_query: str,
    req_id: str,
) -> dict:
    """Validate decomposition quality and suggest improvements."""
    if not conditions:
        return {"quality_score": 0.0, "issues": [], "should_redecompose": True, "suggested_queries": [user_query]}

    conditions_text = "\n".join(
        f"{i}. [{c.angle}] {c.fact} (confidence: {c.confidence:.1f})"
        for i, c in enumerate(conditions)
    )

    messages = [
        {"role": "system", "content": AOT_REFLECTION_PROMPT},
        {"role": "user", "content": (
            f"Original query: {user_query}\n\n"
            f"Current atomic conditions:\n{conditions_text}"
        )},
    ]

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=1024, temperature=0.1)

    if "error" in result:
        return {"quality_score": 0.5, "issues": [], "should_redecompose": False, "suggested_queries": []}

    content = result.get("content", "")
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"quality_score": 0.5, "issues": [], "should_redecompose": False, "suggested_queries": []}

