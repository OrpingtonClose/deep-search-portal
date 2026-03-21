"""Citation verification: self-evaluation and Veritas Inquisitor integration.

Extracted from persistent_deep_research_proxy.py lines 4002-4355.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import veritas_inquisitor

from .config import (
    SUBAGENT_MODEL,
    VERITAS_VERIFY_ENABLED,
    VERITAS_MIN_CONDITIONS,
    VERITAS_HALLUCINATION_THRESHOLD,
)
from .models import AtomicCondition
from .llm import call_llm

log = logging.getLogger("persistent-research")

# ============================================================================
# Entity Extraction (Knowledge Graph)
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from these research findings.

Output ONLY a JSON object:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|concept|technology|place|event|other"}
  ],
  "relationships": [
    {"entity1": "name1", "entity2": "name2", "type": "relationship description", "is_bridge": false}
  ]
}

Rules:
- Extract the most important entities (max 15)
- Identify meaningful relationships between them
- Mark cross-domain relationships as "is_bridge": true
- Output ONLY valid JSON, no markdown fences"""


async def extract_entities_from_conditions(
    conditions: list[AtomicCondition],
    req_id: str,
) -> tuple[list[dict], list[dict]]:
    """Use LLM to extract entities and relationships from atomic conditions."""
    if not conditions:
        return [], []

    conditions_text = "\n".join(
        f"- {c.fact} [angle: {c.angle}, serendipitous: {c.is_serendipitous}]"
        for c in conditions[:30]
    )

    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
        {"role": "user", "content": f"Research findings:\n{conditions_text}"},
    ]

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=2048, temperature=0.1)

    if "error" in result:
        log.warning(f"[{req_id}] Entity extraction error: {result['error']}")
        return [], []

    content = result.get("content", "")
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        return entities, relationships
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[{req_id}] Entity extraction JSON parse error: {e}")
        return [], []


# ============================================================================
# Citation Verification
# ============================================================================

VERIFICATION_PROMPT = """You are a citation verification agent. Verify claims against their stated sources and detect contradictions.

Given these atomic conditions (research findings), analyze them for:
1. Claims that contradict each other
2. Claims where the confidence seems too high or too low given the source quality
3. Claims that are well-supported vs. poorly supported
4. Claims that are speculative but reasonable — flag them but DO NOT discard them
5. Claims that reference fabricated entities (companies, people, studies that don't exist)

IMPORTANT DISTINCTIONS:
- "low_quality" = poorly sourced but the claim itself may be true. Downgrade confidence, don't remove.
- "speculative" = reasonable inference or hypothesis without direct evidence. Label it, keep it.
- "fabricated" = the claim references entities, sources, or data that demonstrably do not exist. Remove these.
- Absence of evidence is NOT evidence of fabrication. Don't mark things as fabricated just because you lack a source.
- Something illegal, unusual, or controversial is NOT fabricated.

Output ONLY a JSON object:
{
  "verified": [
    {"fact_index": 0, "adjusted_confidence": 0.8, "reason": "well-sourced from .edu domain"}
  ],
  "contradictions": [
    {"fact_index_1": 0, "fact_index_2": 3, "description": "Fact 0 says X but Fact 3 says Y"}
  ],
  "low_quality": [
    {"fact_index": 5, "reason": "single uncorroborated forum source"}
  ],
  "speculative": [
    {"fact_index": 2, "reason": "reasonable inference from available data but no direct source"}
  ],
  "fabricated": [
    {"fact_index": 7, "reason": "company 'XYZ Holdings' does not exist in any registry"}
  ]
}

Output ONLY valid JSON, no markdown fences."""


async def verify_conditions(
    conditions: list[AtomicCondition],
    req_id: str,
) -> list[AtomicCondition]:
    """Run citation verification on conditions. Adjusts confidence and flags contradictions."""
    if len(conditions) < 2:
        return conditions

    conditions_text = "\n".join(
        f"{i}. {c.fact} [source: {c.source_url}, confidence: {c.confidence:.1f}, trust: {c.trust_score:.1f}]"
        for i, c in enumerate(conditions)
    )

    messages = [
        {"role": "system", "content": VERIFICATION_PROMPT},
        {"role": "user", "content": f"Verify these {len(conditions)} research findings:\n{conditions_text}"},
    ]

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=2048, temperature=0.1)

    if "error" in result:
        log.warning(f"[{req_id}] Verification error: {result['error']}")
        return conditions

    content = result.get("content", "")
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)

        for v in data.get("verified", []):
            idx = v.get("fact_index", -1)
            if 0 <= idx < len(conditions):
                conditions[idx].confidence = float(v.get("adjusted_confidence", conditions[idx].confidence))

        for c in data.get("contradictions", []):
            idx1 = c.get("fact_index_1", -1)
            idx2 = c.get("fact_index_2", -1)
            if 0 <= idx1 < len(conditions):
                conditions[idx1].confidence = max(0.1, conditions[idx1].confidence - 0.2)
            if 0 <= idx2 < len(conditions):
                conditions[idx2].confidence = max(0.1, conditions[idx2].confidence - 0.2)

        for lq in data.get("low_quality", []):
            idx = lq.get("fact_index", -1)
            if 0 <= idx < len(conditions):
                conditions[idx].confidence = min(conditions[idx].confidence, 0.4)

        for sp in data.get("speculative", []):
            idx = sp.get("fact_index", -1)
            if 0 <= idx < len(conditions):
                conditions[idx].verification_status = "speculative"
                conditions[idx].confidence = min(conditions[idx].confidence, 0.4)

        fabricated_indices: set[int] = set()
        for fab in data.get("fabricated", []):
            idx = fab.get("fact_index", -1)
            if 0 <= idx < len(conditions):
                fabricated_indices.add(idx)
                conditions[idx].verification_status = "fabricated"

        if fabricated_indices:
            conditions = [
                c for i, c in enumerate(conditions)
                if i not in fabricated_indices
            ]
            log.info(
                f"[{req_id}] Self-check: removed {len(fabricated_indices)} "
                f"fabricated conditions"
            )

        return conditions

    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[{req_id}] Verification JSON parse error: {e}")
        return conditions


def _fuzzy_match_claim_to_condition(
    claim_text: str,
    conditions: list[AtomicCondition],
) -> int:
    """Find the best-matching condition index for a Veritas claim.

    Uses token overlap ratio.  Returns -1 if no condition scores above 0.3.
    """
    claim_tokens = set(claim_text.lower().split())
    if not claim_tokens:
        return -1

    best_idx = -1
    best_score = 0.3  # minimum threshold
    for i, cond in enumerate(conditions):
        cond_tokens = set(cond.fact.lower().split())
        if not cond_tokens:
            continue
        overlap = len(claim_tokens & cond_tokens)
        score = overlap / max(len(claim_tokens), len(cond_tokens))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


async def verify_conditions_with_veritas(
    conditions: list[AtomicCondition],
    user_query: str,
    req_id: str,
) -> tuple[list[AtomicCondition], dict]:
    """Run the full Veritas Inquisitor 5-agent reactor on research conditions.

    The Veritas system decomposes the conditions into claims, gathers external
    evidence via web search for each claim, runs a multi-round debate, and
    produces a final verdict classifying each claim as:
      - verified: confirmed by external evidence
      - plausible-unverified: reasonable but no confirming source found
      - speculative: a reasonable inference/hypothesis — kept with label
      - fabricated: references entities/sources that don't exist — removed
      - overconfident: overstates certainty

    Philosophy: anti-hallucination but PRO-SPECULATION.
    - Only *fabricated* claims (invented entities, fake sources) are removed.
    - Speculative claims are kept and labeled — they open investigation paths.
    - Absence of evidence is NOT evidence of fabrication.

    Returns:
        (filtered_conditions, veritas_report)
        - filtered_conditions: conditions with adjusted confidence and
          verification_status set.  Only fabricated ones are removed.
        - veritas_report: the raw Veritas report dict for logging/metrics.
    """
    if len(conditions) < VERITAS_MIN_CONDITIONS:
        return conditions, {}

    # Format conditions as a text block for Veritas to verify.
    target_text = "\n".join(
        f"{i+1}. {c.fact} [source: {c.source_url or 'no source'}]"
        for i, c in enumerate(conditions)
    )

    log.info(
        f"[{req_id}] Running Veritas verification on {len(conditions)} conditions"
    )

    try:
        result = await veritas_inquisitor.verify_output(
            target_text=target_text,
            original_query=user_query,
            req_id=f"{req_id}-veritas",
        )
    except Exception as e:
        log.error(f"[{req_id}] Veritas reactor error: {e}")
        return conditions, {"error": str(e)}

    report = result.get("report", {})
    claims = report.get("claims", [])

    if not claims:
        log.warning(f"[{req_id}] Veritas produced no claim verdicts")
        return conditions, report

    # Map Veritas verdicts back to conditions.
    # Veritas decomposes text into its own claims, so we fuzzy-match each
    # verdict back to the original AtomicCondition by text similarity.
    fabricated_indices: set[int] = set()
    speculative_indices: set[int] = set()
    confidence_overrides: dict[int, float] = {}
    status_overrides: dict[int, str] = {}

    for claim in claims:
        claim_text = claim.get("claim_text", "")
        status = claim.get("status", "")
        claim_confidence = claim.get("confidence", 0.5)
        try:
            claim_confidence = float(claim_confidence)
        except (TypeError, ValueError):
            claim_confidence = 0.5

        idx = _fuzzy_match_claim_to_condition(claim_text, conditions)
        if idx < 0:
            continue

        if status in ("fabricated", "hallucinated"):
            # Only truly fabricated claims (invented entities, fake sources)
            # get removed.  Legacy "hallucinated" status treated as fabricated.
            fabricated_indices.add(idx)
            status_overrides[idx] = "fabricated"
            log.info(
                f"[{req_id}] Veritas: FABRICATED — "
                f"{conditions[idx].fact[:80]}"
            )
        elif status == "speculative":
            # Speculative = reasonable inference without direct proof.
            # Keep the claim but label it and set confidence appropriately.
            speculative_indices.add(idx)
            status_overrides[idx] = "speculative"
            confidence_overrides[idx] = min(
                conditions[idx].confidence,
                max(claim_confidence, 0.2),
            )
            log.info(
                f"[{req_id}] Veritas: SPECULATIVE — "
                f"{conditions[idx].fact[:80]}"
            )
        elif status == "overconfident":
            # Cap confidence at what Veritas measured.
            status_overrides[idx] = "overconfident"
            confidence_overrides[idx] = min(
                conditions[idx].confidence,
                max(claim_confidence, 0.2),
            )
        elif status == "verified":
            # Boost if Veritas confirms it.
            status_overrides[idx] = "verified"
            confidence_overrides[idx] = max(
                conditions[idx].confidence,
                min(claim_confidence, 0.95),
            )
        elif status == "plausible-unverified":
            # Keep with moderate confidence — not confirmed, not fabricated.
            status_overrides[idx] = "plausible-unverified"
            confidence_overrides[idx] = min(
                conditions[idx].confidence,
                max(claim_confidence, 0.3),
            )

    # Apply confidence overrides and verification statuses.
    for idx, conf in confidence_overrides.items():
        if idx not in fabricated_indices:
            conditions[idx].confidence = conf
    for idx, vstatus in status_overrides.items():
        conditions[idx].verification_status = vstatus

    # Remove only fabricated conditions (invented entities, fake sources).
    # Speculative, plausible-unverified, and overconfident are all KEPT.
    filtered = [
        c for i, c in enumerate(conditions)
        if i not in fabricated_indices
    ]

    log.info(
        f"[{req_id}] Veritas results: {len(fabricated_indices)} fabricated "
        f"(removed), {len(speculative_indices)} speculative (kept), "
        f"{len(confidence_overrides)} confidence-adjusted, "
        f"{len(filtered)}/{len(conditions)} conditions retained"
    )

    return filtered, report

