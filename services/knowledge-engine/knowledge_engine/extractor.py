"""Multi-pass LLM entity/relationship extraction using the epistemic ontology.

Pass 1 — Direct extraction: entities, claims, methods, evidence from each chunk.
Pass 2 — Implicit/anomaly extraction: hypotheses, anomalies, cross-domain analogies.
Pass 3 — Cross-chunk relationship inference: relationships between entities across chunks.
"""

import asyncio
import json
import logging
import uuid

import httpx

from .config import UPSTREAM_BASE, UPSTREAM_KEY, EXTRACTION_MODEL, MAX_EXTRACTION_CONCURRENCY

log = logging.getLogger("knowledge-engine")

_semaphore = asyncio.Semaphore(MAX_EXTRACTION_CONCURRENCY)


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ============================================================================
# LLM Call
# ============================================================================

async def _call_llm(
    messages: list[dict],
    client: httpx.AsyncClient,
    temperature: float = 0.1,
) -> str:
    """Call the extraction LLM and return the text response."""
    async with _semaphore:
        try:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json={
                    "model": EXTRACTION_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"LLM extraction call failed: {e}")
            return "{}"


def _parse_json(text: str) -> dict:
    """Safely parse JSON from LLM response."""
    text = text.strip()
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning(f"Failed to parse LLM JSON response: {text[:200]}...")
        return {}


# ============================================================================
# Pass 1: Direct Extraction
# ============================================================================

PASS1_SYSTEM = """You are a knowledge extraction agent. Extract structured knowledge from the given text chunk.

You MUST output valid JSON with these keys:

{
  "concepts": [
    {"name": "...", "domains": ["domain1", "domain2"], "abstraction_level": "concrete|abstract|meta"}
  ],
  "claims": [
    {"statement": "...", "confidence": 0.0-1.0, "polarity": "positive|negative|neutral"}
  ],
  "evidence": [
    {"text": "...", "strength": "strong|moderate|weak|anecdotal", "supports_claim": "..."}
  ],
  "methods": [
    {"name": "...", "domain": "...", "transferable": true|false}
  ]
}

Rules:
- Extract ALL named entities as Concepts with their domains.
- Extract factual assertions as Claims with confidence scores.
- Extract supporting data/quotes as Evidence linked to Claims.
- Extract techniques/approaches as Methods, marking transferable=true if applicable to other domains.
- Be exhaustive. Better to over-extract than miss knowledge.
- Use concise names for Concepts (2-5 words max).
- Claims should be self-contained statements (understandable without the original text)."""

PASS1_USER = """Extract knowledge from this text chunk:

---
{chunk_text}
---

Return ONLY valid JSON."""


async def extract_pass1(
    chunk_text: str,
    client: httpx.AsyncClient,
) -> dict:
    """Pass 1: Direct extraction of concepts, claims, evidence, methods."""
    messages = [
        {"role": "system", "content": PASS1_SYSTEM},
        {"role": "user", "content": PASS1_USER.format(chunk_text=chunk_text)},
    ]
    raw = await _call_llm(messages, client)
    return _parse_json(raw)


# ============================================================================
# Pass 2: Implicit / Anomaly Extraction
# ============================================================================

PASS2_SYSTEM = """You are an advanced knowledge extraction agent focused on finding implicit knowledge, anomalies, and cross-domain connections.

Given a text chunk AND the concepts/claims already extracted from it, identify:

{
  "hypotheses": [
    {"statement": "...", "status": "open", "abductive_origin": "why this hypothesis follows from the text"}
  ],
  "anomalies": [
    {"description": "...", "surprise_score": 0.0-1.0}
  ],
  "analogies": [
    {"concept_a": "...", "concept_b": "...", "cross_domain": true|false, "bridge_score": 0.0-1.0, "explanation": "..."}
  ],
  "implicit_relationships": [
    {"source": "...", "target": "...", "relationship": "...", "confidence": 0.0-1.0}
  ]
}

Rules:
- Hypotheses: What does the author imply but not state? What connections SHOULD exist?
- Anomalies: What doesn't fit? What's surprising or contradictory? What breaks the expected pattern? HIGH surprise_score for things that really don't fit.
- Analogies: What concepts from different domains are structurally similar? cross_domain=true is critical — these are the serendipity seeds.
- Implicit relationships: What relationships does the author assume but never explicitly state?
- If you find nothing implicit, return empty lists. Don't fabricate."""

PASS2_USER = """Text chunk:
---
{chunk_text}
---

Already extracted:
{pass1_json}

Now find the IMPLICIT knowledge, anomalies, and cross-domain analogies. Return ONLY valid JSON."""


async def extract_pass2(
    chunk_text: str,
    pass1_result: dict,
    client: httpx.AsyncClient,
) -> dict:
    """Pass 2: Implicit extraction — hypotheses, anomalies, analogies."""
    messages = [
        {"role": "system", "content": PASS2_SYSTEM},
        {"role": "user", "content": PASS2_USER.format(
            chunk_text=chunk_text,
            pass1_json=json.dumps(pass1_result, indent=2),
        )},
    ]
    raw = await _call_llm(messages, client, temperature=0.3)
    return _parse_json(raw)


# ============================================================================
# Pass 3: Cross-Chunk Relationship Inference
# ============================================================================

PASS3_SYSTEM = """You are a knowledge graph relationship inference agent.

Given a list of concepts and claims extracted from multiple chunks of the SAME document, identify relationships BETWEEN them that span across chunks.

{
  "relationships": [
    {
      "source_name": "...",
      "source_type": "Concept|Claim|Method",
      "target_name": "...",
      "target_type": "Concept|Claim|Method",
      "relationship_type": "RELATED_TO|ANALOGOUS_TO|CONTRADICTS|SUPPORTED_BY|REQUIRES|EXPLAINS|TRANSFERABLE_TO|INSTANCE_OF|UPDATES",
      "cross_domain": true|false,
      "confidence": 0.0-1.0
    }
  ]
}

Rules:
- Focus on cross-chunk connections — things that link different parts of the document.
- Prioritise ANALOGOUS_TO (cross-domain similarities) and CONTRADICTS (conflicting claims).
- Mark cross_domain=true for relationships connecting different subject areas.
- Only include relationships with confidence >= 0.3.
- Keep relationship_type to the listed types only."""

PASS3_USER = """All extracted concepts and claims from this document:

{entities_json}

Find cross-chunk relationships. Return ONLY valid JSON."""


async def extract_pass3_relationships(
    all_entities: dict,
    client: httpx.AsyncClient,
) -> dict:
    """Pass 3: Cross-chunk relationship inference."""
    # Prepare a compact summary of all entities for the LLM
    summary = {
        "concepts": [c.get("name", "") for c in all_entities.get("concepts", [])],
        "claims": [c.get("statement", "")[:100] for c in all_entities.get("claims", [])],
        "methods": [m.get("name", "") for m in all_entities.get("methods", [])],
    }

    messages = [
        {"role": "system", "content": PASS3_SYSTEM},
        {"role": "user", "content": PASS3_USER.format(
            entities_json=json.dumps(summary, indent=2),
        )},
    ]
    raw = await _call_llm(messages, client, temperature=0.2)
    return _parse_json(raw)


# ============================================================================
# Batch Extraction Orchestrator
# ============================================================================

async def extract_all_chunks(
    chunks: list[dict],
) -> list[dict]:
    """Run multi-pass extraction on all chunks concurrently.

    Returns a list of extraction results (one per chunk), each containing
    pass1 and pass2 results merged together.
    """
    async with httpx.AsyncClient() as client:
        # Pass 1 + 2 for each chunk (concurrently)
        tasks = [_extract_chunk(chunk, client) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"Chunk {i} extraction failed: {result}")
                chunk_results.append({"pass1": {}, "pass2": {}, "chunk_id": chunks[i]["id"]})
            else:
                chunk_results.append(result)

        # Pass 3: Cross-chunk relationships
        all_entities = _merge_all_entities(chunk_results)
        if all_entities.get("concepts") or all_entities.get("claims"):
            try:
                cross_rels = await extract_pass3_relationships(all_entities, client)
                # Attach cross-chunk relationships to the first result for simplicity
                if chunk_results:
                    chunk_results[0]["cross_relationships"] = cross_rels.get("relationships", [])
            except Exception as e:
                log.error(f"Pass 3 cross-chunk extraction failed: {e}")

    return chunk_results


async def _extract_chunk(chunk: dict, client: httpx.AsyncClient) -> dict:
    """Run pass 1 and pass 2 on a single chunk."""
    content = chunk["content"]
    chunk_id = chunk["id"]

    # Pass 1
    pass1 = await extract_pass1(content, client)

    # Pass 2 (uses pass 1 results)
    pass2 = await extract_pass2(content, pass1, client)

    return {
        "chunk_id": chunk_id,
        "pass1": pass1,
        "pass2": pass2,
    }


def _merge_all_entities(chunk_results: list[dict]) -> dict:
    """Merge all extracted entities across chunks for pass 3."""
    concepts: list[dict] = []
    claims: list[dict] = []
    methods: list[dict] = []

    for cr in chunk_results:
        p1 = cr.get("pass1", {})
        concepts.extend(p1.get("concepts", []))
        claims.extend(p1.get("claims", []))
        methods.extend(p1.get("methods", []))

    # Deduplicate concepts by name for the summary
    seen_concepts: set[str] = set()
    unique_concepts = []
    for c in concepts:
        name = c.get("name", "").lower().strip()
        if name and name not in seen_concepts:
            seen_concepts.add(name)
            unique_concepts.append(c)

    return {
        "concepts": unique_concepts[:100],  # Cap for LLM context
        "claims": claims[:50],
        "methods": methods[:30],
    }
