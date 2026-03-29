"""
Persistent storage: JSONL flat-file logging, Neo4j-backed persistence,
knowledge graph retrieval, and large document ingestion.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import knowledge_client
import langfuse_config

from shared import make_sse_chunk

from .config import (
    JSONL_LOG_DIR,
    RESEARCH_NAMESPACE,
    log,
)
from .models import AtomicCondition


# ============================================================================
# Persistent Storage (SQLite + FTS5 + Knowledge Graph)
# ============================================================================

# ============================================================================
# JSONL Flat-File Logging (archival)
# ============================================================================

def _ensure_jsonl_dir() -> None:
    """Ensure the JSONL log directory exists."""
    os.makedirs(JSONL_LOG_DIR, exist_ok=True)


def _append_jsonl(session_id: str, record: dict) -> None:
    """Append a single JSON record to the session's JSONL log file."""
    try:
        _ensure_jsonl_dir()
        path = os.path.join(JSONL_LOG_DIR, f"{session_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        log.warning(f"JSONL write error: {e}")


def log_stage_output(
    session_id: str,
    stage: str,
    data: dict,
) -> None:
    """Persist arbitrary pipeline stage output to the session's JSONL log.

    Every pipeline node calls this so that a synthesis failure (or any
    later crash) never destroys earlier work.  The record can be read
    back to resume or retry downstream stages without re-running the
    full pipeline.

    Args:
        session_id: The request / session identifier.
        stage: Pipeline stage name (e.g. ``"comprehend"``, ``"retrieve"``).
        data: Stage-specific payload — must be JSON-serialisable.
    """
    _append_jsonl(session_id, {
        "type": "stage_output",
        "stage": stage,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    })


def _log_conditions_jsonl(
    session_id: str,
    query: str,
    conditions: list["AtomicCondition"],
) -> None:
    """Write conditions to a JSONL archive file."""
    now = datetime.now(timezone.utc).isoformat()
    for c in conditions:
        _append_jsonl(session_id, {
            "type": "condition",
            "session_id": session_id,
            "query": query,
            "fact": c.fact,
            "angle": c.angle,
            "source_url": c.source_url,
            "confidence": c.confidence,
            "trust_score": c.trust_score,
            "domain": c.domain,
            "is_serendipitous": c.is_serendipitous,
            "serendipity_score": c.serendipity_score_val,
            "publication_date": c.publication_date,
            "author": c.author,
            "content_type": c.content_type,
            "source_type": c.source_type,
            "created_at": now,
        })


def _log_entities_jsonl(
    session_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> None:
    """Write entities and relationships to a JSONL archive file."""
    now = datetime.now(timezone.utc).isoformat()
    for ent in entities:
        _append_jsonl(session_id, {
            "type": "entity",
            "name": ent.get("name", ""),
            "entity_type": ent.get("type", "concept"),
            "session_id": session_id,
            "created_at": now,
        })
    for rel in relationships:
        _append_jsonl(session_id, {
            "type": "relationship",
            "entity1": rel.get("entity1", ""),
            "entity2": rel.get("entity2", ""),
            "relationship_type": rel.get("type", "related_to"),
            "is_bridge": rel.get("is_bridge", False),
            "session_id": session_id,
            "created_at": now,
        })


# ============================================================================
# Neo4j-backed Persistence (via Knowledge Engine)
# ============================================================================

def _clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


async def _store_conditions_neo4j(
    session_id: str,
    query: str,
    conditions: list["AtomicCondition"],
) -> tuple[int, str]:
    """Store atomic conditions in Neo4j via the knowledge engine.

    Returns (count_stored, error_message).  error_message is empty on success.
    """
    if not conditions:
        return 0, ""
    span = langfuse_config.start_span(
        session_id, "neo4j:store_conditions",
        input={"count": len(conditions), "query": query[:120]},
    )
    cond_dicts = [
        {
            "fact": c.fact or "",
            "source_url": c.source_url or "",
            "confidence": _clamp01(c.confidence),
            "trust_score": _clamp01(c.trust_score),
            "angle": c.angle or "",
            "domain": c.domain or "",
            "is_serendipitous": bool(c.is_serendipitous),
            "serendipity_score": _clamp01(c.serendipity_score_val),
            "publication_date": c.publication_date or "",
            "author": c.author or "",
            "content_type": c.content_type or "",
            "source_type": c.source_type or "",
        }
        for c in conditions
    ]
    try:
        result = await knowledge_client.store_conditions(
            session_id=session_id,
            query=query,
            conditions=cond_dicts,
            namespace=RESEARCH_NAMESPACE,
        )
        stored = result.get("stored", 0)
        langfuse_config.end_span(span, output={"stored": stored})
        return stored, ""
    except Exception as e:
        log.error(f"Neo4j condition storage error: {e}")
        langfuse_config.end_span(span, output={"error": str(e)}, level="ERROR")
        return 0, str(e)


async def _store_entities_neo4j(
    session_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> tuple[int, int, str]:
    """Store entities and relationships in Neo4j via the knowledge engine.

    Returns (entities_stored, relationships_stored, error_message).
    """
    span = langfuse_config.start_span(
        session_id, "neo4j:store_entities",
        input={"entities": len(entities), "relationships": len(relationships)},
    )
    try:
        result = await knowledge_client.store_entities(
            session_id=session_id,
            entities=entities,
            relationships=relationships,
            namespace=RESEARCH_NAMESPACE,
        )
        e_stored = result.get("entities_stored", 0)
        r_stored = result.get("relationships_stored", 0)
        langfuse_config.end_span(span, output={"entities_stored": e_stored, "relationships_stored": r_stored})
        return e_stored, r_stored, ""
    except Exception as e:
        log.error(f"Neo4j entity storage error: {e}")
        langfuse_config.end_span(span, output={"error": str(e)}, level="ERROR")
        return 0, 0, str(e)


async def _retrieve_related(query: str, limit: int = 20, req_id: str = "") -> list[dict]:
    """Retrieve prior conditions related to the query using Neo4j fulltext search."""
    span = langfuse_config.start_span(
        req_id, "neo4j:retrieve_related",
        input={"query": query[:120], "limit": limit},
    ) if req_id else None
    try:
        results = await knowledge_client.search_conditions(
            query=query,
            namespace=RESEARCH_NAMESPACE,
            limit=limit,
        )
        out = [
            {
                "fact": r.get("fact", ""),
                "source_url": r.get("source_url", ""),
                "confidence": r.get("confidence", 0.0),
                "angle": r.get("angle", ""),
                "is_serendipitous": r.get("is_serendipitous", False),
                "original_query": r.get("query", ""),
                "created_at": r.get("created_at", ""),
                "trust_score": r.get("trust_score", 0.0),
                "serendipity_score": r.get("serendipity_score", 0.0),
            }
            for r in results
        ]
        langfuse_config.end_span(span, output={"results": len(out)})
        return out
    except Exception as e:
        log.warning(f"Neo4j condition search error: {e}")
        langfuse_config.end_span(span, output={"error": str(e)}, level="WARNING")
        return []


async def _retrieve_graph_neighbors(
    entity_names: list[str], max_hops: int = 2, limit: int = 20, req_id: str = "",
) -> list[dict]:
    """Retrieve related conditions via knowledge graph traversal in Neo4j."""
    if not entity_names:
        return []
    span = langfuse_config.start_span(
        req_id, "neo4j:graph_neighbors",
        input={"entities": entity_names[:5], "max_hops": max_hops, "limit": limit},
    ) if req_id else None
    try:
        results = await knowledge_client.graph_neighbors(
            entity_names=entity_names,
            namespace=RESEARCH_NAMESPACE,
            max_hops=max_hops,
            limit=limit,
        )
        out = [
            {
                "fact": r.get("fact", ""),
                "source_url": r.get("source_url", ""),
                "confidence": r.get("confidence", 0.0),
                "angle": r.get("angle", ""),
                "trust_score": r.get("trust_score", 0.0),
                "via_entity": "graph",
            }
            for r in results
        ]
        langfuse_config.end_span(span, output={"results": len(out)})
        return out
    except Exception as e:
        log.warning(f"Neo4j graph neighbor error: {e}")
        langfuse_config.end_span(span, output={"error": str(e)}, level="WARNING")
        return []


# ============================================================================
# Large Document Ingestion
# ============================================================================

LARGE_DOC_CHAR_THRESHOLD = 10000


def _is_large_document(text: str) -> bool:
    """Detect whether a message is a large document rather than a research query.

    Heuristics:
      - Length > LARGE_DOC_CHAR_THRESHOLD chars
      - Low question density (few '?' relative to text length)
    """
    if len(text) < LARGE_DOC_CHAR_THRESHOLD:
        return False
    question_marks = text.count("?")
    question_density = question_marks / max(len(text), 1)
    if question_density < 0.0005:
        return True
    return False


async def run_document_ingestion(
    text: str,
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Distinct pipeline for large document ingestion — not a research query.

    Flow:
      1. Send document text to the Knowledge Engine for ingestion.
      2. Stream progress updates to the user.
      3. After ingestion completes, summarise what was extracted.
    """
    model_id = original_body.get("model", "persistent-miroflow")
    request_id = f"chatcmpl-ingest-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    def reasoning_chunk(content: str) -> str:
        """Emit a reasoning_content delta (collapsible Thinking block)."""
        return make_sse_chunk(
            "",
            request_id=request_id,
            created=created,
            model_id=model_id,
            reasoning_content=content,
        )

    title = text[:80].replace("\n", " ").strip()
    namespace = RESEARCH_NAMESPACE
    doc_chars = len(text)

    yield reasoning_chunk(f"**[Document Ingestion Mode]** Detected large document ({doc_chars:,} chars)\n")
    yield reasoning_chunk(f"Title: {title}...\n")
    yield reasoning_chunk(f"Namespace: {namespace}\n\n")

    # Archive the raw text to a JSONL file
    _append_jsonl(req_id, {
        "type": "document_ingestion",
        "title": title,
        "char_count": doc_chars,
        "namespace": namespace,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    ingestion_ok = False
    try:
        # Step 1: Submit to knowledge engine
        yield reasoning_chunk("**[Step 1: Submitting to Knowledge Engine]**\n")
        ingest_result = await knowledge_client.ingest(
            namespace=namespace,
            title=title,
            text=text,
            source="document-ingestion",
            rebuild=False,  # Append, don't clear existing data
        )
        job_id = ingest_result.get("job_id", "")
        yield reasoning_chunk(f"Ingest job started: {job_id}\n")
        yield reasoning_chunk(f"Total chars: {ingest_result.get('total_chars', doc_chars):,}\n\n")

        # Step 2: Poll for completion
        yield reasoning_chunk("**[Step 2: Processing Document]**\n")
        max_polls = 300  # up to ~10 minutes
        last_status = ""
        for _ in range(max_polls):
            await asyncio.sleep(2)
            try:
                status = await knowledge_client.ingest_status(job_id)
            except Exception as e:
                yield reasoning_chunk(f"  Poll error: {e}\n")
                continue

            current_status = status.get("status", "unknown")
            progress = status.get("progress", "")

            if current_status != last_status:
                yield reasoning_chunk(f"  Status: {current_status}")
                if progress:
                    yield reasoning_chunk(f" — {progress}")
                yield reasoning_chunk("\n")
                last_status = current_status

            if current_status == "completed":
                ingestion_ok = True
                stats = status.get("stats", {})
                yield reasoning_chunk("\n**[Step 3: Ingestion Complete]**\n")
                if stats:
                    yield reasoning_chunk(f"  Chunks: {stats.get('total_chunks', '?')}\n")
                    yield reasoning_chunk(f"  Entities extracted: {stats.get('entities_created', '?')}\n")
                    yield reasoning_chunk(f"  Relationships: {stats.get('relationships_created', '?')}\n")
                    yield reasoning_chunk(f"  Claims: {stats.get('claims_created', '?')}\n")
                break
            elif current_status == "failed":
                error = status.get("error", "Unknown error")
                yield reasoning_chunk(f"\n**Ingestion failed:** {error}\n")
                break
        else:
            yield reasoning_chunk("\n**Warning:** Ingestion is still running (timed out waiting).\n")
            yield reasoning_chunk("You can check status later via the knowledge engine API.\n")

    except Exception as e:
        log.error(f"[{req_id}] Document ingestion error: {e}")
        yield reasoning_chunk(f"\n**Error during ingestion:** {e}\n")

    # Produce a user-facing summary based on actual outcome
    if ingestion_ok:
        yield chunk(
            f"## Document Ingested\n\n"
            f"Your document ({doc_chars:,} characters) has been processed and loaded into "
            f"the knowledge graph under namespace **{namespace}**.\n\n"
            f"The knowledge engine has:\n"
            f"- Chunked the document into overlapping segments\n"
            f"- Extracted concepts, claims, evidence, and relationships\n"
            f"- Resolved duplicate entities\n"
            f"- Loaded everything into Neo4j\n\n"
            f"You can now ask research questions about this document and it will be "
            f"available as prior knowledge for all future research sessions.\n"
        )
    else:
        yield chunk(
            f"## Document Ingestion Failed\n\n"
            f"Your document ({doc_chars:,} characters) could not be fully processed. "
            f"Please check the errors above and try again, or submit a smaller document.\n"
        )

    yield chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# Initialise JSONL log directory
try:
    _ensure_jsonl_dir()
    log.info(f"JSONL log directory ready: {JSONL_LOG_DIR}")
except Exception as e:
    log.warning(f"Failed to create JSONL log directory: {e}")

