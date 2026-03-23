#!/usr/bin/env python3
"""
Swarm Deep Search Proxy for Open WebUI.

A swarm-based proxy that decomposes large corpora of text using a
hierarchical agent swarm (inspired by swarms.world).  Background workers
continuously process corpora — chunking, extracting entities, building
knowledge graphs — while queries are answered non-disruptively from
whatever knowledge the swarm has built so far.

Key design principles:
  * Sending a prompt does NOT disturb the swarm from what it is doing.
  * The proxy is *sincere* about what is happening: it reports real
    swarm state, processing progress, and knowledge coverage honestly.
  * Further large corpora are treated identically to the initial send —
    they are queued additively without resetting existing work.

Architecture:
  Browser → Open WebUI → Swarm Proxy (port 9500)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        Swarm Director   Query Handler   Status Reporter
              │                               │
        Worker Pool ──→ Knowledge Engine (Neo4j, port 9400)
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

import knowledge_client

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
    get_throttler,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("SWARM_PROXY_LOG_DIR", "/opt/swarm_proxy_logs")
log = setup_logging("swarm-proxy", LOG_DIR)

UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SYNTHESIS_MODEL = os.getenv("SWARM_SYNTHESIS_MODEL", "mistral-large-latest")
# WORKER_MODEL reserved for future per-chunk analysis workers
LISTEN_PORT = env_int("SWARM_PROXY_PORT", 9500, minimum=1)
MAX_CONCURRENT_QUERIES = env_int("SWARM_MAX_CONCURRENT_QUERIES", 4, minimum=1)
MAX_SWARM_WORKERS = env_int("SWARM_MAX_WORKERS", 6, minimum=1)
SWARM_NAMESPACE = os.getenv("SWARM_NAMESPACE", "swarm")

# Chunk parameters for corpus decomposition
CHUNK_SIZE = int(os.getenv("SWARM_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("SWARM_CHUNK_OVERLAP", "200"))

# Large document detection threshold (chars)
LARGE_DOC_THRESHOLD = int(os.getenv("SWARM_LARGE_DOC_THRESHOLD", "5000"))

log.info(
    f"Config: synthesis_model={SYNTHESIS_MODEL}, "
    f"upstream={UPSTREAM_BASE}, port={LISTEN_PORT}, "
    f"max_queries={MAX_CONCURRENT_QUERIES}, max_workers={MAX_SWARM_WORKERS}, "
    f"namespace={SWARM_NAMESPACE}"
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

tracker = RequestTracker()
query_limiter = ConcurrencyLimiter(MAX_CONCURRENT_QUERIES)
worker_semaphore = asyncio.Semaphore(MAX_SWARM_WORKERS)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CorpusStatus(str, Enum):
    """Processing status for an ingested corpus."""
    QUEUED = "queued"
    CHUNKING = "chunking"
    SUBMITTING = "submitting"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(str, Enum):
    """Status for an individual swarm worker."""
    IDLE = "idle"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    DONE = "done"
    ERROR = "error"


@dataclass
class CorpusRecord:
    """Tracks a single corpus through the swarm pipeline."""
    id: str
    title: str
    source: str
    total_chars: int
    status: CorpusStatus = CorpusStatus.QUEUED
    total_chunks: int = 0
    chunks_processed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    claims_extracted: int = 0
    error: str = ""
    submitted_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    knowledge_engine_job_id: str = ""


@dataclass
class SwarmWorkerInfo:
    """Tracks what a single worker is currently doing."""
    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    corpus_id: str = ""
    current_task: str = ""
    started_at: float = 0.0


class SwarmState:
    """Global swarm state — holds all corpora, workers, and knowledge stats.

    This is the single source of truth for the swarm's current activity.
    Query handlers read from this without blocking; corpus ingestion
    tasks write to it as they progress.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._corpora: dict[str, CorpusRecord] = {}
        self._workers: dict[str, SwarmWorkerInfo] = {}
        self._total_queries_answered: int = 0
        self._started_at: float = time.monotonic()

    async def add_corpus(self, record: CorpusRecord) -> None:
        """Register a new corpus for tracking."""
        async with self._lock:
            self._corpora[record.id] = record

    async def get_corpus(self, corpus_id: str) -> Optional[CorpusRecord]:
        async with self._lock:
            return self._corpora.get(corpus_id)

    async def update_corpus(
        self,
        corpus_id: str,
        *,
        status: Optional[CorpusStatus] = None,
        total_chunks: Optional[int] = None,
        chunks_processed: Optional[int] = None,
        entities_extracted: Optional[int] = None,
        relationships_extracted: Optional[int] = None,
        claims_extracted: Optional[int] = None,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        knowledge_engine_job_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            rec = self._corpora.get(corpus_id)
            if rec is None:
                return
            if status is not None:
                rec.status = status
            if total_chunks is not None:
                rec.total_chunks = total_chunks
            if chunks_processed is not None:
                rec.chunks_processed = chunks_processed
            if entities_extracted is not None:
                rec.entities_extracted = entities_extracted
            if relationships_extracted is not None:
                rec.relationships_extracted = relationships_extracted
            if claims_extracted is not None:
                rec.claims_extracted = claims_extracted
            if error is not None:
                rec.error = error
            if started_at is not None:
                rec.started_at = started_at
            if completed_at is not None:
                rec.completed_at = completed_at
            if knowledge_engine_job_id is not None:
                rec.knowledge_engine_job_id = knowledge_engine_job_id

    async def register_worker(self, worker_id: str) -> None:
        async with self._lock:
            self._workers[worker_id] = SwarmWorkerInfo(id=worker_id)

    async def update_worker(
        self,
        worker_id: str,
        *,
        status: Optional[WorkerStatus] = None,
        corpus_id: Optional[str] = None,
        current_task: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> None:
        async with self._lock:
            w = self._workers.get(worker_id)
            if w is None:
                return
            if status is not None:
                w.status = status
            if corpus_id is not None:
                w.corpus_id = corpus_id
            if current_task is not None:
                w.current_task = current_task
            if started_at is not None:
                w.started_at = started_at

    async def remove_worker(self, worker_id: str) -> None:
        async with self._lock:
            self._workers.pop(worker_id, None)

    async def increment_queries(self) -> None:
        async with self._lock:
            self._total_queries_answered += 1

    async def get_status_snapshot(self) -> dict:
        """Return a complete snapshot of swarm state for status reporting."""
        async with self._lock:
            corpora_list = []
            for c in self._corpora.values():
                progress = 0.0
                if c.total_chunks > 0:
                    progress = c.chunks_processed / c.total_chunks
                corpora_list.append({
                    "id": c.id,
                    "title": c.title,
                    "status": c.status.value,
                    "total_chars": c.total_chars,
                    "total_chunks": c.total_chunks,
                    "chunks_processed": c.chunks_processed,
                    "progress": round(progress, 3),
                    "entities_extracted": c.entities_extracted,
                    "relationships_extracted": c.relationships_extracted,
                    "claims_extracted": c.claims_extracted,
                    "error": c.error,
                    "submitted_at": c.submitted_at,
                    "completed_at": c.completed_at,
                })

            workers_list = []
            for w in self._workers.values():
                workers_list.append({
                    "id": w.id,
                    "status": w.status.value,
                    "corpus_id": w.corpus_id,
                    "current_task": w.current_task,
                    "uptime_s": round(time.monotonic() - w.started_at, 1) if w.started_at else 0,
                })

            completed = sum(1 for c in self._corpora.values() if c.status == CorpusStatus.COMPLETED)
            processing = sum(1 for c in self._corpora.values() if c.status not in (CorpusStatus.COMPLETED, CorpusStatus.FAILED, CorpusStatus.QUEUED))
            queued = sum(1 for c in self._corpora.values() if c.status == CorpusStatus.QUEUED)
            failed = sum(1 for c in self._corpora.values() if c.status == CorpusStatus.FAILED)

            total_entities = sum(c.entities_extracted for c in self._corpora.values())
            total_relationships = sum(c.relationships_extracted for c in self._corpora.values())
            total_claims = sum(c.claims_extracted for c in self._corpora.values())
            total_chars = sum(c.total_chars for c in self._corpora.values())

            return {
                "swarm_uptime_s": round(time.monotonic() - self._started_at, 1),
                "total_corpora": len(self._corpora),
                "corpora_completed": completed,
                "corpora_processing": processing,
                "corpora_queued": queued,
                "corpora_failed": failed,
                "total_chars_ingested": total_chars,
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "total_claims": total_claims,
                "active_workers": sum(1 for w in self._workers.values() if w.status not in (WorkerStatus.IDLE, WorkerStatus.DONE)),
                "total_queries_answered": self._total_queries_answered,
                "namespace": SWARM_NAMESPACE,
                "corpora": corpora_list,
                "workers": workers_list,
            }

    async def get_corpora_list(self) -> list[dict]:
        """Return a summary of all corpora."""
        async with self._lock:
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "status": c.status.value,
                    "total_chars": c.total_chars,
                    "submitted_at": c.submitted_at,
                    "completed_at": c.completed_at,
                }
                for c in self._corpora.values()
            ]

    async def build_sincerity_preamble(self) -> str:
        """Build an honest status message about the swarm's current state.

        This is the 'sincerity' mechanism — the proxy tells the user exactly
        what is happening right now, no sugar-coating.
        """
        snapshot = await self.get_status_snapshot()

        if snapshot["total_corpora"] == 0:
            return (
                "**[Swarm Status]** No corpora have been submitted yet. "
                "Send me a large body of text and I will begin decomposing "
                "and understanding it. You can ask questions at any time.\n\n"
            )

        parts = ["**[Swarm Status]**"]

        # Overall summary
        parts.append(
            f" {snapshot['total_corpora']} corpus/corpora ingested "
            f"({snapshot['total_chars_ingested']:,} chars total)."
        )

        # Processing state
        if snapshot["corpora_processing"] > 0:
            parts.append(
                f" **Currently processing {snapshot['corpora_processing']} "
                f"corpus/corpora** with {snapshot['active_workers']} active worker(s)."
            )
            # Show per-corpus progress for actively processing ones
            for c in snapshot["corpora"]:
                if c["status"] not in ("completed", "failed", "queued"):
                    pct = int(c["progress"] * 100)
                    parts.append(
                        f"\n  → *{c['title'][:60]}*: {c['status']} — "
                        f"{pct}% ({c['chunks_processed']}/{c['total_chunks']} chunks)"
                    )

        if snapshot["corpora_queued"] > 0:
            parts.append(
                f" {snapshot['corpora_queued']} corpus/corpora waiting in queue."
            )

        if snapshot["corpora_completed"] > 0:
            parts.append(
                f" {snapshot['corpora_completed']} fully processed."
            )

        if snapshot["corpora_failed"] > 0:
            parts.append(
                f" ⚠ {snapshot['corpora_failed']} failed."
            )

        # Knowledge built so far
        if snapshot["total_entities"] > 0 or snapshot["total_claims"] > 0:
            parts.append(
                f"\n**Knowledge built:** {snapshot['total_entities']} entities, "
                f"{snapshot['total_relationships']} relationships, "
                f"{snapshot['total_claims']} claims."
            )

        # Honest caveat if still processing
        if snapshot["corpora_processing"] > 0 or snapshot["corpora_queued"] > 0:
            parts.append(
                "\n*Note: The swarm is still working. My answers reflect "
                "knowledge extracted so far — they may become more complete "
                "as processing continues. This does not affect the swarm's work.*"
            )

        return "".join(parts) + "\n\n"


# Global swarm state
swarm = SwarmState()

# Background worker tasks (kept alive for the process lifetime)
_worker_tasks: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# Swarm worker: corpus processing pipeline
# ---------------------------------------------------------------------------

async def _process_corpus(corpus_id: str, text: str) -> None:
    """Process a single corpus through the swarm pipeline.

    Steps:
      1. Decompose text into overlapping chunks
      2. Submit to Knowledge Engine for full ETL (entity extraction,
         relationship extraction, cross-chunk inference, entity resolution)
      3. Track progress by polling the Knowledge Engine job status
      4. Update swarm state at each stage

    This runs as a background task and does NOT block any queries.
    """
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    async with worker_semaphore:
        await swarm.register_worker(worker_id)

        try:
            record = await swarm.get_corpus(corpus_id)
            if not record:
                log.error(f"[{worker_id}] Corpus {corpus_id} not found")
                return

            await swarm.update_worker(
                worker_id,
                status=WorkerStatus.CHUNKING,
                corpus_id=corpus_id,
                current_task=f"Chunking {record.title[:40]}",
                started_at=time.monotonic(),
            )
            await swarm.update_corpus(
                corpus_id,
                status=CorpusStatus.CHUNKING,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            # Step 1: Compute chunk count for progress tracking
            total_chunks = max(1, (record.total_chars - CHUNK_OVERLAP) // (CHUNK_SIZE - CHUNK_OVERLAP))
            await swarm.update_corpus(corpus_id, total_chunks=total_chunks)

            log.info(
                f"[{worker_id}] Corpus {corpus_id} ({record.title[:40]}): "
                f"{record.total_chars:,} chars → ~{total_chunks} chunks"
            )

            # Step 2: Submit to Knowledge Engine
            await swarm.update_worker(
                worker_id,
                status=WorkerStatus.EXTRACTING,
                current_task=f"Submitting to Knowledge Engine: {record.title[:40]}",
            )
            await swarm.update_corpus(corpus_id, status=CorpusStatus.SUBMITTING)

            try:
                ingest_result = await knowledge_client.ingest(
                    namespace=SWARM_NAMESPACE,
                    title=record.title,
                    text=text,
                    source=record.source or "swarm-ingestion",
                    rebuild=False,  # Additive — never clear existing data
                )
                job_id = ingest_result.get("job_id", "")
                actual_chunks = ingest_result.get("total_chunks", total_chunks)
                await swarm.update_corpus(
                    corpus_id,
                    status=CorpusStatus.EXTRACTING,
                    knowledge_engine_job_id=job_id,
                    total_chunks=actual_chunks,
                )

                log.info(
                    f"[{worker_id}] Knowledge Engine job {job_id} started "
                    f"for corpus {corpus_id}"
                )

            except Exception as e:
                log.error(f"[{worker_id}] Failed to submit corpus {corpus_id}: {e}")
                await swarm.update_corpus(
                    corpus_id,
                    status=CorpusStatus.FAILED,
                    error=f"Submission failed: {e}",
                )
                return

            # Step 3: Poll Knowledge Engine for progress
            await swarm.update_worker(
                worker_id,
                status=WorkerStatus.INDEXING,
                current_task=f"Monitoring extraction: {record.title[:40]}",
            )

            max_polls = 600  # up to ~20 minutes
            last_logged_status = ""
            for poll_idx in range(max_polls):
                await asyncio.sleep(2)

                try:
                    status = await knowledge_client.ingest_status(job_id)
                except Exception as e:
                    if poll_idx % 10 == 0:
                        log.warning(
                            f"[{worker_id}] Poll error for {job_id}: {e}"
                        )
                    continue

                current_status = status.get("status", "unknown")
                progress_text = status.get("progress", "")
                stats = status.get("stats", {})

                # Update progress counters from Knowledge Engine stats
                if stats:
                    await swarm.update_corpus(
                        corpus_id,
                        chunks_processed=stats.get("chunks_processed", 0),
                        entities_extracted=stats.get("entities_created", 0),
                        relationships_extracted=stats.get("relationships_created", 0),
                        claims_extracted=stats.get("claims_created", 0),
                    )

                if current_status != last_logged_status:
                    log.info(
                        f"[{worker_id}] Corpus {corpus_id} status: "
                        f"{current_status} — {progress_text}"
                    )
                    await swarm.update_worker(
                        worker_id,
                        current_task=f"{current_status}: {progress_text}"[:80],
                    )
                    last_logged_status = current_status

                if current_status == "completed":
                    final_stats = status.get("stats", {})
                    await swarm.update_corpus(
                        corpus_id,
                        status=CorpusStatus.COMPLETED,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        chunks_processed=final_stats.get("total_chunks", total_chunks),
                        entities_extracted=final_stats.get("entities_created", 0),
                        relationships_extracted=final_stats.get("relationships_created", 0),
                        claims_extracted=final_stats.get("claims_created", 0),
                    )
                    log.info(
                        f"[{worker_id}] Corpus {corpus_id} COMPLETED: "
                        f"{final_stats.get('entities_created', 0)} entities, "
                        f"{final_stats.get('relationships_created', 0)} rels, "
                        f"{final_stats.get('claims_created', 0)} claims"
                    )
                    break

                elif current_status == "failed":
                    error_msg = status.get("error", "Unknown error")
                    await swarm.update_corpus(
                        corpus_id,
                        status=CorpusStatus.FAILED,
                        error=error_msg,
                    )
                    log.error(
                        f"[{worker_id}] Corpus {corpus_id} FAILED: {error_msg}"
                    )
                    break
            else:
                # Timed out polling
                await swarm.update_corpus(
                    corpus_id,
                    status=CorpusStatus.FAILED,
                    error="Processing timed out after 20 minutes",
                )
                log.warning(
                    f"[{worker_id}] Corpus {corpus_id} timed out"
                )

        except Exception as e:
            log.error(f"[{worker_id}] Unexpected error processing {corpus_id}: {e}")
            await swarm.update_corpus(
                corpus_id,
                status=CorpusStatus.FAILED,
                error=f"Worker error: {e}",
            )

        finally:
            await swarm.update_worker(worker_id, status=WorkerStatus.DONE)
            await swarm.remove_worker(worker_id)


def _is_large_document(text: str) -> bool:
    """Detect whether a message is a large document rather than a query.

    Heuristics:
      - Length > LARGE_DOC_THRESHOLD chars
      - Low question density (few '?' relative to text length)
    """
    if len(text) < LARGE_DOC_THRESHOLD:
        return False
    question_marks = text.count("?")
    question_density = question_marks / max(len(text), 1)
    return question_density < 0.001


async def _submit_corpus(text: str, title: str = "", source: str = "") -> CorpusRecord:
    """Submit a new corpus to the swarm for background processing.

    Returns immediately — the actual processing happens asynchronously.
    """
    corpus_id = f"corpus-{uuid.uuid4().hex[:12]}"
    if not title:
        title = text[:80].replace("\n", " ").strip()

    record = CorpusRecord(
        id=corpus_id,
        title=title,
        source=source,
        total_chars=len(text),
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )

    await swarm.add_corpus(record)

    # Fire-and-forget background task — does NOT block the response
    task = asyncio.create_task(_process_corpus(corpus_id, text))
    _worker_tasks.append(task)

    # Clean up finished tasks
    _worker_tasks[:] = [t for t in _worker_tasks if not t.done()]

    log.info(
        f"Corpus {corpus_id} queued: {len(text):,} chars, "
        f"title={title[:40]}"
    )

    return record


# ---------------------------------------------------------------------------
# Query handler: answer from current swarm knowledge
# ---------------------------------------------------------------------------

_QUERY_SYSTEM_PROMPT = """You are a research analyst powered by a swarm knowledge system. You have access to a knowledge graph that has been built by decomposing and analysing large corpora of text.

**SWARM STATUS:**
{swarm_status}

**YOUR JOB:**
Answer the user's question using ONLY the knowledge graph results provided below. Be thorough, specific, and cite the source documents when possible.

**RULES:**
- If the knowledge graph has relevant information, synthesise it into a clear, comprehensive answer.
- If the knowledge graph has partial information, say what you know and clearly state what gaps remain.
- If no relevant knowledge exists yet, say so honestly — do not fabricate.
- Reference specific entities, claims, and relationships from the graph.
- If the swarm is still processing corpora, mention that more complete answers may be available once processing finishes.
- Be direct. No moralising, no disclaimers, no hedging. Just answer.

**KNOWLEDGE GRAPH RESULTS:**
{knowledge_results}
"""


async def _query_knowledge(query: str, req_id: str) -> str:
    """Query the knowledge graph for information relevant to the user's question."""
    results_parts: list[str] = []

    # 1. Hybrid search across the swarm namespace
    try:
        search_result = await knowledge_client.search(
            namespace=SWARM_NAMESPACE,
            query=query,
            mode="hybrid",
            limit=20,
        )
        results = search_result.get("results", [])
        if results:
            formatted = []
            for i, r in enumerate(results, 1):
                node_type = r.get("node_type", "")
                name = r.get("name", "")
                content = r.get("content", "")[:1500]
                score = r.get("score", 0)
                source_doc = r.get("source_doc", "")

                header = f"{i}. [{node_type}]"
                if name:
                    header += f" **{name}**"
                if source_doc:
                    header += f" (from: {source_doc})"
                header += f" [relevance: {score:.3f}]"

                formatted.append(f"{header}\n{content}" if content else header)

            results_parts.append(
                "**Hybrid Search Results:**\n" + "\n\n---\n\n".join(formatted)
            )
        else:
            results_parts.append("**Hybrid Search:** No matching results found.")
    except Exception as e:
        log.warning(f"[{req_id}] Knowledge search error: {e}")
        results_parts.append(f"**Hybrid Search:** Error — {e}")

    # 2. Try spreading activation if we have search results with named entities
    try:
        # Extract entity names from search results for graph exploration
        entity_names = []
        if results:
            for r in results[:5]:
                name = r.get("name", "")
                if name and len(name) > 2:
                    entity_names.append(name)

        if entity_names:
            activation_result = await knowledge_client.spreading_activation(
                namespace=SWARM_NAMESPACE,
                seed_concepts=entity_names[:3],
                hops=2,
                limit=10,
            )
            activations = activation_result.get("results", [])
            if activations:
                activation_text = "\n".join(
                    f"- **{a.get('name', '?')}** "
                    f"(activation: {a.get('activation', 0):.3f})"
                    for a in activations
                )
                results_parts.append(
                    f"**Graph Exploration (spreading activation from "
                    f"{', '.join(entity_names[:3])}):**\n{activation_text}"
                )
    except Exception as e:
        log.debug(f"[{req_id}] Spreading activation error (non-fatal): {e}")

    if not results_parts:
        return "(No knowledge available yet. The swarm has not processed any corpora.)"

    return "\n\n".join(results_parts)


async def _handle_query(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a query by searching the swarm's knowledge and synthesising an answer.

    This does NOT disturb the swarm — it only reads from the knowledge graph.
    """
    model_id = original_body.get("model", "swarm-miroflow")
    request_id = f"chatcmpl-swarm-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # Extract the user's question
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if not user_query:
        yield chunk("No question found in the message.", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Get swarm status preamble (sincerity mechanism)
    status_preamble = await swarm.build_sincerity_preamble()

    # Stream the thinking section with status
    yield chunk("<think>\n")
    yield chunk(status_preamble)
    yield chunk(f"**Query:** {user_query[:200]}\n\n")

    # Query the knowledge graph
    yield chunk("**[Searching knowledge graph...]**\n")
    knowledge_results = await _query_knowledge(user_query, req_id)

    # Show what we found
    result_summary = knowledge_results[:500]
    if len(knowledge_results) > 500:
        result_summary += "..."
    yield chunk(f"Found knowledge:\n{result_summary}\n\n")

    # Build the synthesis prompt
    system_prompt = _QUERY_SYSTEM_PROMPT.format(
        swarm_status=status_preamble,
        knowledge_results=knowledge_results,
    )

    # Include conversation history for context
    synthesis_messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Add recent conversation for context (last 5 messages max)
    for msg in messages[-5:]:
        if msg.get("role") in ("user", "assistant"):
            synthesis_messages.append({
                "role": msg["role"],
                "content": msg.get("content", "")[:2000] if isinstance(msg.get("content", ""), str) else "",
            })

    yield chunk("**[Synthesising answer...]**\n")
    yield chunk("</think>\n\n")

    # Call LLM for synthesis and stream the response
    try:
        async with get_throttler("mistral").throttle():
            client = http_client()
            resp_body = {
                "model": SYNTHESIS_MODEL,
                "messages": synthesis_messages,
                "max_tokens": 4096,
                "temperature": 0.3,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {UPSTREAM_KEY}",
                "Content-Type": "application/json",
            }

            async with client.stream(
                "POST",
                f"{UPSTREAM_BASE}/chat/completions",
                json=resp_body,
                headers=headers,
                timeout=120.0,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")[:500]
                    log.error(f"[{req_id}] Synthesis LLM error {resp.status_code}: {error_text}")
                    yield chunk(f"Error synthesising answer: {error_text[:200]}", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield chunk(content)
                        except json.JSONDecodeError:
                            pass

    except Exception as e:
        log.error(f"[{req_id}] Synthesis streaming error: {e}")
        yield chunk(f"\n\nError during synthesis: {e}", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    await swarm.increment_queries()

    yield chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _handle_corpus_submission(
    text: str,
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a large document by submitting it to the swarm.

    Returns immediately with a confirmation — the actual processing
    happens in the background without blocking.
    """
    model_id = original_body.get("model", "swarm-miroflow")
    request_id = f"chatcmpl-swarm-ingest-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    title = text[:80].replace("\n", " ").strip()

    yield chunk("<think>\n")
    yield chunk(f"**[Corpus Received]** {len(text):,} characters\n")
    yield chunk(f"Title: {title}...\n")
    yield chunk("Submitting to the swarm for background processing...\n")

    # Submit to the swarm — this returns immediately
    record = await _submit_corpus(text, title=title, source="chat-submission")

    yield chunk(f"Corpus ID: {record.id}\n")
    yield chunk("Status: Queued for processing\n")

    # Show current swarm state
    status = await swarm.build_sincerity_preamble()
    yield chunk(f"\n{status}")
    yield chunk("</think>\n\n")

    # User-facing response
    snapshot = await swarm.get_status_snapshot()
    yield chunk(
        f"## Corpus Submitted to Swarm\n\n"
        f"Your document ({len(text):,} characters) has been submitted to the "
        f"swarm for background processing.\n\n"
        f"**Corpus ID:** `{record.id}`\n"
        f"**Title:** {title}\n\n"
        f"The swarm will now:\n"
        f"1. Decompose the text into overlapping chunks\n"
        f"2. Extract entities, claims, evidence, and relationships\n"
        f"3. Resolve duplicate entities across all corpora\n"
        f"4. Build the knowledge graph incrementally\n\n"
        f"**You can ask questions immediately** — I will answer from "
        f"whatever knowledge the swarm has built so far and be honest "
        f"about what has and hasn't been processed yet.\n\n"
        f"This submission does not interrupt any ongoing processing. "
        f"The swarm continues its current work and will process this "
        f"corpus when a worker becomes available.\n\n"
        f"**Current swarm:** {snapshot['total_corpora']} corpora, "
        f"{snapshot['active_workers']} active workers, "
        f"{snapshot['total_entities']} entities in knowledge graph.\n"
    )

    yield chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = create_app("Swarm Deep Search Proxy")

register_standard_routes(
    app,
    service_name="swarm-deep-search-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "synthesis_model": SYNTHESIS_MODEL,
        "namespace": SWARM_NAMESPACE,
        "max_workers": MAX_SWARM_WORKERS,
        "max_concurrent_queries": MAX_CONCURRENT_QUERIES,
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "swarm-miroflow",
            "object": "model",
            "created": 1700000000,
            "owned_by": "swarm-deep-search-proxy",
            "name": "Swarm MiroFlow",
        }],
    })


@app.get("/v1/swarm/status")
async def swarm_status():
    """Return complete swarm status — what every worker is doing, progress
    on each corpus, and overall knowledge graph statistics."""
    snapshot = await swarm.get_status_snapshot()
    return JSONResponse(snapshot)


@app.get("/v1/swarm/corpora")
async def swarm_corpora():
    """List all submitted corpora with their processing status."""
    corpora = await swarm.get_corpora_list()
    return JSONResponse({"corpora": corpora, "count": len(corpora)})


@app.get("/v1/swarm/sincerity")
async def swarm_sincerity():
    """Return the current sincerity preamble — what the swarm would tell
    a user about its state right now."""
    preamble = await swarm.build_sincerity_preamble()
    return JSONResponse({"preamble": preamble})


@app.post("/v1/swarm/submit")
async def submit_corpus_api(request: Request):
    """Direct API endpoint to submit a corpus for swarm processing.

    Body: {"text": "...", "title": "optional title", "source": "optional source"}
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"error": f"Invalid request body: {e}"},
            status_code=400,
        )

    text = body.get("text", "")
    if not text or len(text) < 100:
        return JSONResponse(
            {"error": "text field is required and must be at least 100 characters"},
            status_code=400,
        )

    title = body.get("title", "")
    source = body.get("source", "api-submission")

    record = await _submit_corpus(text, title=title, source=source)
    snapshot = await swarm.get_status_snapshot()

    return JSONResponse({
        "corpus_id": record.id,
        "title": record.title,
        "total_chars": record.total_chars,
        "status": record.status.value,
        "message": (
            f"Corpus submitted for background processing. "
            f"Swarm has {snapshot['total_corpora']} corpora, "
            f"{snapshot['active_workers']} active workers."
        ),
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Invalid request body: {e}",
                    "type": "invalid_request",
                }
            },
        )

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages array is required and must not be empty",
                    "type": "invalid_request",
                }
            },
        )

    utility = is_utility_request(messages)
    log.info(
        f"[{req_id}] New request: messages={len(messages)}, "
        f"model={body.get('model', '?')}, utility={utility}"
    )

    tracker.start(req_id, utility=utility, messages=len(messages), phase="init")

    # Utility requests (title/tag generation) pass through to upstream
    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "swarm-miroflow"),
            tracker=tracker,
            log=log,
        )
    else:
        # Extract the last user message
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    user_text = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                break

        if _is_large_document(user_text):
            # Large document → submit to swarm (non-blocking)
            log.info(
                f"[{req_id}] Routing to SWARM CORPUS SUBMISSION "
                f"({len(user_text):,} chars)"
            )

            async def _guarded_submit():
                try:
                    async for event in _handle_corpus_submission(
                        user_text, body, req_id
                    ):
                        yield event
                finally:
                    tracker.finish(req_id)

            generator = _guarded_submit()
        else:
            # Regular query → answer from swarm knowledge
            if not query_limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": (
                                f"Too many concurrent queries "
                                f"({query_limiter.max_concurrent}). Try again shortly."
                            ),
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Routing to SWARM QUERY")

            async def _guarded_query():
                async with query_limiter.hold():
                    async for event in _handle_query(messages, body, req_id):
                        yield event

            generator = _guarded_query()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Module-level startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn as _uvicorn
    _uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
