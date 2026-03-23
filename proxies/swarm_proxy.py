#!/usr/bin/env python3
"""
Swarm Deep Search Proxy for Open WebUI.

A fully self-contained swarm-based proxy that decomposes large corpora of text
using an agentic swarm -- no external infrastructure required.  Background
worker agents continuously process corpora -- chunking, extracting entities
and relationships via the LLM, building an in-memory knowledge graph -- while
queries are answered non-disruptively from whatever knowledge the swarm has
built so far.

Key design principles:
  * Sending a prompt does NOT disturb the swarm from what it is doing.
  * The proxy is *sincere* about what is happening: it reports real
    swarm state, processing progress, and knowledge coverage honestly.
  * Further large corpora are treated identically to the initial send --
    they are queued additively without resetting existing work.
  * Zero external infrastructure -- everything runs within this process.

Architecture:
  Browser -> Open WebUI -> Swarm Proxy (port 9500)
                              |
              +---------------+---------------+
              |               |               |
        Swarm Director   Query Handler   Status Reporter
              |
        Worker Pool (LLM-powered agents)
              |
        In-Memory Knowledge Store
         (chunks, entities, relationships, claims)
"""

import asyncio
import collections
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

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
WORKER_MODEL = os.getenv("SWARM_WORKER_MODEL", "mistral-small-latest")
LISTEN_PORT = env_int("SWARM_PROXY_PORT", 9500, minimum=1)
MAX_CONCURRENT_QUERIES = env_int("SWARM_MAX_CONCURRENT_QUERIES", 4, minimum=1)
MAX_SWARM_WORKERS = env_int("SWARM_MAX_WORKERS", 6, minimum=1)

# Chunk parameters for corpus decomposition
CHUNK_SIZE = int(os.getenv("SWARM_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("SWARM_CHUNK_OVERLAP", "200"))

# Large document detection threshold (chars)
LARGE_DOC_THRESHOLD = int(os.getenv("SWARM_LARGE_DOC_THRESHOLD", "5000"))

log.info(
    f"Config: synthesis_model={SYNTHESIS_MODEL}, worker_model={WORKER_MODEL}, "
    f"upstream={UPSTREAM_BASE}, port={LISTEN_PORT}, "
    f"max_queries={MAX_CONCURRENT_QUERIES}, max_workers={MAX_SWARM_WORKERS}"
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

tracker = RequestTracker()
query_limiter = ConcurrencyLimiter(MAX_CONCURRENT_QUERIES)
worker_semaphore = asyncio.Semaphore(MAX_SWARM_WORKERS)


# ---------------------------------------------------------------------------
# In-memory Knowledge Store
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeChunk:
    """A text chunk from a decomposed corpus."""
    id: str
    corpus_id: str
    corpus_title: str
    text: str
    chunk_index: int
    total_chunks: int
    word_freq: dict = field(default_factory=dict)


@dataclass
class Entity:
    """An entity extracted from a chunk by the LLM."""
    name: str
    entity_type: str  # person, org, concept, location, event, etc.
    description: str
    corpus_id: str
    chunk_id: str
    mentions: int = 1


@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str
    target: str
    relation_type: str
    description: str
    corpus_id: str
    chunk_id: str


@dataclass
class Claim:
    """An atomic claim / fact extracted from the corpus."""
    text: str
    confidence: str  # high, medium, low
    source_chunk_id: str
    corpus_id: str
    entities: list[str] = field(default_factory=list)


class KnowledgeStore:
    """Thread-safe in-memory knowledge store.

    This is the swarm's brain -- all extracted knowledge lives here.
    Workers write to it as they process chunks; query handlers read
    from it without blocking.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._entities: dict[str, Entity] = {}  # keyed by lowercase name
        self._relationships: list[Relationship] = []
        self._claims: list[Claim] = []

    async def add_chunk(self, chunk: KnowledgeChunk) -> None:
        async with self._lock:
            self._chunks[chunk.id] = chunk

    async def add_entity(self, entity: Entity) -> None:
        async with self._lock:
            key = entity.name.lower().strip()
            if key in self._entities:
                existing = self._entities[key]
                existing.mentions += 1
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
            else:
                self._entities[key] = entity

    async def add_relationship(self, rel: Relationship) -> None:
        async with self._lock:
            self._relationships.append(rel)

    async def add_claim(self, claim: Claim) -> None:
        async with self._lock:
            self._claims.append(claim)

    async def search(self, query: str, limit: int = 20) -> dict:
        """Search knowledge store using TF-IDF-like scoring.

        Returns matching chunks, entities, relationships, and claims
        ranked by relevance to the query.
        """
        query_terms = _tokenize(query)
        if not query_terms:
            return {
                "chunks": [], "entities": [],
                "relationships": [], "claims": [],
            }

        async with self._lock:
            # Score chunks by term overlap
            chunk_scores: list[tuple[float, KnowledgeChunk]] = []
            total_chunks = max(len(self._chunks), 1)
            for chunk in self._chunks.values():
                score = _score_text(
                    query_terms, chunk.word_freq, total_chunks,
                )
                if score > 0:
                    chunk_scores.append((score, chunk))
            chunk_scores.sort(key=lambda x: x[0], reverse=True)

            # Score entities by name and description match
            entity_scores: list[tuple[float, Entity]] = []
            for entity in self._entities.values():
                name_score = _score_text_raw(
                    query_terms, entity.name.lower(),
                )
                desc_score = (
                    _score_text_raw(
                        query_terms, entity.description.lower(),
                    ) * 0.5
                )
                total = name_score + desc_score
                if total > 0:
                    entity_scores.append((total, entity))
            entity_scores.sort(key=lambda x: x[0], reverse=True)

            # Score claims
            claim_scores: list[tuple[float, Claim]] = []
            for claim in self._claims:
                score = _score_text_raw(query_terms, claim.text.lower())
                if score > 0:
                    claim_scores.append((score, claim))
            claim_scores.sort(key=lambda x: x[0], reverse=True)

            # Gather relevant relationships (connected to top entities)
            top_entity_names = {
                e.name.lower() for _, e in entity_scores[:10]
            }
            relevant_rels = [
                r for r in self._relationships
                if r.source.lower() in top_entity_names
                or r.target.lower() in top_entity_names
            ]

            return {
                "chunks": [
                    {
                        "id": c.id,
                        "corpus_title": c.corpus_title,
                        "text": c.text[:1500],
                        "score": round(s, 4),
                        "chunk_index": c.chunk_index,
                    }
                    for s, c in chunk_scores[:limit]
                ],
                "entities": [
                    {
                        "name": e.name,
                        "type": e.entity_type,
                        "description": e.description,
                        "mentions": e.mentions,
                        "score": round(s, 4),
                    }
                    for s, e in entity_scores[:limit]
                ],
                "relationships": [
                    {
                        "source": r.source,
                        "target": r.target,
                        "type": r.relation_type,
                        "description": r.description,
                    }
                    for r in relevant_rels[:limit]
                ],
                "claims": [
                    {
                        "text": c.text,
                        "confidence": c.confidence,
                        "entities": c.entities,
                        "score": round(s, 4),
                    }
                    for s, c in claim_scores[:limit]
                ],
            }

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "chunks": len(self._chunks),
                "entities": len(self._entities),
                "relationships": len(self._relationships),
                "claims": len(self._claims),
            }


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, removing short/stop words."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "not", "no", "nor", "so", "yet", "both", "each",
        "this", "that", "these", "those", "it", "its", "i", "me", "my",
        "we", "our", "you", "your", "he", "she", "they", "them", "his",
        "her", "their", "what", "which", "who", "whom", "how", "when",
        "where", "why", "if", "then", "than", "more", "very", "just",
        "about", "also", "some", "any", "all", "most", "other", "such",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]


def _build_word_freq(text: str) -> dict[str, int]:
    """Build word frequency map for a text."""
    freq: dict[str, int] = collections.Counter(_tokenize(text))
    return dict(freq)


def _score_text(
    query_terms: list[str], word_freq: dict, total_docs: int,
) -> float:
    """TF-IDF-like scoring of a document against query terms."""
    if not word_freq:
        return 0.0
    total_words = max(sum(word_freq.values()), 1)
    score = 0.0
    for term in query_terms:
        tf = word_freq.get(term, 0) / total_words
        if tf > 0:
            score += tf * (1.0 + math.log(max(total_docs, 1)))
    return score


def _score_text_raw(query_terms: list[str], text: str) -> float:
    """Simple term-overlap scoring against raw text."""
    text_lower = text.lower()
    score = 0.0
    for term in query_terms:
        if term in text_lower:
            score += 1.0
    return score


# Global knowledge store
knowledge = KnowledgeStore()


# ---------------------------------------------------------------------------
# Data models for swarm state
# ---------------------------------------------------------------------------

class CorpusStatus(str, Enum):
    """Processing status for an ingested corpus."""
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(str, Enum):
    """Status for an individual swarm worker."""
    IDLE = "idle"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
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


@dataclass
class SwarmWorkerInfo:
    """Tracks what a single worker is currently doing."""
    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    corpus_id: str = ""
    current_task: str = ""
    started_at: float = 0.0


class SwarmState:
    """Global swarm state -- holds all corpora, workers, and knowledge stats.

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
        """Return a complete snapshot of swarm state for reporting."""
        kstats = await knowledge.stats()

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
                uptime = (
                    round(time.monotonic() - w.started_at, 1)
                    if w.started_at else 0
                )
                workers_list.append({
                    "id": w.id,
                    "status": w.status.value,
                    "corpus_id": w.corpus_id,
                    "current_task": w.current_task,
                    "uptime_s": uptime,
                })

            completed = sum(
                1 for c in self._corpora.values()
                if c.status == CorpusStatus.COMPLETED
            )
            processing = sum(
                1 for c in self._corpora.values()
                if c.status not in (
                    CorpusStatus.COMPLETED, CorpusStatus.FAILED,
                    CorpusStatus.QUEUED,
                )
            )
            queued = sum(
                1 for c in self._corpora.values()
                if c.status == CorpusStatus.QUEUED
            )
            failed = sum(
                1 for c in self._corpora.values()
                if c.status == CorpusStatus.FAILED
            )
            total_chars = sum(
                c.total_chars for c in self._corpora.values()
            )
            active = sum(
                1 for w in self._workers.values()
                if w.status not in (WorkerStatus.IDLE, WorkerStatus.DONE)
            )

            return {
                "swarm_uptime_s": round(
                    time.monotonic() - self._started_at, 1,
                ),
                "total_corpora": len(self._corpora),
                "corpora_completed": completed,
                "corpora_processing": processing,
                "corpora_queued": queued,
                "corpora_failed": failed,
                "total_chars_ingested": total_chars,
                "total_entities": kstats["entities"],
                "total_relationships": kstats["relationships"],
                "total_claims": kstats["claims"],
                "total_chunks": kstats["chunks"],
                "active_workers": active,
                "total_queries_answered": self._total_queries_answered,
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

        This is the 'sincerity' mechanism -- the proxy tells the user
        exactly what is happening right now, no sugar-coating.
        """
        snapshot = await self.get_status_snapshot()

        if snapshot["total_corpora"] == 0:
            return (
                "**[Swarm Status]** No corpora have been submitted yet. "
                "Send me a large body of text and I will begin decomposing "
                "and understanding it. "
                "You can ask questions at any time.\n\n"
            )

        parts = ["**[Swarm Status]**"]

        parts.append(
            f" {snapshot['total_corpora']} corpus/corpora ingested "
            f"({snapshot['total_chars_ingested']:,} chars total)."
        )

        if snapshot["corpora_processing"] > 0:
            parts.append(
                f" **Currently processing "
                f"{snapshot['corpora_processing']} corpus/corpora** "
                f"with {snapshot['active_workers']} active worker(s)."
            )
            for c in snapshot["corpora"]:
                if c["status"] not in ("completed", "failed", "queued"):
                    pct = int(c["progress"] * 100)
                    parts.append(
                        f"\n  -> *{c['title'][:60]}*: "
                        f"{c['status']} -- {pct}% "
                        f"({c['chunks_processed']}/"
                        f"{c['total_chunks']} chunks)"
                    )

        if snapshot["corpora_queued"] > 0:
            parts.append(
                f" {snapshot['corpora_queued']} corpus/corpora "
                f"waiting in queue."
            )

        if snapshot["corpora_completed"] > 0:
            parts.append(
                f" {snapshot['corpora_completed']} fully processed."
            )

        if snapshot["corpora_failed"] > 0:
            parts.append(
                f" WARNING: {snapshot['corpora_failed']} failed."
            )

        if (
            snapshot["total_entities"] > 0
            or snapshot["total_claims"] > 0
        ):
            parts.append(
                f"\n**Knowledge built:** "
                f"{snapshot['total_entities']} entities, "
                f"{snapshot['total_relationships']} relationships, "
                f"{snapshot['total_claims']} claims across "
                f"{snapshot['total_chunks']} chunks."
            )

        if (
            snapshot["corpora_processing"] > 0
            or snapshot["corpora_queued"] > 0
        ):
            parts.append(
                "\n*Note: The swarm is still working. My answers "
                "reflect knowledge extracted so far -- they may become "
                "more complete as processing continues. "
                "This does not affect the swarm's work.*"
            )

        return "".join(parts) + "\n\n"


# Global swarm state
swarm = SwarmState()

# Background worker tasks (kept alive for the process lifetime)
_worker_tasks: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = (
    "You are a knowledge extraction agent in a swarm system. "
    "Your job is to extract structured knowledge from a text chunk.\n\n"
    'Given the following text chunk from "{corpus_title}" '
    "(chunk {chunk_index}/{total_chunks}), extract:\n\n"
    "1. **Entities**: Named things (people, organizations, concepts, "
    "locations, events, technologies, theories, etc.)\n"
    "2. **Relationships**: How entities relate to each other\n"
    "3. **Claims**: Atomic factual statements / assertions made in "
    "the text\n\n"
    "Respond with ONLY valid JSON in this exact format "
    "(no markdown, no code fences):\n"
    "{{\n"
    '  "entities": [\n'
    '    {{"name": "Entity Name", "type": '
    '"person|org|concept|location|event|technology|theory|other", '
    '"description": "Brief description"}}\n'
    "  ],\n"
    '  "relationships": [\n'
    '    {{"source": "Entity A", "target": "Entity B", '
    '"type": "relationship_type", '
    '"description": "How they relate"}}\n'
    "  ],\n"
    '  "claims": [\n'
    '    {{"text": "Atomic factual claim", '
    '"confidence": "high|medium|low", '
    '"entities": ["Entity A", "Entity B"]}}\n'
    "  ]\n"
    "}}\n\n"
    "Extract as many entities, relationships, and claims as the text "
    "supports. Be thorough but accurate. Only extract what the text "
    "actually states or strongly implies.\n\n"
    "TEXT CHUNK:\n{chunk_text}"
)

_QUERY_SYSTEM_PROMPT = (
    "You are a research analyst powered by a swarm knowledge system. "
    "The swarm has decomposed large corpora of text into structured "
    "knowledge: entities, relationships, and claims.\n\n"
    "**SWARM STATUS:**\n{swarm_status}\n\n"
    "**YOUR JOB:**\n"
    "Answer the user's question using ONLY the knowledge results "
    "provided below. Be thorough, specific, and cite the source "
    "documents when possible.\n\n"
    "**RULES:**\n"
    "- If the knowledge store has relevant information, synthesise "
    "it into a clear, comprehensive answer.\n"
    "- If the knowledge store has partial information, say what you "
    "know and clearly state what gaps remain.\n"
    "- If no relevant knowledge exists yet, say so honestly -- "
    "do not fabricate.\n"
    "- Reference specific entities, claims, and relationships from "
    "the results.\n"
    "- If the swarm is still processing corpora, mention that more "
    "complete answers may be available once processing finishes.\n"
    "- Be direct. No moralising, no disclaimers, no hedging. "
    "Just answer.\n\n"
    "**KNOWLEDGE RESULTS:**\n{knowledge_results}"
)


async def _call_llm(
    messages: list[dict],
    *,
    model: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> dict:
    """Call the upstream LLM.

    Returns {"content": str} or {"error": str}.
    """
    resolved_model = model or WORKER_MODEL
    body = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {UPSTREAM_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            async with get_throttler("mistral").throttle():
                client = http_client()
                resp = await client.post(
                    f"{UPSTREAM_BASE}/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=120.0,
                )

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = (
                        choices[0].get("message", {}).get("content", "")
                    )
                    return {"content": content}
                return {"error": "No choices in response"}

            if resp.status_code in (429, 500, 502, 503, 504):
                wait_secs = [5, 15, 30][min(attempt, 2)]
                log.warning(
                    f"LLM {resp.status_code}, retrying in {wait_secs}s "
                    f"(attempt {attempt + 1}/3)"
                )
                await asyncio.sleep(wait_secs)
                continue

            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            }

        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(5)
                continue
            return {"error": str(e)}

    return {"error": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# Corpus chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        start += step
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Swarm worker: corpus processing pipeline (fully self-contained)
# ---------------------------------------------------------------------------

async def _extract_from_chunk(
    chunk_text: str,
    corpus_id: str,
    corpus_title: str,
    chunk_index: int,
    total_chunks: int,
    worker_id: str,
) -> tuple[int, int, int]:
    """Use LLM to extract entities, relationships, and claims.

    Returns (entities_count, relationships_count, claims_count).
    """
    chunk_id = f"{corpus_id}-chunk-{chunk_index}"

    # Store the raw chunk in the knowledge store
    kchunk = KnowledgeChunk(
        id=chunk_id,
        corpus_id=corpus_id,
        corpus_title=corpus_title,
        text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        word_freq=_build_word_freq(chunk_text),
    )
    await knowledge.add_chunk(kchunk)

    # Ask the LLM to extract structured knowledge
    prompt = _EXTRACTION_PROMPT.format(
        corpus_title=corpus_title,
        chunk_index=chunk_index + 1,
        total_chunks=total_chunks,
        chunk_text=chunk_text[:3000],
    )

    result = await _call_llm(
        [{"role": "user", "content": prompt}],
        model=WORKER_MODEL,
        max_tokens=4096,
        temperature=0.1,
    )

    if "error" in result:
        log.warning(
            f"[{worker_id}] Extraction error for chunk "
            f"{chunk_index}: {result['error']}"
        )
        return (0, 0, 0)

    # Parse the JSON response
    content = result["content"].strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        extracted = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                extracted = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning(
                    f"[{worker_id}] Failed to parse extraction "
                    f"for chunk {chunk_index}"
                )
                return (0, 0, 0)
        else:
            log.warning(
                f"[{worker_id}] No JSON found in extraction "
                f"for chunk {chunk_index}"
            )
            return (0, 0, 0)

    entities_count = 0
    rels_count = 0
    claims_count = 0

    # Store entities
    for e in extracted.get("entities", []):
        name = e.get("name", "").strip()
        if not name:
            continue
        await knowledge.add_entity(Entity(
            name=name,
            entity_type=e.get("type", "other"),
            description=e.get("description", ""),
            corpus_id=corpus_id,
            chunk_id=chunk_id,
        ))
        entities_count += 1

    # Store relationships
    for r in extracted.get("relationships", []):
        source = r.get("source", "").strip()
        target = r.get("target", "").strip()
        if not source or not target:
            continue
        await knowledge.add_relationship(Relationship(
            source=source,
            target=target,
            relation_type=r.get("type", "related_to"),
            description=r.get("description", ""),
            corpus_id=corpus_id,
            chunk_id=chunk_id,
        ))
        rels_count += 1

    # Store claims
    for c in extracted.get("claims", []):
        claim_text = c.get("text", "").strip()
        if not claim_text:
            continue
        await knowledge.add_claim(Claim(
            text=claim_text,
            confidence=c.get("confidence", "medium"),
            source_chunk_id=chunk_id,
            corpus_id=corpus_id,
            entities=c.get("entities", []),
        ))
        claims_count += 1

    return (entities_count, rels_count, claims_count)


async def _process_corpus(corpus_id: str, text: str) -> None:
    """Process a single corpus through the swarm pipeline.

    Steps:
      1. Decompose text into overlapping chunks
      2. For each chunk, extract entities/relationships/claims via LLM
      3. Store everything in the in-memory knowledge store
      4. Update swarm state at each stage

    This runs as a background task and does NOT block any queries.
    """
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    async with worker_semaphore:
        await swarm.register_worker(worker_id)

        try:
            record = await swarm.get_corpus(corpus_id)
            if not record:
                log.error(
                    f"[{worker_id}] Corpus {corpus_id} not found",
                )
                return

            # Step 1: Chunking
            await swarm.update_worker(
                worker_id,
                status=WorkerStatus.CHUNKING,
                corpus_id=corpus_id,
                current_task=f"Chunking: {record.title[:40]}",
                started_at=time.monotonic(),
            )
            await swarm.update_corpus(
                corpus_id,
                status=CorpusStatus.CHUNKING,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            chunks = _chunk_text(text)
            total_chunks = len(chunks)
            await swarm.update_corpus(
                corpus_id, total_chunks=total_chunks,
            )

            log.info(
                f"[{worker_id}] Corpus {corpus_id} "
                f"({record.title[:40]}): "
                f"{record.total_chars:,} chars -> "
                f"{total_chunks} chunks"
            )

            # Step 2: Extract knowledge from each chunk
            await swarm.update_worker(
                worker_id,
                status=WorkerStatus.EXTRACTING,
                current_task=f"Extracting: {record.title[:40]}",
            )
            await swarm.update_corpus(
                corpus_id, status=CorpusStatus.EXTRACTING,
            )

            total_entities = 0
            total_rels = 0
            total_claims = 0

            for i, chunk_text in enumerate(chunks):
                await swarm.update_worker(
                    worker_id,
                    current_task=(
                        f"Extracting chunk {i + 1}/{total_chunks}: "
                        f"{record.title[:30]}"
                    ),
                )

                try:
                    ent_count, rel_count, claim_count = (
                        await _extract_from_chunk(
                            chunk_text,
                            corpus_id,
                            record.title,
                            i,
                            total_chunks,
                            worker_id,
                        )
                    )
                    total_entities += ent_count
                    total_rels += rel_count
                    total_claims += claim_count

                except Exception as e:
                    log.warning(
                        f"[{worker_id}] Error extracting chunk "
                        f"{i}/{total_chunks}: {e}"
                    )

                # Update progress after each chunk
                await swarm.update_corpus(
                    corpus_id,
                    chunks_processed=i + 1,
                    entities_extracted=total_entities,
                    relationships_extracted=total_rels,
                    claims_extracted=total_claims,
                )

                if i < total_chunks - 1:
                    await asyncio.sleep(0.1)

            # Step 3: Mark completed
            await swarm.update_corpus(
                corpus_id,
                status=CorpusStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            log.info(
                f"[{worker_id}] Corpus {corpus_id} COMPLETED: "
                f"{total_entities} entities, "
                f"{total_rels} relationships, "
                f"{total_claims} claims from {total_chunks} chunks"
            )

        except Exception as e:
            log.error(
                f"[{worker_id}] Unexpected error processing "
                f"{corpus_id}: {e}"
            )
            await swarm.update_corpus(
                corpus_id,
                status=CorpusStatus.FAILED,
                error=f"Worker error: {e}",
            )

        finally:
            await swarm.update_worker(
                worker_id, status=WorkerStatus.DONE,
            )
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


async def _submit_corpus(
    text: str, title: str = "", source: str = "",
) -> CorpusRecord:
    """Submit a new corpus to the swarm for background processing.

    Returns immediately -- the actual processing happens asynchronously.
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

    # Fire-and-forget background task -- does NOT block the response
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

async def _query_knowledge(query: str, req_id: str) -> str:
    """Query the in-memory knowledge store for relevant information."""
    results = await knowledge.search(query, limit=20)
    parts: list[str] = []

    # Format chunks
    chunks = results.get("chunks", [])
    if chunks:
        formatted = []
        for i, c in enumerate(chunks, 1):
            header = (
                f"{i}. [chunk] from *{c['corpus_title']}* "
                f"[relevance: {c['score']:.3f}]"
            )
            formatted.append(f"{header}\n{c['text']}")
        parts.append(
            "**Matching Text Chunks:**\n"
            + "\n\n---\n\n".join(formatted)
        )

    # Format entities
    entities = results.get("entities", [])
    if entities:
        entity_text = "\n".join(
            f"- **{e['name']}** ({e['type']}) -- "
            f"{e['description']} "
            f"[{e['mentions']} mentions, "
            f"relevance: {e['score']:.3f}]"
            for e in entities[:15]
        )
        parts.append(f"**Entities:**\n{entity_text}")

    # Format relationships
    rels = results.get("relationships", [])
    if rels:
        rel_text = "\n".join(
            f"- {r['source']} --[{r['type']}]--> "
            f"{r['target']}: {r['description']}"
            for r in rels[:15]
        )
        parts.append(f"**Relationships:**\n{rel_text}")

    # Format claims
    claims = results.get("claims", [])
    if claims:
        claim_text = "\n".join(
            f"- [{c['confidence']}] {c['text']}"
            for c in claims[:15]
        )
        parts.append(f"**Claims:**\n{claim_text}")

    if not parts:
        return (
            "(No knowledge available yet. "
            "The swarm has not processed any corpora.)"
        )

    return "\n\n".join(parts)


async def _handle_query(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a query by searching knowledge and synthesising.

    This does NOT disturb the swarm -- it only reads from the store.
    """
    model_id = original_body.get("model", "swarm-miroflow")
    request_id = f"chatcmpl-swarm-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def sse(
        content: str, finish_reason: Optional[str] = None,
    ) -> str:
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
        yield sse(
            "No question found in the message.",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    # Get swarm status preamble (sincerity mechanism)
    status_preamble = await swarm.build_sincerity_preamble()

    # Stream the thinking section with status
    yield sse("<think>\n")
    yield sse(status_preamble)
    yield sse(f"**Query:** {user_query[:200]}\n\n")

    # Query the knowledge store
    yield sse("**[Searching swarm knowledge...]**\n")
    knowledge_results = await _query_knowledge(user_query, req_id)

    # Show what we found
    result_summary = knowledge_results[:500]
    if len(knowledge_results) > 500:
        result_summary += "..."
    yield sse(f"Found knowledge:\n{result_summary}\n\n")

    # Build the synthesis prompt
    system_prompt = _QUERY_SYSTEM_PROMPT.format(
        swarm_status=status_preamble,
        knowledge_results=knowledge_results,
    )

    synthesis_messages = [
        {"role": "system", "content": system_prompt},
    ]
    for msg in messages[-5:]:
        if msg.get("role") in ("user", "assistant"):
            msg_content = msg.get("content", "")
            if isinstance(msg_content, str):
                msg_content = msg_content[:2000]
            else:
                msg_content = ""
            synthesis_messages.append({
                "role": msg["role"],
                "content": msg_content,
            })

    yield sse("**[Synthesising answer...]**\n")
    yield sse("</think>\n\n")

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
                    error_text = error_body.decode(
                        "utf-8", errors="replace",
                    )[:500]
                    log.error(
                        f"[{req_id}] Synthesis LLM error "
                        f"{resp.status_code}: {error_text}"
                    )
                    yield sse(
                        "Error synthesising answer: "
                        f"{error_text[:200]}",
                        finish_reason="stop",
                    )
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
                                    yield sse(content)
                        except json.JSONDecodeError:
                            pass

    except Exception as e:
        log.error(f"[{req_id}] Synthesis streaming error: {e}")
        yield sse(
            f"\n\nError during synthesis: {e}",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    await swarm.increment_queries()

    yield sse("", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _handle_corpus_submission(
    text: str,
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a large document by submitting it to the swarm.

    Returns immediately with a confirmation -- the actual processing
    happens in the background without blocking.
    """
    model_id = original_body.get("model", "swarm-miroflow")
    request_id = f"chatcmpl-swarm-ingest-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def sse(
        content: str, finish_reason: Optional[str] = None,
    ) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    title = text[:80].replace("\n", " ").strip()

    yield sse("<think>\n")
    yield sse(f"**[Corpus Received]** {len(text):,} characters\n")
    yield sse(f"Title: {title}...\n")
    yield sse(
        "Submitting to the swarm for background processing...\n",
    )

    # Submit to the swarm -- this returns immediately
    record = await _submit_corpus(
        text, title=title, source="chat-submission",
    )

    yield sse(f"Corpus ID: {record.id}\n")
    yield sse("Status: Queued for processing\n")

    # Show current swarm state
    status = await swarm.build_sincerity_preamble()
    yield sse(f"\n{status}")
    yield sse("</think>\n\n")

    # User-facing response
    snapshot = await swarm.get_status_snapshot()
    yield sse(
        f"## Corpus Submitted to Swarm\n\n"
        f"Your document ({len(text):,} characters) has been submitted "
        f"to the swarm for background processing.\n\n"
        f"**Corpus ID:** `{record.id}`\n"
        f"**Title:** {title}\n\n"
        f"The swarm will now:\n"
        f"1. Decompose the text into overlapping chunks\n"
        f"2. Use LLM agents to extract entities, claims, and "
        f"relationships from each chunk\n"
        f"3. Merge extracted knowledge into the swarm's in-memory "
        f"knowledge store\n"
        f"4. Make all extracted knowledge immediately queryable\n\n"
        f"**You can ask questions immediately** -- I will answer from "
        f"whatever knowledge the swarm has built so far and be honest "
        f"about what has and hasn't been processed yet.\n\n"
        f"This submission does not interrupt any ongoing processing. "
        f"The swarm continues its current work and will process this "
        f"corpus when a worker becomes available.\n\n"
        f"**Current swarm:** {snapshot['total_corpora']} corpora, "
        f"{snapshot['active_workers']} active workers, "
        f"{snapshot['total_entities']} entities in knowledge store.\n"
    )

    yield sse("", finish_reason="stop")
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
        "worker_model": WORKER_MODEL,
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
    """Return complete swarm status."""
    snapshot = await swarm.get_status_snapshot()
    return JSONResponse(snapshot)


@app.get("/v1/swarm/corpora")
async def swarm_corpora():
    """List all submitted corpora with their processing status."""
    corpora = await swarm.get_corpora_list()
    return JSONResponse({"corpora": corpora, "count": len(corpora)})


@app.get("/v1/swarm/sincerity")
async def swarm_sincerity():
    """Return the current sincerity preamble."""
    preamble = await swarm.build_sincerity_preamble()
    return JSONResponse({"preamble": preamble})


@app.get("/v1/swarm/knowledge")
async def swarm_knowledge_stats():
    """Return statistics about the in-memory knowledge store."""
    stats = await knowledge.stats()
    return JSONResponse(stats)


@app.post("/v1/swarm/submit")
async def submit_corpus_api(request: Request):
    """Direct API endpoint to submit a corpus for swarm processing.

    Body: {"text": "...", "title": "optional", "source": "optional"}
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
            {
                "error": (
                    "text field is required and must be "
                    "at least 100 characters"
                ),
            },
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
                    "message": (
                        "messages array is required "
                        "and must not be empty"
                    ),
                    "type": "invalid_request",
                }
            },
        )

    utility = is_utility_request(messages)
    log.info(
        f"[{req_id}] New request: messages={len(messages)}, "
        f"model={body.get('model', '?')}, utility={utility}"
    )

    tracker.start(
        req_id, utility=utility,
        messages=len(messages), phase="init",
    )

    # Utility requests (title/tag generation) pass through
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
                        if isinstance(p, dict)
                        and p.get("type") == "text"
                    )
                break

        if _is_large_document(user_text):
            # Large document -> submit to swarm (non-blocking)
            log.info(
                f"[{req_id}] Routing to SWARM CORPUS SUBMISSION "
                f"({len(user_text):,} chars)"
            )

            async def _guarded_submit():
                try:
                    async for event in _handle_corpus_submission(
                        user_text, body, req_id,
                    ):
                        yield event
                finally:
                    tracker.finish(req_id)

            generator = _guarded_submit()
        else:
            # Regular query -> answer from swarm knowledge
            if not query_limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": (
                                "Too many concurrent queries "
                                f"({query_limiter.max_concurrent})"
                                ". Try again shortly."
                            ),
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Routing to SWARM QUERY")

            async def _guarded_query():
                try:
                    async with query_limiter.hold():
                        async for event in _handle_query(
                            messages, body, req_id,
                        ):
                            yield event
                finally:
                    tracker.finish(req_id)

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
    _uvicorn.run(
        app, host="0.0.0.0", port=LISTEN_PORT, log_level="info",
    )
