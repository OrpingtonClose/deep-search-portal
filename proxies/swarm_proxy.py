#!/usr/bin/env python3
"""
Ruflo Hive Deep Search Proxy for LibreChat.

A swarm-based proxy implementing the Ruflo Hive architecture -- each worker
holds its own layered knowledge, knowledge propagates through multi-channel
gossip, and the swarm perpetually mines the corpus for deeper insights.

Key design principles:
  * Workers are long-lived and hold layered memory (L0-L3 + pointer map).
  * Multi-channel gossip propagates claims, insights, and contradictions.
  * Perpetual mining re-examines raw chunks with fresh eyes after gossip.
  * Queries route to relevant workers via the pointer network.
  * Sending a prompt does NOT disturb the hive from what it is doing.
  * The proxy is *sincere* about what is happening: it reports real
    hive state, processing progress, and knowledge coverage honestly.
  * Further large corpora are treated identically to the initial send --
    they are queued additively without resetting existing work.

Architecture:
  Browser -> LibreChat -> Swarm Proxy (port 9500)
                              |
              +---------------+---------------+
              |               |               |
        Hive Director    Query Router    Status Reporter
              |               |
        HiveWorker Pool   Pointer Network
         (layered memory)  (topic -> worker routing)
              |
        Multi-Channel Gossip
         (A: understanding, B: claims,
          C: insights, D: contradictions)
              |
        Perpetual Mining Loop
         (reflection, re-reading, sub-swarms)
              |
        Rotating Queen + Hive Oracle
"""

import asyncio
import json
import os
import random
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
    extract_user_text,
    extract_user_text_with_attachments,
    get_throttler,
    http_client,
    is_utility_request,
    make_sse_chunk,
    parse_attachments,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
    utility_passthrough_json,
)

import langfuse_config as lf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("SWARM_PROXY_LOG_DIR", "/opt/swarm_proxy_logs")
log = setup_logging("swarm-proxy", LOG_DIR)

UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.x.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "grok-3-fast")
SYNTHESIS_MODEL = os.getenv("SWARM_SYNTHESIS_MODEL", os.getenv("UPSTREAM_MODEL", "grok-3-fast"))
WORKER_MODEL = os.getenv("SWARM_WORKER_MODEL", os.getenv("SUBAGENT_MODEL", "grok-3-fast"))
LISTEN_PORT = env_int("SWARM_PROXY_PORT", 9500, minimum=1)
MAX_CONCURRENT_QUERIES = env_int("SWARM_MAX_CONCURRENT_QUERIES", 4, minimum=1)
MAX_SWARM_WORKERS = env_int("SWARM_MAX_WORKERS", 6, minimum=1)

# Chunk parameters for corpus decomposition
CHUNK_SIZE = int(os.getenv("SWARM_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("SWARM_CHUNK_OVERLAP", "200"))

# Large document detection threshold (chars)
LARGE_DOC_THRESHOLD = int(os.getenv("SWARM_LARGE_DOC_THRESHOLD", "5000"))

# Hive configuration
HIVE_GOSSIP_ROUNDS = int(os.getenv("SWARM_GOSSIP_ROUNDS", "2"))
HIVE_MINING_INTERVAL = int(os.getenv("SWARM_MINING_INTERVAL", "120"))
HIVE_MAX_MINING_CYCLES = int(os.getenv("SWARM_MAX_MINING_CYCLES", "10"))
HIVE_MAX_PEERS_PER_GOSSIP = int(os.getenv("SWARM_MAX_PEERS_PER_GOSSIP", "5"))
HIVE_CHUNKS_PER_WORKER = int(os.getenv("SWARM_CHUNKS_PER_WORKER", "4"))
HIVE_SUBSWARM_THRESHOLD = int(os.getenv("SWARM_SUBSWARM_THRESHOLD", "5"))
HIVE_ORACLE_INTERVAL = int(os.getenv("SWARM_ORACLE_INTERVAL", "3"))

log.info(
    f"Config: synthesis_model={SYNTHESIS_MODEL}, worker_model={WORKER_MODEL}, "
    f"upstream={UPSTREAM_BASE}, port={LISTEN_PORT}, "
    f"max_queries={MAX_CONCURRENT_QUERIES}, max_workers={MAX_SWARM_WORKERS}, "
    f"gossip_rounds={HIVE_GOSSIP_ROUNDS}, mining_interval={HIVE_MINING_INTERVAL}s, "
    f"max_mining_cycles={HIVE_MAX_MINING_CYCLES}"
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

tracker = RequestTracker()
query_limiter = ConcurrencyLimiter(MAX_CONCURRENT_QUERIES)
worker_semaphore = asyncio.Semaphore(MAX_SWARM_WORKERS)


# ---------------------------------------------------------------------------
# Hive Data Models (4 layers + pointer map)
# ---------------------------------------------------------------------------

@dataclass
class ClaimEntry:
    """A structured claim in a worker's Layer 2 ledger."""
    text: str
    confidence: str  # high, medium, low
    entities: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    provenance_chunk_index: int = 0
    corpus_id: str = ""
    contradiction_flag: bool = False
    contradiction_details: str = ""
    version: int = 0


@dataclass
class InsightEntry:
    """A Layer 3 emergent insight."""
    text: str
    insight_type: str  # pattern, implication, hypothesis, contradiction, open_question
    source_worker_id: str = ""
    cycle_created: int = 0
    replicated_to: list[str] = field(default_factory=list)
    version: int = 0


@dataclass
class Pointer:
    """A soft reference to another worker's expertise."""
    topic: str
    target_worker_id: str
    strength: float = 0.5
    last_verified_cycle: int = 0
    tags: list[str] = field(default_factory=list)
    excerpt: str = ""


@dataclass
class HiveWorker:
    """A long-lived swarm worker that holds its own layered knowledge."""
    id: str
    corpus_id: str
    assigned_chunk_indices: list[int] = field(default_factory=list)

    # Layer 0: Raw chunk anchor (private, never gossiped)
    layer0_raw_chunks: list[str] = field(default_factory=list)

    # Layer 1: Rich local understanding (prose, versioned)
    layer1_understanding: str = ""
    layer1_version: int = 0

    # Layer 2: Structured claim ledger
    layer2_claims: list[ClaimEntry] = field(default_factory=list)
    layer2_version: int = 0

    # Layer 3: Emergent insights & hypotheses
    layer3_insights: list[InsightEntry] = field(default_factory=list)
    layer3_version: int = 0

    # Layer 2.5: Pointer map
    pointers: list[Pointer] = field(default_factory=list)

    # Metadata
    status: str = "initializing"
    current_task: str = ""
    gossip_rounds_completed: int = 0
    mining_cycles_completed: int = 0
    started_at: float = 0.0
    last_activity: float = 0.0
    total_llm_calls: int = 0


# ---------------------------------------------------------------------------
# Swarm state enums and corpus record
# ---------------------------------------------------------------------------

class CorpusStatus(str, Enum):
    """Processing status for an ingested corpus."""
    QUEUED = "queued"
    CHUNKING = "chunking"
    SURVEYING = "surveying"
    GOSSIPING = "gossiping"
    INTERROGATING = "interrogating"
    MINING = "mining"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(str, Enum):
    """Status for an individual hive worker."""
    IDLE = "idle"
    SURVEYING = "surveying"
    GOSSIPING = "gossiping"
    MINING = "mining"
    REFLECTING = "reflecting"
    DEBATING = "debating"
    QUERYING = "querying"
    DONE = "done"
    ERROR = "error"


@dataclass
class CorpusRecord:
    """Tracks a single corpus through the hive pipeline."""
    id: str
    title: str
    source: str
    total_chars: int
    status: CorpusStatus = CorpusStatus.QUEUED
    total_chunks: int = 0
    chunks_processed: int = 0
    workers_assigned: int = 0
    gossip_rounds_done: int = 0
    mining_cycles_done: int = 0
    total_claims: int = 0
    total_insights: int = 0
    error: str = ""
    submitted_at: str = ""
    started_at: str = ""
    completed_at: str = ""


# ---------------------------------------------------------------------------
# HiveState -- the registry of living workers
# ---------------------------------------------------------------------------

class HiveState:
    """Global hive state -- the registry of living workers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._workers: dict[str, HiveWorker] = {}
        self._corpora: dict[str, CorpusRecord] = {}
        self._total_queries_answered: int = 0
        self._started_at: float = time.monotonic()
        self._mining_active: bool = True
        self._current_queen_id: str = ""
        self._hive_cycle: int = 0
        self._global_insights: list[InsightEntry] = []

    async def add_worker(self, worker: HiveWorker) -> None:
        async with self._lock:
            self._workers[worker.id] = worker

    async def get_worker(self, worker_id: str) -> Optional[HiveWorker]:
        async with self._lock:
            return self._workers.get(worker_id)

    async def get_all_workers(self) -> list[HiveWorker]:
        async with self._lock:
            return list(self._workers.values())

    async def get_workers_for_corpus(self, corpus_id: str) -> list[HiveWorker]:
        async with self._lock:
            return [w for w in self._workers.values() if w.corpus_id == corpus_id]

    async def get_workers_for_topic(self, query: str, limit: int = 8) -> list[HiveWorker]:
        """Find workers most relevant to a query via pointer maps and keyword matching."""
        query_lower = query.lower()
        query_terms = set(re.findall(r"[a-z0-9]{3,}", query_lower))

        async with self._lock:
            scored: list[tuple[float, HiveWorker]] = []
            for w in self._workers.values():
                score = 0.0
                # Score from pointers
                for p in w.pointers:
                    topic_lower = p.topic.lower()
                    for term in query_terms:
                        if term in topic_lower:
                            score += p.strength * 2.0
                # Score from Layer 2 claim entities
                for claim in w.layer2_claims:
                    for entity in claim.entities:
                        for term in query_terms:
                            if term in entity.lower():
                                score += 1.0
                    for term in query_terms:
                        if term in claim.text.lower():
                            score += 0.5
                # Score from Layer 3 insights
                for insight in w.layer3_insights:
                    for term in query_terms:
                        if term in insight.text.lower():
                            score += 1.5
                # Score from understanding
                for term in query_terms:
                    if term in w.layer1_understanding.lower():
                        score += 0.3
                if score > 0:
                    scored.append((score, w))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [w for _, w in scored[:limit]]

    async def elect_queen(self) -> str:
        """Elect a new queen from available workers (round-robin)."""
        async with self._lock:
            worker_ids = list(self._workers.keys())
            if not worker_ids:
                return ""
            self._hive_cycle += 1
            idx = self._hive_cycle % len(worker_ids)
            self._current_queen_id = worker_ids[idx]
            return self._current_queen_id

    async def add_corpus(self, record: CorpusRecord) -> None:
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
        workers_assigned: Optional[int] = None,
        gossip_rounds_done: Optional[int] = None,
        mining_cycles_done: Optional[int] = None,
        total_claims: Optional[int] = None,
        total_insights: Optional[int] = None,
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
            if workers_assigned is not None:
                rec.workers_assigned = workers_assigned
            if gossip_rounds_done is not None:
                rec.gossip_rounds_done = gossip_rounds_done
            if mining_cycles_done is not None:
                rec.mining_cycles_done = mining_cycles_done
            if total_claims is not None:
                rec.total_claims = total_claims
            if total_insights is not None:
                rec.total_insights = total_insights
            if error is not None:
                rec.error = error
            if started_at is not None:
                rec.started_at = started_at
            if completed_at is not None:
                rec.completed_at = completed_at

    async def increment_queries(self) -> None:
        async with self._lock:
            self._total_queries_answered += 1

    async def add_global_insight(self, insight: InsightEntry) -> None:
        async with self._lock:
            self._global_insights.append(insight)

    async def get_global_insights(self) -> list[InsightEntry]:
        async with self._lock:
            return list(self._global_insights)

    async def is_mining_active(self) -> bool:
        async with self._lock:
            return self._mining_active

    async def set_mining_active(self, active: bool) -> None:
        async with self._lock:
            self._mining_active = active

    async def get_corpora_list(self) -> list[dict]:
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

    async def get_status_snapshot(self) -> dict:
        """Return a complete snapshot of hive state for reporting."""
        async with self._lock:
            total_claims = sum(
                len(w.layer2_claims) for w in self._workers.values()
            )
            total_insights = sum(
                len(w.layer3_insights) for w in self._workers.values()
            )
            total_pointers = sum(
                len(w.pointers) for w in self._workers.values()
            )

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
                    "workers_assigned": c.workers_assigned,
                    "gossip_rounds_done": c.gossip_rounds_done,
                    "mining_cycles_done": c.mining_cycles_done,
                    "total_claims": c.total_claims,
                    "total_insights": c.total_insights,
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
                    "corpus_id": w.corpus_id,
                    "status": w.status,
                    "current_task": w.current_task,
                    "layer1_version": w.layer1_version,
                    "layer2_claims": len(w.layer2_claims),
                    "layer3_insights": len(w.layer3_insights),
                    "pointers": len(w.pointers),
                    "gossip_rounds": w.gossip_rounds_completed,
                    "mining_cycles": w.mining_cycles_completed,
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
            active_workers = sum(
                1 for w in self._workers.values()
                if w.status not in ("idle", "done")
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
                "total_workers": len(self._workers),
                "active_workers": active_workers,
                "total_claims": total_claims,
                "total_insights": total_insights,
                "total_pointers": total_pointers,
                "global_insights": len(self._global_insights),
                "mining_active": self._mining_active,
                "current_queen": self._current_queen_id,
                "hive_cycle": self._hive_cycle,
                "total_queries_answered": self._total_queries_answered,
                "corpora": corpora_list,
                "workers": workers_list,
            }

    async def build_sincerity_preamble(self) -> str:
        """Build an honest status message about the hive's current state."""
        snapshot = await self.get_status_snapshot()

        if snapshot["total_corpora"] == 0:
            return (
                "**[Hive Status]** No corpora have been submitted yet. "
                "Send me a large body of text and the hive will begin "
                "surveying, gossiping, and mining it for insights. "
                "You can ask questions at any time.\n\n"
            )

        parts = ["**[Hive Status]**"]

        parts.append(
            f" {snapshot['total_workers']} workers alive, "
            f"{snapshot['hive_cycle']} gossip rounds completed, "
            f"{snapshot['total_corpora']} corpus/corpora ingested "
            f"({snapshot['total_chars_ingested']:,} chars total)."
        )

        parts.append(
            f"\n**Knowledge:** "
            f"{snapshot['total_claims']} claims, "
            f"{snapshot['total_insights']} insights, "
            f"{snapshot['total_pointers']} pointers across the hive."
        )

        mining_status = "active" if snapshot["mining_active"] else "paused"
        parts.append(f"\n**Mining:** {mining_status}.")

        if snapshot["current_queen"]:
            parts.append(
                f" Current queen: {snapshot['current_queen']}."
            )

        if snapshot["global_insights"] > 0:
            global_insights = await self.get_global_insights()
            if global_insights:
                latest = global_insights[-1]
                parts.append(
                    f"\n**Latest Hive Oracle insight:** {latest.text[:200]}"
                )

        if snapshot["corpora_processing"] > 0:
            parts.append(
                f"\n**Currently processing "
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
                        f"{c['total_chunks']} chunks, "
                        f"{c['gossip_rounds_done']} gossip rounds, "
                        f"{c['mining_cycles_done']} mining cycles)"
                    )

        if snapshot["corpora_queued"] > 0:
            parts.append(
                f"\n{snapshot['corpora_queued']} corpus/corpora "
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
            snapshot["corpora_processing"] > 0
            or snapshot["corpora_queued"] > 0
            or snapshot["mining_active"]
        ):
            parts.append(
                "\n*Note: The hive is still working. My answers "
                "reflect knowledge extracted so far -- they may become "
                "more complete as gossip and mining continue. "
                "This does not affect the hive's work.*"
            )

        return "".join(parts) + "\n\n"


# Global hive state
hive = HiveState()

# Background tasks (kept alive for the process lifetime)
_background_tasks: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

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


def _parse_llm_json(content: str) -> Optional[dict]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


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
# Phase-specific prompts
# ---------------------------------------------------------------------------

_SURVEY_PROMPT = """\
You are a hive worker in a collective intelligence swarm. You have been \
assigned text chunks from a corpus.
Read the chunks carefully and produce THREE outputs:

1. UNDERSTANDING (Layer 1): A rich narrative of what these chunks contain \
-- themes, arguments, key points, nuances. Be thorough. (~2000 words max)

2. CLAIM LEDGER (Layer 2): A JSON array of structured claims extracted \
from the text. Each claim:
   {"text": "atomic claim", "confidence": "high|medium|low", \
"entities": ["Entity A"], "relationships": [{"source": "A", \
"target": "B", "type": "causes", "description": "..."}], \
"provenance_chunk_index": 0}
   Extract ALL entities, relationships, and claims. Be exhaustive.

3. INSIGHTS (Layer 3): Higher-order observations -- patterns you notice, \
implications, hypotheses, contradictions, open questions. These are your \
most valuable output.
   {"text": "insight text", "insight_type": \
"pattern|implication|hypothesis|contradiction|open_question"}

Output as JSON: {"understanding": "...", "claims": [...], "insights": [...]}

TEXT CHUNKS:
{chunk_texts}"""

_GOSSIP_PROMPT = """\
You are a hive worker refining your understanding through peer gossip.

YOUR CURRENT STATE:
- Understanding (Layer 1): {own_understanding}
- Claim Ledger (Layer 2): {own_claims_json}
- Insights (Layer 3): {own_insights_json}

PEER INPUTS (prioritized):
- Peer Insights (Channel C -- highest priority): {peer_insights}
- Peer Claim Deltas (Channel B): {peer_claim_deltas}
- Peer Understandings (Channel A -- if materially changed): {peer_understandings}
- Open Questions & Contradictions (Channel D): {open_questions}

REFINEMENT RULES:
1. Cross-reference your claims with peers'. Flag contradictions.
2. Absorb peer insights into your Layer 3 -- do NOT drop them.
3. If a peer has deeper knowledge on a topic, create a POINTER instead \
of copying everything.
4. Promote any new patterns or contradictions to Layer 3.
5. Update your understanding (Layer 1) to reflect cross-referenced knowledge.
6. Your claim ledger should GROW, not shrink. New claims from peers get \
added with provenance.

Output JSON: {"understanding": "...", "claims": [...], "insights": [...], \
"pointers": [{"topic": "...", "target_worker_id": "...", "strength": 0.8, \
"excerpt": "..."}]}"""

_REFLECTION_PROMPT = """\
You are a hive worker in a perpetual mining cycle. Re-examine your knowledge.

YOUR RAW CHUNKS (Layer 0 -- re-read these carefully):
{raw_chunks}

YOUR CURRENT UNDERSTANDING (Layer 1):
{understanding}

YOUR CLAIM LEDGER (Layer 2):
{claims_json}

YOUR INSIGHTS (Layer 3):
{insights_json}

SWARM-WIDE INSIGHTS (from peers):
{global_insights}

REFLECTION TASK:
1. Re-read your raw chunks with fresh eyes, informed by everything you \
now know from gossip.
2. Generate up to 3 NEW insights you missed before.
3. Identify up to 2 contradictions (internal or with peer knowledge).
4. Formulate 1 hypothesis that would require focused re-reading.
5. Update your claim ledger with anything you missed on first pass.

Output JSON: {"new_insights": [...], "new_claims": [...], \
"updated_understanding": "...", "focused_questions": ["..."]}"""

_QUEEN_SYNTHESIS_PROMPT = """\
You are the queen synthesizer of a hive swarm. Workers have been processing \
a corpus through multiple gossip rounds and mining cycles.

USER QUERY: {query}

WORKER PERSPECTIVES (routed via pointer network -- these are the most \
relevant workers):
{worker_perspectives}

GLOBAL HIVE INSIGHTS:
{global_insights}

Synthesize a comprehensive answer. Rules:
1. Cross-reference across workers. Note consensus and contradictions.
2. Cite which workers/chunks support each claim.
3. If workers disagree, present both sides with evidence.
4. Be direct, thorough, specific. No moralizing.
5. If the hive's understanding is still evolving (mining active), say so."""

_HIVE_ORACLE_PROMPT = """\
You are the Hive Oracle. All workers have reported their single most \
important new understanding.

WORKER REPORTS:
{worker_reports}

PREVIOUS HIVE INSIGHT:
{previous_hive_insight}

Synthesize: What is the single most important thing the swarm now \
understands that it didn't understand last cycle? This insight will be \
injected into every worker's Layer 3.

Output: {"hive_insight": "...", "confidence": "high|medium|low", \
"key_entities": [...]}"""

_QUERY_SYSTEM_PROMPT = """\
You are a research analyst powered by a hive knowledge system. \
The hive has decomposed large corpora of text into structured \
knowledge through multi-channel gossip and perpetual mining.

**HIVE STATUS:**
{hive_status}

**YOUR JOB:**
Answer the user's question using ONLY the knowledge results \
provided below. Be thorough, specific, and cite the source \
documents when possible.

**RULES:**
- If the hive has relevant information, synthesise \
it into a clear, comprehensive answer.
- If the hive has partial information, say what you \
know and clearly state what gaps remain.
- If no relevant knowledge exists yet, say so honestly -- \
do not fabricate.
- Reference specific claims, insights, and worker perspectives.
- If the hive is still mining, mention that more \
complete answers may be available as mining continues.
- Be direct. No moralising, no disclaimers, no hedging. \
Just answer.

**KNOWLEDGE RESULTS:**
{knowledge_results}"""


# ---------------------------------------------------------------------------
# Hive Ingestion Pipeline
# ---------------------------------------------------------------------------

async def _survey_worker(
    worker: HiveWorker,
    corpus_id: str,
    req_id: str,
) -> None:
    """Phase 1: Worker surveys its assigned chunks and builds initial layers."""
    worker.status = "surveying"
    worker.current_task = "Surveying assigned chunks"
    worker.last_activity = time.monotonic()

    chunk_texts = "\n\n---CHUNK BOUNDARY---\n\n".join(
        f"[Chunk {idx}]\n{text}"
        for idx, text in zip(worker.assigned_chunk_indices, worker.layer0_raw_chunks)
    )

    prompt = _SURVEY_PROMPT.replace("{chunk_texts}", chunk_texts)

    span = lf.start_span(
        req_id, f"hive:survey:{worker.id}",
        input={"worker_id": worker.id, "chunks": len(worker.layer0_raw_chunks)},
    )

    result = await _call_llm(
        [{"role": "user", "content": prompt}],
        model=WORKER_MODEL,
        max_tokens=8192,
        temperature=0.2,
    )
    worker.total_llm_calls += 1

    if "error" in result:
        log.warning(f"[{worker.id}] Survey error: {result['error']}")
        lf.end_span(span, output={"error": result["error"]}, level="ERROR")
        return

    parsed = _parse_llm_json(result["content"])
    if not parsed:
        log.warning(f"[{worker.id}] Survey JSON parse failed")
        # Fall back to using raw content as understanding
        worker.layer1_understanding = result["content"][:4000]
        worker.layer1_version = 1
        lf.end_span(span, output={"error": "json_parse_failed"}, level="WARNING")
        return

    # Populate layers
    worker.layer1_understanding = parsed.get("understanding", "")
    worker.layer1_version = 1

    for c in parsed.get("claims", []):
        if isinstance(c, dict) and c.get("text"):
            worker.layer2_claims.append(ClaimEntry(
                text=c["text"],
                confidence=c.get("confidence", "medium"),
                entities=c.get("entities", []),
                relationships=c.get("relationships", []),
                provenance_chunk_index=c.get("provenance_chunk_index", 0),
                corpus_id=corpus_id,
                version=1,
            ))
    worker.layer2_version = 1

    for i in parsed.get("insights", []):
        if isinstance(i, dict) and i.get("text"):
            worker.layer3_insights.append(InsightEntry(
                text=i["text"],
                insight_type=i.get("insight_type", "pattern"),
                source_worker_id=worker.id,
                cycle_created=0,
                version=1,
            ))
    worker.layer3_version = 1

    worker.last_activity = time.monotonic()
    lf.end_span(span, output={
        "claims": len(worker.layer2_claims),
        "insights": len(worker.layer3_insights),
    })


async def _gossip_round(
    workers: list[HiveWorker],
    round_num: int,
    corpus_id: str,
    req_id: str,
) -> int:
    """Run one gossip round across all workers. Returns count of new insights."""
    new_insights_total = 0

    for worker in workers:
        worker.status = "gossiping"
        worker.current_task = f"Gossip round {round_num + 1}"
        worker.last_activity = time.monotonic()

        # Select peers (up to HIVE_MAX_PEERS_PER_GOSSIP, excluding self)
        peers = [w for w in workers if w.id != worker.id]
        if len(peers) > HIVE_MAX_PEERS_PER_GOSSIP:
            peers = random.sample(peers, HIVE_MAX_PEERS_PER_GOSSIP)

        # Build peer inputs by channel priority
        # Channel C: insights (highest priority, always included)
        peer_insights_parts = []
        for p in peers:
            for ins in p.layer3_insights:
                peer_insights_parts.append(
                    f"[{p.id}] ({ins.insight_type}) {ins.text}"
                )
        peer_insights = "\n".join(peer_insights_parts) or "(none)"

        # Channel B: claim deltas
        peer_claims_parts = []
        for p in peers:
            for cl in p.layer2_claims[:10]:  # limit per peer
                peer_claims_parts.append(
                    f"[{p.id}] [{cl.confidence}] {cl.text}"
                )
        peer_claim_deltas = "\n".join(peer_claims_parts[:30]) or "(none)"

        # Channel A: understandings (if materially changed)
        peer_understandings_parts = []
        for p in peers:
            if p.layer1_version > 0:
                peer_understandings_parts.append(
                    f"[{p.id}] {p.layer1_understanding[:1000]}"
                )
        peer_understandings = "\n".join(peer_understandings_parts) or "(none)"

        # Channel D: open questions and contradictions
        open_q_parts = []
        for p in peers:
            for ins in p.layer3_insights:
                if ins.insight_type in ("contradiction", "open_question"):
                    open_q_parts.append(f"[{p.id}] {ins.text}")
        open_questions = "\n".join(open_q_parts) or "(none)"

        own_claims_json = json.dumps(
            [{"text": c.text, "confidence": c.confidence, "entities": c.entities}
             for c in worker.layer2_claims[:20]],
            indent=None,
        )
        own_insights_json = json.dumps(
            [{"text": i.text, "insight_type": i.insight_type}
             for i in worker.layer3_insights],
            indent=None,
        )

        prompt = (_GOSSIP_PROMPT
            .replace("{own_understanding}", worker.layer1_understanding[:2000])
            .replace("{own_claims_json}", own_claims_json)
            .replace("{own_insights_json}", own_insights_json)
            .replace("{peer_insights}", peer_insights[:3000])
            .replace("{peer_claim_deltas}", peer_claim_deltas[:2000])
            .replace("{peer_understandings}", peer_understandings[:3000])
            .replace("{open_questions}", open_questions[:1000])
        )

        span = lf.start_span(
            req_id, f"hive:gossip:{worker.id}:r{round_num}",
            input={"worker_id": worker.id, "round": round_num, "peers": len(peers)},
        )

        result = await _call_llm(
            [{"role": "user", "content": prompt}],
            model=WORKER_MODEL,
            max_tokens=8192,
            temperature=0.2,
        )
        worker.total_llm_calls += 1

        if "error" in result:
            log.warning(f"[{worker.id}] Gossip round {round_num} error: {result['error']}")
            lf.end_span(span, output={"error": result["error"]}, level="ERROR")
            continue

        parsed = _parse_llm_json(result["content"])
        if not parsed:
            log.warning(f"[{worker.id}] Gossip JSON parse failed")
            lf.end_span(span, output={"error": "json_parse_failed"}, level="WARNING")
            continue

        # Update layers
        new_understanding = parsed.get("understanding", "")
        if new_understanding:
            worker.layer1_understanding = new_understanding
            worker.layer1_version += 1

        prev_claims = len(worker.layer2_claims)
        for c in parsed.get("claims", []):
            if isinstance(c, dict) and c.get("text"):
                worker.layer2_claims.append(ClaimEntry(
                    text=c["text"],
                    confidence=c.get("confidence", "medium"),
                    entities=c.get("entities", []),
                    relationships=c.get("relationships", []),
                    provenance_chunk_index=c.get("provenance_chunk_index", 0),
                    corpus_id=corpus_id,
                    version=worker.layer2_version + 1,
                ))
        worker.layer2_version += 1

        prev_insights = len(worker.layer3_insights)
        for i in parsed.get("insights", []):
            if isinstance(i, dict) and i.get("text"):
                worker.layer3_insights.append(InsightEntry(
                    text=i["text"],
                    insight_type=i.get("insight_type", "pattern"),
                    source_worker_id=worker.id,
                    cycle_created=round_num,
                    version=worker.layer3_version + 1,
                ))
        worker.layer3_version += 1
        new_insights_total += len(worker.layer3_insights) - prev_insights

        # Update pointers
        for p in parsed.get("pointers", []):
            if isinstance(p, dict) and p.get("topic") and p.get("target_worker_id"):
                worker.pointers.append(Pointer(
                    topic=p["topic"],
                    target_worker_id=p["target_worker_id"],
                    strength=float(p.get("strength", 0.5)),
                    excerpt=p.get("excerpt", ""),
                ))

        worker.gossip_rounds_completed = round_num + 1
        worker.last_activity = time.monotonic()
        lf.end_span(span, output={
            "new_claims": len(worker.layer2_claims) - prev_claims,
            "new_insights": len(worker.layer3_insights) - prev_insights,
            "pointers": len(worker.pointers),
        })

    return new_insights_total


async def _interrogation_round(
    workers: list[HiveWorker],
    corpus_id: str,
    req_id: str,
) -> None:
    """Phase 3: Workers re-read raw chunks with focused questions from gossip."""
    global_insights = await hive.get_global_insights()
    global_text = "\n".join(
        f"- ({i.insight_type}) {i.text}" for i in global_insights
    ) or "(none yet)"

    for worker in workers:
        worker.status = "reflecting"
        worker.current_task = "Interrogation: re-reading chunks"
        worker.last_activity = time.monotonic()

        raw_chunks = "\n\n---CHUNK---\n\n".join(worker.layer0_raw_chunks)
        claims_json = json.dumps(
            [{"text": c.text, "confidence": c.confidence}
             for c in worker.layer2_claims[:20]],
            indent=None,
        )
        insights_json = json.dumps(
            [{"text": i.text, "insight_type": i.insight_type}
             for i in worker.layer3_insights],
            indent=None,
        )

        prompt = (_REFLECTION_PROMPT
            .replace("{raw_chunks}", raw_chunks[:6000])
            .replace("{understanding}", worker.layer1_understanding[:2000])
            .replace("{claims_json}", claims_json)
            .replace("{insights_json}", insights_json)
            .replace("{global_insights}", global_text[:2000])
        )

        span = lf.start_span(
            req_id, f"hive:interrogate:{worker.id}",
            input={"worker_id": worker.id},
        )

        result = await _call_llm(
            [{"role": "user", "content": prompt}],
            model=WORKER_MODEL,
            max_tokens=4096,
            temperature=0.3,
        )
        worker.total_llm_calls += 1

        if "error" in result:
            log.warning(f"[{worker.id}] Interrogation error: {result['error']}")
            lf.end_span(span, output={"error": result["error"]}, level="ERROR")
            continue

        parsed = _parse_llm_json(result["content"])
        if not parsed:
            lf.end_span(span, output={"error": "json_parse_failed"}, level="WARNING")
            continue

        # Absorb new findings
        for i in parsed.get("new_insights", []):
            if isinstance(i, dict) and i.get("text"):
                worker.layer3_insights.append(InsightEntry(
                    text=i["text"],
                    insight_type=i.get("insight_type", "pattern"),
                    source_worker_id=worker.id,
                    version=worker.layer3_version + 1,
                ))
        worker.layer3_version += 1

        for c in parsed.get("new_claims", []):
            if isinstance(c, dict) and c.get("text"):
                worker.layer2_claims.append(ClaimEntry(
                    text=c["text"],
                    confidence=c.get("confidence", "medium"),
                    entities=c.get("entities", []),
                    corpus_id=corpus_id,
                    version=worker.layer2_version + 1,
                ))
        worker.layer2_version += 1

        updated = parsed.get("updated_understanding", "")
        if updated:
            worker.layer1_understanding = updated
            worker.layer1_version += 1

        worker.last_activity = time.monotonic()
        lf.end_span(span, output={"status": "complete"})


async def _run_hive_oracle(
    workers: list[HiveWorker],
    req_id: str,
) -> Optional[InsightEntry]:
    """Run the Hive Oracle to produce a global comprehension insight."""
    worker_reports_parts = []
    for w in workers:
        if w.layer3_insights:
            latest = w.layer3_insights[-1]
            worker_reports_parts.append(
                f"[{w.id}] {latest.text}"
            )
    if not worker_reports_parts:
        return None

    worker_reports = "\n".join(worker_reports_parts)

    global_insights = await hive.get_global_insights()
    prev_insight = global_insights[-1].text if global_insights else "(none)"

    prompt = (_HIVE_ORACLE_PROMPT
        .replace("{worker_reports}", worker_reports[:4000])
        .replace("{previous_hive_insight}", prev_insight[:1000])
    )

    span = lf.start_span(req_id, "hive:oracle", input={})

    result = await _call_llm(
        [{"role": "user", "content": prompt}],
        model=SYNTHESIS_MODEL,
        max_tokens=2048,
        temperature=0.3,
    )

    if "error" in result:
        log.warning(f"Hive Oracle error: {result['error']}")
        lf.end_span(span, output={"error": result["error"]}, level="ERROR")
        return None

    parsed = _parse_llm_json(result["content"])
    if not parsed:
        lf.end_span(span, output={"error": "json_parse_failed"}, level="WARNING")
        return None

    insight_text = parsed.get("hive_insight", "")
    if not insight_text:
        lf.end_span(span, output={"error": "empty_insight"}, level="WARNING")
        return None

    insight = InsightEntry(
        text=insight_text,
        insight_type="pattern",
        source_worker_id="hive-oracle",
        version=1,
    )

    await hive.add_global_insight(insight)

    # Inject into all workers' Layer 3
    for w in workers:
        w.layer3_insights.append(InsightEntry(
            text=insight_text,
            insight_type="pattern",
            source_worker_id="hive-oracle",
            version=w.layer3_version + 1,
        ))
        w.layer3_version += 1

    lf.end_span(span, output={"insight": insight_text[:200]})
    return insight


async def _perpetual_mining_loop(
    corpus_id: str,
    req_id: str,
) -> None:
    """Phase 4: Background mining loop for continuous insight extraction."""
    cycle = 0
    while cycle < HIVE_MAX_MINING_CYCLES:
        if not await hive.is_mining_active():
            log.info(f"Mining paused for corpus {corpus_id}")
            await asyncio.sleep(5)
            continue

        await asyncio.sleep(HIVE_MINING_INTERVAL)

        if not await hive.is_mining_active():
            continue

        cycle += 1
        workers = await hive.get_workers_for_corpus(corpus_id)
        if not workers:
            break

        log.info(
            f"Mining cycle {cycle}/{HIVE_MAX_MINING_CYCLES} "
            f"for corpus {corpus_id} with {len(workers)} workers"
        )

        # Each worker reflects
        for worker in workers:
            worker.status = "mining"
            worker.current_task = f"Mining cycle {cycle}"

        await _interrogation_round(workers, corpus_id, req_id)

        # One gossip round to share mining findings
        gossip_round_num = max(
            (w.gossip_rounds_completed for w in workers), default=0,
        )
        new_insights = await _gossip_round(
            workers, gossip_round_num, corpus_id, req_id,
        )

        # Run Hive Oracle periodically
        if cycle % HIVE_ORACLE_INTERVAL == 0:
            await hive.elect_queen()
            await _run_hive_oracle(workers, req_id)

        for w in workers:
            w.mining_cycles_completed = cycle

        await hive.update_corpus(
            corpus_id,
            mining_cycles_done=cycle,
            total_claims=sum(len(w.layer2_claims) for w in workers),
            total_insights=sum(len(w.layer3_insights) for w in workers),
        )

        log.info(
            f"Mining cycle {cycle} complete: "
            f"{new_insights} new insights across {len(workers)} workers"
        )

        # Check for sub-swarm trigger
        if new_insights >= HIVE_SUBSWARM_THRESHOLD:
            log.info(
                f"Sub-swarm threshold met ({new_insights} >= "
                f"{HIVE_SUBSWARM_THRESHOLD}), running focused debate round"
            )
            await _gossip_round(
                workers, gossip_round_num + 1, corpus_id, req_id,
            )

    # Set workers to idle when mining is done
    for w in await hive.get_workers_for_corpus(corpus_id):
        w.status = "idle"
        w.current_task = "Mining complete"


async def _hive_ingest_corpus(
    corpus_id: str, text: str, req_id: str = "",
) -> None:
    """Process a corpus through the Ruflo Hive pipeline.

    Phases:
      1. Survey -- workers read chunks, build initial layered understanding
      2. Gossip -- multi-channel gossip rounds for cross-referencing
      3. Interrogation -- workers re-read chunks with focused questions
      4. Perpetual Mining -- background loop for continuous insight extraction
    """
    corpus_span = lf.start_span(
        req_id, "hive:ingest_corpus",
        input={"corpus_id": corpus_id, "chars": len(text)},
    )

    async with worker_semaphore:
        try:
            record = await hive.get_corpus(corpus_id)
            if not record:
                log.error(f"Corpus {corpus_id} not found")
                lf.end_span(
                    corpus_span, output={"error": "corpus_not_found"},
                    level="ERROR",
                )
                return

            # Step 1: Chunking
            await hive.update_corpus(
                corpus_id,
                status=CorpusStatus.CHUNKING,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            chunks = _chunk_text(text)
            total_chunks = len(chunks)
            await hive.update_corpus(corpus_id, total_chunks=total_chunks)

            log.info(
                f"Corpus {corpus_id} ({record.title[:40]}): "
                f"{record.total_chars:,} chars -> {total_chunks} chunks"
            )

            # Step 2: Assign chunks to workers
            num_workers = max(1, total_chunks // HIVE_CHUNKS_PER_WORKER)
            num_workers = min(num_workers, MAX_SWARM_WORKERS)
            chunks_per_w = max(1, total_chunks // num_workers)

            workers: list[HiveWorker] = []
            for wi in range(num_workers):
                start_idx = wi * chunks_per_w
                end_idx = min(start_idx + chunks_per_w, total_chunks)
                if wi == num_workers - 1:
                    end_idx = total_chunks  # last worker gets remainder

                worker_id = f"worker-{corpus_id[-8:]}-{wi}"
                worker = HiveWorker(
                    id=worker_id,
                    corpus_id=corpus_id,
                    assigned_chunk_indices=list(range(start_idx, end_idx)),
                    layer0_raw_chunks=[chunks[j] for j in range(start_idx, end_idx)],
                    started_at=time.monotonic(),
                    last_activity=time.monotonic(),
                )
                workers.append(worker)
                await hive.add_worker(worker)

            await hive.update_corpus(
                corpus_id, workers_assigned=len(workers),
            )

            # Phase 1: Survey
            await hive.update_corpus(
                corpus_id, status=CorpusStatus.SURVEYING,
            )
            log.info(f"Phase 1 (Survey): {len(workers)} workers surveying")

            survey_tasks = []
            for w in workers:
                survey_tasks.append(_survey_worker(w, corpus_id, req_id))
            await asyncio.gather(*survey_tasks)

            # Update progress
            await hive.update_corpus(
                corpus_id,
                chunks_processed=total_chunks,
                total_claims=sum(len(w.layer2_claims) for w in workers),
                total_insights=sum(len(w.layer3_insights) for w in workers),
            )

            log.info(
                f"Phase 1 complete: "
                f"{sum(len(w.layer2_claims) for w in workers)} claims, "
                f"{sum(len(w.layer3_insights) for w in workers)} insights"
            )

            # Phase 2: Gossip rounds
            await hive.update_corpus(
                corpus_id, status=CorpusStatus.GOSSIPING,
            )
            for gossip_round in range(HIVE_GOSSIP_ROUNDS):
                log.info(
                    f"Phase 2 (Gossip): round {gossip_round + 1}/"
                    f"{HIVE_GOSSIP_ROUNDS}"
                )
                new_insights = await _gossip_round(
                    workers, gossip_round, corpus_id, req_id,
                )
                await hive.update_corpus(
                    corpus_id,
                    gossip_rounds_done=gossip_round + 1,
                    total_claims=sum(len(w.layer2_claims) for w in workers),
                    total_insights=sum(len(w.layer3_insights) for w in workers),
                )
                log.info(
                    f"Gossip round {gossip_round + 1} complete: "
                    f"{new_insights} new insights"
                )
                # Check for convergence
                if new_insights == 0 and gossip_round > 0:
                    log.info("Gossip converged (no new insights)")
                    break

            # Phase 3: Interrogation
            await hive.update_corpus(
                corpus_id, status=CorpusStatus.INTERROGATING,
            )
            log.info("Phase 3 (Interrogation): workers re-reading chunks")
            await _interrogation_round(workers, corpus_id, req_id)

            # One final gossip round to share interrogation findings
            final_gossip_round = max(
                (w.gossip_rounds_completed for w in workers), default=0,
            )
            await _gossip_round(
                workers, final_gossip_round, corpus_id, req_id,
            )

            # Run initial Hive Oracle
            await hive.elect_queen()
            await _run_hive_oracle(workers, req_id)

            # Mark completed (mining will continue in background)
            await hive.update_corpus(
                corpus_id,
                status=CorpusStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_claims=sum(len(w.layer2_claims) for w in workers),
                total_insights=sum(len(w.layer3_insights) for w in workers),
            )

            for w in workers:
                w.status = "idle"
                w.current_task = "Ingestion complete, awaiting mining"

            log.info(
                f"Corpus {corpus_id} ingestion COMPLETED: "
                f"{sum(len(w.layer2_claims) for w in workers)} claims, "
                f"{sum(len(w.layer3_insights) for w in workers)} insights, "
                f"{sum(len(w.pointers) for w in workers)} pointers "
                f"from {len(workers)} workers"
            )
            lf.end_span(corpus_span, output={
                "workers": len(workers),
                "claims": sum(len(w.layer2_claims) for w in workers),
                "insights": sum(len(w.layer3_insights) for w in workers),
                "pointers": sum(len(w.pointers) for w in workers),
            })

            # Phase 4: Launch perpetual mining as a separate background task
            mining_task = asyncio.create_task(
                _perpetual_mining_loop(corpus_id, req_id),
            )
            _background_tasks.append(mining_task)
            _background_tasks[:] = [t for t in _background_tasks if not t.done()]

        except Exception as e:
            log.error(f"Unexpected error processing {corpus_id}: {e}")
            lf.end_span(
                corpus_span, output={"error": str(e)},
                level="ERROR", status_message=str(e),
            )
            await hive.update_corpus(
                corpus_id,
                status=CorpusStatus.FAILED,
                error=f"Hive error: {e}",
            )


# ---------------------------------------------------------------------------
# Document detection and corpus submission
# ---------------------------------------------------------------------------

def _is_large_document(text: str) -> bool:
    """Detect whether a message is a large document rather than a query."""
    if len(text) < LARGE_DOC_THRESHOLD:
        return False
    question_marks = text.count("?")
    question_density = question_marks / max(len(text), 1)
    return question_density < 0.001


async def _submit_corpus(
    text: str, title: str = "", source: str = "",
    req_id: str = "",
) -> CorpusRecord:
    """Submit a new corpus to the hive for background processing."""
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

    await hive.add_corpus(record)

    task = asyncio.create_task(_hive_ingest_corpus(corpus_id, text, req_id))
    _background_tasks.append(task)
    _background_tasks[:] = [t for t in _background_tasks if not t.done()]

    log.info(
        f"Corpus {corpus_id} queued: {len(text):,} chars, "
        f"title={title[:40]}"
    )

    return record


# ---------------------------------------------------------------------------
# Query handler: answer from hive knowledge via pointer routing
# ---------------------------------------------------------------------------

async def _hive_query(query: str, messages: list[dict], req_id: str) -> str:
    """Query the hive by routing to relevant workers via pointers."""
    span = lf.start_span(
        req_id, "hive:query",
        input={"query": query[:200]},
    )

    # Pointer routing: find most relevant workers
    relevant_workers = await hive.get_workers_for_topic(query, limit=8)

    if not relevant_workers:
        # Fall back to all workers
        relevant_workers = await hive.get_all_workers()

    if not relevant_workers:
        lf.end_span(span, output={"result": "no_workers"})
        return (
            "(No knowledge available yet. "
            "The hive has not processed any corpora.)"
        )

    # Phase 1 (Fast): Collect Layer 2 claims + Layer 3 insights
    parts: list[str] = []

    for w in relevant_workers[:8]:
        worker_parts = [f"### Worker {w.id}"]

        # Claims relevant to query
        query_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        relevant_claims = []
        for c in w.layer2_claims:
            claim_lower = c.text.lower()
            entity_match = any(
                term in ent.lower()
                for term in query_terms for ent in c.entities
            )
            text_match = any(term in claim_lower for term in query_terms)
            if entity_match or text_match:
                relevant_claims.append(c)

        if relevant_claims:
            claims_text = "\n".join(
                f"- [{c.confidence}] {c.text}"
                + (f" (contradicted: {c.contradiction_details})"
                   if c.contradiction_flag else "")
                for c in relevant_claims[:10]
            )
            worker_parts.append(f"**Claims:**\n{claims_text}")

        # All insights (high value)
        if w.layer3_insights:
            insights_text = "\n".join(
                f"- ({i.insight_type}) {i.text}"
                for i in w.layer3_insights[:8]
            )
            worker_parts.append(f"**Insights:**\n{insights_text}")

        # Brief understanding excerpt
        if w.layer1_understanding:
            worker_parts.append(
                f"**Understanding excerpt:** {w.layer1_understanding[:500]}"
            )

        if len(worker_parts) > 1:
            parts.append("\n".join(worker_parts))

    # Add global hive insights
    global_insights = await hive.get_global_insights()
    if global_insights:
        gi_text = "\n".join(
            f"- {i.text}" for i in global_insights[-5:]
        )
        parts.append(f"### Hive Oracle Insights\n{gi_text}")

    if not parts:
        lf.end_span(span, output={"result": "no_relevant_knowledge"})
        return (
            "(The hive has workers but no knowledge relevant to this query. "
            "Try a different question or wait for more processing.)"
        )

    answer = "\n\n".join(parts)
    lf.end_span(span, output={
        "workers_queried": len(relevant_workers),
        "sections": len(parts),
    })
    return answer


async def _handle_query(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a query by searching hive knowledge and synthesising."""
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

    def reasoning_sse(content: str) -> str:
        return make_sse_chunk(
            "",
            request_id=request_id,
            created=created,
            model_id=model_id,
            reasoning_content=content,
        )

    # Yield an initial empty reasoning_content data chunk immediately so
    # LibreChat's stream handler receives a valid ``data:`` line within
    # its timeout window before any async delays.
    yield reasoning_sse("")

    raw_query = extract_user_text(messages)
    parsed_q = parse_attachments(raw_query)
    user_query = parsed_q.prompt if parsed_q.has_attachments else raw_query

    if not user_query:
        yield sse(
            "No question found in the message.",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    query_span = lf.start_span(
        req_id, "hive:handle_query",
        input={"query": user_query[:300]},
    )

    # Get hive status preamble
    status_preamble = await hive.build_sincerity_preamble()

    yield reasoning_sse(status_preamble)
    yield reasoning_sse(f"**Query:** {user_query[:200]}\n\n")

    # Query the hive via pointer routing
    yield reasoning_sse("**[Querying hive workers via pointer network...]**\n")
    knowledge_results = await _hive_query(user_query, messages, req_id)

    result_summary = knowledge_results[:500]
    if len(knowledge_results) > 500:
        result_summary += "..."
    yield reasoning_sse(f"Found knowledge:\n{result_summary}\n\n")

    # Build synthesis prompt
    system_prompt = (_QUERY_SYSTEM_PROMPT
        .replace("{hive_status}", status_preamble)
        .replace("{knowledge_results}", knowledge_results)
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

    yield reasoning_sse("**[Synthesising answer...]**\n")

    # Stream synthesis response
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
                    lf.end_span(
                        query_span,
                        output={"error": f"HTTP {resp.status_code}"},
                        level="ERROR",
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
        lf.end_span(query_span, output={"error": str(e)}, level="ERROR")
        yield sse(
            f"\n\nError during synthesis: {e}",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    await hive.increment_queries()
    lf.end_span(query_span, output={"status": "complete"})

    yield sse("", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _handle_corpus_submission(
    text: str,
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Handle a large document by submitting it to the hive."""
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
    submit_span = lf.start_span(
        req_id, "hive:corpus_submission",
        input={"title": title, "chars": len(text)},
    )

    def reasoning_sse(content: str) -> str:
        return make_sse_chunk(
            "",
            request_id=request_id,
            created=created,
            model_id=model_id,
            reasoning_content=content,
        )

    # Yield an initial empty reasoning_content data chunk immediately so
    # LibreChat's stream handler receives a valid ``data:`` line within
    # its timeout window before any async delays.
    yield reasoning_sse("")

    yield reasoning_sse(f"**[Corpus Received]** {len(text):,} characters\n")
    yield reasoning_sse(f"Title: {title}...\n")
    yield reasoning_sse(
        "Submitting to the hive for background processing...\n",
    )

    record = await _submit_corpus(
        text, title=title, source="chat-submission",
        req_id=req_id,
    )

    yield reasoning_sse(f"Corpus ID: {record.id}\n")
    yield reasoning_sse("Status: Queued for hive processing\n")

    status = await hive.build_sincerity_preamble()
    yield reasoning_sse(f"\n{status}")

    snapshot = await hive.get_status_snapshot()
    yield sse(
        f"## Corpus Submitted to Hive\n\n"
        f"Your document ({len(text):,} characters) has been submitted "
        f"to the hive for background processing.\n\n"
        f"**Corpus ID:** `{record.id}`\n"
        f"**Title:** {title}\n\n"
        f"The hive will now:\n"
        f"1. Assign chunks to workers who build layered understanding\n"
        f"2. Workers survey and extract claims, entities, and insights\n"
        f"3. Multi-channel gossip for cross-referencing and contradiction detection\n"
        f"4. Interrogation round: re-read chunks with focused questions\n"
        f"5. Perpetual mining for deeper insights (runs in background)\n\n"
        f"**You can ask questions immediately** -- I will answer from "
        f"whatever knowledge the hive has built so far and be honest "
        f"about what has and hasn't been processed yet.\n\n"
        f"This submission does not interrupt any ongoing processing. "
        f"The hive continues its current work and will process this "
        f"corpus when a worker becomes available.\n\n"
        f"**Current hive:** {snapshot['total_corpora']} corpora, "
        f"{snapshot['total_workers']} workers, "
        f"{snapshot['total_claims']} claims, "
        f"{snapshot['total_insights']} insights.\n"
    )

    lf.end_span(submit_span, output={
        "corpus_id": record.id, "status": record.status.value,
    })

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
        "gossip_rounds": HIVE_GOSSIP_ROUNDS,
        "mining_interval": HIVE_MINING_INTERVAL,
        "max_mining_cycles": HIVE_MAX_MINING_CYCLES,
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
    """Return complete hive status."""
    snapshot = await hive.get_status_snapshot()
    return JSONResponse(snapshot)


@app.get("/v1/swarm/corpora")
async def swarm_corpora():
    """List all submitted corpora with their processing status."""
    corpora = await hive.get_corpora_list()
    return JSONResponse({"corpora": corpora, "count": len(corpora)})


@app.get("/v1/swarm/sincerity")
async def swarm_sincerity():
    """Return the current sincerity preamble."""
    preamble = await hive.build_sincerity_preamble()
    return JSONResponse({"preamble": preamble})


@app.get("/v1/swarm/hive")
async def swarm_hive():
    """Full hive status with worker layer sizes, pointer counts, mining status."""
    snapshot = await hive.get_status_snapshot()
    global_insights = await hive.get_global_insights()
    return JSONResponse({
        **snapshot,
        "global_insight_details": [
            {
                "text": i.text,
                "insight_type": i.insight_type,
                "source_worker_id": i.source_worker_id,
            }
            for i in global_insights
        ],
    })


@app.get("/v1/swarm/insights")
async def swarm_insights():
    """All Layer 3 insights across hive + global Hive Oracle insights."""
    workers = await hive.get_all_workers()
    worker_insights = []
    for w in workers:
        for i in w.layer3_insights:
            worker_insights.append({
                "text": i.text,
                "insight_type": i.insight_type,
                "source_worker_id": i.source_worker_id or w.id,
                "cycle_created": i.cycle_created,
            })

    global_insights = await hive.get_global_insights()
    return JSONResponse({
        "worker_insights": worker_insights,
        "global_insights": [
            {
                "text": i.text,
                "insight_type": i.insight_type,
                "source_worker_id": i.source_worker_id,
            }
            for i in global_insights
        ],
        "total_worker_insights": len(worker_insights),
        "total_global_insights": len(global_insights),
    })


@app.get("/v1/swarm/workers/{worker_id}")
async def swarm_worker_detail(worker_id: str):
    """Detailed view of a single worker's layers."""
    worker = await hive.get_worker(worker_id)
    if not worker:
        return JSONResponse(
            {"error": f"Worker {worker_id} not found"},
            status_code=404,
        )

    return JSONResponse({
        "id": worker.id,
        "corpus_id": worker.corpus_id,
        "status": worker.status,
        "current_task": worker.current_task,
        "assigned_chunk_indices": worker.assigned_chunk_indices,
        "layer1_understanding": worker.layer1_understanding[:2000],
        "layer1_version": worker.layer1_version,
        "layer2_claims": [
            {
                "text": c.text,
                "confidence": c.confidence,
                "entities": c.entities,
                "contradiction_flag": c.contradiction_flag,
                "contradiction_details": c.contradiction_details,
            }
            for c in worker.layer2_claims
        ],
        "layer2_version": worker.layer2_version,
        "layer3_insights": [
            {
                "text": i.text,
                "insight_type": i.insight_type,
                "source_worker_id": i.source_worker_id,
                "cycle_created": i.cycle_created,
            }
            for i in worker.layer3_insights
        ],
        "layer3_version": worker.layer3_version,
        "pointers": [
            {
                "topic": p.topic,
                "target_worker_id": p.target_worker_id,
                "strength": p.strength,
                "excerpt": p.excerpt,
            }
            for p in worker.pointers
        ],
        "gossip_rounds_completed": worker.gossip_rounds_completed,
        "mining_cycles_completed": worker.mining_cycles_completed,
        "total_llm_calls": worker.total_llm_calls,
    })


@app.post("/v1/swarm/mining/pause")
async def mining_pause():
    """Pause the perpetual mining loop."""
    await hive.set_mining_active(False)
    return JSONResponse({"mining_active": False, "message": "Mining paused"})


@app.post("/v1/swarm/mining/resume")
async def mining_resume():
    """Resume the perpetual mining loop."""
    await hive.set_mining_active(True)
    return JSONResponse({"mining_active": True, "message": "Mining resumed"})


@app.post("/v1/swarm/submit")
async def submit_corpus_api(request: Request):
    """Direct API endpoint to submit a corpus for hive processing."""
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
    snapshot = await hive.get_status_snapshot()

    return JSONResponse({
        "corpus_id": record.id,
        "title": record.title,
        "total_chars": record.total_chars,
        "status": record.status.value,
        "message": (
            f"Corpus submitted for hive processing. "
            f"Hive has {snapshot['total_corpora']} corpora, "
            f"{snapshot['total_workers']} workers, "
            f"{snapshot['active_workers']} active."
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

    trace_id = lf.create_trace_id(req_id)
    lf.register_trace(req_id, trace_id)

    # Utility requests pass through
    if utility:
        client_wants_stream = body.get("stream", False)
        if not client_wants_stream:
            log.info(f"[{req_id}] Routing to NON-STREAMING utility passthrough")
            lf.unregister_trace(req_id)
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=UPSTREAM_BASE,
                upstream_key=UPSTREAM_KEY,
                upstream_model=UPSTREAM_MODEL,
                log=log,
            )
            tracker.finish(req_id)
            return result
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        lf.unregister_trace(req_id)
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
        user_text = extract_user_text_with_attachments(messages)
        parsed = parse_attachments(user_text)

        if parsed.has_attachments:
            doc_summary = ", ".join(
                f"{d.filename} ({len(d.content):,} chars)"
                for d in parsed.documents
            )
            log.info(
                f"[{req_id}] ATTACHMENT DETECTED: {len(parsed.documents)} "
                f"doc(s) [{doc_summary}], prompt={parsed.prompt[:80]!r}"
            )

            async def _handle_attachment_submission():
                model_id = body.get("model", "swarm-miroflow")
                request_id = f"chatcmpl-swarm-attach-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                def sse(
                    content: str,
                    finish_reason: Optional[str] = None,
                ) -> str:
                    return make_sse_chunk(
                        content,
                        request_id=request_id,
                        created=created,
                        model_id=model_id,
                        finish_reason=finish_reason,
                    )

                def reasoning_sse(content: str) -> str:
                    return make_sse_chunk(
                        "",
                        request_id=request_id,
                        created=created,
                        model_id=model_id,
                        reasoning_content=content,
                    )

                try:
                    yield reasoning_sse(
                        f"**[Attachments Received]** "
                        f"{len(parsed.documents)} document(s)\n"
                    )

                    submitted_ids = []
                    for doc in parsed.documents:
                        yield reasoning_sse(
                            f"  Submitting: {doc.filename} "
                            f"({len(doc.content):,} chars)\n"
                        )
                        record = await _submit_corpus(
                            doc.content,
                            title=doc.filename,
                            source="attachment",
                            req_id=req_id,
                        )
                        submitted_ids.append(record.id)
                        yield reasoning_sse(
                            f"  -> Corpus {record.id} queued "
                            f"for hive processing\n"
                        )

                    yield reasoning_sse(
                        "\nThe hive will now survey, gossip, "
                        "and mine these documents -- building layered "
                        "knowledge through multi-channel collaboration.\n\n"
                    )

                    status = await hive.build_sincerity_preamble()
                    yield reasoning_sse(f"\n{status}")

                    if parsed.prompt:
                        yield reasoning_sse(
                            f"\n**[Answering query while hive "
                            f"processes documents...]**\n"
                            f"Query: {parsed.prompt[:200]}\n\n"
                        )

                        knowledge_results = await _hive_query(
                            parsed.prompt, messages, req_id,
                        )

                        doc_context = "\n\n".join(
                            f"[From attachment: {d.filename}]\n"
                            f"{d.content}"
                            for d in parsed.documents
                        )

                        system_prompt = (
                            "You are a research analyst powered by a "
                            "hive knowledge system. The user has just "
                            "submitted document(s) to the hive for "
                            "deep analysis.\n\n"
                            "The hive is currently processing these "
                            "documents -- surveying, gossiping, and mining "
                            "them for insights. Meanwhile, answer the user's "
                            "question using the document content provided "
                            "and any knowledge already in the hive.\n\n"
                            "**HIVE STATUS:**\n"
                            f"{status}\n\n"
                            "**DOCUMENT EXCERPTS:**\n"
                            f"{doc_context}\n\n"
                            "**EXISTING KNOWLEDGE:**\n"
                            f"{knowledge_results}\n\n"
                            "**RULES:**\n"
                            "- Answer from the documents and hive "
                            "knowledge. Be thorough and specific.\n"
                            "- Note that the hive is still processing -- "
                            "deeper insights will be available as the "
                            "hive continues its work.\n"
                            "- Be direct. No moralising.\n"
                        )

                        synthesis_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": parsed.prompt},
                        ]

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
                                    yield sse(
                                        f"Error: {error_text[:200]}",
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
                                            choices = data.get(
                                                "choices", [],
                                            )
                                            if choices:
                                                delta = choices[0].get(
                                                    "delta", {},
                                                )
                                                content = delta.get(
                                                    "content", "",
                                                )
                                                if content:
                                                    yield sse(content)
                                        except json.JSONDecodeError:
                                            pass

                        await hive.increment_queries()
                    else:
                        snapshot = await hive.get_status_snapshot()
                        corpus_list = ", ".join(
                            f"`{cid}`" for cid in submitted_ids
                        )
                        yield sse(
                            f"## Documents Submitted to Hive\n\n"
                            f"**{len(parsed.documents)} document(s)** "
                            f"submitted for hive processing.\n\n"
                            f"**Corpus IDs:** {corpus_list}\n\n"
                            f"The hive workers will now:\n"
                            f"1. Survey chunks and build layered "
                            f"understanding\n"
                            f"2. Gossip across channels -- sharing claims, "
                            f"insights, and contradictions\n"
                            f"3. Interrogate raw text with focused "
                            f"questions\n"
                            f"4. Perpetually mine for deeper insights\n\n"
                            f"**Ask questions at any time** -- I'll "
                            f"answer from the hive's current "
                            f"collective understanding.\n\n"
                            f"**Current hive:** "
                            f"{snapshot['total_corpora']} corpora, "
                            f"{snapshot['total_workers']} workers, "
                            f"{snapshot['total_claims']} claims.\n"
                        )

                    yield sse("", finish_reason="stop")
                    yield "data: [DONE]\n\n"

                finally:
                    lf.unregister_trace(req_id)
                    lf.flush()
                    tracker.finish(req_id)

            generator = _handle_attachment_submission()

        elif _is_large_document(parsed.prompt or user_text):
            log.info(
                f"[{req_id}] Routing to HIVE CORPUS SUBMISSION "
                f"({len(user_text):,} chars)"
            )

            async def _guarded_submit():
                try:
                    async for event in _handle_corpus_submission(
                        user_text, body, req_id,
                    ):
                        yield event
                finally:
                    lf.unregister_trace(req_id)
                    lf.flush()
                    tracker.finish(req_id)

            generator = _guarded_submit()
        else:
            if not query_limiter.available():
                lf.unregister_trace(req_id)
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

            log.info(f"[{req_id}] Routing to HIVE QUERY")

            async def _guarded_query():
                try:
                    async with query_limiter.hold():
                        async for event in _handle_query(
                            messages, body, req_id,
                        ):
                            yield event
                finally:
                    lf.unregister_trace(req_id)
                    lf.flush()
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
