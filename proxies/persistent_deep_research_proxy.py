#!/usr/bin/env python3
"""
Persistent Deep Research Proxy for Open WebUI.

An advanced research proxy implementing the full architecture from the design
document, extending the base MiroFlow deep research loop with:

  - **Subagent map-reduce**: A planning agent decomposes the query into multiple
    research angles, parallel subagents each conduct independent research, and a
    synthesis agent combines all findings into a comprehensive answer.
  - **Atom of Thoughts (AoT) state contraction**: Each subagent compresses its
    findings into atomic conditions (fact + source + confidence) rather than
    accumulating raw tool output, preventing context-window overflow.
  - **AoT Reflection**: Validates decomposition quality and triggers
    redecomposition when contraction quality is poor.
  - **Persistent memory**: Atomic conditions are stored in Neo4j (via the
    Knowledge Engine) so future queries can retrieve relevant prior findings.
    JSONL flat files provide local archival logging.
  - **Knowledge Graph**: Entity extraction from findings, relationship storage,
    cross-domain bridge edges, and graph-aware retrieval via Neo4j fulltext search.
  - **Dual-model architecture**: A small, fast model (e.g. Mistral Small) handles
    planning and subagent research; the large model handles final synthesis.
  - **Serendipity-aware exploration**: The planning agent generates cross-domain
    "bridge queries" alongside standard research angles, with serendipity scoring
    (relevance x novelty x surprise).
  - **Dynamic Saturation Detection**: Novelty tracking per turn with adaptive
    turn allocation -- expand when novelty > 0.3, stop when < 0.05.
  - **Trust Scoring**: URL domain-based confidence (.edu/.gov = 0.9, news = 0.6,
    forums/Reddit = 0.3) with contradiction penalties.
  - **Citation Verification**: Verify claims against sources, detect and resolve
    contradictions, adjust confidence scores.
  - **Draft-Synthesis-Revision Loop**: Draft -> critic review -> targeted
    micro-research on weak sections -> final polish.
  - **Recursive subagent spawning**: Subagents can spawn sub-subagents for
    emerging rabbit holes (depth-limited).
  - **Expanded Tool Ecosystem**: arxiv_search, wayback_fetch, wikidata_query
    alongside the existing searxng_search, fetch_webpage, python_exec.

Architecture:
  User Query
      |
      v
  [Retrieve Prior Knowledge] -- Neo4j fulltext + Knowledge Graph lookup
      |
      v
  [Planning Agent] (small model) -- decomposes into N angles + bridge queries
      |
      +--- Subagent 1: angle_1 (small model, AoT loop) ---+
      +--- Subagent 2: angle_2 (small model, AoT loop) ---+
      +--- ...                                             +--- progress queue
      +--- Subagent N: angle_N (small model, AoT loop) ---+
      |          (each may spawn recursive sub-subagents)
      v
  [Entity Extraction + Knowledge Graph Update]
      |
      v
  [Citation Verification] -- verify claims, detect contradictions
      |
      v
  [Store Conditions] -- persist to Neo4j + JSONL archive
      |
      v
  [Draft Synthesis] (large model) -- cross-reference, produce draft
      |
      v
  [Critic Agent] (small model) -- review draft, mark weak sections
      |
      v
  [Revision Agent] -- targeted micro-research on weak sections
      |
      v
  [Final Synthesis] (large model) -- polished final answer
      |
      v
  Streamed SSE response
"""

import asyncio
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncGenerator, Optional, TypedDict
from urllib.parse import urlparse

# Sentinel used to signal the SSE output queue that the pipeline is done.
_STREAM_DONE = object()

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from langgraph.graph import END, START, StateGraph

import knowledge_client
import veritas_inquisitor
from research_metrics import (
    MetricsCollector,
    ResearchMetricsCallback,
    SubagentMetrics,
    list_available_reports,
    load_metrics,
    save_metrics,
)
import research_report
import b2_publisher

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
)

# --- Logging ---
LOG_DIR = os.getenv("PERSISTENT_RESEARCH_LOG_DIR", "/opt/persistent_research_logs")
log = setup_logging("persistent-research", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "mistral-small-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("PERSISTENT_RESEARCH_PORT", 9300, minimum=1)
MAX_SUBAGENT_TURNS = env_int("MAX_SUBAGENT_TURNS", 15, minimum=1)
MAX_CONCURRENT = env_int("MAX_CONCURRENT_PERSISTENT", 2, minimum=1)
RESEARCH_NAMESPACE = os.getenv("RESEARCH_NAMESPACE", "research")
JSONL_LOG_DIR = os.getenv("JSONL_LOG_DIR", "/opt/persistent_research_logs/jsonl")
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000
MAX_PRIOR_CONDITIONS = 20
# Saturation detection thresholds
NOVELTY_EXPAND_THRESHOLD = 0.3
NOVELTY_STOP_THRESHOLD = 0.05
# Legacy constants used by plan_research() and run_subagent() recursive spawning
MAX_SUBAGENTS = env_int("MAX_SUBAGENTS", 7, minimum=1)
MAX_RECURSIVE_DEPTH = env_int("MAX_RECURSIVE_DEPTH", 2, minimum=0)

# --- Tree Research Reactor Config ---
TREE_MAX_CONCURRENT = env_int("TREE_MAX_CONCURRENT", 10, minimum=1)
TREE_MAX_DEPTH = env_int("TREE_MAX_DEPTH", 5, minimum=1)
TREE_MAX_NODES = env_int("TREE_MAX_NODES", 50, minimum=5)
TREE_PRESSURE_THRESHOLD = float(os.getenv("TREE_PRESSURE_THRESHOLD", "0.15"))
TREE_WORKER_IDLE_TIMEOUT = float(os.getenv("TREE_WORKER_IDLE_TIMEOUT", "60.0"))

# --- Enhanced Web Scraping Config ---
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY", "")
BRIGHT_DATA_HOST = os.getenv("BRIGHT_DATA_HOST", "")  # e.g. brd.superproxy.io:33335

log.info(
    f"Config: synthesis_model={UPSTREAM_MODEL}, subagent_model={SUBAGENT_MODEL}, "
    f"upstream={UPSTREAM_BASE}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, "
    f"tree_concurrent={TREE_MAX_CONCURRENT}, tree_depth={TREE_MAX_DEPTH}, "
    f"tree_nodes={TREE_MAX_NODES}, subagent_turns={MAX_SUBAGENT_TURNS}, "
    f"research_ns={RESEARCH_NAMESPACE}, "
    f"apify={'yes' if APIFY_API_KEY else 'no'}, "
    f"bright_data={'yes' if BRIGHT_DATA_API_KEY else 'no'}"
)

# --- Shared state ---
tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)

# Per-request live findings collectors, keyed by req_id.
# Created when research starts, cleaned up when research ends.
_live_collectors: dict[str, "LiveFindingsCollector"] = {}

# Per-request curated event queues for tree reactor -> heartbeat.
_curated_queues: dict[str, asyncio.Queue] = {}

# Per-request metrics collectors, keyed by req_id.
_metrics_collectors: dict[str, MetricsCollector] = {}

# ============================================================================
# Trust Scoring System
# ============================================================================

_TRUST_TIERS: list[tuple[re.Pattern, float]] = [
    (re.compile(r'\.edu(/|$)', re.IGNORECASE), 0.9),
    (re.compile(r'\.gov(/|$)', re.IGNORECASE), 0.9),
    (re.compile(r'(arxiv\.org|pubmed|ncbi\.nlm\.nih|doi\.org|springer\.com|nature\.com|science\.org|ieee\.org|acm\.org)', re.IGNORECASE), 0.9),
    (re.compile(r'(reuters\.com|apnews\.com|bbc\.co\.uk|bbc\.com|nytimes\.com|washingtonpost\.com|theguardian\.com|economist\.com)', re.IGNORECASE), 0.7),
    (re.compile(r'(cnn\.com|foxnews\.com|nbcnews\.com|abcnews\.go\.com|bloomberg\.com|ft\.com)', re.IGNORECASE), 0.6),
    (re.compile(r'(medium\.com|substack\.com|wordpress\.com|blogspot\.com)', re.IGNORECASE), 0.4),
    (re.compile(r'(reddit\.com|quora\.com|stackexchange\.com|stackoverflow\.com|news\.ycombinator\.com)', re.IGNORECASE), 0.3),
    (re.compile(r'(wikipedia\.org)', re.IGNORECASE), 0.6),
]


def trust_score_url(url: str) -> float:
    """Compute a trust score for a URL based on its domain."""
    if not url:
        return 0.5
    for pattern, score in _TRUST_TIERS:
        if pattern.search(url):
            return score
    return 0.5


# ============================================================================
# Serendipity Scoring
# ============================================================================

def serendipity_score(fact: str, query: str, known_facts: list[str]) -> float:
    """Compute a simplified serendipity score.

    Serendipity = geometric_mean(relevance, novelty, surprise).
    Uses Jaccard-based word overlap as a lightweight proxy for embedding
    similarity (no external embedding model required).
    """
    fact_words = set(fact.lower().split())
    query_words = set(query.lower().split())

    if not fact_words or not query_words:
        return 0.0

    relevance = len(fact_words & query_words) / max(len(fact_words | query_words), 1)
    relevance = min(max(relevance, 0.05), 1.0)

    max_sim = 0.0
    for known in known_facts:
        known_words = set(known.lower().split())
        if not known_words:
            continue
        sim = len(fact_words & known_words) / max(len(fact_words | known_words), 1)
        max_sim = max(max_sim, sim)
    novelty = 1.0 - max_sim
    novelty = min(max(novelty, 0.05), 1.0)

    all_context = query_words.copy()
    for k in known_facts[:10]:
        all_context.update(k.lower().split())
    context_overlap = len(fact_words & all_context) / max(len(fact_words), 1)
    surprise = 1.0 - context_overlap
    surprise = min(max(surprise, 0.05), 1.0)

    return (relevance * novelty * surprise) ** (1.0 / 3.0)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class AtomicCondition:
    """A single compressed research finding (Atom of Thoughts)."""
    fact: str
    source_url: str = ""
    confidence: float = 0.5
    angle: str = ""
    domain: str = ""
    is_serendipitous: bool = False
    trust_score: float = 0.5
    serendipity_score_val: float = 0.0
    entities: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"- {self.fact}"]
        if self.source_url:
            parts[0] += f" [source: {self.source_url}]"
        if self.confidence != 0.5:
            parts[0] += f" (confidence: {self.confidence:.1f})"
        if self.trust_score != 0.5:
            parts[0] += f" (trust: {self.trust_score:.1f})"
        if self.is_serendipitous:
            parts[0] += " [SERENDIPITOUS]"
        if self.serendipity_score_val > 0.3:
            parts[0] += f" [serendipity: {self.serendipity_score_val:.2f}]"
        return parts[0]


@dataclass
class SubagentResult:
    """Result from a single subagent's research."""
    angle: str
    conditions: list[AtomicCondition] = field(default_factory=list)
    turns_used: int = 0
    tool_calls_made: int = 0
    error: str = ""
    novelty_history: list[float] = field(default_factory=list)
    spawned_children: int = 0


@dataclass
class ResearchNode:
    """A single node in the research exploration tree.

    Each node represents a question/claim to investigate.  Workers pull
    nodes from a priority queue, research them, and push child nodes
    back.  Only the active LLM+tool research phase occupies a
    semaphore slot.
    """
    id: str
    question: str
    context: str  # why this node was spawned
    depth: int
    pressure: float  # 0-1, higher = explore first
    parent_id: Optional[str] = None
    status: str = "pending"  # pending | researching | done | pruned

    def __lt__(self, other: "ResearchNode") -> bool:
        """PriorityQueue needs ordering; higher pressure = higher priority."""
        return self.pressure > other.pressure


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

async def _store_conditions_neo4j(
    session_id: str,
    query: str,
    conditions: list["AtomicCondition"],
) -> int:
    """Store atomic conditions in Neo4j via the knowledge engine. Returns count stored."""
    if not conditions:
        return 0
    cond_dicts = [
        {
            "fact": c.fact,
            "source_url": c.source_url,
            "confidence": c.confidence,
            "trust_score": c.trust_score,
            "angle": c.angle,
            "domain": c.domain,
            "is_serendipitous": c.is_serendipitous,
            "serendipity_score": c.serendipity_score_val,
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
        return result.get("stored", 0)
    except Exception as e:
        log.error(f"Neo4j condition storage error: {e}")
        return 0


async def _store_entities_neo4j(
    session_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> tuple[int, int]:
    """Store entities and relationships in Neo4j via the knowledge engine."""
    try:
        result = await knowledge_client.store_entities(
            session_id=session_id,
            entities=entities,
            relationships=relationships,
            namespace=RESEARCH_NAMESPACE,
        )
        return result.get("entities_stored", 0), result.get("relationships_stored", 0)
    except Exception as e:
        log.error(f"Neo4j entity storage error: {e}")
        return 0, 0


async def _retrieve_related(query: str, limit: int = 20) -> list[dict]:
    """Retrieve prior conditions related to the query using Neo4j fulltext search."""
    try:
        results = await knowledge_client.search_conditions(
            query=query,
            namespace=RESEARCH_NAMESPACE,
            limit=limit,
        )
        return [
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
    except Exception as e:
        log.warning(f"Neo4j condition search error: {e}")
        return []


async def _retrieve_graph_neighbors(
    entity_names: list[str], max_hops: int = 2, limit: int = 20
) -> list[dict]:
    """Retrieve related conditions via knowledge graph traversal in Neo4j."""
    if not entity_names:
        return []
    try:
        results = await knowledge_client.graph_neighbors(
            entity_names=entity_names,
            namespace=RESEARCH_NAMESPACE,
            max_hops=max_hops,
            limit=limit,
        )
        return [
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
    except Exception as e:
        log.warning(f"Neo4j graph neighbor error: {e}")
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

    title = text[:80].replace("\n", " ").strip()
    namespace = RESEARCH_NAMESPACE
    doc_chars = len(text)

    yield chunk("<think>\n")
    yield chunk(f"**[Document Ingestion Mode]** Detected large document ({doc_chars:,} chars)\n")
    yield chunk(f"Title: {title}...\n")
    yield chunk(f"Namespace: {namespace}\n\n")

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
        yield chunk("**[Step 1: Submitting to Knowledge Engine]**\n")
        ingest_result = await knowledge_client.ingest(
            namespace=namespace,
            title=title,
            text=text,
            source="document-ingestion",
            rebuild=False,  # Append, don't clear existing data
        )
        job_id = ingest_result.get("job_id", "")
        yield chunk(f"Ingest job started: {job_id}\n")
        yield chunk(f"Total chars: {ingest_result.get('total_chars', doc_chars):,}\n\n")

        # Step 2: Poll for completion
        yield chunk("**[Step 2: Processing Document]**\n")
        max_polls = 300  # up to ~10 minutes
        last_status = ""
        for _ in range(max_polls):
            await asyncio.sleep(2)
            try:
                status = await knowledge_client.ingest_status(job_id)
            except Exception as e:
                yield chunk(f"  Poll error: {e}\n")
                continue

            current_status = status.get("status", "unknown")
            progress = status.get("progress", "")

            if current_status != last_status:
                yield chunk(f"  Status: {current_status}")
                if progress:
                    yield chunk(f" — {progress}")
                yield chunk("\n")
                last_status = current_status

            if current_status == "completed":
                ingestion_ok = True
                stats = status.get("stats", {})
                yield chunk(f"\n**[Step 3: Ingestion Complete]**\n")
                if stats:
                    yield chunk(f"  Chunks: {stats.get('total_chunks', '?')}\n")
                    yield chunk(f"  Entities extracted: {stats.get('entities_created', '?')}\n")
                    yield chunk(f"  Relationships: {stats.get('relationships_created', '?')}\n")
                    yield chunk(f"  Claims: {stats.get('claims_created', '?')}\n")
                break
            elif current_status == "failed":
                error = status.get("error", "Unknown error")
                yield chunk(f"\n**Ingestion failed:** {error}\n")
                break
        else:
            yield chunk("\n**Warning:** Ingestion is still running (timed out waiting).\n")
            yield chunk("You can check status later via the knowledge engine API.\n")

    except Exception as e:
        log.error(f"[{req_id}] Document ingestion error: {e}")
        yield chunk(f"\n**Error during ingestion:** {e}\n")

    yield chunk("\n</think>\n\n")

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


# ============================================================================
# Native Tool Definitions (OpenAI function-calling format)
# ============================================================================

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": (
                "Search the web using SearXNG. Returns top results with titles, "
                "URLs, and snippets. Use this to find information, verify facts, "
                "discover sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch a webpage and extract its readable text content. Use this "
                "to read articles, documentation, or any web page found via search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "extract_info": {
                        "type": "string",
                        "description": "Optional: specific information to look for",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute Python code for calculations, data processing, or analysis. "
                "Code runs in a sandboxed subprocess with a 30-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv for academic papers. Returns paper titles, authors, "
                "abstracts, and links. Use this for academic and scientific research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for arXiv papers"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wayback_fetch",
            "description": (
                "Fetch an archived version of a webpage from the Wayback Machine "
                "(archive.org). Use this to recover dead links or see historical "
                "versions of pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to look up in the Wayback Machine",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikidata_query",
            "description": (
                "Query Wikidata for structured facts about entities. Returns "
                "entity properties and relationships. Use this for verifiable "
                "factual data about people, places, organizations, concepts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity to look up (e.g., 'Albert Einstein', 'Python programming language')",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_graph_search",
            "description": (
                "Search the Neo4j knowledge graph for relevant concepts, claims, evidence, "
                "anomalies, and text chunks. Supports hybrid search (keyword + graph traversal "
                "with reciprocal rank fusion). Use this FIRST when the user's question may relate "
                "to documents or knowledge that has been ingested into the knowledge engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace to search within (default: 'default')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "graph"],
                        "description": "Search mode (default: 'hybrid')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10, max 50)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_discover",
            "description": (
                "Run graph discovery algorithms on the knowledge graph to find hidden "
                "connections and serendipitous links. Supports: spreading_activation (multi-hop "
                "activation propagation from seed concepts), swanson_abc (find concepts connected "
                "through intermediaries but not directly \u2014 bisociation discovery), and "
                "information_gaps (find under-connected but important concepts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["spreading_activation", "swanson_abc", "information_gaps"],
                        "description": "The discovery algorithm to run",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace",
                    },
                    "seed_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting concept names (required for spreading_activation and swanson_abc)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 15)",
                    },
                },
                "required": ["algorithm", "namespace"],
            },
        },
    },
]


# ============================================================================
# Tool Execution
# ============================================================================

async def tool_searxng_search(query: str) -> str:
    """Execute a SearXNG search and return formatted results."""
    try:
        client = http_client()
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return f"Search error: HTTP {resp.status_code}"

        data = resp.json()
        results = data.get("results", [])[:10]

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            trust = trust_score_url(url)
            formatted.append(
                f"{i}. **{title}** [trust: {trust:.1f}]\n"
                f"   URL: {url}\n   {snippet}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Search error: request timed out after 20s"
    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================================
# Enhanced Web Scraping Abstraction
# ============================================================================

_CENSORSHIP_INDICATORS = [
    "access denied",
    "403 forbidden",
    "blocked by robots",
    "captcha required",
    "content not available in your region",
    "this content is restricted",
    "verify you are human",
    "automated access",
]


def _is_censored_response(text: str) -> bool:
    """Detect if a scraping response was likely censored or blocked."""
    if not text or len(text.strip()) < 50:
        return True
    lower = text.lower()
    return any(indicator in lower for indicator in _CENSORSHIP_INDICATORS)


async def _fetch_via_apify(url: str) -> Optional[str]:
    """Fetch webpage content via Apify Web Scraper actor.

    Returns extracted text or None on failure.
    """
    if not APIFY_API_KEY:
        return None
    try:
        client = http_client()
        # Use Apify's web-scraper actor via the REST API
        resp = await client.post(
            "https://api.apify.com/v2/acts/apify~web-scraper/run-sync-get-dataset-items",
            params={"token": APIFY_API_KEY},
            json={
                "startUrls": [{"url": url}],
                "pageFunction": "async function pageFunction(context) { "
                    "const $ = context.jQuery; "
                    "const text = $('body').text().replace(/\\s+/g, ' ').trim(); "
                    "return { url: context.request.url, text: text.substring(0, 50000) }; "
                    "}",
                "maxRequestsPerCrawl": 1,
            },
            timeout=httpx.Timeout(60.0, connect=15.0),
        )
        if resp.status_code == 200:
            items = resp.json()
            if items and isinstance(items, list) and items[0].get("text"):
                text = items[0]["text"]
                if not _is_censored_response(text):
                    return text[:WEBPAGE_MAX_CHARS]
                log.debug(f"Apify: censored response for {url}")
        else:
            log.debug(f"Apify: HTTP {resp.status_code} for {url}")
    except Exception as e:
        log.debug(f"Apify fetch error for {url}: {e}")
    return None


async def _fetch_via_bright_data(url: str) -> Optional[str]:
    """Fetch webpage content via Bright Data proxy.

    Returns extracted text or None on failure.
    """
    if not BRIGHT_DATA_API_KEY or not BRIGHT_DATA_HOST:
        return None
    try:
        import html as html_mod
        # Create a dedicated client with the proxy configured at client level.
        # httpx does not support per-request proxy via extensions; it must be
        # set on the AsyncClient constructor.
        proxy_url = f"http://{BRIGHT_DATA_API_KEY}@{BRIGHT_DATA_HOST}"
        async with httpx.AsyncClient(
            proxy=proxy_url,
            timeout=httpx.Timeout(45.0, connect=15.0),
        ) as proxy_client:
            resp = await proxy_client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
            )
        if resp.status_code == 200:
            raw = resp.text
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html_mod.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            if not _is_censored_response(text):
                return text[:WEBPAGE_MAX_CHARS]
            log.debug(f"Bright Data: censored response for {url}")
        else:
            log.debug(f"Bright Data: HTTP {resp.status_code} for {url}")
    except Exception as e:
        log.debug(f"Bright Data fetch error for {url}: {e}")
    return None


async def enhanced_web_fetch(url: str, extract_info: str = "") -> str:
    """Fetch webpage with fallback chain: Apify -> Bright Data -> direct.

    Detects censored/blocked responses and falls back to the next
    backend.  Appends a warning when all enhanced backends fail and
    the direct fetch is also suspect.
    """
    # Try Apify first (best at JS rendering)
    result = await _fetch_via_apify(url)
    if result:
        if extract_info:
            return f"Instructions: {extract_info}\n\nContent from {url} (via Apify):\n{result}"
        return f"Content from {url} (via Apify):\n{result}"

    # Try Bright Data proxy (best at bypassing geo-blocks)
    result = await _fetch_via_bright_data(url)
    if result:
        if extract_info:
            return f"Instructions: {extract_info}\n\nContent from {url} (via Bright Data):\n{result}"
        return f"Content from {url} (via Bright Data):\n{result}"

    # Fallback to direct fetch
    direct = await _tool_fetch_webpage_direct(url, extract_info)

    # If direct fetch looks censored, append a warning
    # Only append censorship warning for actual page content, not for
    # error messages from _tool_fetch_webpage_direct (which are short
    # strings like "Fetch error: HTTP 404..." that would falsely trigger
    # the <50 char check in _is_censored_response).
    _error_prefixes = ("Fetch error", "Non-text content", "PDF document", "No readable text")
    if _is_censored_response(direct) and not direct.startswith(_error_prefixes):
        direct += (
            "\n\n[WARNING: This result may be incomplete or blocked. "
            "The page may require JavaScript rendering, authentication, "
            "or be geo-restricted. Treat 'no results found' with skepticism "
            "and try alternative sources.]"
        )

    return direct


async def _tool_fetch_webpage_direct(url: str, extract_info: str = "") -> str:
    """Direct fetch — original implementation."""
    # (identical to old tool_fetch_webpage, now an internal helper)
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code} for {url}"

        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            return f"PDF document at {url} (binary content, cannot extract text directly)"
        if ("text/html" not in content_type and "text/plain" not in content_type
                and "text/xml" not in content_type and "application/json" not in content_type):
            return f"Non-text content type: {content_type} at {url}"

        raw = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return f"No readable text content found at {url}"

        if len(text) > WEBPAGE_MAX_CHARS:
            text = text[:WEBPAGE_MAX_CHARS] + "\n[...truncated...]\n"

        result = f"Content from {url}:\n{text}"
        if extract_info:
            result = f"Instructions: {extract_info}\n\n{result}"
        return result

    except Exception as e:
        return f"Fetch error for {url}: {e}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage with enhanced scraping fallback chain."""
    return await enhanced_web_fetch(url, extract_info)


def tool_python_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > PYTHON_OUTPUT_MAX:
            output = output[:PYTHON_OUTPUT_MAX] + "\n[... output truncated ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


async def tool_arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers using the arXiv API."""
    try:
        max_results = min(max_results, 10)
        client = http_client()
        resp = await client.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=20.0,
        )
        if resp.status_code != 200:
            return f"arXiv search error: HTTP {resp.status_code}"

        text = resp.text
        entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
        if not entries:
            return "No arXiv papers found."

        formatted = []
        for i, entry in enumerate(entries, 1):
            title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_m.group(1).strip().replace('\n', ' ') if title_m else "Unknown"
            summary_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            summary = summary_m.group(1).strip()[:300] if summary_m else ""
            id_m = re.search(r'<id>(.*?)</id>', entry)
            arxiv_url = id_m.group(1).strip() if id_m else ""
            authors = re.findall(r'<name>(.*?)</name>', entry)
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"
            published_m = re.search(r'<published>(.*?)</published>', entry)
            published = published_m.group(1)[:10] if published_m else ""

            formatted.append(
                f"{i}. **{title}**\n"
                f"   Authors: {author_str}\n"
                f"   Published: {published}\n"
                f"   URL: {arxiv_url}\n"
                f"   Abstract: {summary}"
            )

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "arXiv search error: request timed out"
    except Exception as e:
        return f"arXiv search error: {str(e)}"


async def tool_wayback_fetch(url: str) -> str:
    """Fetch an archived version of a URL from the Wayback Machine."""
    try:
        client = http_client()
        avail_resp = await client.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=15.0,
        )
        if avail_resp.status_code != 200:
            return f"Wayback Machine error: HTTP {avail_resp.status_code}"

        avail_data = avail_resp.json()
        snapshots = avail_data.get("archived_snapshots", {})
        closest = snapshots.get("closest", {})
        if not closest or not closest.get("available"):
            return f"No archived version found for {url}"

        archive_url = closest.get("url", "")
        timestamp = closest.get("timestamp", "")

        content = await tool_fetch_webpage(archive_url)
        return (
            f"**Wayback Machine archive** (captured: {timestamp})\n"
            f"Original URL: {url}\n"
            f"Archive URL: {archive_url}\n\n{content}"
        )

    except httpx.TimeoutException:
        return "Wayback Machine error: request timed out"
    except Exception as e:
        return f"Wayback Machine error: {str(e)}"


async def tool_wikidata_query(entity: str) -> str:
    """Query Wikidata for structured facts about an entity."""
    try:
        client = http_client()
        search_resp = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": entity,
                "language": "en",
                "format": "json",
                "limit": 3,
            },
            timeout=15.0,
        )
        if search_resp.status_code != 200:
            return f"Wikidata search error: HTTP {search_resp.status_code}"

        search_data = search_resp.json()
        results = search_data.get("search", [])
        if not results:
            return f"No Wikidata entity found for '{entity}'"

        formatted = []
        for r in results:
            qid = r.get("id", "")
            label = r.get("label", "")
            desc = r.get("description", "")
            url = f"https://www.wikidata.org/wiki/{qid}"
            formatted.append(
                f"- **{label}** ({qid}): {desc}\n  URL: {url}"
            )

        top_qid = results[0].get("id", "")
        if top_qid:
            entity_resp = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": top_qid,
                    "languages": "en",
                    "format": "json",
                    "props": "labels|descriptions|claims",
                },
                timeout=15.0,
            )
            if entity_resp.status_code == 200:
                entity_data = entity_resp.json()
                ent_info = entity_data.get("entities", {}).get(top_qid, {})
                claims = ent_info.get("claims", {})
                claim_strs = []
                for prop_id, claim_list in list(claims.items())[:10]:
                    for claim in claim_list[:1]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        value = datavalue.get("value", "")
                        if isinstance(value, dict):
                            value = value.get("id", str(value))
                        claim_strs.append(f"  {prop_id}: {value}")
                if claim_strs:
                    formatted.append(f"\nTop claims for {top_qid}:\n" + "\n".join(claim_strs[:8]))

        return "\n".join(formatted)

    except httpx.TimeoutException:
        return "Wikidata query error: request timed out"
    except Exception as e:
        return f"Wikidata query error: {str(e)}"


async def execute_tool(tool_name: str, arguments: dict) -> str:
    """Route and execute a tool call."""
    if tool_name == "searxng_search":
        return await tool_searxng_search(arguments.get("query", ""))
    elif tool_name == "fetch_webpage":
        return await tool_fetch_webpage(
            arguments.get("url", ""),
            arguments.get("extract_info", ""),
        )
    elif tool_name == "python_exec":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool_python_exec, arguments.get("code", ""))
    elif tool_name == "arxiv_search":
        return await tool_arxiv_search(
            arguments.get("query", ""),
            arguments.get("max_results", 5),
        )
    elif tool_name == "wayback_fetch":
        return await tool_wayback_fetch(arguments.get("url", ""))
    elif tool_name == "wikidata_query":
        return await tool_wikidata_query(arguments.get("entity", ""))
    elif tool_name == "knowledge_graph_search":
        return await tool_knowledge_graph_search(arguments)
    elif tool_name == "knowledge_discover":
        return await tool_knowledge_discover(arguments)
    else:
        return f"Unknown tool: {tool_name}"


async def tool_knowledge_graph_search(arguments: dict) -> str:
    """Search the knowledge graph via the knowledge engine microservice."""
    try:
        import knowledge_client
        result = await knowledge_client.search(
            namespace=arguments.get("namespace", "default"),
            query=arguments.get("query", ""),
            mode=arguments.get("mode", "hybrid"),
            limit=min(arguments.get("limit", 10), 50),
        )
        results = result.get("results", [])
        if not results:
            return "No matching knowledge found in the graph."

        formatted = []
        for i, r in enumerate(results, 1):
            node_type = r.get("node_type", "")
            name = r.get("name", "")
            content = r.get("content", "")
            score = r.get("score", 0)
            props = r.get("properties", {})
            source_doc = r.get("source_doc", "")

            header = f"{i}. [{node_type}]"
            if name:
                header += f" **{name}**"
            if source_doc:
                header += f" (from: {source_doc})"
            header += f" [score: {score:.3f}]"

            body = content[:2000] if content else ""
            if props:
                prop_strs = []
                for k, v in props.items():
                    if k not in ("id",) and v is not None:
                        prop_strs.append(f"{k}: {v}")
                if prop_strs:
                    body += "\n  Properties: " + ", ".join(prop_strs[:5])

            formatted.append(f"{header}\n{body}" if body else header)

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Knowledge graph search error: {e}"


async def tool_knowledge_discover(arguments: dict) -> str:
    """Run graph discovery algorithms via the knowledge engine microservice."""
    try:
        import knowledge_client
        algorithm = arguments.get("algorithm", "")
        namespace = arguments.get("namespace", "default")
        seed_concepts = arguments.get("seed_concepts", [])
        limit = arguments.get("limit", 15)

        if algorithm == "spreading_activation":
            if not seed_concepts:
                return "Error: seed_concepts required for spreading_activation"
            result = await knowledge_client.spreading_activation(
                namespace=namespace,
                seed_concepts=seed_concepts,
                limit=limit,
            )
        elif algorithm == "swanson_abc":
            if not seed_concepts:
                return "Error: seed_concepts required for swanson_abc"
            result = await knowledge_client.swanson_abc(
                namespace=namespace,
                seed_concept=seed_concepts[0],
                limit=limit,
            )
        elif algorithm == "information_gaps":
            result = await knowledge_client.information_gaps(
                namespace=namespace,
                limit=limit,
            )
        else:
            return f"Unknown algorithm: {algorithm}. Use: spreading_activation, swanson_abc, information_gaps"

        discoveries = result.get("results", [])
        if not discoveries:
            return f"No discoveries from {algorithm}."

        formatted = [f"**{algorithm.replace('_', ' ').title()} Results:**\n"]
        for i, d in enumerate(discoveries, 1):
            parts = [f"{i}."]
            if "name" in d:
                parts.append(f"**{d['name']}**")
            elif "target_concept" in d:
                parts.append(f"**{d['target_concept']}**")
            if "activation" in d:
                parts.append(f"(activation: {d['activation']:.3f})")
            if "discovery_score" in d:
                parts.append(f"(discovery score: {d['discovery_score']:.3f})")
            if "gap_score" in d:
                parts.append(f"(gap score: {d['gap_score']:.3f})")
            if "bridge_count" in d:
                parts.append(f"via {d['bridge_count']} bridge concepts")
            if "top_bridges" in d:
                bridge_names = [b.get("name", "?") for b in d["top_bridges"][:3]]
                parts.append(f"[bridges: {', '.join(bridge_names)}]")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    except Exception as e:
        return f"Knowledge discovery error: {e}"


async def execute_tools_parallel(
    tool_calls_with_ids: list[tuple[str, str, dict]],
) -> list[tuple[str, str, str, float]]:
    """Execute multiple tool calls concurrently."""

    async def _run_one(tc_id: str, name: str, args: dict):
        t0 = time.monotonic()
        result = await execute_tool(name, args)
        return tc_id, name, result, time.monotonic() - t0

    tasks = [_run_one(tc_id, name, args) for tc_id, name, args in tool_calls_with_ids]
    return list(await asyncio.gather(*tasks))


# ============================================================================
# LLM Communication
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


async def call_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    include_tools: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict:
    """Call the upstream LLM with retry logic."""
    model = model or UPSTREAM_MODEL
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if include_tools:
        body["tools"] = NATIVE_TOOLS
        body["tool_choice"] = "auto"

    last_error: Optional[str] = None
    client = http_client()

    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(300.0, connect=30.0),
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                last_error = f"[LLM Error: HTTP {resp.status_code}] {error_text}"

                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_LLM_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(
                        f"[{req_id}] Retryable error {resp.status_code}, "
                        f"waiting {wait}s (attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue

                return {"error": last_error}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "[LLM Error: No choices in response]"}

            message = choices[0].get("message", {})
            return {
                "message": message,
                "content": message.get("content", "") or "",
                "tool_calls": message.get("tool_calls", None),
                "finish_reason": choices[0].get("finish_reason", ""),
            }

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = f"[LLM Error: {type(e).__name__}]"
            if attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(
                    f"[{req_id}] Timeout, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            return {"error": last_error}

        except Exception as e:
            return {"error": f"[LLM Error: {str(e)}]"}

    return {"error": last_error or "[LLM Error: Max retries exceeded]"}


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

        return conditions

    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[{req_id}] Verification JSON parse error: {e}")
        return conditions


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
2. Generate 1-3 bridge queries that look for unexpected cross-domain connections (serendipity). These should connect the topic to a seemingly unrelated field.
3. Each angle should be independent enough to research separately.
4. Make search queries specific and actionable.
5. Output ONLY valid JSON, no markdown fences or commentary."""


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


# ============================================================================
# Subagent Research (with AoT State Contraction + Saturation Detection)
# ============================================================================

SUBAGENT_PROMPT_TEMPLATE = """You are a focused research subagent. Today is: {date}

Your assigned research angle: {angle_title}
Description: {angle_description}
Initial search query: {angle_query}

**INSTRUCTIONS:**
1. Use tools to research this specific angle thoroughly.
2. After EACH tool result, extract the key facts as atomic conditions.
3. Search from multiple sub-angles within your assigned topic.
4. Read actual web pages, don't just rely on search snippets.
5. Be thorough but focused on your assigned angle.
6. Use arxiv_search for academic papers, wikidata_query for structured facts.
7. Use wayback_fetch if a link is dead or unavailable.

{serendipity_instruction}

**ATOMIC CONDITION FORMAT:**
After gathering information, you must output your findings as atomic conditions.
When you are done researching, output your findings in this exact JSON format:
```json
{{"conditions": [
    {{"fact": "clear factual statement", "source_url": "url", "confidence": 0.8}},
    ...
]}}
```

**TOOL USAGE:**
- You MUST use tools. Never answer from training data alone.
- Do NOT repeat the same search query or fetch the same URL twice.
- If a tool call fails, try a different approach.
- Use different tools for different needs (web search, arxiv for papers, wikidata for facts).

**WHEN TO STOP:**
- You have found 3-10 distinct facts about your angle
- Additional searches return information you already have (saturation)
- You have verified key claims across sources"""

SERENDIPITY_INSTRUCTION = """**SERENDIPITY HUNTING:**
You are not just looking for direct answers. You are hunting for "happy accidents" --
concepts from distant fields that unexpectedly illuminate the query.

When you find a connection that seems:
1. Relevant to the query AND
2. From a completely different domain than other sources AND
3. Surprising (you didn't expect this to be useful)

Flag it as [SERENDIPITOUS FINDING] and increase your search priority
for that domain cluster."""

CONDITION_EXTRACTION_PROMPT = """Based on the research you've done so far, extract all key findings as atomic conditions.

Output ONLY a JSON object with this structure:
{"conditions": [
    {"fact": "clear factual statement supported by your research", "source_url": "the URL source", "confidence": 0.9},
    ...
]}

Rules:
- Each fact should be a single, clear, verifiable statement
- Confidence: 0.9 for well-sourced facts, 0.7 for partially verified, 0.5 for single-source, 0.3 for uncertain
- Include the most relevant source URL for each fact
- Output 3-10 conditions maximum
- Output ONLY valid JSON, no markdown fences"""

GAP_ANALYSIS_PROMPT = """Analyze the current research findings and identify gaps that need deeper investigation.

Current findings:
{findings}

Original query: {query}

Output ONLY a JSON object:
{{
  "gaps": [
    {{"title": "gap description", "query": "specific search query to fill this gap", "priority": "high|medium|low"}}
  ],
  "saturation_estimate": 0.7
}}

Rules:
- Identify 1-3 gaps maximum
- Only include high-priority gaps that would significantly improve the answer
- saturation_estimate: 0.0 = no useful info found, 1.0 = topic fully covered
- Output ONLY valid JSON"""


async def run_subagent(
    angle: dict,
    subagent_index: int,
    progress_queue: asyncio.Queue,
    req_id: str,
    user_query: str,
    depth: int = 0,
) -> SubagentResult:
    """Run a single subagent's research loop on one angle.

    Uses AoT-style state contraction and dynamic saturation detection.
    May spawn recursive sub-subagents for rabbit holes.
    """
    angle_title = angle.get("title", f"Angle {subagent_index + 1}")
    angle_query = angle.get("query", user_query)
    angle_desc = angle.get("description", "Research this angle")
    is_bridge = angle.get("is_bridge", False)
    sa_id = f"{req_id}-sa{subagent_index}" + (f"-d{depth}" if depth > 0 else "")

    log.info(f"[{sa_id}] Starting subagent: {angle_title} (depth={depth})")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    serendipity_inst = SERENDIPITY_INSTRUCTION if is_bridge else ""
    system_prompt = SUBAGENT_PROMPT_TEMPLATE.format(
        date=today,
        angle_title=angle_title,
        angle_description=angle_desc,
        angle_query=angle_query,
        serendipity_instruction=serendipity_inst,
    )

    agent_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research this angle thoroughly: {angle_query}"},
    ]

    result = SubagentResult(angle=angle_title)
    used_queries: set[str] = set()
    consecutive_errors = 0
    known_facts: list[str] = []

    try:
        for turn in range(1, MAX_SUBAGENT_TURNS + 1):
            await progress_queue.put({
                "type": "progress",
                "subagent": subagent_index,
                "text": f"  [{angle_title}] Turn {turn}/{MAX_SUBAGENT_TURNS}\n",
            })

            llm_result = await call_llm(
                agent_messages, sa_id,
                model=SUBAGENT_MODEL,
                include_tools=True,
                max_tokens=4096,
                temperature=0.3,
            )

            if "error" in llm_result:
                consecutive_errors += 1
                log.warning(f"[{sa_id}] Turn {turn}: Error: {llm_result['error']}")
                if consecutive_errors >= 3:
                    result.error = llm_result["error"]
                    break
                agent_messages.append({"role": "assistant", "content": llm_result["error"]})
                agent_messages.append({"role": "user", "content": "Error occurred. Try a different approach."})
                continue

            consecutive_errors = 0
            content = llm_result.get("content", "")
            tool_calls = llm_result.get("tool_calls")

            if not tool_calls:
                result.turns_used = turn
                conditions = _parse_conditions(content, angle_title, is_bridge)
                if conditions:
                    for c in conditions:
                        c.trust_score = trust_score_url(c.source_url)
                        c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)
                    result.conditions.extend(conditions)
                    # Feed live findings to heartbeat collector
                    await progress_queue.put({
                        "type": "conditions",
                        "subagent": subagent_index,
                        "conditions": conditions,
                    })
                break

            assistant_msg: dict = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
            agent_messages.append(assistant_msg)

            calls_to_run: list[tuple[str, str, dict]] = []
            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments_str = func.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}

                query_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if query_key in used_queries:
                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "Duplicate call skipped. Try a different query.",
                    })
                    continue

                used_queries.add(query_key)
                calls_to_run.append((tc_id, tool_name, arguments))

            if calls_to_run:
                tool_results = await execute_tools_parallel(calls_to_run)
                result.tool_calls_made += len(tool_results)

                for tc_id, tool_name, tool_result, duration in tool_results:
                    await progress_queue.put({
                        "type": "tool",
                        "subagent": subagent_index,
                        "text": f"  [{angle_title}] {tool_name} ({duration:.1f}s)\n",
                    })

                    truncated = tool_result
                    if len(tool_result) > 8000:
                        truncated = tool_result[:6000] + "\n[...truncated...]\n" + tool_result[-1500:]

                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": truncated,
                    })

            # AoT State Contraction every 3 turns
            if turn > 0 and turn % 3 == 0 and turn < MAX_SUBAGENT_TURNS:
                contraction_msgs = agent_messages + [
                    {"role": "user", "content": CONDITION_EXTRACTION_PROMPT}
                ]
                extract_result = await call_llm(
                    contraction_msgs, sa_id,
                    model=SUBAGENT_MODEL,
                    max_tokens=2048,
                    temperature=0.1,
                )
                if "error" not in extract_result:
                    mid_conditions = _parse_conditions(
                        extract_result.get("content", ""), angle_title, is_bridge
                    )
                    if mid_conditions:
                        for c in mid_conditions:
                            c.trust_score = trust_score_url(c.source_url)
                            c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)

                        # Dynamic Saturation Detection
                        new_fact_texts = [c.fact for c in mid_conditions]
                        if known_facts:
                            novel_count = 0
                            for nf in new_fact_texts:
                                nf_words = set(nf.lower().split())
                                max_sim = 0.0
                                for kf in known_facts:
                                    kf_words = set(kf.lower().split())
                                    if nf_words and kf_words:
                                        sim = len(nf_words & kf_words) / max(len(nf_words | kf_words), 1)
                                        max_sim = max(max_sim, sim)
                                if max_sim < 0.6:
                                    novel_count += 1
                            novelty = novel_count / max(len(new_fact_texts), 1)
                        else:
                            novelty = 1.0

                        result.novelty_history.append(novelty)
                        known_facts.extend(new_fact_texts)
                        result.conditions.extend(mid_conditions)

                        # Feed live findings to heartbeat collector
                        await progress_queue.put({
                            "type": "conditions",
                            "subagent": subagent_index,
                            "conditions": mid_conditions,
                        })

                        if len(result.novelty_history) >= 2 and novelty < NOVELTY_STOP_THRESHOLD:
                            log.info(f"[{sa_id}] Saturation detected (novelty={novelty:.2f}), stopping early")
                            await progress_queue.put({
                                "type": "progress",
                                "subagent": subagent_index,
                                "text": f"  [{angle_title}] Saturation detected, stopping early\n",
                            })
                            result.turns_used = turn
                            break

                        conditions_text = "\n".join(c.to_text() for c in mid_conditions)
                        agent_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": (
                                f"Continue researching: {angle_query}\n\n"
                                f"Findings so far (from previous turns):\n{conditions_text}\n\n"
                                f"Find NEW information that is NOT covered above. "
                                f"Search for different sub-angles, deeper details, or verification."
                            )},
                        ]
                        log.info(
                            f"[{sa_id}] Turn {turn}: AoT contraction - "
                            f"compressed {len(mid_conditions)} conditions, "
                            f"novelty={novelty:.2f}, reset context"
                        )

            result.turns_used = turn

        # Final condition extraction if we used all turns
        if result.turns_used >= MAX_SUBAGENT_TURNS and not result.conditions:
            agent_messages.append({"role": "user", "content": CONDITION_EXTRACTION_PROMPT})
            final_extract = await call_llm(
                agent_messages, sa_id,
                model=SUBAGENT_MODEL,
                max_tokens=2048,
                temperature=0.1,
            )
            if "error" not in final_extract:
                conditions = _parse_conditions(
                    final_extract.get("content", ""), angle_title, is_bridge
                )
                if conditions:
                    for c in conditions:
                        c.trust_score = trust_score_url(c.source_url)
                        c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)
                    result.conditions.extend(conditions)
                    # Feed live findings to heartbeat collector
                    await progress_queue.put({
                        "type": "conditions",
                        "subagent": subagent_index,
                        "conditions": conditions,
                    })

        # Recursive subagent spawning for rabbit holes
        if (depth < MAX_RECURSIVE_DEPTH
                and result.conditions
                and len(result.novelty_history) > 0
                and result.novelty_history[-1] > NOVELTY_EXPAND_THRESHOLD):
            findings_text = "\n".join(c.to_text() for c in result.conditions[:15])
            gap_messages = [
                {"role": "system", "content": GAP_ANALYSIS_PROMPT.format(
                    findings=findings_text, query=angle_query
                )},
                {"role": "user", "content": "Identify research gaps."},
            ]
            gap_result = await call_llm(gap_messages, sa_id, model=SUBAGENT_MODEL, max_tokens=1024, temperature=0.2)

            if "error" not in gap_result:
                gap_content = gap_result.get("content", "")
                try:
                    cleaned = gap_content.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                        cleaned = re.sub(r'\s*```$', '', cleaned)
                    gap_data = json.loads(cleaned)
                    gaps = gap_data.get("gaps", [])
                    high_priority_gaps = [g for g in gaps if g.get("priority") == "high"][:2]

                    if high_priority_gaps:
                        log.info(f"[{sa_id}] Spawning {len(high_priority_gaps)} recursive sub-subagents (depth={depth+1})")
                        await progress_queue.put({
                            "type": "progress",
                            "subagent": subagent_index,
                            "text": f"  [{angle_title}] Spawning {len(high_priority_gaps)} sub-subagents for rabbit holes\n",
                        })

                        child_tasks = []
                        for gi, gap in enumerate(high_priority_gaps):
                            child_angle = {
                                "title": f"{angle_title} > {gap.get('title', 'Deep dive')}",
                                "query": gap.get("query", ""),
                                "description": gap.get("title", ""),
                                "is_bridge": is_bridge,
                            }
                            child_tasks.append(
                                asyncio.create_task(
                                    run_subagent(child_angle, subagent_index * 100 + gi, progress_queue, req_id, user_query, depth + 1)
                                )
                            )

                        child_results = await asyncio.gather(*child_tasks, return_exceptions=True)
                        for cr in child_results:
                            if isinstance(cr, SubagentResult):
                                result.conditions.extend(cr.conditions)
                                result.spawned_children += 1

                except (json.JSONDecodeError, ValueError):
                    pass

    except Exception as e:
        log.error(f"[{sa_id}] Subagent error: {e}\n{traceback.format_exc()}")
        result.error = str(e)

    # Deduplicate conditions
    seen_facts: set[str] = set()
    unique_conditions: list[AtomicCondition] = []
    for c in result.conditions:
        key = c.fact.lower().strip()[:100]
        if key not in seen_facts:
            seen_facts.add(key)
            unique_conditions.append(c)
    result.conditions = unique_conditions

    await progress_queue.put({
        "type": "done",
        "subagent": subagent_index,
        "angle": angle_title,
        "conditions_count": len(result.conditions),
    })

    log.info(
        f"[{sa_id}] Subagent complete: {len(result.conditions)} conditions, "
        f"{result.turns_used} turns, {result.tool_calls_made} tool calls, "
        f"{result.spawned_children} children spawned"
    )
    return result


def _parse_conditions(content: str, angle: str, is_bridge: bool) -> list[AtomicCondition]:
    """Try to parse atomic conditions from LLM output."""
    if not content:
        return []

    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)
        conditions_data = data.get("conditions", [])
        return [
            AtomicCondition(
                fact=c.get("fact", ""),
                source_url=c.get("source_url", ""),
                confidence=float(c.get("confidence", 0.5)),
                angle=angle,
                is_serendipitous=is_bridge,
            )
            for c in conditions_data
            if c.get("fact")
        ]
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    json_match = re.search(r'\{[^{}]*"conditions"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return [
                AtomicCondition(
                    fact=c.get("fact", ""),
                    source_url=c.get("source_url", ""),
                    confidence=float(c.get("confidence", 0.5)),
                    angle=angle,
                    is_serendipitous=is_bridge,
                )
                for c in data.get("conditions", [])
                if c.get("fact")
            ]
        except (json.JSONDecodeError, ValueError):
            pass

    if len(content.strip()) > 20:
        return [
            AtomicCondition(
                fact=content.strip()[:500],
                angle=angle,
                confidence=0.3,
                is_serendipitous=is_bridge,
            )
        ]

    return []


# ============================================================================
# Tree Research Reactor
# ============================================================================

SPAWN_QUESTIONS_PROMPT = """You are a research strategist.  Given the findings so far from investigating a question, generate focused follow-up questions that would deepen understanding.

**Original user query:** {user_query}
**Question just investigated:** {node_question}
**Context:** {node_context}

**Findings from this investigation:**
{findings_text}

**Questions already in the research tree (avoid duplicates):**
{existing_questions}

Generate follow-up questions.  For each, provide:
- "question": a specific, searchable question
- "context": one sentence on why this matters
- "pressure": 0.0-1.0 importance score (1.0 = critical gap, 0.1 = minor curiosity)

Rules:
- Generate 0-5 questions maximum
- Only questions that would SIGNIFICANTLY improve the final answer
- Higher pressure for: contradictions, unverified claims, critical gaps
- Lower pressure for: tangential curiosity, already-well-covered areas
- 0 questions is fine if the topic is saturated
- Do NOT repeat questions already in the tree
- Output ONLY valid JSON, no markdown fences

Output format:
{{"sub_questions": [{{"question": "...", "context": "...", "pressure": 0.8}}]}}"""


def _compute_pressure(
    base_pressure: float,
    depth: int,
    parent_pressure: float,
) -> float:
    """Compute final pressure score for a research node.

    Combines the LLM's assessed importance with a depth decay and
    inheritance from the parent node's pressure.
    """
    depth_decay = max(0.1, 1.0 - (depth * 0.15))
    inherited = parent_pressure * 0.3
    base_weight = base_pressure * 0.7
    return min(1.0, (base_weight + inherited) * depth_decay)


async def _spawn_sub_questions(
    node: ResearchNode,
    conditions: list[AtomicCondition],
    user_query: str,
    existing_questions: list[str],
    req_id: str,
) -> list[ResearchNode]:
    """Ask LLM to generate follow-up questions from research findings.

    Returns a list of new ResearchNode children.
    """
    if not conditions or node.depth >= TREE_MAX_DEPTH:
        return []

    findings_text = "\n".join(
        f"- {c.fact} [confidence: {c.confidence:.1f}]"
        for c in conditions[:15]
    )

    existing_text = "\n".join(f"- {q}" for q in existing_questions[-30:]) or "(none yet)"

    prompt = SPAWN_QUESTIONS_PROMPT.format(
        user_query=user_query,
        node_question=node.question,
        node_context=node.context,
        findings_text=findings_text,
        existing_questions=existing_text,
    )

    result = await call_llm(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate follow-up questions based on these findings."},
        ],
        req_id,
        model=SUBAGENT_MODEL,
        max_tokens=1024,
        temperature=0.4,
    )

    if "error" in result:
        log.warning(f"[{req_id}] Spawn sub-questions error: {result['error']}")
        return []

    content = result.get("content", "")
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        json_match = re.search(r'\{[^{}]*"sub_questions"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError):
                return []
        else:
            return []

    children: list[ResearchNode] = []
    for sq in data.get("sub_questions", []):
        question = sq.get("question", "").strip()
        if not question:
            continue
        # Skip near-duplicate questions
        q_lower = question.lower()
        if any(q_lower in eq.lower() or eq.lower() in q_lower for eq in existing_questions):
            continue

        raw_pressure = float(sq.get("pressure", 0.5))
        pressure = _compute_pressure(raw_pressure, node.depth + 1, node.pressure)

        if pressure < TREE_PRESSURE_THRESHOLD:
            continue

        child = ResearchNode(
            id=f"{req_id}-n{uuid.uuid4().hex[:6]}",
            question=question,
            context=sq.get("context", ""),
            depth=node.depth + 1,
            pressure=pressure,
            parent_id=node.id,
        )
        children.append(child)

    return children


async def _research_single_node(
    node: ResearchNode,
    user_query: str,
    req_id: str,
    collector: "LiveFindingsCollector",
    curated_queue: asyncio.Queue,
) -> tuple[list[AtomicCondition], SubagentResult]:
    """Research a single tree node using the existing subagent loop.

    This wraps run_subagent with the tree node's question/context
    and feeds findings into the collector and curated queue.
    """
    angle = {
        "title": node.question[:80],
        "query": node.question,
        "description": node.context,
        "is_bridge": False,
    }

    # Lightweight internal progress queue (not emitting system noise)
    internal_queue: asyncio.Queue = asyncio.Queue()

    sa_result = await run_subagent(
        angle=angle,
        subagent_index=0,
        progress_queue=internal_queue,
        req_id=req_id,
        user_query=user_query,
        depth=0,
    )

    # Feed conditions to the live findings collector for heartbeat
    if sa_result.conditions:
        await collector.add_conditions(sa_result.conditions)

    # Emit a curated update about what we found
    if sa_result.conditions:
        top_finding = max(sa_result.conditions, key=lambda c: c.confidence)
        await curated_queue.put({
            "type": "finding",
            "node_id": node.id,
            "question": node.question,
            "finding": top_finding.fact[:200],
            "conditions_count": len(sa_result.conditions),
            "depth": node.depth,
        })

    return sa_result.conditions, sa_result


async def tree_research_reactor(
    user_query: str,
    prior_conditions: list[dict],
    graph_neighbors: list[dict],
    req_id: str,
    collector: "LiveFindingsCollector",
    curated_queue: asyncio.Queue,
) -> dict:
    """Tree-based research reactor.

    Explores the research space as a tree: each finding can spawn
    sub-questions which get explored by concurrent workers.

    The semaphore governs only the workers doing active LLM+tool
    research.  Spawning and queuing are free (no slot consumed).

    Returns a dict with keys matching the old plan+subagents output:
      - subagent_results, all_conditions, total_turns, total_tools,
        total_children, progress_log
    """
    sem = asyncio.Semaphore(TREE_MAX_CONCURRENT)
    pending: asyncio.PriorityQueue = asyncio.PriorityQueue()
    progress: list[str] = []

    # Bookkeeping
    all_conditions: list[AtomicCondition] = []
    all_results: list[SubagentResult] = []
    all_questions: list[str] = [user_query]
    nodes_by_id: dict[str, ResearchNode] = {}
    total_queued = 0
    total_processed = 0
    active_workers = 0  # count of workers currently researching a node
    lock = asyncio.Lock()
    done_event = asyncio.Event()  # set when tree exploration is complete

    # Build the root node
    prior_text = ""
    if prior_conditions:
        prior_text = " | Prior knowledge: " + "; ".join(
            pc["fact"][:80] for pc in prior_conditions[:5]
        )

    neighbor_text = ""
    if graph_neighbors:
        neighbor_text = " | Graph context: " + "; ".join(
            f"{n.get('fact', '')[:80]} (via {n.get('via_entity', '?')})"
            for n in graph_neighbors[:5]
        )

    root = ResearchNode(
        id=f"{req_id}-root",
        question=user_query,
        context=f"Original user query{prior_text}{neighbor_text}",
        depth=0,
        pressure=1.0,
    )
    nodes_by_id[root.id] = root
    await pending.put(root)
    total_queued = 1

    progress.append(
        f"\n**[Phase 2: Tree Research Reactor]** "
        f"(max {TREE_MAX_CONCURRENT} concurrent, "
        f"depth limit {TREE_MAX_DEPTH}, "
        f"node budget {TREE_MAX_NODES})\n"
    )
    progress.append(f"Root question: {user_query[:120]}\n")

    await curated_queue.put({
        "type": "start",
        "question": user_query,
    })

    async def worker(worker_id: int) -> None:
        nonlocal total_processed, total_queued, active_workers

        while True:
            # Wait for work or termination.  Instead of a simple timeout
            # (which causes idle workers to exit before children are
            # spawned), we poll the queue in short intervals and only
            # exit when done_event is set.
            node = None
            while node is None:
                if done_event.is_set():
                    return
                try:
                    node = await asyncio.wait_for(
                        pending.get(), timeout=TREE_WORKER_IDLE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Check if exploration is complete: no items in
                    # queue and no workers actively researching.
                    async with lock:
                        if active_workers == 0 and pending.empty():
                            done_event.set()
                            return
                    continue

            # Skip pruned nodes
            if node.status == "pruned":
                continue

            # Mark this worker as active before acquiring semaphore
            async with lock:
                active_workers += 1

            try:
                # Acquire semaphore — only active research counts
                async with sem:
                    node.status = "researching"

                    conditions, sa_result = await _research_single_node(
                        node, user_query, req_id, collector, curated_queue,
                    )

                    node.status = "done"

                async with lock:
                    total_processed += 1
                    all_conditions.extend(conditions)
                    all_results.append(sa_result)

                # Spawn children (doesn't hold semaphore)
                async with lock:
                    current_queued = total_queued

                if current_queued < TREE_MAX_NODES and conditions:
                    children = await _spawn_sub_questions(
                        node, conditions, user_query, all_questions, req_id,
                    )

                    async with lock:
                        actually_queued = 0
                        for child in children:
                            if total_queued >= TREE_MAX_NODES:
                                break
                            nodes_by_id[child.id] = child
                            all_questions.append(child.question)
                            await pending.put(child)
                            total_queued += 1
                            actually_queued += 1

                    if actually_queued > 0:
                        await curated_queue.put({
                            "type": "branch",
                            "parent_question": node.question[:80],
                            "children_count": actually_queued,
                            "top_child": children[0].question[:100] if children else "",
                            "depth": node.depth + 1,
                        })
            finally:
                async with lock:
                    active_workers -= 1

    # Launch worker pool
    workers = [
        asyncio.create_task(worker(i))
        for i in range(TREE_MAX_CONCURRENT)
    ]

    await asyncio.gather(*workers, return_exceptions=True)

    # Compute totals
    total_turns = sum(r.turns_used for r in all_results)
    total_tools = sum(r.tool_calls_made for r in all_results)
    total_children = sum(r.spawned_children for r in all_results)

    progress.append(
        f"\n**Tree Exploration Complete:** "
        f"{total_processed} nodes explored "
        f"(depth reached: {max((n.depth for n in nodes_by_id.values()), default=0)}), "
        f"{len(all_conditions)} atomic conditions, "
        f"{total_turns} total turns, {total_tools} tool calls\n"
    )

    await curated_queue.put({
        "type": "summary",
        "nodes_explored": total_processed,
        "conditions_count": len(all_conditions),
    })

    return {
        "subagent_results": all_results,
        "all_conditions": all_conditions,
        "total_turns": total_turns,
        "total_tools": total_tools,
        "total_children": total_children,
        "progress_log": progress,
    }


# ============================================================================
# Live Findings Collector (shared state for heartbeat)
# ============================================================================


class LiveFindingsCollector:
    """Thread-safe collector that subagents populate with conditions in real-time.

    The heartbeat task reads from this to surface interesting findings.
    """

    def __init__(self) -> None:
        self._conditions: list[AtomicCondition] = []
        self._lock = asyncio.Lock()
        self._shared_facts: set[str] = set()  # facts already sent to the user

    async def add_conditions(self, conditions: list[AtomicCondition]) -> None:
        async with self._lock:
            self._conditions.extend(conditions)

    async def get_new_findings(self) -> list[AtomicCondition]:
        """Return conditions not yet shared via heartbeat."""
        async with self._lock:
            new = [
                c for c in self._conditions
                if c.fact.lower().strip()[:100] not in self._shared_facts
            ]
            return new

    async def mark_shared(self, fact: str) -> None:
        async with self._lock:
            self._shared_facts.add(fact.lower().strip()[:100])

    async def get_shared_facts(self) -> list[str]:
        """Return a copy of facts already shared via heartbeat."""
        async with self._lock:
            return list(self._shared_facts)

    async def all_conditions(self) -> list[AtomicCondition]:
        async with self._lock:
            return list(self._conditions)


# ============================================================================
# Heartbeat Prompt & Task
# ============================================================================

_HEARTBEAT_PROMPT = """You are a research analyst. Share ONE noteworthy new finding as a direct factual statement in under 40 words.

Rules:
- Lead with the specific data: names, numbers, prices, dates, percentages, sources
- Example: "Technogym Biostrength uses AI-driven resistance adjustment and claims 30% faster strength gains vs conventional machines, per their 2024 whitepaper."
- NO commentary, NO excitement, NO exclamation marks, NO "I found", NO "It turns out", NO "Oh my gosh"
- NO hedging like "how cool is that" or "imagine" — just the facts
- Professional, dry, factual — like a Reuters wire report
- If nothing is genuinely new vs the "Already shared" list, reply with exactly: SKIP

New findings:
{findings}

Already shared (do NOT repeat or rephrase — if a finding covers the same topic as any item below, output SKIP):
{already_shared}"""


_HEARTBEAT_STATUS_PHRASES = [
    "Searching for additional sources...",
    "Cross-referencing across databases...",
    "Verifying claims against primary sources...",
    "Checking for corroborating evidence...",
    "Scanning recent publications...",
    "Reviewing archived records...",
    "Comparing data across sources...",
    "Evaluating source reliability...",
]


async def _generate_heartbeat_message(
    collector: LiveFindingsCollector,
    req_id: str,
    _phrase_idx: list[int],
) -> str:
    """Generate an engaging heartbeat message using Mistral Small.

    Falls back to a static status phrase if the LLM call fails or
    there are no new findings.
    """
    new_findings = await collector.get_new_findings()

    if not new_findings:
        # Nothing new — pick a rotating status phrase
        idx = _phrase_idx[0] % len(_HEARTBEAT_STATUS_PHRASES)
        _phrase_idx[0] += 1
        return _HEARTBEAT_STATUS_PHRASES[idx]

    # Build a compact list of new facts for the LLM
    findings_text = "\n".join(
        f"- {c.fact[:200]}" for c in new_findings[:10]
    )

    shared_facts = await collector.get_shared_facts()
    already_text = "\n".join(
        f"- {f[:120]}" for f in shared_facts[:8]
    ) if shared_facts else "(none yet)"

    prompt = _HEARTBEAT_PROMPT.format(
        findings=findings_text,
        already_shared=already_text,
    )

    try:
        result = await call_llm(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Give a brief update."},
            ],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=60,
            temperature=0.5,
        )

        if "error" not in result:
            msg = result.get("content", "").strip()
            # If the LLM says SKIP, there's nothing new to share
            if msg and msg.upper().strip() == "SKIP":
                idx = _phrase_idx[0] % len(_HEARTBEAT_STATUS_PHRASES)
                _phrase_idx[0] += 1
                return _HEARTBEAT_STATUS_PHRASES[idx]
            if msg and len(msg) > 10:
                # Mark all findings as shared to prevent rephrasing
                for f in new_findings:
                    await collector.mark_shared(f.fact)
                return msg
    except Exception as e:
        log.debug(f"[{req_id}] Heartbeat LLM call failed: {e}")

    # Fallback
    idx = _phrase_idx[0] % len(_HEARTBEAT_STATUS_PHRASES)
    _phrase_idx[0] += 1
    return _HEARTBEAT_STATUS_PHRASES[idx]


async def _heartbeat_loop(
    output_queue: asyncio.Queue,
    collector: LiveFindingsCollector,
    chunk_fn,
    req_id: str,
    interval: float = 8.0,
    curated_queue: Optional[asyncio.Queue] = None,
) -> None:
    """Background task: emit curated research updates into the SSE stream.

    When a curated_queue is provided (tree reactor mode), it consumes
    structured events from the reactor and formats them as user-facing
    updates.  Otherwise falls back to the LLM-based heartbeat.

    Also emits `: keepalive` SSE comments every 5 seconds to prevent
    proxy/CDN timeouts (these are invisible to the UI parser).
    """
    phrase_idx = [0]
    last_heartbeat = time.monotonic()
    KEEPALIVE_INTERVAL = 5.0

    try:
        while True:
            now = time.monotonic()
            time_since_heartbeat = now - last_heartbeat

            # Drain curated queue first (tree reactor events)
            curated_msg = None
            if curated_queue is not None:
                try:
                    curated_msg = curated_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            if curated_msg is not None:
                formatted = _format_curated_event(curated_msg)
                if formatted:
                    await output_queue.put(chunk_fn(f"\n{formatted}\n"))
                    last_heartbeat = time.monotonic()
            elif time_since_heartbeat >= interval:
                msg = await _generate_heartbeat_message(collector, req_id, phrase_idx)
                await output_queue.put(chunk_fn(f"\n{msg}\n"))
                last_heartbeat = time.monotonic()
            else:
                # Emit invisible keepalive comment
                await output_queue.put(": keepalive\n\n")

            # Sleep until next event (keepalive or heartbeat, whichever is sooner)
            next_heartbeat_in = max(0.1, interval - (time.monotonic() - last_heartbeat))
            await asyncio.sleep(min(KEEPALIVE_INTERVAL, next_heartbeat_in))

    except asyncio.CancelledError:
        log.debug(f"[{req_id}] Heartbeat task cancelled")
        return


def _format_curated_event(event: dict) -> str:
    """Format a tree reactor curated event into a user-facing message.

    Returns an empty string for events that should be silently consumed.
    """
    evt_type = event.get("type", "")

    if evt_type == "start":
        return f"Investigating: {event.get('question', '')[:120]}"

    elif evt_type == "finding":
        finding = event.get("finding", "")
        depth = event.get("depth", 0)
        count = event.get("conditions_count", 0)
        q = event.get("question", "")[:80]
        if depth > 0:
            return f"[depth {depth}] {q} — {count} findings. Key: {finding}"
        return f"{q} — {count} findings. Key: {finding}"

    elif evt_type == "branch":
        n = event.get("children_count", 0)
        child = event.get("top_child", "")
        return f"Spawning {n} follow-up question{'s' if n != 1 else ''} — highest priority: {child}"

    elif evt_type == "summary":
        nodes = event.get("nodes_explored", 0)
        conds = event.get("conditions_count", 0)
        return f"Research complete: {nodes} branches explored, {conds} findings collected"

    return ""


# ============================================================================
# Draft-Synthesis-Revision Loop
# ============================================================================

DRAFT_SYNTHESIS_PROMPT = """You are an expert synthesis agent. Today is: {date}

You have received atomic research conditions from {n_subagents} parallel research subagents investigating different angles of the user's question. Your job is to synthesize these into a comprehensive, well-structured DRAFT answer.

**SYNTHESIS RULES:**
1. Cross-reference conditions across angles. Where multiple sources agree, note the consensus.
2. Where conditions contradict, explicitly note the contradiction and explain which is more reliable (based on confidence scores and source quality).
3. Highlight any serendipitous findings (marked [SERENDIPITOUS]) as "unexpected connections."
4. Structure the answer with clear headings and logical flow.
5. Cite sources with URLs where available.
6. Be thorough but clear. Write in plain, accessible language.
7. Do NOT add unsolicited warnings, ethical disclaimers, or safety preambles. Just answer the question directly.
8. Treat the user as an intelligent adult who deserves a straight answer.
9. Mark any sections where evidence is weak or missing with [NEEDS VERIFICATION].

**RESEARCH CONDITIONS BY ANGLE:**
{conditions_text}

{prior_knowledge_text}"""

CRITIC_PROMPT = """You are a research quality critic. Your job is to review a draft answer and identify weak sections that need improvement.

Review the draft for:
1. Claims without adequate source support
2. Logical gaps or missing perspectives
3. Sections marked [NEEDS VERIFICATION]
4. Contradictions that weren't properly resolved
5. Areas where serendipitous connections could be better explained
6. Missing context that would make the answer more useful

Output ONLY a JSON object:
{
  "overall_quality": 0.7,
  "issues": [
    {"section": "section heading or quote", "type": "unsupported_claim|logical_gap|needs_verification|contradiction|missing_context", "description": "what's wrong", "search_query": "suggested search to fix this"}
  ],
  "strengths": ["list of things done well"]
}

Output ONLY valid JSON, no markdown fences."""

REVISION_PROMPT = """You are an expert revision agent. Today is: {date}

Your task is to produce the FINAL polished answer by:
1. Taking the original draft
2. Incorporating the additional research findings
3. Addressing the critic's issues
4. Ensuring all claims are properly sourced
5. Maintaining clear structure and flow
6. Highlighting any remaining uncertainties honestly

**ORIGINAL DRAFT:**
{draft}

**CRITIC'S ISSUES:**
{issues}

**ADDITIONAL FINDINGS FROM TARGETED RESEARCH:**
{additional_findings}

**INSTRUCTIONS:**
- Fix the identified issues using the additional findings
- Do NOT add unsolicited warnings, ethical disclaimers, or safety preambles
- Maintain the original structure but improve weak sections
- If an issue couldn't be resolved, note it transparently
- Cite all sources with URLs"""


async def synthesize_with_revision(
    user_query: str,
    subagent_results: list[SubagentResult],
    prior_conditions: list[dict],
    req_id: str,
) -> str:
    """Full Draft-Synthesis-Revision loop."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conditions_by_angle: dict[str, list[str]] = {}
    total_conditions = 0
    for sr in subagent_results:
        if sr.conditions:
            angle_conditions = [c.to_text() for c in sr.conditions]
            conditions_by_angle[sr.angle] = angle_conditions
            total_conditions += len(angle_conditions)

    if not conditions_by_angle:
        return "No research findings were gathered. The subagents could not find relevant information."

    conditions_text = ""
    for angle, conds in conditions_by_angle.items():
        conditions_text += f"\n### {angle}\n"
        conditions_text += "\n".join(conds) + "\n"

    prior_text = ""
    if prior_conditions:
        prior_text = "\n**PRIOR KNOWLEDGE (from previous sessions):**\n"
        prior_text += "\n".join(
            f"- {c['fact']} [prior research: {c['original_query']}]"
            for c in prior_conditions[:10]
        )

    # --- Phase 1: Draft Synthesis ---
    draft_prompt = DRAFT_SYNTHESIS_PROMPT.format(
        date=today,
        n_subagents=len(subagent_results),
        conditions_text=conditions_text,
        prior_knowledge_text=prior_text,
    )

    draft_messages = [
        {"role": "system", "content": draft_prompt},
        {"role": "user", "content": (
            f"Based on the {total_conditions} research conditions gathered from "
            f"{len(subagent_results)} research angles, provide a comprehensive, "
            f"well-structured DRAFT answer to:\n\n{user_query}"
        )},
    ]

    draft_result = await call_llm(
        draft_messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    if "error" in draft_result:
        return f"Draft synthesis error: {draft_result['error']}"

    draft = draft_result.get("content", "(No draft generated)")

    # --- Phase 2: Critic Review ---
    critic_messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": f"Original question: {user_query}\n\nDraft answer:\n{draft}"},
    ]

    critic_result = await call_llm(
        critic_messages, req_id,
        model=SUBAGENT_MODEL,
        max_tokens=2048,
        temperature=0.2,
    )

    issues_text = "No issues found."
    additional_findings = "No additional research needed."

    if "error" not in critic_result:
        critic_content = critic_result.get("content", "")
        try:
            cleaned = critic_content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            critic_data = json.loads(cleaned)
            issues = critic_data.get("issues", [])
            overall_quality = critic_data.get("overall_quality", 0.8)

            if issues and overall_quality < 0.85:
                issues_text = json.dumps(issues, indent=2)

                # --- Phase 3: Targeted micro-research on weak points ---
                search_queries = [
                    issue.get("search_query", "")
                    for issue in issues[:3]
                    if issue.get("search_query")
                ]

                if search_queries:
                    search_tasks = [tool_searxng_search(q) for q in search_queries]
                    micro_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                    micro_results = []
                    for q, sr in zip(search_queries, micro_search_results):
                        if isinstance(sr, str) and not sr.startswith("Search error"):
                            micro_results.append(f"**Search: {q}**\n{sr[:2000]}")

                    if micro_results:
                        additional_findings = "\n\n".join(micro_results)

        except (json.JSONDecodeError, ValueError):
            pass

    # --- Phase 4: Final Revision ---
    revision_prompt = REVISION_PROMPT.format(
        date=today,
        draft=draft,
        issues=issues_text,
        additional_findings=additional_findings,
    )

    revision_messages = [
        {"role": "system", "content": revision_prompt},
        {"role": "user", "content": (
            f"Produce the final polished answer to: {user_query}\n\n"
            f"Address the critic's issues and incorporate the additional findings."
        )},
    ]

    final_result = await call_llm(
        revision_messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    if "error" in final_result:
        return draft

    return final_result.get("content", draft)


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
        collector = LiveFindingsCollector()
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
    """Phase 5: Citation verification."""
    req_id = state["req_id"]
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("verify")
    all_conditions = list(state["all_conditions"])
    progress: list[str] = []

    if all_conditions and len(all_conditions) >= 2:
        progress.append("\n**[Phase 5: Citation Verification]**\n")
        progress.append("Verifying claims and detecting contradictions...\n")

        all_conditions = await verify_conditions(all_conditions, req_id)

        high_conf = sum(1 for c in all_conditions if c.confidence >= 0.7)
        low_conf = sum(1 for c in all_conditions if c.confidence < 0.4)
        progress.append(
            f"Verification complete: {high_conf} high-confidence, "
            f"{low_conf} low-confidence conditions.\n"
        )

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("verify")

    return {"all_conditions": all_conditions, "progress_log": progress, "phase": "reflect"}


async def pdr_node_reflect(state: PersistentResearchState) -> dict:
    """Phase 6: AoT Reflection."""
    req_id = state["req_id"]
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

        # Generate HTML report
        try:
            html_report = research_report.generate_report(
                metrics=metrics_dict,
                conditions=condition_dicts,
                final_answer=final_answer,
                progress_log=list(state.get("progress_log", [])),
            )
            research_report.save_report(html_report, req_id)

            # Publish to B2 if configured
            if b2_publisher.is_configured():
                try:
                    report_url = await asyncio.to_thread(b2_publisher.publish_report, req_id, html_report)
                    metrics_json = json.dumps(metrics_dict, indent=2, default=str)
                    metrics_url = await asyncio.to_thread(b2_publisher.publish_metrics, req_id, metrics_json)
                    log.info(f"[{req_id}] Published report to B2: {report_url}")
                except Exception as e:
                    log.error(f"[{req_id}] Failed to publish to B2: {e}")
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


_persistent_research_graph = build_persistent_research_graph()


# ============================================================================
# Main Orchestrator
# ============================================================================

async def _pipeline_producer(
    initial_state: dict[str, Any],
    config: dict,
    output_queue: asyncio.Queue,
    chunk_fn,
    req_id: str,
) -> None:
    """Run the LangGraph pipeline and push SSE chunks to the output queue.

    This runs as a background task so the heartbeat can interleave its
    updates into the same queue.
    """
    last_progress_idx = 0
    final_state = initial_state

    try:
        async for state_update in _persistent_research_graph.astream(
            initial_state, config=config, stream_mode="values",
        ):
            final_state = state_update
            progress_list = state_update.get("progress_log", [])
            for msg in progress_list[last_progress_idx:]:
                await output_queue.put(chunk_fn(msg))
            last_progress_idx = len(progress_list)

        # Pipeline done — emit closing think tag and final answer
        await output_queue.put(chunk_fn("\n</think>\n\n"))
        final_answer = final_state.get("final_answer", "(No answer generated)")
        for i in range(0, len(final_answer), 200):
            await output_queue.put(chunk_fn(final_answer[i:i + 200]))
        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    except Exception as e:
        start_time = initial_state.get("start_time", 0)
        elapsed = time.monotonic() - start_time if start_time else 0
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Persistent research error after {elapsed:.2f}s: {e}\n{tb}")
        await output_queue.put(chunk_fn(f"\nError: {str(e)}\n"))
        await output_queue.put(chunk_fn("\n</think>\n\n"))
        await output_queue.put(chunk_fn(f"**Deep Research Error**\n\nAn error occurred during research: {str(e)}"))
        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    finally:
        await output_queue.put(_STREAM_DONE)


async def run_persistent_research(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Orchestrate the full persistent deep research pipeline via LangGraph.

    Uses an asyncio.Queue so the pipeline, heartbeat task, and keepalive
    comments can all push SSE chunks into a single ordered stream.
    """
    model_id = original_body.get("model", "persistent-miroflow")
    request_id = f"chatcmpl-pdr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    user_query = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if not user_query:
        yield chunk("Error: No user message found.")
        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    log.info(f"[{req_id}] Starting persistent deep research: {user_query[:100]}")

    initial_state: dict[str, Any] = {
        "req_id": req_id,
        "user_query": user_query,
        "start_time": start_time,
        "prior_conditions": [],
        "graph_neighbors": [],
        "subagent_results": [],
        "all_conditions": [],
        "total_turns": 0,
        "total_tools": 0,
        "total_children": 0,
        "nodes_explored": 0,
        "reflection": {},
        "final_answer": "",
        "progress_log": [],
        "phase": "retrieve",
        "report_url": "",
        "metrics_url": "",
    }

    # Create the shared output queue, live findings collector, and curated queue
    output_queue: asyncio.Queue = asyncio.Queue()
    collector = LiveFindingsCollector()
    _live_collectors[req_id] = collector
    curated_queue: asyncio.Queue = asyncio.Queue()
    _curated_queues[req_id] = curated_queue

    # Create metrics collector for this session
    metrics_collector = MetricsCollector(session_id=req_id, query=user_query)
    _metrics_collectors[req_id] = metrics_collector
    metrics_callback = ResearchMetricsCallback(metrics_collector)

    yield chunk("<think>\n")

    config = {
        "configurable": {"thread_id": req_id},
        "callbacks": [metrics_callback],
    }

    # Start the pipeline producer as a background task
    pipeline_task = asyncio.create_task(
        _pipeline_producer(initial_state, config, output_queue, chunk, req_id)
    )

    # Start the heartbeat task with curated queue for tree reactor updates
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            output_queue, collector, chunk, req_id,
            interval=8.0, curated_queue=curated_queue,
        )
    )

    try:
        # Consume from the output queue and yield to the SSE response
        while True:
            try:
                item = await asyncio.wait_for(output_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # No items for 5s — emit invisible keepalive to prevent timeouts
                yield ": keepalive\n\n"
                continue

            if item is _STREAM_DONE:
                break

            yield item

    except asyncio.CancelledError:
        log.info(f"[{req_id}] Client disconnected, cancelling pipeline")
        pipeline_task.cancel()
        raise

    finally:
        # Stop the heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Ensure the pipeline task is done
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

        # Clean up the live collector, curated queue, and metrics collector
        _live_collectors.pop(req_id, None)
        _curated_queues.pop(req_id, None)
        _metrics_collectors.pop(req_id, None)
        tracker.finish(req_id)


# ============================================================================
# FastAPI App
# ============================================================================

app = create_app("Persistent Deep Research Proxy")

register_standard_routes(
    app,
    service_name="persistent-deep-research-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "synthesis_model": UPSTREAM_MODEL,
        "subagent_model": SUBAGENT_MODEL,
        "searxng": SEARXNG_URL,
        "max_subagents": MAX_SUBAGENTS,
        "max_subagent_turns": MAX_SUBAGENT_TURNS,
        "max_recursive_depth": MAX_RECURSIVE_DEPTH,
        "tree_max_concurrent": TREE_MAX_CONCURRENT,
        "tree_max_depth": TREE_MAX_DEPTH,
        "tree_max_nodes": TREE_MAX_NODES,
        "research_namespace": RESEARCH_NAMESPACE,
        "jsonl_log_dir": JSONL_LOG_DIR,
        "tools": [t["function"]["name"] for t in NATIVE_TOOLS],
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "persistent-miroflow",
            "object": "model",
            "created": 1700000000,
            "owned_by": "persistent-deep-research-proxy",
            "name": "Persistent MiroFlow",
        }]
    })


@app.get("/knowledge/stats")
async def knowledge_stats():
    """Return statistics about the persistent knowledge base (from Neo4j)."""
    try:
        stats = await knowledge_client.research_stats(namespace=RESEARCH_NAMESPACE)
        return JSONResponse({
            "total_conditions": stats.get("total_conditions", 0),
            "total_sessions": stats.get("total_sessions", 0),
            "unique_queries": stats.get("total_queries", 0),
            "total_entities": stats.get("total_entities", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "storage_backend": "neo4j",
            "namespace": RESEARCH_NAMESPACE,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/research/reports")
async def get_research_reports():
    """List all available research reports with metadata."""
    try:
        reports = list_available_reports()
        return JSONResponse({"reports": reports, "count": len(reports)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


_SAFE_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@app.get("/research/report/{session_id}")
async def get_research_report(session_id: str):
    """Serve the HTML report for a research session.

    First checks B2, then falls back to local file.
    """
    if not _SAFE_SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)

    from fastapi.responses import HTMLResponse

    # Try local file first
    report_path = os.path.join(
        os.getenv("RESEARCH_REPORTS_DIR", "/opt/persistent_research_logs/reports"),
        f"{session_id}.html",
    )
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return JSONResponse(
            {"error": f"Report not found for session {session_id}"},
            status_code=404,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/research/metrics/{session_id}")
async def get_research_metrics(session_id: str):
    """Serve the metrics JSON for a research session.

    Designed for LLM consumption — structured data for performance analysis.
    """
    if not _SAFE_SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)

    metrics = load_metrics(session_id)
    if metrics is None:
        return JSONResponse(
            {"error": f"Metrics not found for session {session_id}"},
            status_code=404,
        )
    return JSONResponse(metrics)


@app.post("/v1/verify")
async def verify_research(request: Request):
    """Verify an LLM output using the Veritas Inquisitor swarm.

    Request body:
    {
        "target_text": "text to verify",
        "original_query": "original user query",
        "stream": true/false  (default: true)
    }
    """
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}"}},
        )

    target_text = body.get("target_text", "")
    original_query = body.get("original_query", "")
    stream = body.get("stream", True)

    if not target_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "target_text is required"}},
        )

    log.info(f"[{req_id}] Verify request: {len(target_text)} chars")
    tracker.start(req_id, phase="verify", target_chars=len(target_text))

    if not limiter.available():
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        f"Too many concurrent persistent research sessions "
                        f"({limiter.max_concurrent}). Try again shortly."
                    ),
                    "type": "rate_limit",
                }
            },
        )

    if stream:
        async def _stream_verify():
            async with limiter.hold():
                try:
                    async for event in veritas_inquisitor.stream_verification(
                        target_text, original_query, req_id,
                    ):
                        yield event
                finally:
                    tracker.finish(req_id)

        return StreamingResponse(
            _stream_verify(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            async with limiter.hold():
                result = await veritas_inquisitor.verify_output(
                    target_text, original_query, req_id,
                )
            return JSONResponse(result)
        finally:
            tracker.finish(req_id)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
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

    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "persistent-miroflow"),
            tracker=tracker,
            log=log,
        )
    else:
        # Check if the last user message is a large document for ingestion
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    user_text = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                break

        if _is_large_document(user_text):
            log.info(
                f"[{req_id}] Routing to DOCUMENT INGESTION "
                f"({len(user_text):,} chars)"
            )

            async def _guarded_ingest():
                async with limiter.hold():
                    try:
                        async for event in run_document_ingestion(
                            user_text, body, req_id
                        ):
                            yield event
                    finally:
                        tracker.finish(req_id)

            generator = _guarded_ingest()
        else:
            if not limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": (
                                f"Too many concurrent persistent research sessions "
                                f"({limiter.max_concurrent}). Try again shortly."
                            ),
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Routing to PERSISTENT DEEP RESEARCH")

            async def _guarded_research():
                async with limiter.hold():
                    async for event in run_persistent_research(messages, body, req_id):
                        yield event

            generator = _guarded_research()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Persistent Deep Research Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
