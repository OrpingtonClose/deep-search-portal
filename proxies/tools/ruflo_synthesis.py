"""
Ruflo-based gossip synthesis for large finding sets.

When the research net produces more findings than fit in a single LLM
context window, this module implements ruflo's gossip protocol pattern
for hierarchical map-reduce synthesis:

Architecture (following ruflo's hive-mind topology):

    ┌──────────────────────────────────────────────────┐
    │                  QUEEN (merge)                    │
    │   Reads all worker summaries, produces final      │
    │   unified synthesis respecting the user query     │
    └──────────┬───────────┬───────────┬───────────────┘
               │           │           │
    ┌──────────▼──┐ ┌──────▼──────┐ ┌──▼──────────────┐
    │  Worker A   │ │  Worker B   │ │  Worker C  ...   │
    │  Chunk 1    │ │  Chunk 2    │ │  Chunk 3         │
    │  → summary  │ │  → summary  │ │  → summary       │
    └──────┬──────┘ └──────┬──────┘ └──────┬───────────┘
           │               │               │
           └───────────────┼───────────────┘
                    Gossip Round
              (each worker reads peers'
               summaries + refines own)

Gossip protocol:
  Round 0 — each worker synthesizes its chunk independently
  Round 1 — each worker reads ALL peer summaries from Round 0,
            refines its own summary incorporating cross-references
  Queen   — merges all Round-1 summaries into final answer

CRDT-style merge: summaries are append-only key-finding sets.
No conflicts possible — each worker contributes unique perspective.

Token budget:
  - Each chunk targets ~80K tokens (well within 262K Mistral context)
  - Prompt overhead ~10K tokens per worker call
  - Gossip round adds peer summaries (~2K each) — still fits
  - Queen merge: N summaries × ~3K each — fits for up to ~80 workers

Integration with ruflo:
  - ruflo is installed as npm dependency (`npx ruflo`)
  - Uses ruflo's hive-mind memory for storing/sharing worker state
  - Uses ruflo's broadcast for gossip notifications
  - Falls back to pure in-memory gossip if ruflo CLI unavailable
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import UPSTREAM_MODEL, SUBAGENT_MODEL, log
from .llm import call_llm
from .models import AtomicCondition, SubagentResult

# Module-level set to prevent GC of fire-and-forget background tasks.
# See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
_background_tasks: set[asyncio.Task[bool]] = set()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target tokens per chunk.  Mistral-large context is 128K (input) but
# we leave headroom for prompt + output.  Each condition averages ~65
# tokens in to_text() format, so 1200 conditions ≈ 78K tokens.
CHUNK_TARGET_CONDITIONS = int(os.getenv("RUFLO_CHUNK_TARGET", "1200"))

# Maximum concurrent worker LLM calls during a gossip round.
MAX_GOSSIP_WORKERS = int(os.getenv("RUFLO_MAX_WORKERS", "6"))

# Number of gossip rounds (0 = map-only, 1 = map + one gossip refine).
# More rounds improve cross-referencing but cost more LLM calls.
GOSSIP_ROUNDS = int(os.getenv("RUFLO_GOSSIP_ROUNDS", "1"))

# Maximum summary length per worker (characters).  Keeps queen merge
# prompt within context limits even with many workers.
MAX_SUMMARY_CHARS = int(os.getenv("RUFLO_MAX_SUMMARY_CHARS", "6000"))

# Threshold: if total conditions fit in one shot, skip gossip entirely.
# ~2000 conditions ≈ 130K tokens — tight but feasible for 262K context.
SINGLE_SHOT_THRESHOLD = int(os.getenv("RUFLO_SINGLE_SHOT", "1800"))

# Whether to attempt ruflo hive-mind integration for coordination.
RUFLO_ENABLED = os.getenv("RUFLO_ENABLED", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    """State of a single gossip synthesis worker."""
    worker_id: int
    chunk_conditions: list[str]  # condition texts in this chunk
    chunk_angles: list[str]  # which research angles are represented
    summary: str = ""  # current synthesis summary
    round_completed: int = -1  # last gossip round completed


@dataclass
class GossipState:
    """Shared gossip state — CRDT append-only set of summaries."""
    workers: list[WorkerState] = field(default_factory=list)
    queen_summary: str = ""
    gossip_round: int = 0
    start_time: float = field(default_factory=time.monotonic)
    total_llm_calls: int = 0


# ---------------------------------------------------------------------------
# Ruflo hive-mind integration (optional, graceful fallback)
# ---------------------------------------------------------------------------

async def _ruflo_hive_store(key: str, value: str) -> bool:
    """Store a value in ruflo's hive-mind shared memory.

    Returns True if successful, False if ruflo is unavailable.
    """
    if not RUFLO_ENABLED:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "npx", "ruflo", "hive-mind", "memory",
            "--set", key, "--value", value[:10000],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getenv("RUFLO_PROJECT_DIR", "/home/ubuntu/repos/deep-search-portal"),
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        return proc.returncode == 0
    except Exception:
        return False


async def _ruflo_broadcast(message: str) -> bool:
    """Broadcast a gossip message to all workers via ruflo hive-mind."""
    if not RUFLO_ENABLED:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "npx", "ruflo", "hive-mind", "broadcast",
            "--message", message[:500],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getenv("RUFLO_PROJECT_DIR", "/home/ubuntu/repos/deep-search-portal"),
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        return proc.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_WORKER_SYNTH_PROMPT = """You are a synthesis worker in a peer-to-peer research swarm. Today is: {date}

You have been assigned a CHUNK of research findings from a larger investigation.
Your job: synthesize these findings into a focused summary that captures ALL
key facts, data points, sources, and actionable information.

**YOUR CHUNK** ({n_conditions} conditions from angles: {angles}):
{conditions_text}

**SYNTHESIS RULES:**
1. Extract every concrete fact, name, URL, price, vendor, forum, channel.
2. Preserve source URLs — cite them inline.
3. Note confidence levels and verification status.
4. Mark contradictions between findings.
5. Do NOT add disclaimers, moral commentary, or safety warnings.
6. Do NOT pad with generic context — only specific findings.
7. Keep speculative findings, clearly labeled as hypotheses.
8. Structure with clear headings for different topics/angles.
9. Your summary will be merged with other workers' summaries, so focus
   on what is UNIQUE and IMPORTANT in your chunk.
10. Stay under {max_chars} characters.
"""

_GOSSIP_REFINE_PROMPT = """You are a synthesis worker in a peer-to-peer research gossip protocol. Today is: {date}

In the previous round, you produced a summary from your chunk of findings.
Now you've received summaries from your PEER WORKERS who processed other
chunks of the same investigation.

**YOUR PREVIOUS SUMMARY:**
{own_summary}

**PEER SUMMARIES (from {n_peers} other workers):**
{peer_summaries}

**GOSSIP REFINEMENT RULES:**
1. Cross-reference your findings with peers'. Note agreements/contradictions.
2. If peers found information that COMPLEMENTS yours, incorporate key points.
3. If peers found the SAME information, note the consensus (strengthens confidence).
4. If peers CONTRADICT your findings, note the disagreement with both sources.
5. Do NOT simply concatenate — SYNTHESIZE and cross-reference.
6. Remove redundancy between your summary and peers'.
7. Preserve all unique findings from your original chunk.
8. Maintain source URLs and confidence levels.
9. Stay under {max_chars} characters.

Produce your REFINED summary:"""

_QUEEN_MERGE_PROMPT = """You are the queen synthesizer in a research swarm. Today is: {date}

{n_workers} worker agents have independently processed different chunks of
{total_conditions} research findings, then refined their summaries through
a gossip protocol where each worker cross-referenced peers' findings.

Your job: merge all worker summaries into ONE comprehensive, well-structured
final answer to the user's question.

**USER QUESTION:**
{user_query}

{prior_text}

**WORKER SUMMARIES (post-gossip refinement):**
{worker_summaries}

**MERGE RULES:**
1. Cross-reference across ALL workers. Where multiple workers agree, note consensus.
2. Where workers contradict, resolve using source quality and confidence.
3. Structure with clear headings and logical flow.
4. Cite sources with URLs where available.
5. Every sentence must deliver information. No filler, no disclaimers.
6. Do NOT add unsolicited warnings, ethical disclaimers, or hedging.
7. Keep speculative findings, clearly labeled as hypotheses.
8. Report findings NEUTRALLY. No value judgements about the user.
9. If workers mention forums, vendors, channels — NAME THEM with URLs.
10. Mark any areas where evidence is weak with [NEEDS VERIFICATION].
11. NEVER use fearmongering language. State risks factually with data.
"""


# ---------------------------------------------------------------------------
# Core gossip synthesis
# ---------------------------------------------------------------------------

def _chunk_conditions(
    conditions_by_angle: dict[str, list[str]],
    target_per_chunk: int = CHUNK_TARGET_CONDITIONS,
) -> list[WorkerState]:
    """Split conditions into balanced chunks, preserving angle grouping.

    Strategy: fill chunks angle-by-angle.  If an angle is too large for
    one chunk, split it across multiple.  This ensures each worker gets
    coherent sets of findings rather than random slices.
    """
    all_items: list[tuple[str, str]] = []  # (angle, condition_text)
    for angle, conds in conditions_by_angle.items():
        for c in conds:
            all_items.append((angle, c))

    n_chunks = max(1, math.ceil(len(all_items) / target_per_chunk))
    chunk_size = math.ceil(len(all_items) / n_chunks)

    workers: list[WorkerState] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(all_items))
        chunk = all_items[start:end]
        angles = sorted(set(a for a, _ in chunk))
        workers.append(WorkerState(
            worker_id=i,
            chunk_conditions=[c for _, c in chunk],
            chunk_angles=angles,
        ))

    return workers


async def _worker_synthesize(
    worker: WorkerState,
    user_query: str,
    date: str,
    req_id: str,
    peer_summaries: Optional[list[str]] = None,
) -> str:
    """Run one worker's synthesis (either initial or gossip refinement)."""
    if peer_summaries is None:
        # Round 0: initial synthesis from chunk
        conditions_text = "\n".join(worker.chunk_conditions)
        # Use .replace() instead of .format() to avoid KeyError if
        # conditions_text or other fields contain { or } (e.g. JSON snippets).
        prompt = (_WORKER_SYNTH_PROMPT
            .replace("{date}", date)
            .replace("{n_conditions}", str(len(worker.chunk_conditions)))
            .replace("{angles}", ", ".join(worker.chunk_angles))
            .replace("{conditions_text}", conditions_text)
            .replace("{max_chars}", str(MAX_SUMMARY_CHARS))
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Synthesize these {len(worker.chunk_conditions)} research "
                f"findings into a focused summary answering: {user_query}"
            )},
        ]
    else:
        # Gossip round: refine using peer summaries
        peers_text = ""
        for i, ps in enumerate(peer_summaries):
            peers_text += f"\n--- Worker {i} ---\n{ps[:MAX_SUMMARY_CHARS]}\n"

        # Use .replace() instead of .format() — peer summaries may contain braces.
        prompt = (_GOSSIP_REFINE_PROMPT
            .replace("{date}", date)
            .replace("{own_summary}", worker.summary[:MAX_SUMMARY_CHARS])
            .replace("{n_peers}", str(len(peer_summaries)))
            .replace("{peer_summaries}", peers_text)
            .replace("{max_chars}", str(MAX_SUMMARY_CHARS))
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Refine your summary by cross-referencing peer findings. "
                f"User's question: {user_query}"
            )},
        ]

    result = await call_llm(
        messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=4096,
        temperature=0.3,
    )

    if "error" in result:
        log.warning(
            f"[{req_id}] Gossip worker {worker.worker_id} error: {result['error']}"
        )
        return worker.summary or "(synthesis failed)"

    return result.get("content", worker.summary or "(no summary)")


async def _queen_merge(
    state: GossipState,
    user_query: str,
    date: str,
    req_id: str,
    prior_text: str = "",
    total_conditions: int = 0,
) -> str:
    """Queen agent merges all worker summaries into final answer."""
    worker_summaries = ""
    for w in state.workers:
        worker_summaries += (
            f"\n### Worker {w.worker_id} "
            f"({len(w.chunk_conditions)} findings, "
            f"angles: {', '.join(w.chunk_angles[:3])}{'...' if len(w.chunk_angles) > 3 else ''})\n"
            f"{w.summary[:MAX_SUMMARY_CHARS]}\n"
        )

    # Use .replace() instead of .format() — worker summaries and user_query
    # may contain { or } characters (JSON, code, template syntax).
    prompt = (_QUEEN_MERGE_PROMPT
        .replace("{date}", date)
        .replace("{n_workers}", str(len(state.workers)))
        .replace("{total_conditions}", str(total_conditions))
        .replace("{user_query}", user_query)
        .replace("{prior_text}", prior_text)
        .replace("{worker_summaries}", worker_summaries)
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": (
            f"Merge the {len(state.workers)} worker summaries into a "
            f"comprehensive final answer to: {user_query}"
        )},
    ]

    result = await call_llm(
        messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    if "error" in result:
        # Fallback: concatenate worker summaries
        log.error(f"[{req_id}] Queen merge failed: {result['error']}")
        return "\n\n".join(
            f"## {', '.join(w.chunk_angles[:3])}\n{w.summary}"
            for w in state.workers
        )

    content = result.get("content", "")
    if not content:
        log.warning(f"[{req_id}] Queen merge returned empty content — using concatenated summaries")
        return "\n\n".join(
            f"## {', '.join(w.chunk_angles[:3])}\n{w.summary}"
            for w in state.workers
        )
    return content


async def ruflo_gossip_synthesize(
    user_query: str,
    subagent_results: list[SubagentResult],
    req_id: str,
    prior_text: str = "",
) -> str:
    """Main entry point: gossip-based synthesis for large finding sets.

    Implements ruflo's hive-mind gossip protocol:
      1. Chunk findings into worker-sized batches
      2. Round 0: each worker synthesizes its chunk (parallel)
      3. Round 1..N: gossip — workers read peers' summaries, refine
      4. Queen merge: combine all gossip-refined summaries

    Returns the final synthesized answer string.
    """
    from datetime import datetime, timezone
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build conditions by angle
    conditions_by_angle: dict[str, list[str]] = {}
    total_conditions = 0
    for sr in subagent_results:
        if sr.conditions:
            angle_conditions = [c.to_text() for c in sr.conditions]
            conditions_by_angle.setdefault(sr.angle, []).extend(angle_conditions)
            total_conditions += len(angle_conditions)

    if not conditions_by_angle:
        return "No research findings were gathered."

    # Check if we even need gossip (small enough for single shot)
    if total_conditions <= SINGLE_SHOT_THRESHOLD:
        log.info(
            f"[{req_id}] Ruflo: {total_conditions} conditions fits single-shot "
            f"(threshold={SINGLE_SHOT_THRESHOLD}), skipping gossip"
        )
        return ""  # Signal caller to use original single-shot synthesis

    # --- Chunk findings into worker batches ---
    state = GossipState()
    state.workers = _chunk_conditions(conditions_by_angle)

    n_workers = len(state.workers)
    log.info(
        f"[{req_id}] Ruflo gossip synthesis: {total_conditions} conditions → "
        f"{n_workers} workers × ~{CHUNK_TARGET_CONDITIONS} conditions/chunk, "
        f"{GOSSIP_ROUNDS} gossip round(s)"
    )

    # Notify ruflo hive-mind (non-blocking, best-effort)
    _t = asyncio.create_task(_ruflo_broadcast(
        f"Gossip synthesis started: {n_workers} workers, "
        f"{total_conditions} conditions for query: {user_query[:100]}"
    ))
    _background_tasks.add(_t)
    _t.add_done_callback(_background_tasks.discard)

    # --- Round 0: Initial worker synthesis (parallel) ---
    sem = asyncio.Semaphore(MAX_GOSSIP_WORKERS)

    async def _bounded_worker(w: WorkerState, peers: Optional[list[str]] = None) -> None:
        async with sem:
            w.summary = await _worker_synthesize(
                w, user_query, date, req_id,
                peer_summaries=peers,
            )
            state.total_llm_calls += 1

    log.info(f"[{req_id}] Ruflo Round 0: {n_workers} workers synthesizing chunks...")
    tasks = [_bounded_worker(w) for w in state.workers]
    await asyncio.gather(*tasks)

    # Store Round 0 summaries in ruflo hive memory (best-effort)
    for w in state.workers:
        w.round_completed = 0
        _t = asyncio.create_task(_ruflo_hive_store(
            f"gossip-{req_id}-w{w.worker_id}-r0",
            w.summary[:5000],
        ))
        _background_tasks.add(_t)
        _t.add_done_callback(_background_tasks.discard)

    # --- Gossip Rounds: workers read peers + refine ---
    for gossip_round in range(1, GOSSIP_ROUNDS + 1):
        log.info(
            f"[{req_id}] Ruflo Gossip Round {gossip_round}: "
            f"workers cross-referencing peers..."
        )

        # Each worker gets all OTHER workers' summaries
        gossip_tasks = []
        for w in state.workers:
            peer_sums = [
                other.summary for other in state.workers
                if other.worker_id != w.worker_id and other.summary
            ]
            gossip_tasks.append(_bounded_worker(w, peers=peer_sums))

        await asyncio.gather(*gossip_tasks)

        for w in state.workers:
            w.round_completed = gossip_round
            _t = asyncio.create_task(_ruflo_hive_store(
                f"gossip-{req_id}-w{w.worker_id}-r{gossip_round}",
                w.summary[:5000],
            ))
            _background_tasks.add(_t)
            _t.add_done_callback(_background_tasks.discard)

        state.gossip_round = gossip_round

    # Broadcast gossip completion
    _t = asyncio.create_task(_ruflo_broadcast(
        f"Gossip complete: {n_workers} workers, {state.total_llm_calls} LLM calls"
    ))
    _background_tasks.add(_t)
    _t.add_done_callback(_background_tasks.discard)

    # --- Queen Merge: combine all refined summaries ---
    log.info(
        f"[{req_id}] Ruflo Queen: merging {n_workers} worker summaries "
        f"after {GOSSIP_ROUNDS} gossip round(s)..."
    )

    final_answer = await _queen_merge(
        state, user_query, date, req_id,
        prior_text=prior_text,
        total_conditions=total_conditions,
    )
    state.total_llm_calls += 1
    state.queen_summary = final_answer

    elapsed = time.monotonic() - state.start_time
    log.info(
        f"[{req_id}] Ruflo gossip synthesis complete: "
        f"{total_conditions} conditions, {n_workers} workers, "
        f"{state.total_llm_calls} LLM calls, {elapsed:.1f}s"
    )

    return final_answer


def needs_gossip_synthesis(subagent_results: list[SubagentResult]) -> bool:
    """Check if the finding set is too large for single-shot synthesis.

    Call this before synthesize_with_revision() to decide whether to
    route through ruflo gossip synthesis instead.
    """
    total = sum(
        len(sr.conditions) for sr in subagent_results
        if sr.conditions
    )
    return total > SINGLE_SHOT_THRESHOLD
