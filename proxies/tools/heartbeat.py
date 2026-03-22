"""LiveFindingsCollector and heartbeat generation for SSE streaming.

Extracted from persistent_deep_research_proxy.py lines 5489-5888.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from .config import SUBAGENT_MODEL
from .models import AtomicCondition
from .llm import call_llm

log = logging.getLogger("persistent-research")

# ============================================================================
# Live Findings Collector (shared state for heartbeat)
# ============================================================================


class LiveFindingsCollector:
    """Thread-safe collector that subagents populate with conditions in real-time.

    The heartbeat task reads from this to surface interesting findings.
    Also tracks live tool-call activity so heartbeat messages can reference
    actual sources, tools, and queries instead of generic filler.
    """

    def __init__(self, user_query: str = "") -> None:
        self._conditions: list[AtomicCondition] = []
        self._lock = asyncio.Lock()
        self._shared_facts: set[str] = set()  # facts already sent to the user
        self.user_query: str = user_query  # the original prompt for relevance checks
        # Activity tracking for context-aware heartbeat
        self._recent_tool_calls: list[dict] = []  # {tool, query, duration, ts}
        self._active_questions: list[str] = []  # questions currently being researched
        self._sources_checked: list[str] = []  # URLs/sources that were fetched
        self._current_phase: str = ""  # current pipeline phase

    async def add_conditions(self, conditions: list[AtomicCondition]) -> None:
        async with self._lock:
            self._conditions.extend(conditions)

    async def log_tool_call(self, tool_name: str, query: str, duration: float) -> None:
        """Record a tool invocation for heartbeat context."""
        async with self._lock:
            self._recent_tool_calls.append({
                "tool": tool_name, "query": query[:120],
                "duration": duration, "ts": time.monotonic(),
            })
            # Keep only last 20 entries
            if len(self._recent_tool_calls) > 20:
                self._recent_tool_calls = self._recent_tool_calls[-20:]

    async def set_active_question(self, question: str) -> None:
        """Record a question currently under investigation."""
        async with self._lock:
            if question not in self._active_questions:
                self._active_questions.append(question)
                if len(self._active_questions) > 10:
                    self._active_questions = self._active_questions[-10:]

    async def clear_active_question(self, question: str) -> None:
        """Remove a question that's done being investigated."""
        async with self._lock:
            self._active_questions = [
                q for q in self._active_questions if q != question
            ]

    async def log_source(self, url: str) -> None:
        """Record a source URL that was checked."""
        async with self._lock:
            if url not in self._sources_checked:
                self._sources_checked.append(url)
                if len(self._sources_checked) > 30:
                    self._sources_checked = self._sources_checked[-30:]

    async def set_phase(self, phase: str) -> None:
        """Update current pipeline phase."""
        async with self._lock:
            self._current_phase = phase

    async def get_activity_context(self) -> dict:
        """Return a snapshot of current activity for heartbeat generation."""
        async with self._lock:
            now = time.monotonic()
            recent = [t for t in self._recent_tool_calls if now - t["ts"] < 30]
            return {
                "recent_tools": recent[-5:],
                "active_questions": list(self._active_questions),
                "sources_count": len(self._sources_checked),
                "recent_sources": self._sources_checked[-5:],
                "phase": self._current_phase,
                "total_tool_calls": len(self._recent_tool_calls),
            }

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
- The finding MUST directly help answer the user's query: "{user_query}"
- If the finding is tangential or doesn't help answer the query, output SKIP
- If nothing is genuinely new vs the "Already shared" list, reply with exactly: SKIP

New findings:
{findings}

Already shared (do NOT repeat or rephrase — if a finding covers the same topic as any item below, output SKIP):
{already_shared}"""


_TOOL_DISPLAY_NAMES = {
    "searxng_search": "SearXNG",
    "news_search": "news engines (Google News, Bing News)",
    "fetch_webpage": "web page",
    "arxiv_search": "arXiv",
    "wayback_fetch": "Wayback Machine",
    "wikidata_query": "Wikidata",
    "knowledge_graph_search": "Neo4j knowledge graph",
    "knowledge_discover": "knowledge graph discovery",
    "chan_4plebs_search": "4plebs /pol/ archive",
    "chan_b4k_search": "b4k /biz/ archive",
    "chan_warosu_search": "warosu archive",
    "twitter_search": "Twitter/X",
    "python_exec": "Python sandbox",
    "web_search": "commercial SERP APIs",
}


def _build_context_aware_status(activity: dict) -> str:
    """Build a specific heartbeat status from actual activity context."""
    phase = activity.get("phase", "")
    recent_tools = activity.get("recent_tools", [])
    active_qs = activity.get("active_questions", [])
    sources_count = activity.get("sources_count", 0)
    recent_sources = activity.get("recent_sources", [])

    # Phase-specific messages
    if phase == "verify":
        return f"Cross-checking {sources_count} sources for contradictions via Veritas fact-check swarm"
    if phase == "entities":
        return f"Extracting entities and relationships into Neo4j knowledge graph from {sources_count} sources"
    if phase == "reflect":
        return "Running AoT reflection — checking for gaps, overlaps, and non-atomic claims"
    if phase == "synthesize":
        return "Drafting synthesis from verified conditions"

    # Tool-based messages
    if recent_tools:
        last = recent_tools[-1]
        tool_name = _TOOL_DISPLAY_NAMES.get(last["tool"], last["tool"])
        query = last["query"]
        if last["tool"] in ("fetch_webpage",):
            # Show domain, not full URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(query).netloc or query
                return f"Reading {domain} ({last['duration']:.1f}s)"
            except Exception:
                pass
            return f"Fetching page content ({last['duration']:.1f}s)"
        if query:
            return f"Querying {tool_name}: \"{query}\" ({last['duration']:.1f}s)"
        return f"Calling {tool_name} ({last['duration']:.1f}s)"

    # Active question messages
    if active_qs:
        return f"Investigating: {active_qs[-1]}"

    # Source count fallback (still specific)
    if sources_count > 0:
        return f"Checked {sources_count} sources so far, searching for more"

    return "Initialising research tree"


async def _generate_heartbeat_message(
    collector: LiveFindingsCollector,
    req_id: str,
    _phrase_idx: list[int],
) -> str:
    """Generate a heartbeat message using Mistral Small.

    Falls back to a context-aware status message built from actual
    tool calls, active questions, and sources — never generic filler.
    """
    new_findings = await collector.get_new_findings()

    if not new_findings:
        # Nothing new — build a context-aware status from actual activity
        activity = await collector.get_activity_context()
        return _build_context_aware_status(activity)

    # Build a compact list of new facts for the LLM
    findings_text = "\n".join(
        f"- {c.fact[:200]}" for c in new_findings[:10]
    )

    shared_facts = await collector.get_shared_facts()
    already_text = "\n".join(
        f"- {f[:120]}" for f in shared_facts[:8]
    ) if shared_facts else "(none yet)"

    # Use replace instead of .format() to avoid KeyError if user_query contains { or }
    prompt = _HEARTBEAT_PROMPT.replace(
        "{findings}", findings_text
    ).replace(
        "{already_shared}", already_text
    ).replace(
        "{user_query}", collector.user_query[:200]
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
                activity = await collector.get_activity_context()
                return _build_context_aware_status(activity)
            if msg and len(msg) > 10:
                # Mark all findings as shared to prevent rephrasing
                for f in new_findings:
                    await collector.mark_shared(f.fact)
                return msg
    except Exception as e:
        log.debug(f"[{req_id}] Heartbeat LLM call failed: {e}")

    # Fallback — context-aware, not generic
    activity = await collector.get_activity_context()
    return _build_context_aware_status(activity)


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
                formatted = await _format_curated_event_llm(curated_msg, req_id)
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


_CURATED_EVENT_PROMPT = """You are a research progress formatter. Convert the raw research event below into a single concise, informative status message for the user.

Rules:
- One sentence, under 40 words
- Lead with the most specific, useful information (names, numbers, sources, topics)
- Professional and factual — like a Reuters wire ticker
- NO excitement, NO commentary, NO hedging
- Include depth/branch context if relevant
- If the event has no useful information for the user, output exactly: SKIP

Raw event:
{event_json}"""


async def _format_curated_event_llm(event: dict, req_id: str) -> str:
    """Format a tree reactor curated event via LLM prompt.

    Uses a fast model to produce a concise, naturally-worded status
    message instead of programmatic string truncation.
    Falls back to a simple template if the LLM call fails.
    """
    evt_type = event.get("type", "")
    if not evt_type:
        return ""

    # Summary events are already concise — no LLM needed
    if evt_type == "summary":
        nodes = event.get("nodes_explored", 0)
        conds = event.get("conditions_count", 0)
        return f"Research complete: {nodes} branches explored, {conds} findings collected"

    # Build the prompt with the full event data (no truncation)
    event_json = json.dumps(event, default=str, ensure_ascii=False)
    prompt = _CURATED_EVENT_PROMPT.replace("{event_json}", event_json)

    try:
        result = await call_llm(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Format this event."},
            ],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=60,
            temperature=0.2,
        )
        if "error" not in result:
            msg = result.get("content", "").strip()
            if msg and msg.upper().strip() != "SKIP" and len(msg) > 5:
                return msg
    except Exception as e:
        log.debug(f"[{req_id}] Curated event LLM format failed: {e}")

    # Fallback — simple templates without character truncation
    return _format_curated_event_fallback(event)


def _format_curated_event_fallback(event: dict) -> str:
    """Fallback template formatter when LLM is unavailable."""
    evt_type = event.get("type", "")

    if evt_type == "start":
        return f"Investigating: {event.get('question', '')}"

    elif evt_type == "finding":
        finding = event.get("finding", "")
        depth = event.get("depth", 0)
        count = event.get("conditions_count", 0)
        q = event.get("question", "")
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


