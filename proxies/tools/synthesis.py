"""
Live findings collector, heartbeat, draft-synthesis-revision loop,
LangGraph state & pipeline graph, and main research orchestrator.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncGenerator, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from research_metrics import (
    MetricsCollector,
    ResearchMetricsCallback,
    SubagentMetrics,
    save_metrics,
)
import research_report
import langfuse_config

from shared import make_sse_chunk

from .config import (
    MAX_PRIOR_CONDITIONS,
    PORTAL_PUBLIC_URL,
    SUBAGENT_MODEL,
    UPSTREAM_MODEL,
    VERITAS_MIN_CONDITIONS,
    VERITAS_VERIFY_ENABLED,
    _STREAM_DONE,
    _curated_queues,
    _live_collectors,
    _metrics_collectors,
    log,
    tracker,
)
from .models import AtomicCondition, SubagentResult
from .persistence import (
    _log_conditions_jsonl,
    _log_entities_jsonl,
    _retrieve_graph_neighbors,
    _retrieve_related,
    _store_conditions_neo4j,
    _store_entities_neo4j,
)
from .llm import call_llm, _request_configs
from .planning import (
    extract_entities_from_conditions,
    verify_conditions,
    verify_conditions_with_veritas,
)
from .search_tools import tool_searxng_search
from .subagent import reflect_on_conditions
from .tree_reactor import tree_research_reactor


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
7. Do NOT add unsolicited warnings, ethical disclaimers, safety preambles, moral commentary, or hedging language.
8. Do NOT use phrases like "it's important to note", "please consult a professional", "exercise caution", "it should be noted that", "while controversial".
9. Treat the user as an intelligent adult. Answer the question directly. Every sentence must add information.
10. Mark any sections where evidence is weak or missing with [NEEDS VERIFICATION].
11. Do NOT pad the answer with generic context the user already knows. Get to the specific findings.
12. Include speculative findings (marked [SPECULATIVE]) — present them clearly as hypotheses or inferences, not as established facts. Speculation is valuable when labeled honestly.
13. Do NOT discard speculative findings just because they lack direct evidence. Present them with appropriate caveats like "evidence suggests", "this may indicate", or "one plausible interpretation is".
14. Report findings NEUTRALLY. Do NOT make value judgements about the user's intentions, morality, or choices. If the user asks how to do X, report how to do X — do not lecture them about whether they should.
15. NEVER use fearmongering language ("you could die", "extremely dangerous", "this will destroy"). State risks factually with data (e.g. "hypoglycemia occurs in X% of cases per [source]") not emotionally.
16. When research sources mention forums, vendors, communities, Telegram channels, or other concrete resources — NAME THEM SPECIFICALLY. Do not say "some forums exist" — say which forums, which channels, which vendors, with URLs.
17. If the user's query mentions forums, communities, or channels — investigating those is a PRIMARY research objective, not a footnote.

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
7. FEARMONGERING: emotional language designed to scare rather than inform (e.g. "you could die", "extremely dangerous"). Flag these as type "fearmongering" — risks should be stated with data, not drama.
8. VALUE JUDGEMENTS: moralizing about the user's intentions or choices. Flag these as type "value_judgement".
9. VAGUE REFERENCES: saying "some forums" or "certain vendors" without naming them specifically. Flag these as type "vague_reference" — the user wants names, URLs, specifics.

Output ONLY a JSON object:
{
  "overall_quality": 0.7,
  "issues": [
    {"section": "section heading or quote", "type": "unsupported_claim|logical_gap|needs_verification|contradiction|missing_context|fearmongering|value_judgement|vague_reference", "description": "what's wrong", "search_query": "suggested search to fix this"}
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
- Do NOT add unsolicited warnings, ethical disclaimers, safety preambles, moral commentary, or hedging
- Do NOT use phrases like "it's important to note", "please consult a professional", "exercise caution"
- Every sentence must deliver information that helps answer the query. Strip filler.
- Maintain the original structure but improve weak sections
- If an issue couldn't be resolved, note it transparently
- Cite all sources with URLs
- KEEP speculative findings — present them clearly as hypotheses with caveats, not as facts. Do NOT remove them just because they lack direct evidence.
- Something being illegal, unusual, or controversial does NOT make it wrong or worth removing.
- If the draft contains fearmongering, emotional language, or value judgements — REMOVE them. Replace with neutral, data-backed risk statements.
- If the draft says "some forums" or "certain vendors" without specifics — that is a FAILURE. Either add the specifics from the findings or mark it [NEEDS VERIFICATION]."""


_RELEVANCE_GATE_PROMPT = """You are a strict relevance filter. The user asked:
"{user_query}"

Does the following text directly help answer that query? Answer ONLY "YES" or "NO".
- YES = contains specific facts, data, or analysis that answers the query
- NO = off-topic, generic filler, moral disclaimers, ethical warnings, or tangential

Text to evaluate:
"{text}"
"""


async def relevance_gate(text: str, user_query: str, req_id: str) -> bool:
    """Cheap LLM check: does this text help answer the user's query?

    Uses the small/fast model. Returns True if relevant, False if not.
    On error, defaults to True (let content through rather than block).
    """
    if not text or not user_query:
        return True

    # Use replace instead of .format() to avoid KeyError if text contains { or }
    prompt = _RELEVANCE_GATE_PROMPT.replace(
        "{user_query}", user_query[:300]
    ).replace(
        "{text}", text[:500]
    )

    try:
        result = await call_llm(
            [{"role": "user", "content": prompt}],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=5,
            temperature=0.0,
        )
        if "error" not in result:
            answer = result.get("content", "").strip().upper()
            return answer.startswith("YES")
    except Exception:
        pass

    return True  # fail-open


async def strip_moralizing(text: str, user_query: str, req_id: str) -> str:
    """Post-process LLM output to remove moralizing, disclaimers, and filler.

    Uses the small/fast model to strip content that doesn't help answer
    the user's query. Returns the cleaned text.
    """
    if not text or len(text) < 100:
        return text

    strip_prompt = (
        f"The user asked: \"{user_query[:300]}\"\n\n"
        f"Below is a research answer. Remove ALL of the following:\n"
        f"- Ethical disclaimers or safety warnings\n"
        f"- Phrases like \"it's important to note\", \"please consult a professional\", "
        f"\"exercise caution\", \"it should be noted\"\n"
        f"- Moral commentary or unsolicited advice\n"
        f"- Generic filler that doesn't add specific information\n"
        f"- Hedging that weakens otherwise supported claims\n"
        f"- Fearmongering language (\"you could die\", \"extremely dangerous\", "
        f"\"this will destroy your health\") — replace with neutral data-backed "
        f"risk statements where the underlying fact is real\n"
        f"- Value judgements about the user's choices or intentions\n"
        f"- \"Final Verdict\" or \"Should You\" sections that lecture rather than inform\n\n"
        f"Keep ALL specific facts, data, sources, URLs, vendor names, forum names, "
        f"Telegram channels, and analysis intact. "
        f"Output ONLY the cleaned text, nothing else.\n\n"
        f"Text:\n{text}"
    )

    try:
        result = await call_llm(
            [{"role": "user", "content": strip_prompt}],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=8192,
            temperature=0.1,
        )
        if "error" not in result:
            cleaned = result.get("content", "").strip()
            if cleaned and len(cleaned) > len(text) * 0.3:
                return cleaned
    except Exception:
        pass

    return text  # fail-open


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
        return await strip_moralizing(draft, user_query, req_id)

    final_text = final_result.get("content", draft)
    return await strip_moralizing(final_text, user_query, req_id)


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
        collector = LiveFindingsCollector(user_query=state["user_query"])
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
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("entities")
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
    """Phase 5: Citation verification.

    Verification is now primarily done at admission time (per-condition)
    via the ConditionStore, plus inline verification during the tree phase
    (cross-referencing concrete entities during research).

    This phase runs a lightweight self-evaluation pass on the already-admitted
    conditions to catch any remaining contradictions or confidence adjustments.

    Veritas Inquisitor (the 5-agent post-hoc swarm) is DEPRECATED by default:
    admission-time + inline verification replaced it.  Can be forced via
    VERITAS_FORCE_POST_HOC=true env var.
    """
    req_id = state["req_id"]
    user_query = state["user_query"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("verify")
    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.start_node("verify")
    all_conditions = list(state["all_conditions"])
    progress: list[str] = []
    pre_count = len(all_conditions)

    if all_conditions and len(all_conditions) >= 2:
        # Lightweight self-evaluation: cross-check for contradictions
        progress.append("\n**[Phase 5: Citation Cross-Check]**\n")
        progress.append(
            f"Cross-checking {len(all_conditions)} pre-admitted conditions "
            f"for contradictions (conditions already passed admission pipeline)...\n"
        )

        all_conditions = await verify_conditions(all_conditions, req_id)

        stage1_removed = pre_count - len(all_conditions)
        high_conf = sum(1 for c in all_conditions if c.confidence >= 0.7)
        low_conf = sum(1 for c in all_conditions if c.confidence < 0.4)
        speculative = sum(1 for c in all_conditions if c.verification_status == "speculative")
        summary = (f"Cross-check complete: {high_conf} high-confidence, "
                   f"{low_conf} low-confidence, {speculative} speculative.")
        if stage1_removed > 0:
            summary += f" {stage1_removed} fabricated removed."
        progress.append(summary + "\n")

    # Veritas Inquisitor — only runs if explicitly forced.
    # Admission-time + inline verification during the tree phase replaces this.
    force_veritas = os.getenv("VERITAS_FORCE_POST_HOC", "").lower() in ("1", "true", "yes")
    veritas_report: dict = {}
    if force_veritas and VERITAS_VERIFY_ENABLED and len(all_conditions) >= VERITAS_MIN_CONDITIONS:
        progress.append("\n**[Phase 5b: Veritas Fact-Check]** (forced via VERITAS_FORCE_POST_HOC)\n")
        progress.append(
            f"Running Veritas Inquisitor on {len(all_conditions)} conditions "
            f"(5-agent swarm with web search)...\n"
        )

        pre_veritas_count = len(all_conditions)
        all_conditions, veritas_report = await verify_conditions_with_veritas(
            all_conditions, user_query, req_id,
        )
        progress.append(
            f"{len(all_conditions)} conditions retained out of {pre_veritas_count}.\n"
        )
    elif VERITAS_VERIFY_ENABLED and not force_veritas:
        progress.append(
            "\n**[Phase 5b: Inline Verification]** "
            "Entity cross-referencing was performed during the tree research phase. "
            "Skipping post-hoc Veritas pass (set VERITAS_FORCE_POST_HOC=true to force).\n"
        )

    mc = _metrics_collectors.get(req_id)
    if mc:
        mc.end_node("verify")

    return {"all_conditions": all_conditions, "progress_log": progress, "phase": "reflect"}


async def pdr_node_reflect(state: PersistentResearchState) -> dict:
    """Phase 6: AoT Reflection."""
    req_id = state["req_id"]
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("reflect")
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
    collector = _live_collectors.get(req_id)
    if collector:
        await collector.set_phase("synthesize")
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

    # Relevance gate: check if the final answer actually addresses the query
    is_relevant = await relevance_gate(final_answer, state["user_query"], req_id)
    if not is_relevant:
        log.warning(f"[{req_id}] Final answer failed relevance gate — re-running synthesis")
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

        # Generate Markdown report (user-readable)
        try:
            md_report = research_report.generate_report(
                metrics=metrics_dict,
                conditions=condition_dicts,
                final_answer=final_answer,
                progress_log=list(state.get("progress_log", [])),
            )
            research_report.save_report(md_report, req_id)

            # Save metrics JSON alongside report
            metrics_json = json.dumps(metrics_dict, indent=2, default=str)
            research_report.save_metrics_json(metrics_json, req_id)

            # Build portal URLs for the report and metrics
            base = PORTAL_PUBLIC_URL
            if not base:
                log.warning(
                    "[%s] PORTAL_PUBLIC_URL not set — report links will be relative",
                    req_id,
                )
                base = ""
            report_url = f"{base}/research/report/{req_id}"
            metrics_url = f"{base}/research/metrics/{req_id}"
            log.info(f"[{req_id}] Report available at: {report_url}")
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

        # Pipeline done — emit closing think tag, links header, and final answer
        await output_queue.put(chunk_fn("\n</think>\n\n"))

        # Emit report + trace links as the first visible lines
        report_url = final_state.get("report_url", "")
        metrics_url = final_state.get("metrics_url", "")
        langfuse_url = initial_state.get("_langfuse_trace_url", "")
        link_lines = []
        if report_url:
            link_lines.append(f"**[Full Report]({report_url})**")
        if metrics_url:
            link_lines.append(f"[Metrics JSON]({metrics_url})")
        if langfuse_url:
            link_lines.append(f"[Langfuse Trace]({langfuse_url})")
        if link_lines:
            await output_queue.put(chunk_fn(" | ".join(link_lines) + "\n\n"))

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

    # --- Langfuse tracing: generate trace URL early so it goes into initial_state ---
    langfuse_trace_id = langfuse_config.create_trace_id(req_id)
    langfuse_trace_url = langfuse_config.get_trace_url(langfuse_trace_id)
    langfuse_handler = langfuse_config.create_callback_handler(
        trace_id=langfuse_trace_id,
        session_id=req_id,
        tags=["persistent-research"],
    )

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
        "_langfuse_trace_url": langfuse_trace_url or "",
    }

    # Create the shared output queue, live findings collector, and curated queue
    output_queue: asyncio.Queue = asyncio.Queue()
    collector = LiveFindingsCollector(user_query=user_query)
    _live_collectors[req_id] = collector
    curated_queue: asyncio.Queue = asyncio.Queue()
    _curated_queues[req_id] = curated_queue

    # Create metrics collector for this session
    metrics_collector = MetricsCollector(session_id=req_id, query=user_query)
    _metrics_collectors[req_id] = metrics_collector
    metrics_callback = ResearchMetricsCallback(metrics_collector)

    yield chunk("<think>\n")

    callbacks = [metrics_callback]
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    config = {
        "configurable": {"thread_id": req_id},
        "callbacks": callbacks,
    }

    # Register the config so call_llm and execute_tool can look it up
    # by req_id and fire callbacks on every LLM/tool invocation.
    _request_configs[req_id] = config

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

        # Clean up the live collector, curated queue, metrics collector, and config
        _live_collectors.pop(req_id, None)
        _curated_queues.pop(req_id, None)
        _metrics_collectors.pop(req_id, None)
        _request_configs.pop(req_id, None)
        langfuse_config.flush()
        tracker.finish(req_id)

