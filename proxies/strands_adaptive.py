"""Adaptive loop prevention for the Strands multi-agent researcher.

Uses Strands SDK primitives (Plugin + @hook) to detect when the planner
sends repeated/similar queries to the researcher and:

  1. **Escalate temperature** on intermediate repeats so the researcher
     explores different search strategies (transparent to the planner).
  2. **Cancel the tool call** with cached results after MAX_SIMILAR_CALLS
     near-identical queries, instructing the planner to synthesize.

The plugin is registered on the **planner** agent.  It intercepts
``BeforeToolCallEvent`` (before the researcher runs) and
``AfterToolCallEvent`` (to cache results and restore temperature).

Designed to be imported by the strands-agent (in MiroThinker) via
PYTHONPATH — same pattern as strands_observability.py.

Configuration (env vars):
    BASE_TEMPERATURE      – starting researcher temp (default 0.7)
    TEMP_ESCALATION_STEP  – bump per repeat (default 0.2)
    MAX_TEMPERATURE       – ceiling (default 1.5)
    SIMILARITY_THRESHOLD  – Jaccard threshold for "similar" (default 0.6)
    MAX_SIMILAR_CALLS     – after this many, return cache (default 3)
"""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any

from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent, BeforeInvocationEvent
from strands.plugins import Plugin, hook

log = logging.getLogger("strands_adaptive")

# ── Configuration ─────────────────────────────────────────────────────

BASE_TEMPERATURE = float(os.environ.get("BASE_TEMPERATURE", "0.7"))
TEMP_ESCALATION_STEP = float(os.environ.get("TEMP_ESCALATION_STEP", "0.2"))
MAX_TEMPERATURE = float(os.environ.get("MAX_TEMPERATURE", "1.5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.6"))
MAX_SIMILAR_CALLS = int(os.environ.get("MAX_SIMILAR_CALLS", "3"))

# ── Text helpers ──────────────────────────────────────────────────────


def normalize_query(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ── Plugin ────────────────────────────────────────────────────────────


class AdaptiveLoopPlugin(Plugin):
    """Strands Plugin that prevents multi-agent loops via temperature escalation.

    Register on the **planner** agent::

        plugin = AdaptiveLoopPlugin(researcher_model)
        planner = Agent(..., plugins=[plugin])

    The plugin intercepts calls to the ``researcher`` tool and:
      - On repeat 1–2: bumps researcher temperature + appends diversity hint
      - On repeat 3+: cancels the call with cached results
    """

    name = "adaptive-loop-prevention"

    def __init__(
        self,
        researcher_model: Any,
        researcher_tool_name: str = "researcher",
        *,
        base_temperature: float = BASE_TEMPERATURE,
        temp_step: float = TEMP_ESCALATION_STEP,
        max_temperature: float = MAX_TEMPERATURE,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_similar_calls: int = MAX_SIMILAR_CALLS,
    ) -> None:
        self._researcher_model = researcher_model
        self._tool_name = researcher_tool_name
        self._base_temp = base_temperature
        self._temp_step = temp_step
        self._max_temp = max_temperature
        self._sim_threshold = similarity_threshold
        self._max_similar = max_similar_calls

        # Per-request state (cleared on each new invocation)
        self._query_history: list[tuple[str, str]] = []  # (normalized, result)
        self._old_params: dict | None = None  # for temperature restore

    # ── Similarity helpers ────────────────────────────────────────

    def _count_similar(self, normalized: str) -> int:
        return sum(
            1
            for pq, _ in self._query_history
            if jaccard_similarity(normalized, pq) >= self._sim_threshold
        )

    def _best_cached(self, normalized: str) -> str:
        best_sim, best_result = 0.0, "(no prior results)"
        for pq, result in self._query_history:
            sim = jaccard_similarity(normalized, pq)
            if sim > best_sim:
                best_sim = sim
                best_result = result
        return best_result if best_sim >= self._sim_threshold else "(no prior results)"

    # ── Temperature control ───────────────────────────────────────

    def _escalate_temperature(self, similar_count: int) -> None:
        new_temp = min(
            self._base_temp + (similar_count * self._temp_step),
            self._max_temp,
        )
        self._old_params = copy.deepcopy(
            dict(self._researcher_model.config.get("params", {}))
        )
        new_params = copy.deepcopy(self._old_params)
        new_params["temperature"] = new_temp
        self._researcher_model.config["params"] = new_params
        log.info("Researcher temperature escalated to %.2f", new_temp)

    def _restore_temperature(self) -> None:
        if self._old_params is not None:
            self._researcher_model.config["params"] = self._old_params
            self._old_params = None

    # ── Hook: reset state on new request ──────────────────────────

    @hook
    def on_invocation_start(self, event: BeforeInvocationEvent) -> None:
        """Clear query history at the start of each planner invocation."""
        self._query_history.clear()
        self._old_params = None
        log.debug("Adaptive loop plugin reset for new invocation")

    # ── Hook: intercept researcher calls ──────────────────────────

    @hook
    def before_researcher_call(self, event: BeforeToolCallEvent) -> None:
        """Detect repeated queries; escalate temperature or cancel."""
        if event.tool_use.get("name") != self._tool_name:
            return

        # Extract query from tool input
        tool_input = event.tool_use.get("input", {})
        if isinstance(tool_input, dict):
            query = tool_input.get("input", "")
        elif isinstance(tool_input, str):
            query = tool_input
        else:
            query = str(tool_input)

        normalized = normalize_query(query)
        similar_count = self._count_similar(normalized)

        # Case 1: too many repeats → cancel with cached result
        if similar_count >= self._max_similar:
            cached = self._best_cached(normalized)
            log.warning(
                "Researcher called %d times with similar query — cancelling. "
                "Query: %.100s",
                similar_count + 1,
                query,
            )
            event.cancel_tool = (
                f"[DUPLICATE QUERY — returning cached result from "
                f"attempt {similar_count}]\n\n{cached}\n\n---\n"
                f"You have already searched for this same topic "
                f"{similar_count} times. STOP delegating and "
                f"SYNTHESIZE your answer from the data above NOW."
            )
            return

        # Case 2: similar query → escalate temperature + diversity hint
        if similar_count > 0:
            self._escalate_temperature(similar_count)
            log.info(
                "Similar query #%d detected (sim>%.2f). Query: %.100s",
                similar_count + 1,
                self._sim_threshold,
                query,
            )

            diversity_hint = (
                "\n\n[SYSTEM NOTE: Your previous search on this topic returned "
                "limited or duplicate results. Try COMPLETELY DIFFERENT search "
                "terms, alternative tools (e.g. Exa instead of Brave, or "
                "Firecrawl for deep scraping), different languages, or "
                "unconventional angles. Do NOT repeat the same queries.]"
            )
            if isinstance(tool_input, dict):
                event.tool_use["input"] = {"input": query + diversity_hint}
            else:
                event.tool_use["input"] = query + diversity_hint

        # Case 3: novel query → proceed normally (no changes)

    # ── Hook: cache results + restore temperature ─────────────────

    @hook
    def after_researcher_call(self, event: AfterToolCallEvent) -> None:
        """Cache researcher results and restore base temperature."""
        if event.tool_use.get("name") != self._tool_name:
            return

        # Restore temperature if it was escalated
        self._restore_temperature()

        # Extract result text for caching
        result_text = ""
        if event.cancel_message:
            # Tool was cancelled — don't cache the cancel message
            return

        result = event.result
        if result:
            for block in result.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    result_text += block["text"]

        # Extract original query
        tool_input = event.tool_use.get("input", {})
        if isinstance(tool_input, dict):
            query = tool_input.get("input", "")
        elif isinstance(tool_input, str):
            query = tool_input
        else:
            query = str(tool_input)

        # Strip any diversity hint we appended
        hint_marker = "\n\n[SYSTEM NOTE: Your previous search"
        if hint_marker in query:
            query = query[: query.index(hint_marker)]

        normalized = normalize_query(query)
        self._query_history.append((normalized, result_text[:2000]))
        log.info(
            "Researcher call #%d completed. %d chars cached. Query: %.80s",
            len(self._query_history),
            len(result_text),
            query,
        )

    # ── Public API ────────────────────────────────────────────────

    def reset(self) -> None:
        """Manually clear state (called by reset_budget in agent.py)."""
        self._query_history.clear()
        self._restore_temperature()
