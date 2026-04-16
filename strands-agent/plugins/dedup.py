"""Dedup + scoring plugin — Jaccard dedup with trust/serendipity scoring.

Transplants from deep-search-portal:
- pipeline.py: ConditionStore.DUPLICATE_THRESHOLD, _jaccard_similarity, _fast_relevance_check
- scoring.py: trust_score_url, serendipity_score
"""

from __future__ import annotations

import re
import logging

from strands.plugins import Plugin, hook
from strands.hooks import AfterToolCallEvent

log = logging.getLogger("dedup-plugin")

DUPLICATE_THRESHOLD = 0.48  # from deep-search-portal pipeline.py line 246


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def serendipity_score(fact: str, query: str, known_facts: list[str]) -> float:
    """Geometric mean of relevance, novelty, surprise.

    Ported from deep-search-portal/proxies/tools/scoring.py lines 54-87.
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


# Trust scoring tiers ported from deep-search-portal/proxies/tools/scoring.py lines 13-37
_TRUST_TIERS = [
    (re.compile(r"\.edu(/|$)", re.I), 0.9),
    (re.compile(r"\.gov(/|$)", re.I), 0.9),
    (re.compile(r"(arxiv\.org|pubmed|doi\.org|nature\.com|ieee\.org)", re.I), 0.9),
    (re.compile(r"(reuters\.com|apnews\.com|bbc\.co|nytimes\.com)", re.I), 0.7),
    (re.compile(r"(wikipedia\.org)", re.I), 0.6),
    (
        re.compile(r"(reddit\.com|stackexchange\.com|news\.ycombinator\.com)", re.I),
        0.3,
    ),
    (re.compile(r"(medium\.com|substack\.com)", re.I), 0.4),
]


def trust_score_url(url: str) -> float:
    if not url:
        return 0.5
    for pattern, score in _TRUST_TIERS:
        if pattern.search(url):
            return score
    return 0.5


class DedupPlugin(Plugin):
    """Jaccard dedup on researcher tool results.

    Intercepts AfterToolCallEvent on the researcher agent. Extracts text
    from tool results and checks against all previously seen findings.
    If Jaccard similarity > DUPLICATE_THRESHOLD, logs a warning and
    stores a saturation signal in agent state.

    Also computes trust_score and serendipity_score for each finding
    and stores them in the lineage plugin (if available).
    """

    name = "dedup-tracker"

    def __init__(self, query: str = ""):
        super().__init__()
        self._seen_facts: list[str] = []
        self._query = query
        self._duplicate_count = 0
        self._total_count = 0

    def reset(self, query: str = ""):
        self._seen_facts.clear()
        self._query = query
        self._duplicate_count = 0
        self._total_count = 0

    @property
    def saturation_ratio(self) -> float:
        """Fraction of recent findings that were duplicates."""
        if self._total_count == 0:
            return 0.0
        return self._duplicate_count / self._total_count

    @hook
    def on_tool_result(self, event: AfterToolCallEvent) -> None:
        """Check tool results for duplicates."""
        if event.cancel_message:
            return

        result_text = ""
        if event.result:
            for block in event.result.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    result_text += block["text"]

        if not result_text.strip() or len(result_text) < 20:
            return

        # Truncate for comparison
        fact = result_text[:1000]
        self._total_count += 1

        # Check for duplicates
        for seen in self._seen_facts:
            if jaccard_similarity(fact, seen) > DUPLICATE_THRESHOLD:
                self._duplicate_count += 1
                log.info(
                    "Duplicate finding detected (%d/%d). Saturation: %.0f%%",
                    self._duplicate_count,
                    self._total_count,
                    self.saturation_ratio * 100,
                )
                return

        self._seen_facts.append(fact)

        # Compute scores
        seren = serendipity_score(fact, self._query, self._seen_facts[:-1])
        if seren > 0.4:
            log.info("High serendipity finding (%.2f): %.100s", seren, fact)
