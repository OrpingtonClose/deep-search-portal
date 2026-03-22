"""
Condition admission pipeline: QueryComprehension, ConditionStore,
condition validation, deduplication, and saturation detection.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .config import (
    SUBAGENT_MODEL,
    log,
)
from .models import AtomicCondition, CrossRef
from .scoring import serendipity_score, trust_score_url
from .llm import call_llm

# Lazy import to avoid circular dependency (synthesis → pipeline → synthesis)
def _get_relevance_gate():
    from .synthesis import relevance_gate
    return relevance_gate


# ============================================================================
# Condition Admission Pipeline
# ============================================================================

# Patterns that indicate a fake/placeholder source URL (internal tool labels)
_FAKE_URL_PATTERNS = re.compile(
    r'(searxng_search_results|reddit_search_results|forum_search_results|'
    r'twitter_search_results|chan_\w+_results|news_search_results|'
    r'scholar_search_results|substack_search_results|tool_results|'
    r'search_results|No source|no source|N/A|n/a|none|unknown)',
    re.IGNORECASE,
)


@dataclass
class QueryComprehension:
    """Deep semantic understanding of a research query.

    Produced once at reactor start by an LLM analysis pass.  This is NOT
    just keyword extraction — it maps the full knowledge territory:
    what entities exist, what domains are relevant, what implicit questions
    the user is really asking, and crucially what *adjacent* knowledge
    territories might contain the deep/rare information the user needs.

    The comprehension map is used to:
      - Loosen the relevance gate (a condition about enforcement actions
        passes even if the query only names a substance)
      - Guide condition spawning toward unexplored deep territories
      - Help the question router pick tools for information needs
    """
    # Core entities mentioned or implied by the query
    entities: list[str] = field(default_factory=list)
    # Knowledge domains the query touches (e.g., "pharmacology", "law enforcement")
    domains: list[str] = field(default_factory=list)
    # Implicit questions the user is really asking (not just the literal query)
    implicit_questions: list[str] = field(default_factory=list)
    # Adjacent territories — topics NOT in the query but likely to contain
    # the deep/rare knowledge the user needs
    adjacent_territories: list[str] = field(default_factory=list)
    # Keywords and phrases that indicate relevance (broader than the query)
    relevance_keywords: list[str] = field(default_factory=list)
    # What kind of deep knowledge would actually be valuable here
    deep_knowledge_targets: list[str] = field(default_factory=list)
    # One-paragraph summary of what this query is *really* about
    semantic_summary: str = ""
    # Intent classification: "transactional" (user wants to DO something),
    # "informational" (user wants to UNDERSTAND something),
    # "exploratory" (user wants to EXPLORE a topic broadly)
    intent_type: str = "informational"
    # One-sentence summary of what the user ultimately needs to accomplish
    core_need: str = ""


_QUERY_COMPREHENSION_PROMPT = """You are a deep research analyst. Your job is to DEEPLY understand what a research query is really about — not just the surface words, but the full knowledge territory.

The goal is to understand the query well enough to guide researchers toward RARE, DEEP, EMBEDDED knowledge — the kind found in community discussions, practitioner experiences, court documents, underground forums, academic papers, and obscure archives. NOT surface-level Wikipedia summaries.

Research query: {query}

Analyze this query and output ONLY valid JSON:
{{
  "entities": ["every entity, person, substance, organization, concept mentioned or implied"],
  "domains": ["every knowledge domain this touches — be expansive, include adjacent fields"],
  "implicit_questions": ["what is the user REALLY trying to understand? list 5-10 implicit questions they haven't asked but need answered"],
  "adjacent_territories": ["topics NOT in the query but where the DEEP knowledge lives — practitioner communities, enforcement databases, underground discussions, academic niches, historical archives"],
  "relevance_keywords": ["broad set of 20-30 keywords/phrases that indicate a piece of information is relevant to this query — include slang, technical terms, community jargon, legal terms"],
  "deep_knowledge_targets": ["specific types of deep knowledge that would be valuable — e.g., 'court case outcomes', 'practitioner dosing discussions', 'supply chain vendor reviews', 'regulatory enforcement actions'"],
  "semantic_summary": "one paragraph explaining what this query is REALLY about at the deepest level — what knowledge gap is the user trying to fill?",
  "intent_type": "one of: transactional | informational | exploratory. transactional = user wants to DO/BUY/GET something (e.g., 'where can I buy X', 'how to set up Y'). informational = user wants to UNDERSTAND something (e.g., 'what causes X', 'how does Y work'). exploratory = user wants to broadly EXPLORE a topic (e.g., 'tell me about X', 'what's the state of Y').",
  "core_need": "one sentence describing what the user ultimately needs to accomplish or understand — this is the ACTIONABLE goal behind the query"
}}"""


async def comprehend_query(user_query: str, req_id: str) -> QueryComprehension:
    """Produce a deep semantic understanding of the research query.

    This runs ONCE at reactor start.  The resulting QueryComprehension
    is shared with the ConditionStore, relevance gate, question router,
    and spawn logic so they all operate from the same understanding of
    what the query is really about.
    """
    prompt = _QUERY_COMPREHENSION_PROMPT.replace("{query}", user_query[:2000])
    try:
        result = await call_llm(
            [{"role": "user", "content": prompt}],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=2048,
            temperature=0.3,
        )
        if "error" not in result:
            content = result.get("content", "").strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            data = json.loads(content)
            # Validate intent_type
            raw_intent = data.get("intent_type", "informational").strip().lower()
            if raw_intent not in ("transactional", "informational", "exploratory"):
                raw_intent = "informational"
            return QueryComprehension(
                entities=data.get("entities", [])[:30],
                domains=data.get("domains", [])[:20],
                implicit_questions=data.get("implicit_questions", [])[:10],
                adjacent_territories=data.get("adjacent_territories", [])[:15],
                relevance_keywords=data.get("relevance_keywords", [])[:40],
                deep_knowledge_targets=data.get("deep_knowledge_targets", [])[:15],
                semantic_summary=data.get("semantic_summary", ""),
                intent_type=raw_intent,
                core_need=data.get("core_need", "")[:500],
            )
    except Exception as e:
        log.warning(f"[{req_id}] Query comprehension failed (non-fatal): {e}")

    # Fallback: minimal comprehension from the query itself
    words = [w for w in re.split(r'\W+', user_query.lower()) if len(w) > 3]
    return QueryComprehension(
        entities=words[:10],
        domains=[],
        implicit_questions=[],
        adjacent_territories=[],
        relevance_keywords=words[:20],
        deep_knowledge_targets=[],
        semantic_summary=user_query,
    )


@dataclass
class AdmissionResult:
    """Result of attempting to admit a condition into the global store."""
    admitted: bool
    reason: str  # "admitted", "duplicate", "irrelevant", "fabricated_url"
    condition: Optional[AtomicCondition] = None
    similar_to: Optional[str] = None  # fact text of the most similar existing condition
    saturation_signal: str = ""  # guidance for the subagent on what to explore instead
    serendipity_score_val: float = 0.0


def _validate_source_url(url: str) -> str:
    """Validate and clean a source URL. Returns cleaned URL or empty string."""
    if not url:
        return ""
    url = url.strip()
    # Strip known placeholder patterns
    if _FAKE_URL_PATTERNS.search(url):
        return ""
    # Must look like a real URL
    if not url.startswith(("http://", "https://")):
        return ""
    # Basic URL structure check
    try:
        parsed = urlparse(url)
        if not parsed.netloc or "." not in parsed.netloc:
            return ""
    except Exception:
        return ""
    return url


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two text strings using word sets."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / max(union, 1)


def _compute_topic_buckets(conditions: list[AtomicCondition]) -> dict[str, int]:
    """Group conditions into topic buckets by keyword clustering.

    Returns a dict mapping topic_label -> condition_count.
    Topics are derived from the most frequent 2-word phrases.
    """
    from collections import Counter
    bigrams: Counter = Counter()
    for c in conditions:
        words = c.fact.lower().split()
        for i in range(len(words) - 1):
            # Skip very short words (articles, prepositions)
            if len(words[i]) > 2 and len(words[i + 1]) > 2:
                bigrams[f"{words[i]} {words[i + 1]}"] += 1

    # Top bigrams become topic labels
    topics: dict[str, int] = {}
    for bigram, count in bigrams.most_common(20):
        if count >= 2:
            topics[bigram] = count
    return topics


class ConditionStore:
    """Global condition store with admission pipeline.

    Every condition must pass through admit() before entering the store.
    The admission pipeline performs:
      1. Source URL validation (strip fakes)
      2. Relevance check (comprehension-aware: keyword pre-check + LLM gate)
      3. Novelty check (Jaccard dedup against all existing conditions)
      4. Serendipity scoring (query adherence + novelty + surprise)
      5. Verification-at-birth (trust scoring, cross-reference)

    Duplicate conditions are rejected with a saturation signal telling
    the subagent what to explore instead.

    Query comprehension:
      The store holds a QueryComprehension that maps the full knowledge
      territory.  Understanding conditions (entities, domains, implicit
      questions) go through the same admission pipeline.  The comprehension
      evolves as research progresses — new understanding is admitted just
      like any other condition.
    """

    # Jaccard threshold above which a new condition is considered a duplicate
    DUPLICATE_THRESHOLD = 0.55
    # Number of conditions on a topic before it's considered saturated
    SATURATION_THRESHOLD = 10

    # Number of actionable conditions on an entity before that entity is saturated
    ENTITY_SATURATION_THRESHOLD = 6

    def __init__(self, user_query: str, req_id: str, comprehension: Optional["QueryComprehension"] = None):
        self._user_query = user_query
        self._req_id = req_id
        self._conditions: list[AtomicCondition] = []
        self._fact_word_sets: list[set[str]] = []  # parallel to _conditions for fast Jaccard
        self._lock = asyncio.Lock()
        self._admitted_count = 0
        self._rejected_duplicate = 0
        self._rejected_irrelevant = 0
        self._rejected_fabricated = 0
        # Deep query comprehension — evolves as understanding conditions are admitted
        self.comprehension: Optional["QueryComprehension"] = comprehension
        # Build the relevance keyword set from comprehension for fast pre-check
        self._relevance_words: set[str] = set()
        if comprehension:
            self._rebuild_relevance_words()
        # Entity-level saturation tracking: entity_name -> count of conditions mentioning it
        self._entity_condition_counts: dict[str, int] = {}
        # Set of entity names that have reached saturation threshold
        self._saturated_entities: set[str] = set()

    def _rebuild_relevance_words(self) -> None:
        """Rebuild the fast relevance keyword set from comprehension."""
        words: set[str] = set()
        if self.comprehension:
            for kw in self.comprehension.relevance_keywords:
                words.update(w.lower() for w in re.split(r'\W+', kw) if len(w) > 2)
            for ent in self.comprehension.entities:
                words.update(w.lower() for w in re.split(r'\W+', ent) if len(w) > 2)
            for dom in self.comprehension.domains:
                words.update(w.lower() for w in re.split(r'\W+', dom) if len(w) > 2)
            for terr in self.comprehension.adjacent_territories:
                words.update(w.lower() for w in re.split(r'\W+', terr) if len(w) > 2)
        # Always include words from the query itself
        words.update(w.lower() for w in re.split(r'\W+', self._user_query) if len(w) > 2)
        self._relevance_words = words

    def _fast_relevance_check(self, fact: str) -> bool:
        """Fast keyword-based relevance pre-check using comprehension map.

        If the fact shares ANY keywords with the comprehension's relevance
        set, it passes.  This is deliberately LOOSE — the point is to avoid
        rejecting conditions that are in adjacent territories identified by
        the comprehension.  Only truly unrelated facts get blocked here.

        Returns True if the fact is likely relevant (should proceed to LLM gate
        or be admitted directly), False if clearly irrelevant.
        """
        if not self._relevance_words:
            return True  # no comprehension = let everything through
        fact_words = set(w.lower() for w in re.split(r'\W+', fact) if len(w) > 2)
        overlap = fact_words & self._relevance_words
        # If ANY keyword matches, it's potentially relevant
        return len(overlap) >= 1

    async def admit_understanding(self, comprehension: "QueryComprehension") -> list[AdmissionResult]:
        """Admit understanding conditions derived from query comprehension.

        Each piece of understanding (entity, domain, implicit question,
        adjacent territory) is treated as a condition and admitted through
        the same pipeline.  This lets the system's understanding of the
        query evolve as research progresses.
        """
        self.comprehension = comprehension
        self._rebuild_relevance_words()

        understanding_conditions: list[AtomicCondition] = []

        # Entities as conditions
        for ent in comprehension.entities:
            understanding_conditions.append(AtomicCondition(
                fact=f"[ENTITY] {ent}",
                confidence=0.9,
                angle="query_comprehension",
                source_url="",
                verification_status="understanding",
            ))

        # Domains as conditions
        for dom in comprehension.domains:
            understanding_conditions.append(AtomicCondition(
                fact=f"[DOMAIN] {dom} — relevant knowledge domain for this query",
                confidence=0.8,
                angle="query_comprehension",
                source_url="",
                verification_status="understanding",
            ))

        # Implicit questions as conditions
        for q in comprehension.implicit_questions:
            understanding_conditions.append(AtomicCondition(
                fact=f"[IMPLICIT_QUESTION] {q}",
                confidence=0.7,
                angle="query_comprehension",
                source_url="",
                verification_status="understanding",
            ))

        # Adjacent territories as conditions
        for terr in comprehension.adjacent_territories:
            understanding_conditions.append(AtomicCondition(
                fact=f"[ADJACENT_TERRITORY] {terr} — deep knowledge likely found here",
                confidence=0.6,
                angle="query_comprehension",
                source_url="",
                verification_status="understanding",
            ))

        # Deep knowledge targets as conditions
        for target in comprehension.deep_knowledge_targets:
            understanding_conditions.append(AtomicCondition(
                fact=f"[DEEP_TARGET] {target}",
                confidence=0.7,
                angle="query_comprehension",
                source_url="",
                verification_status="understanding",
            ))

        # Admit them through the pipeline (skip LLM relevance — they ARE the relevance)
        return await self.admit_batch(understanding_conditions, skip_relevance_llm=True)

    @property
    def conditions(self) -> list[AtomicCondition]:
        return list(self._conditions)

    @property
    def stats(self) -> dict:
        return {
            "admitted": self._admitted_count,
            "rejected_duplicate": self._rejected_duplicate,
            "rejected_irrelevant": self._rejected_irrelevant,
            "rejected_fabricated": self._rejected_fabricated,
            "total_stored": len(self._conditions),
            "saturated_entities": sorted(self._saturated_entities),
            "entity_coverage": dict(self._entity_condition_counts),
        }

    def _find_most_similar(self, fact_words: set[str]) -> tuple[float, str]:
        """Find the most similar existing condition by Jaccard similarity."""
        best_sim = 0.0
        best_fact = ""
        for i, existing_words in enumerate(self._fact_word_sets):
            if not existing_words:
                continue
            intersection = len(fact_words & existing_words)
            union = len(fact_words | existing_words)
            sim = intersection / max(union, 1)
            if sim > best_sim:
                best_sim = sim
                best_fact = self._conditions[i].fact
        return best_sim, best_fact

    def _update_entity_counts(self, condition: AtomicCondition) -> None:
        """Track which entities this condition mentions and update counts.

        Called inside _lock after admission.  Uses the condition's own
        entities list plus simple keyword matching against comprehension
        entities.
        """
        mentioned: set[str] = set()

        # 1. Use condition's own entity list if populated
        for ent in condition.entities:
            mentioned.add(ent.lower().strip())

        # 2. Check comprehension entities against the fact text
        if self.comprehension:
            fact_lower = condition.fact.lower()
            for ent in self.comprehension.entities:
                if ent.lower() in fact_lower:
                    mentioned.add(ent.lower().strip())

        for ent in mentioned:
            if not ent:
                continue
            self._entity_condition_counts[ent] = self._entity_condition_counts.get(ent, 0) + 1
            if self._entity_condition_counts[ent] >= self.ENTITY_SATURATION_THRESHOLD:
                self._saturated_entities.add(ent)

    def get_saturated_entities(self) -> set[str]:
        """Return the set of entity names that have reached saturation."""
        return set(self._saturated_entities)

    def entity_saturation_ratio(self, question: str) -> float:
        """Return 0.0-1.0 indicating how saturated the entities in a question are.

        If the question mentions entities that are already saturated,
        returns a high value (closer to 1.0).  Used by the tree reactor
        to reduce pressure on branches exploring already-covered ground.
        """
        if not self._saturated_entities:
            return 0.0
        q_lower = question.lower()
        total_entities = 0
        saturated_count = 0
        for ent in self._entity_condition_counts:
            if ent in q_lower:
                total_entities += 1
                if ent in self._saturated_entities:
                    saturated_count += 1
        if total_entities == 0:
            return 0.0
        return saturated_count / total_entities

    def _get_saturation_signal(self) -> str:
        """Compute a saturation signal describing well-covered topics and entities."""
        # Topic-level saturation (bigram-based)
        topics = _compute_topic_buckets(self._conditions)
        saturated_topics = [
            f"\"{topic}\" ({count} conditions)"
            for topic, count in topics.items()
            if count >= self.SATURATION_THRESHOLD
        ]

        # Entity-level saturation
        saturated_ents = [
            f"\"{ent}\" ({self._entity_condition_counts[ent]} conditions)"
            for ent in sorted(self._saturated_entities)
        ]

        parts = []
        if saturated_topics:
            parts.append(f"SATURATED topics: {', '.join(saturated_topics[:5])}")
        if saturated_ents:
            parts.append(f"SATURATED entities: {', '.join(saturated_ents[:5])}")

        if not parts:
            return ""
        return (
            f"{'; '.join(parts)}. "
            f"Do NOT explore these further. Redirect research to unexplored angles: "
            f"enforcement cases, practitioner experiences, court rulings, vendor reviews, "
            f"community discussions, underground sources."
        )

    async def admit(
        self,
        condition: AtomicCondition,
        skip_relevance_llm: bool = False,
    ) -> AdmissionResult:
        """Attempt to admit a single condition into the global store.

        Runs the full admission pipeline:
          1. Source URL validation
          2. Relevance gate (cheap LLM call)
          3. Novelty check (Jaccard dedup)
          4. Serendipity scoring
          5. Trust scoring + cross-reference

        Returns an AdmissionResult with admission decision and guidance.
        """
        # Step 1: Validate source URL
        condition.source_url = _validate_source_url(condition.source_url)

        # Step 2: Basic content check
        if not condition.fact or len(condition.fact.strip()) < 10:
            return AdmissionResult(
                admitted=False,
                reason="empty",
                condition=condition,
            )

        # Step 3: Relevance gate (comprehension-aware)
        if not skip_relevance_llm:
            # Fast pre-check using comprehension keywords — deliberately loose
            fast_pass = self._fast_relevance_check(condition.fact)
            if fast_pass:
                # Comprehension says it's in the knowledge territory — admit
                # without the expensive LLM call
                pass
            else:
                # Not in the comprehension's keyword territory — use LLM gate
                is_relevant = await _get_relevance_gate()(
                    condition.fact, self._user_query, self._req_id,
                )
                if not is_relevant:
                    self._rejected_irrelevant += 1
                    return AdmissionResult(
                        admitted=False,
                        reason="irrelevant",
                        condition=condition,
                    )

        # Step 4: Novelty check (global Jaccard dedup)
        fact_words = set(condition.fact.lower().split())
        async with self._lock:
            best_sim, similar_fact = self._find_most_similar(fact_words)

            if best_sim > self.DUPLICATE_THRESHOLD:
                self._rejected_duplicate += 1
                saturation = self._get_saturation_signal()
                return AdmissionResult(
                    admitted=False,
                    reason="duplicate",
                    condition=condition,
                    similar_to=similar_fact,
                    saturation_signal=saturation,
                )

            # Step 5: Serendipity scoring
            known_facts = [c.fact for c in self._conditions[-50:]]
            seren = serendipity_score(
                condition.fact, self._user_query, known_facts,
            )
            condition.serendipity_score_val = seren

            # Step 6: Trust scoring
            condition.trust_score = trust_score_url(condition.source_url)

            # Step 7: Cross-reference — build bidirectional links (knowledge net)
            # Every new condition checks against all existing ones for overlap.
            # Partial overlap (0.3-0.55 Jaccard) means they discuss the same
            # topic but say different things — potential confirm or contradict.
            new_idx = len(self._conditions)  # index the new condition will have
            for i, existing in enumerate(self._conditions):
                sim = _jaccard_similarity(condition.fact, existing.fact)
                if sim < 0.2:
                    continue  # too dissimilar to be related
                if sim >= self.DUPLICATE_THRESHOLD:
                    continue  # duplicate — already caught above

                # Determine relationship: same confidence direction = confirms,
                # opposite direction = contradicts, otherwise = related
                conf_diff = abs(condition.confidence - existing.confidence)
                if conf_diff > 0.3:
                    relation = "contradicts"
                    # Contradicting claims reduce confidence on the less-sourced one
                    if condition.trust_score < existing.trust_score:
                        condition.confidence = max(condition.confidence - 0.1, 0.2)
                    elif condition.trust_score > existing.trust_score:
                        existing.confidence = max(existing.confidence - 0.1, 0.2)
                elif sim > 0.35:
                    relation = "confirms"
                    # Corroborating claims boost confidence on both
                    condition.confidence = min(condition.confidence + 0.05, 1.0)
                    existing.confidence = min(existing.confidence + 0.05, 1.0)
                else:
                    relation = "related"

                # Bidirectional links: new → existing AND existing → new
                condition.cross_refs.append(CrossRef(
                    relation=relation, target_idx=i, similarity=sim,
                ))
                existing.cross_refs.append(CrossRef(
                    relation=relation, target_idx=new_idx, similarity=sim,
                ))

            # Admit the condition
            self._conditions.append(condition)
            self._fact_word_sets.append(fact_words)
            self._admitted_count += 1
            self._update_entity_counts(condition)

            return AdmissionResult(
                admitted=True,
                reason="admitted",
                condition=condition,
                serendipity_score_val=seren,
                saturation_signal=self._get_saturation_signal(),
            )

    async def admit_batch(
        self,
        conditions: list[AtomicCondition],
        skip_relevance_llm: bool = False,
    ) -> list[AdmissionResult]:
        """Admit multiple conditions, returning results for each."""
        results: list[AdmissionResult] = []
        for c in conditions:
            result = await self.admit(c, skip_relevance_llm=skip_relevance_llm)
            results.append(result)
        return results

    def get_net_summary(self, max_items: int = 20) -> str:
        """Summarize the cross-reference knowledge net for downstream use.

        Returns a human-readable summary of the most-linked conditions and
        their relationships, suitable for injecting into synthesis or spawn
        prompts.  This exposes the net structure so downstream components
        know what confirms, contradicts, or relates to what.
        """
        if not self._conditions:
            return "(no conditions yet)"

        # Sort by number of cross-refs (most-connected first)
        indexed = [(i, c) for i, c in enumerate(self._conditions) if c.cross_refs]
        indexed.sort(key=lambda x: len(x[1].cross_refs), reverse=True)

        lines: list[str] = []
        for idx, cond in indexed[:max_items]:
            confirms = [r for r in cond.cross_refs if r.relation == "confirms"]
            contradicts = [r for r in cond.cross_refs if r.relation == "contradicts"]
            related = [r for r in cond.cross_refs if r.relation == "related"]

            line = f"- {cond.fact[:120]}"
            parts = []
            if confirms:
                parts.append(f"confirmed by {len(confirms)} other(s)")
            if contradicts:
                parts.append(f"contradicted by {len(contradicts)} other(s)")
            if related:
                parts.append(f"related to {len(related)} other(s)")
            if parts:
                line += f"  [{', '.join(parts)}]"
            lines.append(line)

        total_links = sum(len(c.cross_refs) for c in self._conditions)
        header = (
            f"Knowledge net: {len(self._conditions)} conditions, "
            f"{total_links} cross-reference links"
        )
        return header + "\n" + "\n".join(lines)


