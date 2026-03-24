"""
Research net reactor: priority-based concurrent research with semantic
deduplication.  Questions form a net (not a tree) — before spawning a
new question, the reactor checks the QuestionRegistry for semantic
matches and CONNECTS to existing nodes instead of duplicating research.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Optional

import langfuse_config

from .config import (
    RESEARCH_TIME_LIMIT,
    SUBAGENT_MODEL,
    TREE_MAX_CONCURRENT,
    TREE_MAX_DEPTH,
    TREE_MAX_NODES,
    TREE_PRESSURE_THRESHOLD,
    TREE_WORKER_IDLE_TIMEOUT,
    log,
)
from .models import AtomicCondition, ResearchNode, SubagentResult
from .llm import call_llm
from .pipeline import ConditionStore, QuestionRegistry, QueryComprehension, comprehend_query
from .subagent import run_subagent

if TYPE_CHECKING:
    from .synthesis import LiveFindingsCollector


# ============================================================================
# Tree Research Reactor
# ============================================================================

SPAWN_QUESTIONS_PROMPT = """You are a research strategist who generates focused follow-up questions.

Given the findings so far, generate follow-up questions that help answer the ORIGINAL USER QUERY more completely. Your goal is to chase DEEP, RARE, EMBEDDED knowledge — the kind found in community discussions, practitioner experiences, court documents, underground forums, academic papers, and obscure archives. NOT surface-level summaries.

**Original user query:** {user_query}
**Question just investigated:** {node_question}
**Context:** {node_context}

**Deep understanding of the query:**
{comprehension_context}

**Knowledge net state (findings):**
{net_summary}

**Research question net (already asked/being researched):**
{question_net_summary}

**Findings from this investigation:**
{findings_text}

Generate follow-up questions. For each, provide:
- "question": a specific, searchable question
- "context": one sentence on why this matters
- "pressure": 0.0-1.0 importance score (1.0 = critical gap, 0.1 = minor curiosity)
- "strategy": one of "deepen" | "verify" | "lateral" | "contrarian" | "historical" | "cross-domain"

STRATEGY RULES:
- "deepen": drill further into a specific finding (preferred — most questions should be this)
- "verify": cross-reference a specific entity/claim by searching for independent mentions (REQUIRED for any concrete entity discovered — vendor, person, product, website). Example: "Has anyone on Reddit/forums confirmed buying from [vendor X]?" or "What do reviews say about [product Y]?"
- "lateral": explore a related angle that DIRECTLY helps answer the original query from a different perspective
- "contrarian": investigate the opposite claim or a dissenting viewpoint ON THE SAME TOPIC. Pay special attention to claims that CONTRADICT each other in the knowledge net — these need resolution.
- "historical": look at historical precedents directly relevant to the query
- "cross-domain": ONLY use if there is a genuinely useful parallel — do NOT force random associations
- "verify" questions are HIGH PRIORITY — they should be generated for EVERY concrete entity found. If findings mention specific vendors, products, organizations, or individuals, there MUST be verify questions for them.
- Non-deepen strategies are optional. Only use them if they genuinely serve the user's question.
- CRITICAL: "lateral" does NOT mean "free association with a keyword". If the user asks about buying insulin, a lateral question is about alternative purchasing channels — NOT about bodybuilding or side effects.

RESEARCH NET RULES (CRITICAL — read carefully):
The questions above form a RESEARCH NET. Before generating a new question:
1. CHECK if a similar question is already [DONE], [ACTIVE], or [QUEUED] in the net
2. If a question is ALREADY ANSWERED ([DONE]) — do NOT ask a rephrased version of it. Instead, if the answer needs deeper investigation, generate a MORE SPECIFIC follow-up that drills into a particular detail.
3. If a question is [ACTIVE] or [QUEUED] — do NOT generate anything similar. It will be answered soon.
4. Two questions are "similar" if they would produce the same search results. Examples of SIMILAR questions:
   - "Are there documented cases of insulin importation prosecution in Poland?" ≈ "What do Polish court records show about insulin importation cases?"
   - "Has anyone reviewed vendor X?" ≈ "What do people say about vendor X on forums?"
5. If contradictions exist in the net, generate ONE targeted question to resolve them — not multiple variations.

PRESSURE RULES:
- Higher pressure for: contradictions in the knowledge net, unverified claims, critical gaps directly relevant to the query, unexplored adjacent territories
- HIGHEST pressure (0.9-1.0) for: verify questions about concrete entities that haven't been cross-referenced yet
- Lower pressure for: already-well-confirmed areas, tangential topics
- 0 questions is fine if the topic is well-covered AND all concrete entities have been verified

Other rules:
- Generate 0-5 questions maximum
- Each question MUST be semantically DISTINCT from all questions in the research net
- Output ONLY valid JSON, no markdown fences

Output format:
{{"sub_questions": [{{"question": "...", "context": "...", "pressure": 0.8, "strategy": "lateral"}}]}}"""


# ============================================================================
# Prompt-Distance Scoring
# ============================================================================

# Intent-specific keyword boosters: questions containing these patterns
# get a relevance boost for the corresponding intent type.
_INTENT_BOOSTERS: dict[str, list[str]] = {
    "transactional": [
        "buy", "purchase", "order", "get", "acquire", "obtain", "source",
        "vendor", "supplier", "price", "cost", "shipping", "delivery",
        "how to", "setup", "install", "configure", "step by step",
        "where can", "who sells", "cheapest", "best deal", "coupon",
        "subscribe", "sign up", "register", "apply", "book",
    ],
    "informational": [
        "what is", "how does", "why", "explain", "understand",
        "mechanism", "cause", "effect", "research", "study",
        "evidence", "data", "statistics", "history", "background",
        "difference between", "compare", "analysis", "review",
    ],
    "exploratory": [
        "overview", "landscape", "state of", "trends", "future",
        "possibilities", "options", "alternatives", "emerging",
        "what are", "survey", "map", "scope", "range",
    ],
}


def _score_prompt_distance(
    question: str,
    core_need: str,
    intent_type: str,
) -> float:
    """Score how closely a research question serves the user's core need.

    Returns a value in [0.0, 1.0] where 1.0 means the question directly
    addresses the core need and intent.

    The score combines:
      1. Word overlap between question and core_need (Jaccard-like)
      2. Intent-specific keyword boost
    """
    if not core_need:
        return 0.5  # no core_need available — neutral score

    q_words = set(question.lower().split())
    need_words = set(core_need.lower().split())

    # Filter out very short words (articles, prepositions)
    q_words = {w for w in q_words if len(w) > 2}
    need_words = {w for w in need_words if len(w) > 2}

    if not q_words or not need_words:
        return 0.5

    # 1. Word overlap component (0-1)
    overlap = len(q_words & need_words)
    union = len(q_words | need_words)
    overlap_score = overlap / max(union, 1)

    # 2. Intent-specific keyword boost (0 or 0.15)
    intent_boost = 0.0
    q_lower = question.lower()
    boosters = _INTENT_BOOSTERS.get(intent_type, [])
    for keyword in boosters:
        if keyword in q_lower:
            intent_boost = 0.15
            break

    # Combine: overlap is primary, intent boost is additive
    raw = overlap_score * 0.85 + intent_boost
    return min(1.0, max(0.0, raw))


def _compute_pressure(
    base_pressure: float,
    depth: int,
    parent_pressure: float,
    prompt_distance: float = 0.5,
) -> float:
    """Compute final pressure score for a research node.

    Combines the LLM's assessed importance with a depth decay,
    inheritance from the parent node's pressure, and prompt-distance
    scoring (how closely the question serves the user's core need).

    prompt_distance in [0, 1]: 1.0 = directly addresses core need.
    Questions closer to the core need get a pressure boost.
    """
    depth_decay = max(0.1, 1.0 - (depth * 0.15))
    inherited = parent_pressure * 0.3
    base_weight = base_pressure * 0.7
    # prompt_distance modulates base: questions far from core_need
    # get dampened, questions close get boosted (range: 0.7x to 1.3x)
    distance_factor = 0.7 + (prompt_distance * 0.6)
    return min(1.0, (base_weight + inherited) * depth_decay * distance_factor)


async def _spawn_sub_questions(
    node: ResearchNode,
    conditions: list[AtomicCondition],
    user_query: str,
    existing_questions: list[str],
    req_id: str,
    condition_store: Optional["ConditionStore"] = None,
    question_registry: Optional["QuestionRegistry"] = None,
) -> list[ResearchNode]:
    """Ask LLM to generate follow-up questions, then apply connect-or-spawn gate.

    Uses the condition store's comprehension map, knowledge net state,
    AND the question registry to guide question generation toward deep,
    rare knowledge while avoiding near-duplicate questions.

    The connect-or-spawn gate (powered by QuestionRegistry) checks every
    LLM-proposed question against all existing questions in the net:
      - If a semantic match exists → CONNECT (add edge) instead of spawning
      - If no match → SPAWN a new node

    Returns a list of new ResearchNode children (only truly novel questions).
    """
    if not conditions or node.depth >= TREE_MAX_DEPTH:
        return []

    findings_text = "\n".join(
        f"- {c.fact} [confidence: {c.confidence:.1f}]"
        for c in conditions[:15]
    )

    # Build comprehension context for the spawn prompt
    comprehension_context = "(no deep comprehension available)"
    net_summary = "(no knowledge net yet)"
    question_net_summary = "(no questions in the net yet)"
    if condition_store:
        if condition_store.comprehension:
            comp = condition_store.comprehension
            parts = []
            if comp.semantic_summary:
                parts.append(f"Summary: {comp.semantic_summary[:300]}")
            if comp.adjacent_territories:
                parts.append(f"Adjacent territories to explore: {', '.join(comp.adjacent_territories[:8])}")
            if comp.deep_knowledge_targets:
                parts.append(f"Deep knowledge targets: {', '.join(comp.deep_knowledge_targets[:8])}")
            if comp.implicit_questions:
                unanswered = [q for q in comp.implicit_questions if q not in existing_questions]
                if unanswered:
                    parts.append(f"Still-unanswered implicit questions: {', '.join(unanswered[:5])}")
            comprehension_context = "\n".join(parts) if parts else comprehension_context
        net_summary = condition_store.get_net_summary(max_items=10)

    if question_registry:
        question_net_summary = question_registry.get_net_question_summary(max_items=20)

    prompt = SPAWN_QUESTIONS_PROMPT.format(
        user_query=user_query,
        node_question=node.question,
        node_context=node.context,
        comprehension_context=comprehension_context,
        net_summary=net_summary,
        question_net_summary=question_net_summary,
        findings_text=findings_text,
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
    connected_count = 0

    for sq in data.get("sub_questions", []):
        question = sq.get("question", "").strip()
        if not question:
            continue

        # ---- Connect-or-Spawn Gate ----
        # Check the question registry for semantic matches BEFORE spawning.
        # This is the core "net" mechanism: similar questions connect
        # instead of duplicating research.
        if question_registry:
            matches = await question_registry.find_similar(question)
            if matches:
                best = matches[0]
                # Connect to the existing node instead of spawning
                await question_registry.add_edge(node.id, best.node_id)
                node.connected_to.append(best.node_id)
                connected_count += 1
                log.info(
                    f"[{req_id}] NET CONNECT: \"{question[:80]}\" → "
                    f"existing \"{best.question[:80]}\" "
                    f"(sim={best.similarity:.2f}, status={best.status})"
                )
                continue  # Do NOT spawn — already covered

        # Legacy substring check as fallback (in case registry not available)
        q_lower = question.lower()
        if any(q_lower in eq.lower() or eq.lower() in q_lower for eq in existing_questions):
            continue

        raw_pressure = float(sq.get("pressure", 0.5))

        # Compute prompt-distance for this question
        pd_score = 0.5
        if condition_store and condition_store.comprehension:
            pd_score = _score_prompt_distance(
                question,
                condition_store.comprehension.core_need,
                condition_store.comprehension.intent_type,
            )

        pressure = _compute_pressure(
            raw_pressure, node.depth + 1, node.pressure,
            prompt_distance=pd_score,
        )

        # Saturation-aware pressure decay — if this question
        # covers entities we've already thoroughly researched, reduce
        # pressure so the net redirects budget to unexplored ground.
        if condition_store:
            sat_ratio = condition_store.entity_saturation_ratio(question)
            if sat_ratio > 0:
                # Reduce pressure proportionally: 50% reduction at full saturation
                pressure *= (1.0 - sat_ratio * 0.5)

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

    if connected_count > 0:
        log.info(
            f"[{req_id}] Net dedup: {connected_count} questions connected to "
            f"existing nodes, {len(children)} new questions spawned"
        )

    return children


_ENTITY_EXTRACTION_PROMPT = """Extract CONCRETE ENTITIES from these research findings that need independent verification.

A concrete entity is: a specific vendor/website, product name, person, organization, or service that was discovered during research and could be verified by searching for independent mentions.

Do NOT include: general concepts, countries, well-known companies (Google, Amazon, etc.), or abstract ideas.

Findings:
{findings_text}

Output ONLY valid JSON:
{{"entities": [
    {{"name": "exact entity name", "type": "vendor|product|person|organization|website|service", "fact_index": 0, "search_queries": ["entity_name review", "entity_name reddit"]}}
]}}

If no concrete entities need verification, return: {{"entities": []}}"""


async def _extract_entities_for_verification(
    conditions: list[AtomicCondition],
    req_id: str,
) -> list[dict]:
    """Extract concrete entities from conditions that need cross-referencing.

    Uses a lightweight LLM call to identify vendor names, product names,
    specific websites, persons, or organizations mentioned in findings.
    Returns a list of entity dicts with search queries for verification.
    """
    if not conditions:
        return []

    findings_text = "\n".join(
        f"{i}. {c.fact} [source: {c.source_url or 'no source'}]"
        for i, c in enumerate(conditions)
    )

    prompt = _ENTITY_EXTRACTION_PROMPT.format(findings_text=findings_text)

    result = await call_llm(
        [{"role": "user", "content": prompt}],
        req_id,
        model=SUBAGENT_MODEL,
        max_tokens=1024,
        temperature=0.1,
    )

    if "error" in result:
        log.warning(f"[{req_id}] Entity extraction error: {result['error']}")
        return []

    content = result.get("content", "").strip()
    try:
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []

    return data.get("entities", [])


def _spawn_verification_nodes(
    entities: list[dict],
    parent_node: ResearchNode,
    existing_questions: list[str],
    req_id: str,
) -> list[ResearchNode]:
    """Create verification ResearchNodes for concrete entities.

    Each entity gets a dedicated tree node that will cross-reference it
    across forums, reviews, and social media.  These nodes get high
    pressure so the tree prioritizes them.
    """
    children: list[ResearchNode] = []
    for ent in entities:
        name = ent.get("name", "").strip()
        ent_type = ent.get("type", "entity")
        if not name:
            continue

        # Build verification question
        question = (
            f'Verify "{name}": search for independent mentions, reviews, '
            f"complaints, or discussions about this {ent_type} across "
            f"Reddit, forums, social media, and review sites"
        )

        # Skip if we already have a similar question
        if any(
            name.lower() in eq.lower() and "verify" in eq.lower()
            for eq in existing_questions
        ):
            continue

        # Verification nodes get high pressure (0.85) and inherit
        # parent depth + 1 so they can still spawn further branches
        # based on what the verification discovers
        child = ResearchNode(
            id=f"{req_id}-verify-{uuid.uuid4().hex[:6]}",
            question=question,
            context=f"Cross-reference {ent_type} '{name}' found in parent research. "
                    f"Search queries to try: {', '.join(ent.get('search_queries', []))}",
            depth=parent_node.depth + 1,
            pressure=_compute_pressure(0.85, parent_node.depth + 1, parent_node.pressure),
            parent_id=parent_node.id,
        )
        children.append(child)

    return children


async def _research_single_node(
    node: ResearchNode,
    user_query: str,
    req_id: str,
    collector: "LiveFindingsCollector",
    curated_queue: asyncio.Queue,
    condition_store: Optional["ConditionStore"] = None,
) -> tuple[list[AtomicCondition], SubagentResult]:
    """Research a single tree node using the existing subagent loop.

    This wraps run_subagent with the tree node's question/context
    and feeds findings into the collector and curated queue.
    Conditions pass through the global ConditionStore at birth.
    """
    span = langfuse_config.start_span(
        req_id, f"tree:research_node:{node.id}",
        input={"question": node.question[:200], "depth": node.depth, "pressure": node.pressure},
    )
    angle = {
        "title": node.question,
        "query": node.question,
        "description": node.context,
        "is_bridge": False,
    }

    # Track this question as actively being researched
    await collector.set_active_question(node.question)

    # Lightweight internal progress queue (not emitting system noise)
    internal_queue: asyncio.Queue = asyncio.Queue()

    sa_result = await run_subagent(
        angle=angle,
        subagent_index=0,
        progress_queue=internal_queue,
        req_id=req_id,
        user_query=user_query,
        depth=0,
        collector=collector,
        condition_store=condition_store,
    )

    # Clear the active question now that research is done
    await collector.clear_active_question(node.question)

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
            "finding": top_finding.fact,
            "conditions_count": len(sa_result.conditions),
            "depth": node.depth,
        })

    langfuse_config.end_span(span, output={
        "conditions": len(sa_result.conditions),
        "turns": sa_result.turns_used,
        "tools": sa_result.tool_calls_made,
    })
    return sa_result.conditions, sa_result


async def tree_research_reactor(
    user_query: str,
    prior_conditions: list[dict],
    graph_neighbors: list[dict],
    req_id: str,
    collector: "LiveFindingsCollector",
    curated_queue: asyncio.Queue,
    start_time: float = 0.0,
) -> dict:
    """Research net reactor with global condition admission pipeline.

    Explores the research space as a NET: each finding can spawn
    sub-questions, but before spawning, every candidate question is
    checked against the QuestionRegistry for semantic matches.  If a
    similar question already exists, the new question CONNECTS to the
    existing node instead of duplicating research.

    All conditions pass through a global ConditionStore which handles
    dedup, relevance gating, serendipity scoring, and saturation signaling.

    The semaphore governs only the workers doing active LLM+tool
    research.  Spawning and queuing are free (no slot consumed).

    Returns a dict with keys matching the old plan+subagents output:
      - subagent_results, all_conditions, total_turns, total_tools,
        total_children, progress_log, admission_stats, net_stats
    """
    sem = asyncio.Semaphore(TREE_MAX_CONCURRENT)
    pending: asyncio.PriorityQueue = asyncio.PriorityQueue()
    progress: list[str] = []

    # Step 0: Deep query comprehension — understand what the query is REALLY about
    # This runs once and produces a semantic map that guides all downstream decisions
    progress.append("\n**[Phase 2a: Query Comprehension]**\n")
    progress.append("Building deep semantic understanding of the research query...\n")
    comp_span = langfuse_config.start_span(
        req_id, "tree:query_comprehension",
        input={"query": user_query[:200]},
    )
    comprehension = await comprehend_query(user_query, req_id)
    langfuse_config.end_span(comp_span, output={
        "intent": comprehension.intent_type,
        "entities": len(comprehension.entities),
        "domains": len(comprehension.domains),
        "implicit_questions": len(comprehension.implicit_questions),
    })
    if comprehension.semantic_summary:
        progress.append(
            f"Understanding: {comprehension.semantic_summary[:300]}\n"
            f"Intent: **{comprehension.intent_type}**\n"
            f"Core need: {comprehension.core_need[:200]}\n"
            f"Entities: {', '.join(comprehension.entities[:10])}\n"
            f"Domains: {', '.join(comprehension.domains[:8])}\n"
            f"Adjacent territories: {', '.join(comprehension.adjacent_territories[:6])}\n"
        )
    log.info(
        f"[{req_id}] Query comprehension: intent={comprehension.intent_type}, "
        f"{len(comprehension.entities)} entities, "
        f"{len(comprehension.domains)} domains, "
        f"{len(comprehension.implicit_questions)} implicit questions, "
        f"{len(comprehension.adjacent_territories)} adjacent territories, "
        f"{len(comprehension.relevance_keywords)} relevance keywords, "
        f"core_need={comprehension.core_need[:100]}"
    )

    # Global condition store — seeded with comprehension for relevance-aware admission
    condition_store = ConditionStore(
        user_query=user_query, req_id=req_id, comprehension=comprehension,
    )

    # Global question registry — tracks all questions for semantic dedup (the "net")
    question_registry = QuestionRegistry()

    # Admit understanding conditions — they go through the same pipeline
    understanding_results = await condition_store.admit_understanding(comprehension)
    understanding_admitted = sum(1 for r in understanding_results if r.admitted)
    progress.append(
        f"Admitted {understanding_admitted} understanding conditions "
        f"(entities, domains, implicit questions, adjacent territories, deep targets)\n"
    )

    # Bookkeeping
    all_conditions: list[AtomicCondition] = []
    all_results: list[SubagentResult] = []
    all_questions: list[str] = [user_query]
    nodes_by_id: dict[str, ResearchNode] = {}
    total_queued = 0
    total_processed = 0
    active_workers = 0  # count of workers currently researching a node
    lock = asyncio.Lock()
    done_event = asyncio.Event()  # set when net exploration is complete

    # Build the root node
    prior_text = ""
    if prior_conditions:
        prior_text = " | Prior knowledge: " + "; ".join(
            pc["fact"] for pc in prior_conditions[:5]
        )

    neighbor_text = ""
    if graph_neighbors:
        neighbor_text = " | Graph context: " + "; ".join(
            f"{n.get('fact', '')} (via {n.get('via_entity', '?')})"
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
    await question_registry.register(user_query, root.id, status="pending")
    total_queued = 1

    # --- Pre-seed: comprehension-guided initial angles ---
    # Use the query comprehension's implicit questions and adjacent territories
    # to seed parallel research angles.  This replaces the old generic
    # "decompose into 3-5 angles" prompt — now the angles come from deep
    # understanding of what the query is really about.
    try:
        seed_angles: list[tuple[str, str]] = []  # (question, context)

        # Implicit questions from comprehension — these are the questions
        # the user is REALLY asking but didn't spell out
        for q in comprehension.implicit_questions[:4]:
            if q.strip() and q.lower() != user_query.lower():
                seed_angles.append((q, "Implicit question from query comprehension"))

        # Adjacent territories — where the DEEP knowledge lives
        for terr in comprehension.adjacent_territories[:3]:
            if terr.strip():
                seed_angles.append((
                    f"What do {terr} reveal about {user_query[:100]}?",
                    f"Adjacent territory: {terr}",
                ))

        # Deep knowledge targets — specific types of rare knowledge
        for target in comprehension.deep_knowledge_targets[:2]:
            if target.strip():
                seed_angles.append((
                    f"Find {target} related to {user_query[:100]}",
                    f"Deep knowledge target: {target}",
                ))

        # If comprehension didn't produce enough angles, fall back to LLM decomposition
        if len(seed_angles) < 3:
            seed_prompt = (
                f"Decompose this research query into 3-5 DISTINCT research angles "
                f"that can be investigated IN PARALLEL. Focus on angles that would "
                f"find DEEP, RARE knowledge — practitioner experiences, community "
                f"discussions, enforcement data, obscure archives.\n\n"
                f"Query: {user_query}\n\n"
                f"Output ONLY valid JSON:\n"
                f'{{"angles": [{{"question": "specific searchable question", '
                f'"context": "why this angle matters"}}]}}'
            )
            seed_result = await call_llm(
                [{"role": "user", "content": seed_prompt}],
                req_id, model=SUBAGENT_MODEL, max_tokens=1024, temperature=0.4,
            )
            if "error" not in seed_result:
                seed_content = seed_result.get("content", "").strip()
                if seed_content.startswith("```"):
                    seed_content = re.sub(r'^```(?:json)?\s*', '', seed_content)
                    seed_content = re.sub(r'\s*```$', '', seed_content)
                seed_data = json.loads(seed_content)
                for angle in seed_data.get("angles", [])[:5]:
                    q = angle.get("question", "").strip()
                    if q and q.lower() != user_query.lower():
                        seed_angles.append((q, angle.get("context", "")))

        # Create seed nodes from the angles — use prompt-distance
        # scoring instead of uniform 0.9 pressure so questions closer
        # to the core_need get explored first.
        # Each seed angle goes through the question registry's connect-or-spawn
        # gate to prevent the comprehension from seeding near-duplicate angles.
        for i, (q, ctx) in enumerate(seed_angles[:8]):
            # Check the registry first — even seed angles can be near-duplicates
            gate_span = langfuse_config.start_span(
                req_id, f"tree:connect_or_spawn:seed{i}",
                input={"question": q[:120]},
            )
            matches = await question_registry.find_similar(q)
            if matches:
                # Connect to existing instead of seeding a duplicate
                await question_registry.add_edge(root.id, matches[0].node_id)
                root.connected_to.append(matches[0].node_id)
                log.info(
                    f"[{req_id}] Seed angle deduped: \"{q[:60]}\" → "
                    f"existing \"{matches[0].question[:60]}\" "
                    f"(sim={matches[0].similarity:.2f})"
                )
                langfuse_config.end_span(gate_span, output={
                    "decision": "connect",
                    "matched_node": matches[0].node_id,
                    "similarity": matches[0].similarity,
                })
                continue

            pd_score = _score_prompt_distance(
                q, comprehension.core_need, comprehension.intent_type,
            )
            # Seed pressure: base 0.9 modulated by prompt distance
            # Range: ~0.63 (distant) to ~1.0 (directly on core_need)
            seed_pressure = min(1.0, 0.9 * (0.7 + pd_score * 0.6))
            seed_node = ResearchNode(
                id=f"{req_id}-seed{i}",
                question=q,
                context=ctx,
                depth=0,
                pressure=seed_pressure,
                parent_id=root.id,
            )
            nodes_by_id[seed_node.id] = seed_node
            all_questions.append(q)
            await pending.put(seed_node)
            await question_registry.register(q, seed_node.id, status="pending")
            total_queued += 1
            langfuse_config.end_span(gate_span, output={
                "decision": "spawn",
                "node_id": seed_node.id,
                "pressure": seed_pressure,
            })

        log.info(
            f"[{req_id}] Seeded {total_queued - 1} comprehension-guided research "
            f"angles (after net dedup from {len(seed_angles)} candidates)"
        )
    except Exception as e:
        log.warning(f"[{req_id}] Pre-seed decomposition failed (non-fatal): {e}")

    progress.append(
        f"\n**[Phase 2: Research Net Reactor]** "
        f"(max {TREE_MAX_CONCURRENT} concurrent, "
        f"depth limit {TREE_MAX_DEPTH}, "
        f"node budget {TREE_MAX_NODES})\n"
    )
    progress.append(f"Root question: {user_query}\n")

    await curated_queue.put({
        "type": "start",
        "question": user_query,
    })

    _reactor_start = start_time or time.monotonic()

    def _time_exceeded() -> bool:
        """Return True if RESEARCH_TIME_LIMIT has been exceeded."""
        if RESEARCH_TIME_LIMIT <= 0:
            return False
        return (time.monotonic() - _reactor_start) >= RESEARCH_TIME_LIMIT

    async def worker(worker_id: int) -> None:
        nonlocal total_processed, total_queued, active_workers

        while True:
            # Check time limit before picking up new work
            if _time_exceeded():
                log.info(
                    f"[{req_id}] Worker {worker_id}: research time limit "
                    f"({RESEARCH_TIME_LIMIT:.0f}s) reached — stopping"
                )
                done_event.set()
                return

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
                    # Also check time limit while waiting
                    if _time_exceeded():
                        done_event.set()
                        return
                    continue

            # Skip pruned nodes
            if node.status == "pruned":
                continue

            # Mark this worker as active before acquiring semaphore
            async with lock:
                active_workers += 1

            worker_span = langfuse_config.start_span(
                req_id, f"tree:worker:{worker_id}:{node.id}",
                input={"question": node.question[:200], "depth": node.depth},
            )
            try:
                # Acquire semaphore — only active research counts
                async with sem:
                    node.status = "researching"
                    await question_registry.update_status(node.id, "researching")

                    conditions, sa_result = await _research_single_node(
                        node, user_query, req_id, collector, curated_queue,
                        condition_store=condition_store,
                    )

                    node.status = "done"
                    await question_registry.update_status(node.id, "done")

                    # Store top finding in the registry for net summary
                    if conditions:
                        top = max(conditions, key=lambda c: c.confidence)
                        await question_registry.update_finding(
                            node.id, top.fact,
                        )

                async with lock:
                    total_processed += 1
                    all_conditions.extend(conditions)
                    all_results.append(sa_result)

                # Spawn children (doesn't hold semaphore)
                # Skip spawning if time limit exceeded — let tree wind down
                if _time_exceeded():
                    log.info(
                        f"[{req_id}] Skipping child spawn for {node.id} "
                        f"— time limit exceeded"
                    )
                    continue

                async with lock:
                    current_queued = total_queued

                if current_queued < TREE_MAX_NODES and conditions:
                    # 1. Regular sub-questions from LLM — now with
                    #    connect-or-spawn gate via question_registry
                    children = await _spawn_sub_questions(
                        node, conditions, user_query, all_questions, req_id,
                        condition_store=condition_store,
                        question_registry=question_registry,
                    )

                    # 2. Auto-spawn verification nodes for concrete
                    #    entities discovered in this node's findings.
                    #    These also go through the question registry
                    #    to prevent duplicate verification.
                    if node.depth < TREE_MAX_DEPTH:
                        try:
                            ent_span = langfuse_config.start_span(
                                req_id, f"tree:entity_extraction:{node.id}",
                                input={"conditions_count": len(conditions)},
                            )
                            entities = await _extract_entities_for_verification(
                                conditions, req_id,
                            )
                            langfuse_config.end_span(ent_span, output={
                                "entities_found": len(entities) if entities else 0,
                            })
                            if entities:
                                raw_verify = _spawn_verification_nodes(
                                    entities, node, all_questions, req_id,
                                )
                                # Filter verification nodes through the
                                # question registry too
                                for vc in raw_verify:
                                    matches = await question_registry.find_similar(
                                        vc.question,
                                    )
                                    if matches:
                                        await question_registry.add_edge(
                                            node.id, matches[0].node_id,
                                        )
                                        node.connected_to.append(
                                            matches[0].node_id,
                                        )
                                        log.info(
                                            f"[{req_id}] Verify node deduped: "
                                            f"\"{vc.question[:60]}\" → "
                                            f"\"{matches[0].question[:60]}\""
                                        )
                                    else:
                                        children.append(vc)
                                log.info(
                                    f"[{req_id}] Verification: "
                                    f"{len(raw_verify)} candidates, "
                                    f"{len([c for c in children if c in raw_verify])} "
                                    f"after net dedup"
                                )
                        except Exception as e:
                            log.warning(
                                f"[{req_id}] Entity verification spawn failed "
                                f"(non-fatal): {e}"
                            )

                    async with lock:
                        actually_queued = 0
                        for child in children:
                            if total_queued >= TREE_MAX_NODES:
                                break
                            nodes_by_id[child.id] = child
                            all_questions.append(child.question)
                            await pending.put(child)
                            await question_registry.register(
                                child.question, child.id, status="pending",
                            )
                            total_queued += 1
                            actually_queued += 1

                    if actually_queued > 0:
                        await curated_queue.put({
                            "type": "branch",
                            "parent_question": node.question,
                            "children_count": actually_queued,
                            "top_child": children[0].question if children else "",
                            "depth": node.depth + 1,
                        })
                langfuse_config.end_span(worker_span, output={
                    "conditions": len(all_conditions) - (total_processed - 1) if total_processed else 0,
                    "children_spawned": actually_queued if 'actually_queued' in dir() else 0,
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

    # Compute net structure stats
    net_stats = question_registry.stats
    total_net_edges = sum(
        len(n.connected_to) for n in nodes_by_id.values()
    )

    _elapsed = time.monotonic() - _reactor_start
    _time_note = ""
    if _time_exceeded():
        _time_note = (
            f" **[TIME LIMIT: research capped at {RESEARCH_TIME_LIMIT:.0f}s "
            f"— forcing synthesis with findings so far]**"
        )
    progress.append(
        f"\n**Research Net Exploration Complete** ({_elapsed:.0f}s):{_time_note} "
        f"{total_processed} nodes explored "
        f"(depth reached: {max((n.depth for n in nodes_by_id.values()), default=0)}), "
        f"{len(all_conditions)} atomic conditions, "
        f"{total_turns} total turns, {total_tools} tool calls\n"
    )
    progress.append(
        f"**Net Structure:** {net_stats['unique_questions']} unique questions, "
        f"{net_stats['total_connected']} questions connected to existing nodes "
        f"(saved from duplicate research), "
        f"{net_stats['net_edges']} net edges\n"
    )

    await curated_queue.put({
        "type": "summary",
        "nodes_explored": total_processed,
        "conditions_count": len(all_conditions),
        "net_connections": net_stats["total_connected"],
        "net_edges": total_net_edges,
    })

    # When using admission pipeline, the ConditionStore has the canonical set
    admission_stats = condition_store.stats
    store_conditions = condition_store.conditions
    if store_conditions:
        # Use the globally-admitted conditions instead of the raw all_conditions
        all_conditions = store_conditions

    progress.append(
        f"\n**Admission Pipeline Stats:** "
        f"{admission_stats['admitted']} admitted, "
        f"{admission_stats['rejected_duplicate']} duplicates rejected, "
        f"{admission_stats['rejected_irrelevant']} irrelevant rejected\n"
    )

    return {
        "subagent_results": all_results,
        "all_conditions": all_conditions,
        "total_turns": total_turns,
        "total_tools": total_tools,
        "total_children": total_children,
        "progress_log": progress,
        "admission_stats": admission_stats,
        "net_stats": net_stats,
    }

