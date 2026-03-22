"""
Tree research reactor: priority-based concurrent research with depth control.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import TYPE_CHECKING, Optional

from .config import (
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
from .pipeline import ConditionStore, comprehend_query
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

**Knowledge net state:**
{net_summary}

**Findings from this investigation:**
{findings_text}

**Questions already in the research tree (avoid duplicates):**
{existing_questions}

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

KNOWLEDGE NET RULES:
- If two claims CONTRADICT each other, generate a question that resolves the contradiction from an independent source
- If a claim has ZERO cross-references, it's unverified — consider generating a question to corroborate or refute it
- If a claim is confirmed by multiple sources, it's well-established — lower pressure for that area
- Use the adjacent territories and deep knowledge targets from the query comprehension to guide where to look next

PRESSURE RULES:
- Higher pressure for: contradictions in the knowledge net, unverified claims, critical gaps directly relevant to the query, unexplored adjacent territories
- HIGHEST pressure (0.9-1.0) for: verify questions about concrete entities that haven't been cross-referenced yet
- Lower pressure for: already-well-confirmed areas, tangential topics
- 0 questions is fine if the topic is well-covered AND all concrete entities have been verified

Other rules:
- Generate 0-5 questions maximum
- Do NOT repeat questions already in the tree
- Output ONLY valid JSON, no markdown fences

Output format:
{{"sub_questions": [{{"question": "...", "context": "...", "pressure": 0.8, "strategy": "lateral"}}]}}"""


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
    condition_store: Optional["ConditionStore"] = None,
) -> list[ResearchNode]:
    """Ask LLM to generate follow-up questions from research findings.

    Uses the condition store's comprehension map and knowledge net state
    to guide question generation toward deep, rare knowledge.

    Returns a list of new ResearchNode children.
    """
    if not conditions or node.depth >= TREE_MAX_DEPTH:
        return []

    findings_text = "\n".join(
        f"- {c.fact} [confidence: {c.confidence:.1f}]"
        for c in conditions[:15]
    )

    existing_text = "\n".join(f"- {q}" for q in existing_questions[-30:]) or "(none yet)"

    # Build comprehension context for the spawn prompt
    comprehension_context = "(no deep comprehension available)"
    net_summary = "(no knowledge net yet)"
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

    prompt = SPAWN_QUESTIONS_PROMPT.format(
        user_query=user_query,
        node_question=node.question,
        node_context=node.context,
        comprehension_context=comprehension_context,
        net_summary=net_summary,
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

    return sa_result.conditions, sa_result


async def tree_research_reactor(
    user_query: str,
    prior_conditions: list[dict],
    graph_neighbors: list[dict],
    req_id: str,
    collector: "LiveFindingsCollector",
    curated_queue: asyncio.Queue,
) -> dict:
    """Tree-based research reactor with global condition admission pipeline.

    Explores the research space as a tree: each finding can spawn
    sub-questions which get explored by concurrent workers.
    All conditions pass through a global ConditionStore which handles
    dedup, relevance gating, serendipity scoring, and saturation signaling.

    The semaphore governs only the workers doing active LLM+tool
    research.  Spawning and queuing are free (no slot consumed).

    Returns a dict with keys matching the old plan+subagents output:
      - subagent_results, all_conditions, total_turns, total_tools,
        total_children, progress_log, admission_stats
    """
    sem = asyncio.Semaphore(TREE_MAX_CONCURRENT)
    pending: asyncio.PriorityQueue = asyncio.PriorityQueue()
    progress: list[str] = []

    # Step 0: Deep query comprehension — understand what the query is REALLY about
    # This runs once and produces a semantic map that guides all downstream decisions
    progress.append("\n**[Phase 2a: Query Comprehension]**\n")
    progress.append("Building deep semantic understanding of the research query...\n")
    comprehension = await comprehend_query(user_query, req_id)
    if comprehension.semantic_summary:
        progress.append(
            f"Understanding: {comprehension.semantic_summary[:300]}\n"
            f"Entities: {', '.join(comprehension.entities[:10])}\n"
            f"Domains: {', '.join(comprehension.domains[:8])}\n"
            f"Adjacent territories: {', '.join(comprehension.adjacent_territories[:6])}\n"
        )
    log.info(
        f"[{req_id}] Query comprehension: {len(comprehension.entities)} entities, "
        f"{len(comprehension.domains)} domains, "
        f"{len(comprehension.implicit_questions)} implicit questions, "
        f"{len(comprehension.adjacent_territories)} adjacent territories, "
        f"{len(comprehension.relevance_keywords)} relevance keywords"
    )

    # Global condition store — seeded with comprehension for relevance-aware admission
    condition_store = ConditionStore(
        user_query=user_query, req_id=req_id, comprehension=comprehension,
    )

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
    done_event = asyncio.Event()  # set when tree exploration is complete

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

        # Create seed nodes from the angles
        for i, (q, ctx) in enumerate(seed_angles[:8]):
            seed_node = ResearchNode(
                id=f"{req_id}-seed{i}",
                question=q,
                context=ctx,
                depth=0,
                pressure=0.9,
                parent_id=root.id,
            )
            nodes_by_id[seed_node.id] = seed_node
            all_questions.append(q)
            await pending.put(seed_node)
            total_queued += 1

        log.info(
            f"[{req_id}] Seeded {len(seed_angles)} comprehension-guided research angles"
        )
    except Exception as e:
        log.warning(f"[{req_id}] Pre-seed decomposition failed (non-fatal): {e}")

    progress.append(
        f"\n**[Phase 2: Tree Research Reactor]** "
        f"(max {TREE_MAX_CONCURRENT} concurrent, "
        f"depth limit {TREE_MAX_DEPTH}, "
        f"node budget {TREE_MAX_NODES})\n"
    )
    progress.append(f"Root question: {user_query}\n")

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
                        condition_store=condition_store,
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
                    # 1. Regular sub-questions from LLM (includes
                    #    "verify" strategy now)
                    children = await _spawn_sub_questions(
                        node, conditions, user_query, all_questions, req_id,
                        condition_store=condition_store,
                    )

                    # 2. Auto-spawn verification nodes for concrete
                    #    entities discovered in this node's findings.
                    #    This ensures every vendor/person/product gets
                    #    cross-referenced even if the LLM doesn't
                    #    explicitly ask for it.
                    #    Respects TREE_MAX_DEPTH to prevent unbounded
                    #    depth cascading from verification → verification.
                    if node.depth < TREE_MAX_DEPTH:
                        try:
                            entities = await _extract_entities_for_verification(
                                conditions, req_id,
                            )
                            if entities:
                                verify_children = _spawn_verification_nodes(
                                    entities, node, all_questions, req_id,
                                )
                                children.extend(verify_children)
                                log.info(
                                    f"[{req_id}] Auto-spawned {len(verify_children)} "
                                    f"verification nodes for {len(entities)} entities"
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
    }

