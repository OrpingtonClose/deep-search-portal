"""Tree-based research reactor: parallel exploration with pressure-based spawning.

Extracted from persistent_deep_research_proxy.py lines 5046-5486.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

from .config import (
    SUBAGENT_MODEL,
    TREE_MAX_CONCURRENT,
    TREE_MAX_DEPTH,
    TREE_MAX_NODES,
    TREE_PRESSURE_THRESHOLD,
    TREE_WORKER_IDLE_TIMEOUT,
)
from .models import AtomicCondition, ResearchNode, SubagentResult
from .llm import call_llm
from .subagent import run_subagent

if TYPE_CHECKING:
    from .heartbeat import LiveFindingsCollector

log = logging.getLogger("persistent-research")

# ============================================================================
# Tree Research Reactor
# ============================================================================

SPAWN_QUESTIONS_PROMPT = """You are a research strategist who generates focused follow-up questions.

Given the findings so far, generate follow-up questions that help answer the ORIGINAL USER QUERY more completely. Diversity is good, but every question must be directly useful for answering what the user actually asked.

**Original user query:** {user_query}
**Question just investigated:** {node_question}
**Context:** {node_context}

**Findings from this investigation:**
{findings_text}

**Questions already in the research tree (avoid duplicates):**
{existing_questions}

Generate follow-up questions. For each, provide:
- "question": a specific, searchable question
- "context": one sentence on why this matters
- "pressure": 0.0-1.0 importance score (1.0 = critical gap, 0.1 = minor curiosity)
- "strategy": one of "deepen" | "lateral" | "contrarian" | "historical" | "cross-domain"

STRATEGY RULES:
- "deepen": drill further into a specific finding (preferred — most questions should be this)
- "lateral": explore a related angle that DIRECTLY helps answer the original query from a different perspective
- "contrarian": investigate the opposite claim or a dissenting viewpoint ON THE SAME TOPIC
- "historical": look at historical precedents directly relevant to the query
- "cross-domain": ONLY use if there is a genuinely useful parallel — do NOT force random associations
- Non-deepen strategies are optional. Only use them if they genuinely serve the user's question.
- CRITICAL: "lateral" does NOT mean "free association with a keyword". If the user asks about buying insulin, a lateral question is about alternative purchasing channels — NOT about bodybuilding or side effects.

PRESSURE RULES:
- Higher pressure for: contradictions, unverified claims, critical gaps directly relevant to the query
- Lower pressure for: already-well-covered areas, tangential topics
- 0 questions is fine if the topic is well-covered

Other rules:
- Generate 0-5 questions maximum
- Do NOT repeat questions already in the tree
- Output ONLY valid JSON, no markdown fences

Output format:
{"sub_questions": [{"question": "...", "context": "...", "pressure": 0.8, "strategy": "lateral"}]}"""


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

    # Use .replace() instead of .format() to avoid KeyError when
    # web-scraped findings_text contains { or } characters.
    prompt = SPAWN_QUESTIONS_PROMPT.replace(
        "{user_query}", user_query
    ).replace(
        "{node_question}", node.question
    ).replace(
        "{node_context}", node.context
    ).replace(
        "{findings_text}", findings_text
    ).replace(
        "{existing_questions}", existing_text
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

    # --- Pre-seed: decompose into parallel initial angles ---
    # Generate 3-5 initial research angles so workers start in parallel
    # instead of waiting for the single root node to finish first.
    try:
        seed_prompt = (
            f"Decompose this research query into 3-5 DISTINCT research angles "
            f"that can be investigated IN PARALLEL. Each angle should cover a "
            f"different aspect, perspective, or source type.\n\n"
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
            for i, angle in enumerate(seed_data.get("angles", [])[:5]):
                q = angle.get("question", "").strip()
                if not q or q.lower() == user_query.lower():
                    continue
                seed_node = ResearchNode(
                    id=f"{req_id}-seed{i}",
                    question=q,
                    context=angle.get("context", ""),
                    depth=0,
                    pressure=0.9,
                    parent_id=root.id,
                )
                nodes_by_id[seed_node.id] = seed_node
                all_questions.append(q)
                await pending.put(seed_node)
                total_queued += 1
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

    return {
        "subagent_results": all_results,
        "all_conditions": all_conditions,
        "total_turns": total_turns,
        "total_tools": total_tools,
        "total_children": total_children,
        "progress_log": progress,
    }

