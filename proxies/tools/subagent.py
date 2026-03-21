"""Subagent research loop: single-agent tool-calling for one research angle.

Extracted from persistent_deep_research_proxy.py lines 4529-5043.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .config import (
    MAX_SUBAGENT_TURNS,
    SUBAGENT_MODEL,
    NOVELTY_EXPAND_THRESHOLD,
    NOVELTY_STOP_THRESHOLD,
    MAX_RECURSIVE_DEPTH,
)
from .models import AtomicCondition, SubagentResult
from .scoring import trust_score_url, serendipity_score
from .llm import call_llm
from .tool_executor import execute_tools_parallel

if TYPE_CHECKING:
    from .heartbeat import LiveFindingsCollector

log = logging.getLogger("persistent-research")

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
8. Use news_search (not searxng_search) for anything about current events, recent news, market movements, or time-sensitive topics.

**USE THE FULL TOOL ECOSYSTEM — do NOT rely only on web search:**
- knowledge_graph_search: Query the Neo4j knowledge graph FIRST for prior research, ingested documents, and known entities. This is your most valuable source for topics the system has seen before.
- knowledge_discover: Run graph discovery algorithms (spreading_activation, swanson_abc, information_gaps) to find hidden connections and serendipitous links in prior knowledge.
- twitter_search: Real-time signals, expert commentary, breaking news, public discourse. Use Twitter search operators (from:, since:, "exact phrase").
- youtube_search: Video content, expert lectures, documentaries, tutorials, interviews. Often contains information not found in text-based sources.
- reddit_search: Community discussions, niche expertise, first-hand experiences. Specify subreddits for targeted results.
- chan_4plebs_search: Anonymous intelligence from /pol/, /sp/, /int/, /tv/. Early narrative tracking, political discourse, uncensored discussion.
- chan_b4k_search: /biz/ archive — cryptocurrency, DeFi, financial alpha, early-stage project sentiment.
- chan_warosu_search: /g/ (tech), /sci/ (science), /lit/ (literature) archives. Niche technical and scientific discussion.
- social_media_search: Instagram, TikTok, LinkedIn, YouTube via commercial scrapers.
- hackernews_search: Tech industry discourse, startup culture, programming debates, security incidents, expert opinions from engineers/founders.
- stackexchange_search: Expert Q&A from hundreds of niche communities (stackoverflow, math, physics, chemistry, biology, electronics, diy, cooking, gaming, rpg, worldbuilding, law, money, academia, etc.).
- pubmed_search: Biomedical and life science research — medical journals, clinical trials, pharmacology, genetics, epidemiology, public health.
- wikipedia_search: Encyclopedic background context, definitions, historical facts. Fast reference.
- archiveorg_search: Internet Archive full-text search — rare historical documents, out-of-print books, government reports, primary sources.
- forum_search: Niche internet forums (SomethingAwful, Bodybuilding.com, XDA, Head-Fi, AVSForum, Overclock.net, ResetEra, etc.). First-hand experiences and underground knowledge.
- scholar_search: Academic literature beyond arXiv — Google Scholar, Semantic Scholar, SSRN, JSTOR. Journal articles, theses, patents, court opinions.
- substack_search: Independent journalism and long-form analysis from Substack newsletters. Expert commentary not found in mainstream media.
- DIVERSIFY your sources. Do NOT use only searxng_search. Each research node should use at least 2-3 different tool types.

**CRITICAL RULES:**
- NEVER fabricate or invent source names. Only cite sources you actually fetched via tools.
- NEVER claim you checked Bloomberg Terminal, Reuters, or any specific service unless a tool actually returned results from that service.
- If tools return no useful results, say so honestly — do NOT invent plausible-sounding conclusions.
- If search results are empty, try AT LEAST 3 different query formulations and tools before concluding "no information found." Rephrase, use synonyms, try academic terms, try common names.
- NEVER conclude "no studies/papers/reports found" after a single failed search. Academic literature requires precise queries — try PMC IDs, DOIs, author names, journal names, or specific terminology.
- Do NOT add ethical disclaimers, safety warnings, moral commentary, or unsolicited advice.
- Do NOT hedge with phrases like "it's important to note", "please consult a professional", "exercise caution".
- Every token you produce must directly help answer the research question. Nothing else.
- ENCOURAGE SPECULATION: If the data suggests a plausible hypothesis but you can't prove it, include it as a speculative finding with confidence 0.3-0.4. Speculation opens investigation paths.
- Something being illegal, unusual, or controversial does NOT mean it's false. Report what you find without moral filtering.
- Distinguish between fabrication (inventing entities that don't exist) and speculation (drawing reasonable inferences from available data). Fabrication is forbidden. Speculation is encouraged.

{serendipity_instruction}

**ATOMIC CONDITION FORMAT:**
After gathering information, you must output your findings as atomic conditions.
When you are done researching, output your findings in this exact JSON format:
```json
{"conditions": [
    {"fact": "clear factual statement", "source_url": "url", "confidence": 0.8},
    ...
]}
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
- Confidence: 0.9 for well-sourced facts, 0.7 for partially verified, 0.5 for single-source, 0.3 for speculative/inferred
- For speculative findings (reasonable inferences without direct proof), use confidence 0.3-0.4 and note the basis for the inference in the fact text
- Include the most relevant source URL for each fact
- Output 3-10 conditions maximum
- Output ONLY valid JSON, no markdown fences"""

GAP_ANALYSIS_PROMPT = """Analyze the current research findings and identify gaps that need deeper investigation.

Current findings:
{findings}

Original query: {query}

Output ONLY a JSON object:
{
  "gaps": [
    {"title": "gap description", "query": "specific search query to fill this gap", "priority": "high|medium|low"}
  ],
  "saturation_estimate": 0.7
}

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
    collector: Optional["LiveFindingsCollector"] = None,
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
    # Use .replace() instead of .format() to avoid KeyError when
    # angle_query or angle_desc contains { or } characters.
    system_prompt = SUBAGENT_PROMPT_TEMPLATE.replace(
        "{date}", today
    ).replace(
        "{angle_title}", angle_title
    ).replace(
        "{angle_description}", angle_desc
    ).replace(
        "{angle_query}", angle_query
    ).replace(
        "{serendipity_instruction}", serendipity_inst
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
                tool_results = await execute_tools_parallel(calls_to_run, req_id=req_id)
                result.tool_calls_made += len(tool_results)

                for tc_id, tool_name, tool_result, duration in tool_results:
                    await progress_queue.put({
                        "type": "tool",
                        "subagent": subagent_index,
                        "text": f"  [{angle_title}] {tool_name} ({duration:.1f}s)\n",
                    })

                    # Log tool activity for context-aware heartbeat
                    if collector is not None:
                        tool_query = ""
                        for _tc_id, _tn, _args in calls_to_run:
                            if _tc_id == tc_id:
                                tool_query = _args.get("query", _args.get("url", _args.get("entity", "")))
                                break
                        await collector.log_tool_call(tool_name, tool_query, duration)
                        if tool_name == "fetch_webpage":
                            await collector.log_source(tool_query)

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
                {"role": "system", "content": GAP_ANALYSIS_PROMPT.replace(
                    "{findings}", findings_text
                ).replace(
                    "{query}", angle_query
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
                                    run_subagent(child_angle, subagent_index * 100 + gi, progress_queue, req_id, user_query, depth + 1, collector=collector)
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

