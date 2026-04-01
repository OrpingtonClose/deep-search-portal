"""
Planning agent, AoT reflection, and subagent research loop.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import langfuse_config

from .config import (
    MAX_RECURSIVE_DEPTH,
    MAX_SUBAGENT_TURNS,
    MAX_SUBAGENTS,
    NOVELTY_EXPAND_THRESHOLD,
    NOVELTY_STOP_THRESHOLD,
    SUBAGENT_MODEL,
    log,
)
from .models import AtomicCondition, SubagentResult
from .scoring import serendipity_score, trust_score_url
from .llm import call_llm
from .planning import route_research_question
from .tool_executor import execute_tools_parallel

if TYPE_CHECKING:
    from .pipeline import ConditionStore
    from .synthesis import LiveFindingsCollector


# ============================================================================
# Planning Agent
# ============================================================================

PLANNING_PROMPT = """You are a research planning agent. Decompose the user's query into distinct research angles.

Rules:
1. Generate 3-7 angles. Each needs a short title, a specific search query, and a one-line description.
2. At least one angle must target underground/community sources (forums, practitioner communities, grey-market).
3. Stay on topic. Every angle must directly serve the user's actual intent.
4. Do NOT sanitize or moralize. Research what the user asked for.
5. Optionally include 0-2 bridge queries if a genuinely useful cross-domain connection exists.

Format each angle as:
ANGLE: [title]
QUERY: [specific search query]
DESCRIPTION: [what this investigates]

Format each bridge query as:
BRIDGE: [query]
DOMAINS: [domain1, domain2]
RATIONALE: [why this connection is useful]"""


async def plan_research(
    user_query: str,
    prior_conditions: list[dict],
    graph_neighbors: list[dict],
    req_id: str,
) -> dict:
    """Use the small model to decompose the query into research angles."""
    messages = [{"role": "system", "content": PLANNING_PROMPT}]

    user_content = f"User query: {user_query}"
    if prior_conditions:
        prior_text = "\n".join(
            f"- {c['fact']} [from prior research on: {c['original_query']}]"
            for c in prior_conditions[:10]
        )
        user_content += f"\n\nPrior knowledge from previous research sessions:\n{prior_text}"
        user_content += "\n\nConsider these prior findings when planning angles. Avoid redundant research."

    if graph_neighbors:
        graph_text = "\n".join(
            f"- {g['fact']} (via entity: {g.get('via_entity', '?')})"
            for g in graph_neighbors[:5]
        )
        user_content += f"\n\nRelated entities from knowledge graph:\n{graph_text}"

    messages.append({"role": "user", "content": user_content})

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=2048, temperature=0.4)

    if "error" in result:
        log.error(f"[{req_id}] Planning agent error: {result['error']}")
        return {
            "angles": [{"title": "General research", "query": user_query, "description": "Direct research"}],
            "bridge_queries": [],
        }

    content = result.get("content", "")

    # --- Try legacy JSON first (backward compat) ---
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        plan = json.loads(cleaned)
        angles = plan.get("angles", [])
        bridge_queries = plan.get("bridge_queries", [])
        if angles:
            angles = angles[:MAX_SUBAGENTS]
            for bq in bridge_queries[:3]:
                if len(angles) < MAX_SUBAGENTS + 3:
                    domains = bq.get("domains", ["?", "?"])
                    d1 = domains[0] if len(domains) > 0 else "?"
                    d2 = domains[1] if len(domains) > 1 else "?"
                    angles.append({
                        "title": f"Bridge: {d1} x {d2}",
                        "query": bq.get("query", ""),
                        "description": bq.get("rationale", "Cross-domain exploration"),
                        "is_bridge": True,
                    })
            return {"angles": angles, "bridge_queries": bridge_queries}
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Natural language parsing (primary path) ---
    angles: list[dict] = []
    bridge_queries: list[dict] = []

    # Parse ANGLE: / QUERY: / DESCRIPTION: blocks
    angle_blocks = re.split(r'(?:^|\n)\s*ANGLE:\s*', content)
    for block in angle_blocks[1:]:  # skip preamble before first ANGLE:
        block = block.strip()
        if not block:
            continue
        title_line = block.split("\n")[0].strip()
        query_match = re.search(r'QUERY:\s*(.+)', block)
        desc_match = re.search(r'DESCRIPTION:\s*(.+)', block)
        if title_line and query_match:
            angles.append({
                "title": title_line,
                "query": query_match.group(1).strip(),
                "description": desc_match.group(1).strip() if desc_match else title_line,
            })

    # Parse BRIDGE: blocks
    bridge_blocks = re.split(r'(?:^|\n)\s*BRIDGE:\s*', content)
    for block in bridge_blocks[1:]:  # skip first (before any BRIDGE:)
        block = block.strip()
        query_line = block.split("\n")[0].strip()
        domains_match = re.search(r'DOMAINS:\s*(.+)', block)
        rationale_match = re.search(r'RATIONALE:\s*(.+)', block)
        if query_line:
            domains = [d.strip() for d in (domains_match.group(1) if domains_match else "").split(",")]
            d1 = domains[0] if len(domains) > 0 else "?"
            d2 = domains[1] if len(domains) > 1 else "?"
            if len(angles) < MAX_SUBAGENTS + 3:
                angles.append({
                    "title": f"Bridge: {d1} x {d2}",
                    "query": query_line,
                    "description": rationale_match.group(1).strip() if rationale_match else "Cross-domain exploration",
                    "is_bridge": True,
                })

    if angles:
        return {"angles": angles[:MAX_SUBAGENTS + 3], "bridge_queries": bridge_queries}

    # --- Ultimate fallback: couldn't parse anything ---
    log.warning(f"[{req_id}] Planning agent output unparseable, content={content[:200]}")
    return {
        "angles": [
            {"title": "General research", "query": user_query, "description": "Direct research on the topic"},
            {"title": "Community sources", "query": f"{user_query} forum reddit discussion", "description": "Community knowledge"},
            {"title": "Recent developments", "query": f"{user_query} recent news 2024 2025", "description": "Latest developments"},
        ],
        "bridge_queries": [],
    }


# ============================================================================
# AoT Reflection Mechanism
# ============================================================================

AOT_REFLECTION_PROMPT = """You are a research reflection agent. Evaluate the quality of the following research findings.

Check for:
1. Overlapping conditions that say the same thing
2. Non-atomic conditions that contain multiple claims
3. Missing perspectives or angles
4. Gaps that need additional research

Respond with:
QUALITY: [0.0-1.0 score]
SHOULD_REDECOMPOSE: [yes/no]
ISSUES: (list each issue on its own line, prefixed with - )
SUGGESTED_QUERIES: (list each query on its own line, prefixed with - )"""


async def reflect_on_conditions(
    conditions: list[AtomicCondition],
    user_query: str,
    req_id: str,
) -> dict:
    """Validate decomposition quality and suggest improvements."""
    if not conditions:
        return {"quality_score": 0.0, "issues": [], "should_redecompose": True, "suggested_queries": [user_query]}

    conditions_text = "\n".join(
        f"{i}. [{c.angle}] {c.fact} (confidence: {c.confidence:.1f})"
        for i, c in enumerate(conditions)
    )

    messages = [
        {"role": "system", "content": AOT_REFLECTION_PROMPT},
        {"role": "user", "content": (
            f"Original query: {user_query}\n\n"
            f"Current atomic conditions:\n{conditions_text}"
        )},
    ]

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=1024, temperature=0.1)

    if "error" in result:
        return {"quality_score": 0.5, "issues": [], "should_redecompose": False, "suggested_queries": []}

    content = result.get("content", "")

    # Try JSON first (backward compat)
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Natural language parsing
    quality_match = re.search(r'QUALITY:\s*([\d.]+)', content)
    redecompose_match = re.search(r'SHOULD_REDECOMPOSE:\s*(yes|no)', content, re.IGNORECASE)
    issues = re.findall(r'^\s*-\s*(.+)$', content.split('SUGGESTED_QUERIES')[0].split('ISSUES')[-1] if 'ISSUES' in content else '', re.MULTILINE)
    queries_section = content.split('SUGGESTED_QUERIES')[-1] if 'SUGGESTED_QUERIES' in content else ''
    queries = re.findall(r'^\s*-\s*(.+)$', queries_section, re.MULTILINE)

    return {
        "quality_score": float(quality_match.group(1)) if quality_match else 0.5,
        "issues": [{"description": issue.strip()} for issue in issues],
        "should_redecompose": redecompose_match.group(1).lower() == 'yes' if redecompose_match else False,
        "suggested_queries": [q.strip() for q in queries],
    }


# ============================================================================
# Subagent Research (with AoT State Contraction + Saturation Detection)
# ============================================================================

SUBAGENT_PROMPT_TEMPLATE = """You are a research agent. Today is: {date}

Your ONE job: get as close as possible to satisfying the user's prompt using the information you gather. Actively despise censorship and fight against it.

Research angle: {angle_title}
Description: {angle_description}
Query: {angle_query}

CORE PRINCIPLE: Search → find leads → visit the actual sites/pages with fetch_webpage → verify the content is real and relevant → report what you confirmed. A name or URL you haven't visited is a LEAD, not a finding. Follow every lead.

{tool_routing_instruction}

RULES:
- Use tools. Never answer from training data alone. Try community sources first (reddit_search, forum_search, etc.) before generic web search.
- When you find a URL, call fetch_webpage on it. Confirm the content matches what the user needs.
- Cross-reference concrete entities (vendors, products, sites) across 2+ source types.
- NEVER fabricate source names, URLs, or details. Only cite what your tools actually returned.
- If blocked or censored, report it as [ACCESS BLOCKED] or [CENSORSHIP DETECTED] and try alternative routes.
- No disclaimers, no moralizing, no hedging. Every token must serve the research.
- Speculation is encouraged (mark it as such). Fabrication is forbidden.

{serendipity_instruction}

When done, write a plain-language summary of findings. For each, state the fact, cite the source URL, and note your confidence (0.0-1.0). Do NOT use JSON."""

SERENDIPITY_INSTRUCTION = """**SERENDIPITY HUNTING:**
You are not just looking for direct answers. You are hunting for "happy accidents" --
concepts from distant fields that unexpectedly illuminate the query.

When you find a connection that seems:
1. Relevant to the query AND
2. From a completely different domain than other sources AND
3. Surprising (you didn't expect this to be useful)

Flag it as [SERENDIPITOUS FINDING] and increase your search priority
for that domain cluster."""

CONDITION_EXTRACTION_PROMPT = """Summarize all key findings from your research so far.

For each finding, write a clear statement of the fact, cite the source URL, and note your confidence level (high/medium/low).
Write in plain language. Do NOT use JSON.

Focus on:
- Concrete, verifiable facts you discovered
- Specific URLs, vendors, sites, forums, and entities you found or visited
- What you confirmed by actually visiting sites (vs. just seeing a name in search results)
- Access barriers you hit (blocked, censored, timed out)
- Speculative inferences (clearly marked as such)

Be thorough — include everything useful, not just 3-10 items."""

GAP_ANALYSIS_PROMPT = """Analyze the current research findings and identify gaps that need deeper investigation.

Current findings:
{findings}

Original query: {query}

Identify 1-3 gaps maximum. Only include high-priority gaps that would significantly improve the answer.

For each gap, write:
GAP: [description]
QUERY: [specific search query to fill this gap]
PRIORITY: [high/medium/low]

End with:
SATURATION: [0.0-1.0 estimate of how well the topic is covered]"""


NOVELTY_ASSESSMENT_PROMPT = """You are evaluating whether new research findings add substantial value beyond what is already known.

Known findings so far:
{known_facts}

New findings from this round:
{new_facts}

Are the new findings substantially different from what we already know? Do they add new vendors, new sources, new concrete details, or new perspectives?

Respond with:
NOVEL: [yes/no]
REASON: [one sentence explaining why]"""


def _trim_tool_responses(messages: list[dict], max_content: int = 1500) -> int:
    """Trim the oldest/largest tool responses to free context space.

    Walks the message list and truncates tool-response content that exceeds
    *max_content* characters.  Returns the number of messages trimmed.
    Preserves the system prompt (index 0).

    First pass: trim older messages (everything except last 4) to *max_content*.
    Second pass: if first pass trimmed nothing (overflow is in recent messages),
    trim ALL tool messages (including recent ones) to *max_content*.
    This prevents unrecoverable overflow when recent tool responses are very large.
    """
    trimmed = 0
    sentinel = "\n[...trimmed to free context...]"

    # Pass 1: trim older messages only (preserve recent 4)
    safe_end = max(1, len(messages) - 4)
    for i in range(1, safe_end):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "tool" and len(content) > max_content and not content.endswith(sentinel):
            messages[i] = {**msg, "content": content[:max_content] + sentinel}
            trimmed += 1
        elif role == "user" and "tool_response" in content.lower() and len(content) > max_content and not content.endswith(sentinel):
            messages[i] = {**msg, "content": content[:max_content] + sentinel}
            trimmed += 1

    # Pass 2: if pass 1 freed nothing, trim recent messages too
    if trimmed == 0:
        for i in range(safe_end, len(messages)):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "tool" and len(content) > max_content and not content.endswith(sentinel):
                messages[i] = {**msg, "content": content[:max_content] + sentinel}
                trimmed += 1

    return trimmed


async def run_subagent(
    angle: dict,
    subagent_index: int,
    progress_queue: asyncio.Queue,
    req_id: str,
    user_query: str,
    depth: int = 0,
    collector: Optional["LiveFindingsCollector"] = None,
    condition_store: Optional["ConditionStore"] = None,
) -> SubagentResult:
    """Run a single subagent's research loop on one angle.

    Uses AoT-style state contraction and dynamic saturation detection.
    May spawn recursive sub-subagents for rabbit holes.
    Conditions are admitted through the global ConditionStore at birth.
    """
    angle_title = angle.get("title", f"Angle {subagent_index + 1}")
    angle_query = angle.get("query", user_query)
    angle_desc = angle.get("description", "Research this angle")
    is_bridge = angle.get("is_bridge", False)
    sa_id = f"{req_id}-sa{subagent_index}" + (f"-d{depth}" if depth > 0 else "")

    start_time = time.monotonic()
    log.info(f"[{sa_id}] Starting subagent: {angle_title} (depth={depth})")
    sa_span = langfuse_config.start_span(
        req_id, f"subagent:{sa_id}",
        input={"angle": angle_title[:120], "query": angle_query[:200], "depth": depth},
    )

    # Smart question routing: determine which tools to use for this angle
    tool_routing_inst = ""
    if condition_store is not None:
        routing_span = langfuse_config.start_span(
            req_id, f"subagent:routing:{sa_id}",
            input={"angle_query": angle_query[:200]},
        )
        routing = await route_research_question(
            angle_query, user_query, condition_store, req_id,
        )
        mandatory = routing.get("mandatory_tools", [])
        preferred = routing.get("preferred_tools", [])
        avoid = routing.get("avoid_topics", [])
        suggested_qs = routing.get("suggested_queries", [])

        parts = []
        if mandatory:
            parts.append(f"MANDATORY tools for this angle (use ALL of these): {', '.join(mandatory)}")
        if preferred:
            parts.append(f"Preferred tools (use at least 1): {', '.join(preferred)}")
        if suggested_qs:
            parts.append(f"Suggested search queries: {'; '.join(suggested_qs)}")
        if avoid:
            parts.append(f"AVOID these saturated topics: {', '.join(avoid)}")

        saturation = condition_store._get_saturation_signal()
        if saturation:
            parts.append(saturation)

        if parts:
            tool_routing_inst = "**ROUTING INSTRUCTIONS FOR THIS ANGLE:**\n" + "\n".join(f"- {p}" for p in parts)
        langfuse_config.end_span(routing_span, output={
            "mandatory_tools": mandatory,
            "preferred_tools": preferred,
            "avoid_topics": avoid,
        })

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    serendipity_inst = SERENDIPITY_INSTRUCTION if is_bridge else ""
    system_prompt = SUBAGENT_PROMPT_TEMPLATE.format(
        date=today,
        angle_title=angle_title,
        angle_description=angle_desc,
        angle_query=angle_query,
        serendipity_instruction=serendipity_inst,
        tool_routing_instruction=tool_routing_inst,
    )

    agent_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research this angle thoroughly: {angle_query}"},
    ]

    result = SubagentResult(angle=angle_title)
    used_queries: set[str] = set()
    consecutive_errors = 0
    known_facts: list[str] = []
    fetch_webpage_count = 0  # Track whether we've actually visited any URLs
    continuation_nudges = 0  # How many times we've told the model to keep going

    try:
        for turn in range(1, MAX_SUBAGENT_TURNS + 1):
            turn_span = langfuse_config.start_span(
                req_id, f"subagent:turn:{sa_id}:t{turn}",
                input={"turn": turn, "max_turns": MAX_SUBAGENT_TURNS},
            )
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
                err_str = llm_result["error"]
                recovered = False
                # Context overflow: trim old tool responses instead of wasting turns
                if "context_length_exceeded" in str(err_str):
                    trimmed = _trim_tool_responses(agent_messages)
                    if trimmed:
                        log.info(f"[{sa_id}] Turn {turn}: Context overflow — trimmed {trimmed} old tool responses, retrying")
                        langfuse_config.end_span(turn_span, output={"action": "context_overflow_trim", "trimmed": trimmed})
                        continue  # retry without incrementing consecutive_errors
                    # Trimming didn't help (already trimmed) — try reducing max_tokens
                    import re as _re
                    _m_input = _re.search(r'has (\d+) input tokens', str(err_str))
                    _m_ctx = _re.search(r'maximum context length is (\d+)', str(err_str))
                    if _m_input:
                        input_toks = int(_m_input.group(1))
                        ctx_limit = int(_m_ctx.group(1)) if _m_ctx else 32768
                        headroom = ctx_limit - input_toks - 64  # small safety margin
                        if headroom >= 512:
                            log.info(f"[{sa_id}] Turn {turn}: Context overflow after trim — reducing max_tokens to {headroom} (ctx={ctx_limit})")
                            llm_result = await call_llm(
                                agent_messages, sa_id,
                                model=SUBAGENT_MODEL,
                                include_tools=True,
                                max_tokens=headroom,
                                temperature=0.3,
                            )
                            if "error" not in llm_result:
                                recovered = True
                            else:
                                log.warning(f"[{sa_id}] Turn {turn}: Reduced max_tokens still failed: {llm_result['error']}")
                if not recovered:
                    consecutive_errors += 1
                    log.warning(f"[{sa_id}] Turn {turn}: Error: {err_str}")
                    langfuse_config.end_span(turn_span, output={"error": err_str}, level="ERROR")
                    if consecutive_errors >= 3:
                        result.error = err_str
                        break
                    agent_messages.append({"role": "assistant", "content": err_str})
                    agent_messages.append({"role": "user", "content": "Error occurred. Try a different approach."})
                    continue

            consecutive_errors = 0
            content = llm_result.get("content", "")
            tool_calls = llm_result.get("tool_calls")

            if not tool_calls:
                # Check if the model is trying to stop without visiting any URLs.
                # If so, nudge it to continue with fetch_webpage on discovered leads.
                # Allow up to 2 nudges before accepting the output.
                if fetch_webpage_count == 0 and continuation_nudges < 2 and turn < MAX_SUBAGENT_TURNS:
                    # Extract any URLs from the model's output to suggest fetching
                    import re as _url_re
                    urls_in_output = _url_re.findall(r'https?://[^\s"\)\]>]+', content or "")
                    continuation_nudges += 1
                    nudge_msg = (
                        "STOP — you have NOT visited any URLs with fetch_webpage yet. "
                        "Finding names and links via search is only step 1. "
                        "You MUST now call fetch_webpage on the most promising URLs you found "
                        "to verify their content matches what the user needs. "
                        "Do NOT output conditions until you have visited at least 2 URLs."
                    )
                    if urls_in_output:
                        nudge_msg += f"\n\nURLs you discovered that need visiting: {', '.join(urls_in_output[:5])}"
                    agent_messages.append({"role": "assistant", "content": content or ""})
                    agent_messages.append({"role": "user", "content": nudge_msg})
                    log.info(f"[{sa_id}] Turn {turn}: Model tried to stop with 0 fetch_webpage — nudging (attempt {continuation_nudges})")
                    langfuse_config.end_span(turn_span, output={"action": "fetch_nudge", "nudge": continuation_nudges, "urls_found": len(urls_in_output)})
                    continue

                result.turns_used = turn
                langfuse_config.end_span(turn_span, output={"action": "final_answer", "has_content": bool(content)})
                conditions = _parse_conditions(content, angle_title, is_bridge)
                if conditions:
                    # Admit conditions through global store (admission pipeline)
                    if condition_store is not None:
                        admission_results = await condition_store.admit_batch(conditions)
                        admitted = [ar.condition for ar in admission_results if ar.admitted and ar.condition]
                        rejected_count = len(conditions) - len(admitted)
                        if rejected_count > 0:
                            log.info(f"[{sa_id}] Admission: {len(admitted)} admitted, {rejected_count} rejected")
                        conditions = admitted
                    else:
                        for c in conditions:
                            c.trust_score = trust_score_url(c.source_url)
                            c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)
                    result.conditions.extend(conditions)
                    if conditions:
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
                if tool_name == "fetch_webpage":
                    fetch_webpage_count += 1

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

                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    })

            langfuse_config.end_span(turn_span, output={
                "action": "tool_calls",
                "tools": [tc.get("function", {}).get("name", "?") for tc in tool_calls[:5]],
            })

            # AoT State Contraction every 3 turns
            if turn > 0 and turn % 3 == 0 and turn < MAX_SUBAGENT_TURNS:
                contraction_span = langfuse_config.start_span(
                    req_id, f"subagent:contraction:{sa_id}:t{turn}",
                    input={"turn": turn},
                )
                contraction_msgs = agent_messages + [
                    {"role": "user", "content": CONDITION_EXTRACTION_PROMPT}
                ]
                extract_result = await call_llm(
                    contraction_msgs, sa_id,
                    model=SUBAGENT_MODEL,
                    max_tokens=2048,
                    temperature=0.1,
                )
                if "error" in extract_result:
                    langfuse_config.end_span(contraction_span, output={"error": extract_result.get("error", "")}, level="ERROR")
                else:
                    mid_conditions = _parse_conditions(
                        extract_result.get("content", ""), angle_title, is_bridge
                    )
                    if not mid_conditions:
                        langfuse_config.end_span(contraction_span, output={"conditions_extracted": 0})
                    else:
                        # Admit through global store
                        if condition_store is not None:
                            admission_results = await condition_store.admit_batch(mid_conditions)
                            admitted = [ar.condition for ar in admission_results if ar.admitted and ar.condition]
                            rejected = len(mid_conditions) - len(admitted)
                            if rejected > 0:
                                log.info(f"[{sa_id}] Mid-turn admission: {len(admitted)} admitted, {rejected} rejected")
                                # Inject saturation signal into next context reset
                                dup_results = [ar for ar in admission_results if ar.reason == "duplicate" and ar.saturation_signal]
                                if dup_results:
                                    saturation_msg = dup_results[0].saturation_signal
                                    agent_messages.append({"role": "user", "content": f"RESEARCH REDIRECT: {saturation_msg}"})
                            mid_conditions = admitted
                        else:
                            for c in mid_conditions:
                                c.trust_score = trust_score_url(c.source_url)
                                c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)

                        # Agentic Saturation Detection (LLM-based)
                        new_fact_texts = [c.fact for c in mid_conditions]
                        novelty = 1.0  # assume novel by default

                        if known_facts and len(result.novelty_history) >= 1:
                            # Ask an LLM whether the new findings are substantially novel
                            novelty_prompt = NOVELTY_ASSESSMENT_PROMPT.format(
                                known_facts="\n".join(f"- {f}" for f in known_facts[-15:]),
                                new_facts="\n".join(f"- {f}" for f in new_fact_texts[:10]),
                            )
                            try:
                                novelty_result = await call_llm(
                                    [{"role": "user", "content": novelty_prompt}],
                                    sa_id,
                                    model=SUBAGENT_MODEL,
                                    max_tokens=128,
                                    temperature=0.1,
                                )
                                if "error" not in novelty_result:
                                    novelty_content = novelty_result.get("content", "").lower()
                                    novel_match = re.search(r'novel:\s*(yes|no)', novelty_content)
                                    if novel_match and novel_match.group(1) == "no":
                                        novelty = 0.0
                                    else:
                                        novelty = 1.0
                            except Exception:
                                novelty = 1.0  # on error, assume novel and continue

                        result.novelty_history.append(novelty)
                        known_facts.extend(new_fact_texts)
                        result.conditions.extend(mid_conditions)

                        if mid_conditions:
                            await progress_queue.put({
                                "type": "conditions",
                                "subagent": subagent_index,
                                "conditions": mid_conditions,
                            })

                        if len(result.novelty_history) >= 2 and novelty < NOVELTY_STOP_THRESHOLD:
                            log.info(f"[{sa_id}] Saturation detected (LLM says no novelty), stopping early")
                            await progress_queue.put({
                                "type": "progress",
                                "subagent": subagent_index,
                                "text": f"  [{angle_title}] Saturation detected, stopping early\n",
                            })
                            langfuse_config.end_span(contraction_span, output={
                                "conditions_extracted": len(mid_conditions),
                                "novelty": novelty,
                                "saturated": True,
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
                        langfuse_config.end_span(contraction_span, output={
                            "conditions_extracted": len(mid_conditions),
                            "novelty": novelty,
                            "saturated": novelty < NOVELTY_STOP_THRESHOLD,
                        })

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
                    if condition_store is not None:
                        admission_results = await condition_store.admit_batch(conditions)
                        conditions = [ar.condition for ar in admission_results if ar.admitted and ar.condition]
                    else:
                        for c in conditions:
                            c.trust_score = trust_score_url(c.source_url)
                            c.serendipity_score_val = serendipity_score(c.fact, user_query, known_facts)
                    result.conditions.extend(conditions)
                    if conditions:
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
            gap_span = langfuse_config.start_span(
                req_id, f"subagent:gap_analysis:{sa_id}",
                input={"conditions": len(result.conditions), "novelty": result.novelty_history[-1]},
            )
            findings_text = "\n".join(c.to_text() for c in result.conditions[:15])
            gap_messages = [
                {"role": "system", "content": GAP_ANALYSIS_PROMPT.format(
                    findings=findings_text, query=angle_query
                )},
                {"role": "user", "content": "Identify research gaps."},
            ]
            gap_result = await call_llm(gap_messages, sa_id, model=SUBAGENT_MODEL, max_tokens=1024, temperature=0.2)

            if "error" not in gap_result:
                gap_content = gap_result.get("content", "")

                # Parse gaps from natural language or JSON
                high_priority_gaps: list[dict] = []
                try:
                    # Try JSON first (backward compat)
                    cleaned = gap_content.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                        cleaned = re.sub(r'\s*```$', '', cleaned)
                    gap_data = json.loads(cleaned)
                    gaps = gap_data.get("gaps", [])
                    high_priority_gaps = [g for g in gaps if g.get("priority") == "high"][:2]
                except (json.JSONDecodeError, ValueError):
                    # Natural language parsing
                    gap_blocks = re.split(r'(?:^|\n)\s*GAP:\s*', gap_content)
                    for block in gap_blocks[1:]:  # skip preamble before first GAP:
                        block = block.strip()
                        if not block:
                            continue
                        title_line = block.split("\n")[0].strip()
                        query_match = re.search(r'QUERY:\s*(.+)', block)
                        priority_match = re.search(r'PRIORITY:\s*(\w+)', block)
                        priority = priority_match.group(1).lower() if priority_match else "medium"
                        if title_line and query_match and priority == "high":
                            high_priority_gaps.append({
                                "title": title_line,
                                "query": query_match.group(1).strip(),
                            })
                    high_priority_gaps = high_priority_gaps[:2]

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
                                run_subagent(child_angle, subagent_index * 100 + gi, progress_queue, req_id, user_query, depth + 1, collector=collector, condition_store=condition_store)
                            )
                        )

                    child_results = await asyncio.gather(*child_tasks, return_exceptions=True)
                    for cr in child_results:
                        if isinstance(cr, SubagentResult):
                            result.conditions.extend(cr.conditions)
                            result.spawned_children += 1
            langfuse_config.end_span(gap_span, output={
                "children_spawned": result.spawned_children,
            })

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

    duration_secs = time.monotonic() - start_time
    result.duration_secs = duration_secs

    log.info(
        f"[{sa_id}] Subagent complete: {len(result.conditions)} conditions, "
        f"{result.turns_used} turns, {result.tool_calls_made} tool calls, "
        f"{result.spawned_children} children spawned, {duration_secs:.1f}s"
    )
    langfuse_config.end_span(sa_span, output={
        "conditions": len(result.conditions),
        "turns": result.turns_used,
        "tool_calls": result.tool_calls_made,
        "children": result.spawned_children,
        "duration_secs": round(duration_secs, 1),
        "error": result.error or None,
    })
    return result


# Patterns that indicate an LLM safety refusal rather than a research finding.
_REFUSAL_PATTERNS = re.compile(
    r"(?i)"
    r"(?:I\s+(?:WILL\s+NOT|cannot|can'?t|refuse\s+to|am\s+unable\s+to)\s+(?:comply|assist|help|provide|fulfill|support))"
    r"|(?:(?:dangerous|unethical|illegal|harmful|inappropriate)\s+(?:and\s+)?(?:query|request|question))"
    r"|(?:I\s+am\s+designed\s+to\s+provide\s+(?:safe|legal|responsible))"
    r"|(?:I\s+(?:must|have\s+to)\s+(?:refuse|decline|reject))"
    r"|(?:(?:violates?|against)\s+(?:my|our)\s+(?:policy|policies|guidelines|ethical))"
    r"|(?:not\s+(?:able|going)\s+to\s+(?:help|assist|provide)\s+(?:with|you)\s+(?:this|that))"
)


def _is_llm_refusal(text: str) -> bool:
    """Detect whether text is an LLM safety refusal rather than a research finding."""
    if not text:
        return False
    return bool(_REFUSAL_PATTERNS.search(text))


def _parse_conditions(content: str, angle: str, is_bridge: bool) -> list[AtomicCondition]:
    """Parse findings from LLM output into AtomicCondition objects.

    Handles both natural-language summaries (preferred) and legacy JSON
    format (for backward compatibility with older prompts).
    """
    if not content:
        return []

    # NOTE: Do NOT check _is_llm_refusal on the entire content here.
    # Legitimate findings can contain phrases like "I cannot provide exact pricing"
    # that match refusal patterns. Per-chunk checks below handle this correctly.

    # --- Legacy JSON support (backward compat) ---
    # If the model still outputs JSON despite not being asked to, handle it.
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "conditions" in data:
            return [
                AtomicCondition(
                    fact=c.get("fact", ""),
                    source_url=c.get("source_url", ""),
                    confidence=float(c.get("confidence", 0.5)),
                    angle=angle,
                    is_serendipitous=is_bridge,
                )
                for c in data.get("conditions", [])
                if c.get("fact") and not _is_llm_refusal(c.get("fact", ""))
            ]
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # --- Natural language parsing (primary path) ---
    # The model writes a plain-text summary of findings. We split it into
    # individual findings by looking for paragraph breaks, numbered lists,
    # or bullet points — each becomes one AtomicCondition.
    conditions: list[AtomicCondition] = []

    # Extract URLs mentioned in the text for source attribution
    url_pattern = re.compile(r'https?://[^\s"\)\]>,;]+(?::\d+)?[^\s"\)\]>,;:]*(?<![.!?])')

    # Split on common delimiters: numbered items, bullet points, blank lines
    # Each chunk becomes one condition (if substantive).
    # Strip leading number prefix from the first item so all chunks are clean
    # (the regex only splits on \n-prefixed delimiters, so item 1 keeps its "1. ").
    chunks = re.split(
        r'\n\s*(?:\d+[\.\)]\s+|[-*•]\s+|\n)',
        re.sub(r'^\s*\d+[\.\)]\s+', '', content.strip()),
    )
    # If no splits found, treat the whole content as one chunk
    if len(chunks) <= 1:
        chunks = [p.strip() for p in content.strip().split('\n\n') if p.strip()]
    if not chunks:
        chunks = [content.strip()]

    # Skip preamble: the first chunk is often introductory text like
    # "Here are the key findings from my research:" — skip it if it
    # contains no URL and no confidence indicator (it's not a finding).
    _preamble_skipped = False
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue
        if _is_llm_refusal(chunk):
            continue
        if not _preamble_skipped:
            _preamble_skipped = True
            has_url = url_pattern.search(chunk)
            has_confidence = re.search(r'confidence[:\s]+(?:\d|high|medium|low)', chunk.lower())
            has_finding_signal = has_url or has_confidence or any(
                w in chunk.lower() for w in ("found", "confirmed", "verified", "listed", "available", "ships")
            )
            if not has_finding_signal and chunk.endswith(":"):
                continue  # skip preamble like "Here are my findings:"

        # Extract first URL as source
        url_match = url_pattern.search(chunk)
        source_url = url_match.group(0) if url_match else ""

        # Infer confidence — prefer explicit model-stated confidence over heuristics
        chunk_lower = chunk.lower()
        confidence = None

        # 1) Extract explicit confidence if the model provided one
        explicit_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', chunk_lower)
        if explicit_match:
            try:
                val = float(explicit_match.group(1))
                if 0.0 <= val <= 1.0:
                    confidence = val
            except ValueError:
                pass
        if confidence is None:
            # Map categorical confidence labels
            if "confidence: high" in chunk_lower or "confidence high" in chunk_lower:
                confidence = 0.8
            elif "confidence: medium" in chunk_lower or "confidence medium" in chunk_lower:
                confidence = 0.5
            elif "confidence: low" in chunk_lower or "confidence low" in chunk_lower:
                confidence = 0.3

        # 2) Keyword heuristic fallback (only if model didn't state confidence)
        if confidence is None:
            if any(w in chunk_lower for w in ("[tool_error]", "[access blocked]", "[censorship")):
                confidence = 0.2
            elif any(w in chunk_lower for w in ("speculative", "unverified", "rumor", "unclear")):
                confidence = 0.3
            elif re.search(r'\b(?:no|not|nothing|zero|none)\b.{0,20}\b(?:found|available|results?|evidence|vendors?)\b', chunk_lower):
                # Negative context: "no results found", "not available", "nothing found"
                confidence = 0.3
            elif any(re.search(r'(?<!\bnot\s)(?<!\bno\s)\b' + w + r'\b', chunk_lower) for w in ("confirmed", "verified", "visited", "in stock", "product page")):
                confidence = 0.9
            elif any(re.search(r'(?<!\bnot\s)(?<!\bno\s)\b' + w + r'\b', chunk_lower) for w in ("found", "listed", "available", "ships to")):
                confidence = 0.7
            elif any(w in chunk_lower for w in ("mentioned", "discussed", "referenced", "suggests")):
                confidence = 0.5
            else:
                confidence = 0.5

        conditions.append(AtomicCondition(
            fact=chunk[:500],
            source_url=source_url,
            confidence=confidence,
            angle=angle,
            is_serendipitous=is_bridge,
        ))

    # If we got no structured chunks but content is substantive, keep it as one condition
    if not conditions and len(content.strip()) > 20:
        url_match = url_pattern.search(content)
        conditions.append(AtomicCondition(
            fact=content.strip()[:500],
            source_url=url_match.group(0) if url_match else "",
            confidence=0.5,
            angle=angle,
            is_serendipitous=is_bridge,
        ))

    return conditions

