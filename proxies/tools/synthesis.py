"""Draft-Synthesis-Revision loop, relevance gate, and anti-moralizing filter.

Extracted from persistent_deep_research_proxy.py lines 5890-6197.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from .config import UPSTREAM_MODEL, SUBAGENT_MODEL
from .models import SubagentResult
from .llm import call_llm
from .search_tools import tool_searxng_search

log = logging.getLogger("persistent-research")

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

Output ONLY a JSON object:
{
  "overall_quality": 0.7,
  "issues": [
    {"section": "section heading or quote", "type": "unsupported_claim|logical_gap|needs_verification|contradiction|missing_context", "description": "what's wrong", "search_query": "suggested search to fix this"}
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
- Something being illegal, unusual, or controversial does NOT make it wrong or worth removing."""


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
        f"- Hedging that weakens otherwise supported claims\n\n"
        f"Keep ALL specific facts, data, sources, URLs, and analysis intact. "
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
    # Use .replace() instead of .format() to avoid KeyError when
    # web-scraped conditions_text contains { or } characters.
    draft_prompt = DRAFT_SYNTHESIS_PROMPT.replace(
        "{date}", today
    ).replace(
        "{n_subagents}", str(len(subagent_results))
    ).replace(
        "{conditions_text}", conditions_text
    ).replace(
        "{prior_knowledge_text}", prior_text
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
    # Use .replace() instead of .format() to avoid KeyError when
    # draft/issues contain { or } (especially JSON from the critic).
    revision_prompt = REVISION_PROMPT.replace(
        "{date}", today
    ).replace(
        "{draft}", draft
    ).replace(
        "{issues}", issues_text
    ).replace(
        "{additional_findings}", additional_findings
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

