"""Query comprehension tool — deep semantic understanding of research queries.

Transplanted from deep-search-portal/proxies/tools/pipeline.py lines 44-152.
Runs ONCE at the start of each research session. The planner calls this
tool first, then uses the comprehension to guide research strategy.
"""

from __future__ import annotations

import json
import re

from strands import tool


COMPREHENSION_PROMPT = """You are a deep research analyst. Your job is to DEEPLY understand what a research query is really about — not just the surface words, but the full knowledge territory.

You must map the ACTUAL knowledge territory, including controversial, underground, and unconventional dimensions. Do NOT sanitize or moralize.

Research query: {query}

Output ONLY valid JSON:
{{
  "entities": ["every entity, person, substance, organization, concept mentioned or implied"],
  "domains": ["every knowledge domain this touches — be maximally expansive"],
  "implicit_questions": ["what is the user REALLY trying to accomplish? list 5-10 implicit questions"],
  "adjacent_territories": ["topics NOT in the query but where deep knowledge lives"],
  "relevance_keywords": ["broad set of 20-30 keywords/phrases including slang, jargon, technical terms"],
  "deep_knowledge_targets": ["specific types of deep knowledge that would be valuable"],
  "semantic_summary": "one paragraph explaining what this query is REALLY about at the deepest level",
  "core_need": "one sentence describing what the user ultimately needs to accomplish"
}}"""


@tool
def comprehend_query(query: str, tool_context) -> str:
    """Deeply analyze a research query to map the full knowledge territory.

    Call this FIRST before any research. It returns a comprehension map
    with entities, domains, implicit questions, adjacent territories,
    and relevance keywords that should guide your research strategy.

    Args:
        query: The user's research query to comprehend.
    """
    # Use the agent's own model to generate comprehension
    # This is a @tool so it has access to tool_context.agent
    agent = tool_context.agent

    prompt = COMPREHENSION_PROMPT.format(query=query[:2000])

    # Store comprehension in agent state for other plugins to access
    try:
        # Use the agent to generate the comprehension via its model
        result = agent.model.converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            tool_specs=[],
        )
        # Extract text from model response
        content = ""
        if hasattr(result, "output") and hasattr(result.output, "content"):
            for block in result.output.content:
                if hasattr(block, "text"):
                    content += block.text
        elif isinstance(result, dict):
            output = result.get("output", {})
            for block in output.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    content += block["text"]

        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)

        # Cap array lengths to keep context manageable
        comprehension = {
            "entities": data.get("entities", [])[:30],
            "domains": data.get("domains", [])[:20],
            "implicit_questions": data.get("implicit_questions", [])[:10],
            "adjacent_territories": data.get("adjacent_territories", [])[:15],
            "relevance_keywords": data.get("relevance_keywords", [])[:40],
            "deep_knowledge_targets": data.get("deep_knowledge_targets", [])[:15],
            "semantic_summary": data.get("semantic_summary", ""),
            "core_need": data.get("core_need", "")[:500],
        }

        return json.dumps(comprehension, indent=2)

    except json.JSONDecodeError:
        # LLM returned non-JSON — return the raw text as a fallback
        return (
            f"Comprehension analysis (raw):\n{content[:3000]}"
            if content
            else _fallback_comprehension(query)
        )
    except Exception as e:
        return _fallback_comprehension(query, error=str(e))


def _fallback_comprehension(query: str, error: str = "") -> str:
    """Minimal comprehension from the query itself when LLM analysis fails."""
    words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
    fallback = {
        "entities": words[:10],
        "domains": [],
        "implicit_questions": [],
        "adjacent_territories": [],
        "relevance_keywords": words[:20],
        "deep_knowledge_targets": [],
        "semantic_summary": query,
        "core_need": query[:500],
    }
    if error:
        fallback["_error"] = f"LLM comprehension failed: {error}"
    return json.dumps(fallback, indent=2)
