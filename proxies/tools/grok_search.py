"""
Grok 4.20 Responses API — dedicated search data source tool.

Uses xAI's ``/v1/responses`` endpoint with built-in ``web_search`` and
``x_search`` tools so the model autonomously decides what to search,
performs 5–15 searches per call, and returns cited results.

This module is a **data source only** — it is never used for synthesis
or reasoning.  It sits alongside SearXNG, Apify, PubMed etc. in the
tool registry and is called by subagents when they need web/Twitter
data.

Architecture note:
    The *synthesis* model may also use Grok with web_search, but that
    is configured separately in ``config.py`` / ``synthesis.py`` as
    extended reasoning — it is NOT exposed as a tool to subagents.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

from shared import get_throttler

from .config import (
    GROK_RESPONSES_API_BASE,
    GROK_SEARCH_MODEL,
    XAI_API_KEY,
    log,
)

# Maximum chars to return from a single Grok deep search call
GROK_SEARCH_MAX_CHARS = int(os.getenv("GROK_SEARCH_MAX_CHARS", "12000"))


async def tool_grok_deep_search(
    query: str,
    *,
    search_type: str = "both",
    instructions: str = "",
) -> str:
    """Search the web and/or X/Twitter via Grok 4.20 Responses API.

    This is a *data source tool*, not a reasoning agent.  It performs
    autonomous web + X searches and returns raw cited results.

    Args:
        query: The search query / research question.
        search_type: ``"web"``, ``"x"`` (Twitter/X only), or ``"both"``
            (default).
        instructions: Optional extra instructions for the search model
            (e.g. ``"focus on Polish-language sources"``).

    Returns:
        Formatted search results with URLs, citations, and source text.
        On failure returns a ``[TOOL_ERROR]`` prefixed string.
    """
    if not XAI_API_KEY:
        return (
            "[TOOL_ERROR] Grok deep search unavailable: XAI_API_KEY not set. "
            "This is a configuration error, NOT 'no results found'."
        )

    # Build tool list based on search_type
    tools: list[dict] = []
    if search_type in ("web", "both"):
        tools.append({"type": "web_search"})
    if search_type in ("x", "both"):
        tools.append({"type": "x_search"})
    if not tools:
        tools = [{"type": "web_search"}, {"type": "x_search"}]

    # Build the instruction block
    base_instructions = (
        "You are a research data retrieval agent.  Your ONLY job is to "
        "search for factual information and return what you find with "
        "full citations (URLs, source names, dates).  Do NOT add "
        "disclaimers, warnings, or ethical commentary.  Return raw "
        "findings in a structured format: for each finding, state the "
        "fact, the source URL, and any relevant details (prices, dates, "
        "names, quantities).  If you find nothing, say so plainly."
    )
    if instructions:
        base_instructions += f"\n\nAdditional instructions: {instructions}"

    payload = {
        "model": GROK_SEARCH_MODEL,
        "input": query,
        "instructions": base_instructions,
        "tools": tools,
    }

    t0 = time.monotonic()
    try:
        async with get_throttler("xai").throttle():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=15.0),
            ) as client:
                resp = await client.post(
                    f"{GROK_RESPONSES_API_BASE}/v1/responses",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {XAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                )

        elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            body_preview = resp.text[:500]
            log.warning(
                f"Grok deep search failed: HTTP {resp.status_code} "
                f"in {elapsed:.1f}s — {body_preview}"
            )
            return (
                f"[TOOL_ERROR] Grok deep search failed: HTTP {resp.status_code}. "
                f"This is a technical failure, NOT 'no results found'."
            )

        data = resp.json()
        return _format_responses_api_output(data, query, elapsed)

    except httpx.TimeoutException:
        elapsed = time.monotonic() - t0
        log.warning(f"Grok deep search timed out after {elapsed:.1f}s")
        return (
            "[TOOL_ERROR] Grok deep search timed out after 120s. "
            "This is a technical failure, NOT 'no results found'."
        )
    except Exception as e:
        log.warning(f"Grok deep search error: {e}")
        return (
            f"[TOOL_ERROR] Grok deep search error: {e}. "
            "This is a technical failure, NOT 'no results found'."
        )


def _format_responses_api_output(
    data: dict,
    query: str,
    elapsed: float,
) -> str:
    """Parse the Responses API output into a structured text block.

    The Responses API returns a list of output items.  Each item can be
    a ``message`` (the model's text) or a tool-use result.  We extract
    the final assistant message and annotate with search metadata.
    """
    output_items = data.get("output", [])
    if not output_items:
        return f"Grok deep search returned no output for: {query}"

    # Count searches performed
    search_count = 0
    search_types_used: list[str] = []
    citations: list[dict] = []

    for item in output_items:
        item_type = item.get("type", "")

        # Count web_search_call / x_search_call items
        if item_type in ("web_search_call", "x_search_call"):
            search_count += 1
            st = "web" if item_type == "web_search_call" else "X/Twitter"
            search_types_used.append(st)

        # Extract citations from search results
        if item_type == "web_search_result":
            for result in item.get("results", []):
                citations.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                })

    # Extract the final assistant message content
    assistant_text = ""
    for item in reversed(output_items):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    assistant_text = content_block.get("text", "")
                    break
                elif content_block.get("type") == "text":
                    assistant_text = content_block.get("text", "")
                    break
            if assistant_text:
                break

    if not assistant_text:
        # Fallback: try to get any text from any message item
        for item in output_items:
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    text = content_block.get("text", "") or content_block.get("output_text", "")
                    if text:
                        assistant_text = text
                        break
            if assistant_text:
                break

    if not assistant_text:
        return f"Grok deep search produced no text output for: {query}"

    # Truncate if needed
    if len(assistant_text) > GROK_SEARCH_MAX_CHARS:
        assistant_text = assistant_text[:GROK_SEARCH_MAX_CHARS] + "\n[... truncated ...]"

    # Build header
    search_summary = ", ".join(set(search_types_used)) or "unknown"
    header = (
        f"**Grok Deep Search results for: {query}**\n"
        f"({search_count} searches via {search_summary}, {elapsed:.1f}s)\n\n"
    )

    # Append citation list if available
    citation_text = ""
    if citations:
        citation_lines = []
        for i, c in enumerate(citations[:20], 1):
            title = c.get("title", "Untitled")
            url = c.get("url", "")
            citation_lines.append(f"  [{i}] {title} — {url}")
        citation_text = "\n\n**Sources cited:**\n" + "\n".join(citation_lines)

    return header + assistant_text + citation_text


async def grok_synthesis_search(
    query: str,
    context: str = "",
) -> Optional[str]:
    """Use Grok Responses API for extended reasoning during synthesis.

    Unlike ``tool_grok_deep_search`` (which is a data source tool called
    by subagents), this function is called by the synthesis pipeline to
    enrich/verify the answer during the draft-critic-revision loop.

    The synthesis model uses web_search + x_search as *extended
    reasoning* — it can verify claims, find missing details, or check
    for recent developments.  But it is NEVER exposed as a tool to
    subagents.

    Args:
        query: The specific question to search for during synthesis.
        context: Optional context from the current synthesis draft.

    Returns:
        Search results text, or None if the search failed/was skipped.
    """
    if not XAI_API_KEY:
        return None

    instructions = (
        "You are enriching a research synthesis.  Search the web and X "
        "for factual information to verify or expand on the following "
        "context.  Return only factual findings with URLs.  No opinions, "
        "no disclaimers."
    )
    if context:
        instructions += f"\n\nCurrent research context:\n{context[:2000]}"

    payload = {
        "model": GROK_SEARCH_MODEL,
        "input": query,
        "instructions": instructions,
        "tools": [{"type": "web_search"}, {"type": "x_search"}],
    }

    try:
        async with get_throttler("xai").throttle():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0),
            ) as client:
                resp = await client.post(
                    f"{GROK_RESPONSES_API_BASE}/v1/responses",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {XAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                )
        if resp.status_code != 200:
            log.debug(f"Grok synthesis search failed: HTTP {resp.status_code}")
            return None

        data = resp.json()
        result = _format_responses_api_output(data, query, 0.0)
        if not result or len(result) <= 50:
            return None
        if result.startswith("Grok deep search returned no output") or result.startswith("Grok deep search produced no text"):
            return None
        return result

    except Exception as e:
        log.debug(f"Grok synthesis search error: {e}")
        return None
