# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""
Thought-refinement middleware — transforms chaotic agent reasoning into
user-friendly status updates.

The raw ``💭 Thinking`` blocks from Strands agents contain LLM chain-of-thought
that is often repetitive, self-contradictory, and hard to follow.  This module
intercepts those blocks and rewrites them via a fast LLM into concise,
informative summaries that:

- Tell the user **what the agent is doing** right now
- Highlight **interesting decisions** and workarounds
- Stay short (target: 2-4 sentences per thinking block)
- Preserve technical substance without the noise

Architecture
~~~~~~~~~~~~
Two modes of operation:

1. **Blocking** (``refine_sync``): collects the full thinking block, sends it
   to the refiner LLM, returns the refined text.  Used by the non-streaming
   ``/v1/chat/completions`` path.

2. **Async** (``refine_async``): same logic but awaitable.  Used by the SSE
   streaming path — the thinking buffer is collected during streaming, then
   refined in one shot when the block closes (before tool/answer tokens are
   emitted).

The refiner uses the cheapest available model (Venice ``qwen3.5-9b`` by
default) so the latency/cost overhead is minimal.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("thought-refiner")

# ── Configuration ─────────────────────────────────────────────────────

# The refiner model should be fast and cheap.  Venice qwen3.5-9b is ideal:
# near-zero cost, good at summarisation.  We disable reasoning mode
# (venice_parameters.include_venice_system_prompt=False + reasoning.effort="none")
# to avoid the model wasting time on chain-of-thought for a simple rewrite.
REFINER_API_BASE = os.environ.get(
    "REFINER_API_BASE",
    os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1"),
)
REFINER_MODEL = os.environ.get("REFINER_MODEL", "qwen3.5-9b")
REFINER_API_KEY = os.environ.get(
    "REFINER_API_KEY",
    os.environ.get("VENICE_API_KEY", ""),
)

# Skip refinement for very short thinking blocks (not worth the API call)
MIN_THINKING_LENGTH = int(os.environ.get("REFINER_MIN_LENGTH", "100"))

# Maximum raw thinking chars to send to the refiner (truncate the middle)
MAX_THINKING_INPUT = int(os.environ.get("REFINER_MAX_INPUT", "4000"))

# Feature toggle — set to "0" or "false" to disable refinement entirely
REFINER_ENABLED = os.environ.get("REFINER_ENABLED", "1").lower() not in ("0", "false", "no")


# ── Refiner prompt ────────────────────────────────────────────────────

REFINER_SYSTEM_PROMPT = """\
You are a research assistant UX writer. Your job is to take the raw, chaotic \
internal reasoning of an AI research agent and rewrite it as a concise, \
user-friendly status update.

Rules:
- Write 2-5 sentences maximum
- Use present tense ("The agent is searching...", "Found an interesting lead...")
- Highlight interesting decisions, strategy changes, or unexpected findings
- If the agent is working around a problem, mention it briefly
- Do NOT include disclaimers, warnings, or meta-commentary about the process
- Do NOT repeat the user's original question
- Do NOT use bullet points or headers — write flowing prose
- If the thinking is mostly planning/strategy, summarise the approach
- If the thinking contains actual findings, highlight the most interesting ones
- Keep technical terms when they add value, but explain jargon briefly
- Write in a tone that's informative and slightly conversational — like a \
knowledgeable colleague giving you a quick update
- Do NOT wrap your output in any tags or formatting — just write the summary text\
"""

REFINER_USER_TEMPLATE = """\
Here is the raw thinking from the research agent. Rewrite it as a brief, \
user-friendly status update:

---
{thinking}
---

Write your concise summary (2-5 sentences):"""


def _truncate_middle(text: str, max_len: int) -> str:
    """Truncate text by removing the middle if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    # Keep first and last portions, remove middle
    keep = max_len // 2
    return text[:keep] + "\n\n[...internal reasoning truncated...]\n\n" + text[-keep:]


def _build_refiner_messages(raw_thinking: str) -> list[dict]:
    """Build the chat messages for the refiner LLM call."""
    truncated = _truncate_middle(raw_thinking.strip(), MAX_THINKING_INPUT)
    return [
        {"role": "system", "content": REFINER_SYSTEM_PROMPT},
        {"role": "user", "content": REFINER_USER_TEMPLATE.format(thinking=truncated)},
    ]


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper tags that some models add."""
    import re
    # Remove leading <think> and trailing </think>
    text = re.sub(r"^\s*<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*$", "", text)
    return text.strip()


def _extract_content(data: dict) -> str:
    """Extract text content from a chat completion response.

    Venice models may return content in ``reasoning_content`` instead of
    ``content`` when reasoning mode is active.  This helper checks both
    fields.
    """
    msg = data.get("choices", [{}])[0].get("message", {})
    # Prefer regular content; fall back to reasoning_content
    text = msg.get("content", "") or ""
    if not text.strip():
        text = msg.get("reasoning_content", "") or ""
    return text.strip()


def _refiner_body(raw_thinking: str) -> dict:
    """Build the JSON request body for the refiner API call."""
    return {
        "model": REFINER_MODEL,
        "messages": _build_refiner_messages(raw_thinking),
        "max_tokens": 300,
        "temperature": 0.3,
        "stream": False,
        # Disable reasoning/thinking mode so the model responds directly
        # without wasting tokens on chain-of-thought.
        "venice_parameters": {"include_venice_system_prompt": False},
        "reasoning": {"effort": "none"},
    }


def refine_sync(raw_thinking: str, timeout: float = 20.0) -> str:
    """Refine a raw thinking block synchronously.

    Returns the refined text, or the original text (truncated) if
    refinement fails or is disabled.

    Args:
        raw_thinking: The raw chain-of-thought text from the agent.
        timeout: HTTP timeout in seconds for the refiner API call.

    Returns:
        Refined, user-friendly summary of the thinking.
    """
    if not REFINER_ENABLED:
        return raw_thinking

    if len(raw_thinking.strip()) < MIN_THINKING_LENGTH:
        return raw_thinking

    if not REFINER_API_KEY:
        logger.warning("Refiner API key not set — skipping refinement")
        return raw_thinking

    start = time.time()
    try:
        resp = httpx.post(
            f"{REFINER_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {REFINER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=_refiner_body(raw_thinking),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        refined = _strip_think_tags(_extract_content(data))
        if refined:
            elapsed = time.time() - start
            logger.info("Thought refined in %.1fs (%d→%d chars)", elapsed, len(raw_thinking), len(refined))
            return refined
    except httpx.TimeoutException:
        logger.warning("Refiner timed out after %.1fs — using raw thinking", timeout)
    except Exception:
        logger.exception("Refiner failed — using raw thinking")

    # Fallback: return a trimmed version of the raw thinking
    return _truncate_middle(raw_thinking.strip(), 1000)


async def refine_async(raw_thinking: str, timeout: float = 20.0) -> str:
    """Refine a raw thinking block asynchronously.

    Same as ``refine_sync`` but uses ``httpx.AsyncClient`` for use in
    async streaming generators.

    Args:
        raw_thinking: The raw chain-of-thought text from the agent.
        timeout: HTTP timeout in seconds for the refiner API call.

    Returns:
        Refined, user-friendly summary of the thinking.
    """
    if not REFINER_ENABLED:
        return raw_thinking

    if len(raw_thinking.strip()) < MIN_THINKING_LENGTH:
        return raw_thinking

    if not REFINER_API_KEY:
        logger.warning("Refiner API key not set — skipping refinement")
        return raw_thinking

    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{REFINER_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {REFINER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=_refiner_body(raw_thinking),
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            refined = _strip_think_tags(_extract_content(data))
            if refined:
                elapsed = time.time() - start
                logger.info("Thought refined (async) in %.1fs (%d→%d chars)", elapsed, len(raw_thinking), len(refined))
                return refined
    except httpx.TimeoutException:
        logger.warning("Refiner timed out (async) after %.1fs — using raw thinking", timeout)
    except Exception:
        logger.exception("Refiner failed (async) — using raw thinking")

    # Fallback: return a trimmed version of the raw thinking
    return _truncate_middle(raw_thinking.strip(), 1000)
