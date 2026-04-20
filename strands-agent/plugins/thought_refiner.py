# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native thought-refinement plugin.

Replaces the standalone ``thought_refiner.py`` module with a Strands SDK
Plugin.  The refinement logic (Venice ``qwen3.5-9b`` via raw httpx) is
preserved but wrapped in the plugin lifecycle so it can be composed with
other plugins via the ``Agent(plugins=[...])`` constructor.

The plugin provides both sync and async refinement methods that the SSE
streaming handler in ``main.py`` calls when flushing thinking buffers.
"""

from __future__ import annotations

import logging
import os
import re
import time

import httpx
from strands.plugins import Plugin

logger = logging.getLogger("thought-refiner")

# ── Configuration ─────────────────────────────────────────────────────

REFINER_API_BASE = os.environ.get(
    "REFINER_API_BASE",
    os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1"),
)
REFINER_MODEL = os.environ.get("REFINER_MODEL", "qwen3.5-9b")
REFINER_API_KEY = os.environ.get(
    "REFINER_API_KEY",
    os.environ.get("VENICE_API_KEY", ""),
)
MIN_THINKING_LENGTH = int(os.environ.get("REFINER_MIN_LENGTH", "100"))
MAX_THINKING_INPUT = int(os.environ.get("REFINER_MAX_INPUT", "4000"))
REFINER_ENABLED = os.environ.get("REFINER_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)

# ── Refiner prompt ────────────────────────────────────────────────────

REFINER_SYSTEM_PROMPT = """\
You rewrite raw AI agent reasoning into engaging, readable prose.

Rules:
- 2-5 sentences, flowing prose — like a narrator describing a detective at work
- Present tense: "Searching for...", "Found that...", "Looking into..."
- Sprinkle in interesting specifics from the reasoning — a surprising fact, \
a notable source, a clever connection the agent is making
- Sound curious and engaged, not clinical — make the reader want to keep reading
- NO quotes, NO code, NO backticks, NO markdown formatting, NO asterisks
- NO disclaimers or meta-commentary
- NO bullet points, headers, or lists
- NO references to "the agent", "the model", "the system" — just describe the action
- Do NOT repeat the user's question
- Do NOT wrap output in tags, quotes, or formatting — plain text only\
"""

REFINER_USER_TEMPLATE = """\
Raw thinking:

{thinking}

Rewrite as a plain-text status update (2-5 sentences, no formatting):"""


def _truncate_middle(text: str, max_len: int) -> str:
    """Truncate text by removing the middle if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return text[:keep] + "\n\n[...internal reasoning truncated...]\n\n" + text[-keep:]


def _build_refiner_messages(raw_thinking: str) -> list[dict[str, str]]:
    """Build the chat messages for the refiner LLM call."""
    truncated = _truncate_middle(raw_thinking.strip(), MAX_THINKING_INPUT)
    return [
        {"role": "system", "content": REFINER_SYSTEM_PROMPT},
        {"role": "user", "content": REFINER_USER_TEMPLATE.format(thinking=truncated)},
    ]


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper tags that some models add."""
    text = re.sub(r"^\s*<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*$", "", text)
    return text.strip()


def _extract_content(data: dict) -> str:
    """Extract text content from a chat completion response.

    Venice models may return content in ``reasoning_content`` instead of
    ``content`` when reasoning mode is active.
    """
    msg = data.get("choices", [{}])[0].get("message", {})
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
        "venice_parameters": {"include_venice_system_prompt": False},
        "reasoning": {"effort": "none"},
    }


class ThoughtRefinerPlugin(Plugin):
    """Refines raw chain-of-thought into user-friendly status updates.

    Uses Venice ``qwen3.5-9b`` (or configurable model) to rewrite chaotic
    agent reasoning into 2-4 sentence summaries.  Provides both sync and
    async methods for use by the SSE streaming handler and the non-streaming
    response path.

    The plugin does NOT hook into model events directly because thought
    refinement happens at the SSE presentation layer (buffering thinking
    tokens, then flushing as refined text).  Instead, it exposes
    ``refine_sync`` and ``refine_async`` methods that the streaming
    handler calls explicitly.
    """

    name: str = "thought-refiner"

    def __init__(self, enabled: bool | None = None) -> None:
        """Initialise the thought refiner plugin.

        Args:
            enabled: Override the REFINER_ENABLED env var.  When None,
                uses the env var.
        """
        super().__init__()
        self.enabled = enabled if enabled is not None else REFINER_ENABLED

    @property
    def is_available(self) -> bool:
        """Whether refinement is available (enabled + API key present)."""
        return self.enabled and bool(REFINER_API_KEY)

    def refine_sync(self, raw_thinking: str, timeout: float = 20.0) -> str:
        """Refine a raw thinking block synchronously.

        Args:
            raw_thinking: The raw chain-of-thought text from the agent.
            timeout: HTTP timeout in seconds.

        Returns:
            Refined, user-friendly summary.
        """
        if not self.enabled:
            return _truncate_middle(raw_thinking.strip(), 500)

        if len(raw_thinking.strip()) < MIN_THINKING_LENGTH:
            return raw_thinking

        if not REFINER_API_KEY:
            logger.warning("refiner API key not set — skipping refinement")
            return _truncate_middle(raw_thinking.strip(), 500)

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
                logger.info(
                    "chars_in=<%d>, chars_out=<%d>, elapsed=<%.1fs> | thought refined",
                    len(raw_thinking),
                    len(refined),
                    elapsed,
                )
                return refined
        except httpx.TimeoutException:
            logger.warning("timeout=<%.1fs> | refiner timed out", timeout)
        except Exception:
            logger.exception("refiner failed — using raw thinking")

        return _truncate_middle(raw_thinking.strip(), 1000)

    async def refine_async(self, raw_thinking: str, timeout: float = 20.0) -> str:
        """Refine a raw thinking block asynchronously.

        Args:
            raw_thinking: The raw chain-of-thought text from the agent.
            timeout: HTTP timeout in seconds.

        Returns:
            Refined, user-friendly summary.
        """
        if not self.enabled:
            return _truncate_middle(raw_thinking.strip(), 500)

        if len(raw_thinking.strip()) < MIN_THINKING_LENGTH:
            return raw_thinking

        if not REFINER_API_KEY:
            logger.warning("refiner API key not set — skipping refinement")
            return _truncate_middle(raw_thinking.strip(), 500)

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
                    logger.info(
                        "chars_in=<%d>, chars_out=<%d>, elapsed=<%.1fs> | thought refined (async)",
                        len(raw_thinking),
                        len(refined),
                        elapsed,
                    )
                    return refined
        except httpx.TimeoutException:
            logger.warning("timeout=<%.1fs> | refiner timed out (async)", timeout)
        except Exception:
            logger.exception("refiner failed (async) — using raw thinking")

        return _truncate_middle(raw_thinking.strip(), 1000)
