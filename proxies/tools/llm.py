"""LLM factory functions, message conversion, and call_llm with retry logic.

Extracted from persistent_deep_research_proxy.py lines 222-252, 3867-3999.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from shared import get_throttler

from .config import (
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    UPSTREAM_MODEL,
    SUBAGENT_MODEL,
)

log = logging.getLogger("persistent-research")

# Per-request LangGraph callback config, keyed by req_id.
# call_llm looks up the config for the current request so that
# ResearchMetricsCallback fires on every LLM call automatically.
_request_configs: dict[str, dict] = {}

def _get_llm(
    model: str = "",
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: float = 300.0,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance pointing at the Mistral API.

    Note: We pass max_tokens via extra_body instead of the native parameter
    because langchain-openai >=1.0 converts max_tokens to
    max_completion_tokens, which the Mistral API rejects with a 422.
    """
    return ChatOpenAI(
        model=model or UPSTREAM_MODEL,
        api_key=UPSTREAM_KEY,
        base_url=UPSTREAM_BASE,
        temperature=temperature,
        timeout=timeout,
        extra_body={"max_tokens": max_tokens},
    )


def _get_synthesis_llm(**kwargs: Any) -> ChatOpenAI:
    """LLM for synthesis / revision (upstream large model)."""
    return _get_llm(model=UPSTREAM_MODEL, max_tokens=8192, temperature=0.3, **kwargs)


def _get_subagent_llm(**kwargs: Any) -> ChatOpenAI:
    """LLM for subagents, heartbeat, relevance gate (small/fast model)."""
    return _get_llm(model=SUBAGENT_MODEL, max_tokens=4096, temperature=0.3, **kwargs)


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


def _dicts_to_langchain_messages(
    messages: list[dict],
) -> list[SystemMessage | HumanMessage | AIMessage | ToolMessage]:
    """Convert OpenAI-format message dicts to LangChain message objects."""
    lc_msgs: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            # Preserve tool_calls if present so bind_tools round-trips work
            tc = m.get("tool_calls")
            if tc:
                lc_msgs.append(AIMessage(
                    content=content,
                    additional_kwargs={"tool_calls": tc},
                ))
            else:
                lc_msgs.append(AIMessage(content=content))
        elif role == "tool":
            lc_msgs.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
            ))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs



async def call_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    include_tools: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict:
    """Call the upstream LLM via LangChain ChatOpenAI (fires callbacks).

    Returns the same dict format as the old raw-httpx version for
    backward compatibility:
        {"content": str, "tool_calls": list|None, "message": dict, "finish_reason": str}
    or  {"error": str}
    """
    resolved_model = model or UPSTREAM_MODEL
    llm = _get_llm(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if include_tools:
        from .tool_defs import LANGCHAIN_TOOLS
        llm = llm.bind_tools(LANGCHAIN_TOOLS)

    lc_messages = _dicts_to_langchain_messages(messages)

    # Look up per-request config (contains callbacks list with
    # ResearchMetricsCallback) so metrics fire automatically.
    config = _request_configs.get(req_id, {})

    _mistral_throttle = get_throttler("mistral")
    last_error: Optional[str] = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            async with _mistral_throttle.throttle():
                ai_msg: AIMessage = await llm.ainvoke(lc_messages, config=config)

                content = ai_msg.content or ""

            # Extract tool_calls in OpenAI format for backward compat
            tool_calls_out = None
            if ai_msg.tool_calls:
                tool_calls_out = [
                    {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for tc in ai_msg.tool_calls
                ]

            # Build backward-compatible "message" dict
            message_dict: dict[str, Any] = {"content": content}
            if tool_calls_out:
                message_dict["tool_calls"] = tool_calls_out

            return {
                "message": message_dict,
                "content": content,
                "tool_calls": tool_calls_out,
                "finish_reason": ai_msg.response_metadata.get(
                    "finish_reason", "stop"
                ),
            }

        except Exception as e:
            err_str = str(e)
            # Detect retryable HTTP status codes from the error message
            _codes_pattern = "|".join(str(c) for c in RETRYABLE_STATUS_CODES)
            retryable = bool(
                re.search(rf"\b({_codes_pattern})\b", err_str)
            ) or isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout))

            last_error = f"[LLM Error: {err_str[:500]}]"

            if retryable and attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(
                    f"[{req_id}] Retryable LLM error, waiting {wait}s "
                    f"(attempt {attempt + 1}/{MAX_LLM_RETRIES}): {err_str[:200]}"
                )
                await asyncio.sleep(wait)
                continue

            return {"error": last_error}

    return {"error": last_error or "[LLM Error: Max retries exceeded]"}

