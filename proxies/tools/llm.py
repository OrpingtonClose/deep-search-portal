"""
LLM communication: message conversion, call_llm with retry logic.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from shared import get_throttler

from .config import (
    UPSTREAM_MODEL,
    _get_llm,
    log,
)
from .tool_defs import LANGCHAIN_TOOLS


# ============================================================================
# LLM Communication
# ============================================================================

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


# Per-request LangGraph callback config, keyed by req_id.
# call_llm looks up the config for the current request so that
# ResearchMetricsCallback fires on every LLM call automatically.
_request_configs: dict[str, dict] = {}


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


