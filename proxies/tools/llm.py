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
    SEARCH_BACKEND,
    UPSTREAM_MODEL,
    _get_llm,
    _get_llm_via_litellm,
    build_xml_tools_system_prompt,
    is_xml_tool_model,
    log,
    parse_xml_tool_calls,
)
from .tool_defs import LANGCHAIN_TOOLS


# ============================================================================
# LLM Communication
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


def _is_cloudflare_challenge(error_text: str) -> bool:
    """Detect Cloudflare 'Just a moment...' challenge pages.

    Mistral's API sits behind Cloudflare.  When rate-limited at the CDN
    layer, Cloudflare serves an HTML challenge page instead of the JSON
    API response.  LangChain receives the HTML, fails to parse it, and
    raises an exception whose str() contains the raw HTML.

    These are transient rate-limits and MUST be retried.
    """
    lower = error_text[:500].lower()
    return "just a moment" in lower or ("<!doctype html" in lower and "cloudflare" in lower)


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


def _inject_xml_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Prepend Hermes-3 XML tool instructions to the system message.

    If the first message is a system message, the XML prompt is prepended.
    Otherwise a new system message is inserted at position 0.
    Also converts any ``tool`` role messages to ``user`` role with
    ``<tool_response>`` XML wrapper so the model understands them.
    """
    xml_prompt = build_xml_tools_system_prompt(tools)
    out: list[dict] = []
    system_injected = False

    for m in messages:
        role = m.get("role", "user")
        if role == "system" and not system_injected:
            out.append({
                "role": "system",
                "content": xml_prompt + "\n\n" + (m.get("content", "") or ""),
            })
            system_injected = True
        elif role == "tool":
            # Convert tool responses to user messages with XML wrapper
            tc_id = m.get("tool_call_id", "unknown")
            content = m.get("content", "") or ""
            out.append({
                "role": "user",
                "content": f"<tool_response>\n{content}\n</tool_response>",
            })
        elif role == "assistant" and m.get("tool_calls"):
            # Strip native tool_calls from assistant messages — the XML
            # model already emitted <tool_call> tags in its content.
            out.append({
                "role": "assistant",
                "content": m.get("content", "") or "",
            })
        else:
            out.append(m)

    if not system_injected:
        out.insert(0, {"role": "system", "content": xml_prompt})

    return out


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
    use_xml = include_tools and is_xml_tool_model(resolved_model)

    if SEARCH_BACKEND == "mcp":
        # --- MCP path: LiteLLM handles retries, fallbacks, cost tracking ---
        llm = _get_llm_via_litellm(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if include_tools and not use_xml:
            llm = llm.bind_tools(LANGCHAIN_TOOLS)

        msgs = _inject_xml_tools(messages, LANGCHAIN_TOOLS) if use_xml else messages
        lc_messages = _dicts_to_langchain_messages(msgs)
        config = _request_configs.get(req_id, {})

        try:
            ai_msg: AIMessage = await llm.ainvoke(lc_messages, config=config)
            content = ai_msg.content or ""

            tool_calls_out = None
            if use_xml:
                tool_calls_out = parse_xml_tool_calls(content)
            elif ai_msg.tool_calls:
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
            return {"error": f"[LLM Error: {str(e)[:500]}]"}

    # --- Legacy path: manual retries with Cloudflare detection ---
    llm = _get_llm(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if include_tools and not use_xml:
        llm = llm.bind_tools(LANGCHAIN_TOOLS)

    msgs = _inject_xml_tools(messages, LANGCHAIN_TOOLS) if use_xml else messages
    lc_messages = _dicts_to_langchain_messages(msgs)

    # Look up per-request config (contains callbacks list with
    # ResearchMetricsCallback) so metrics fire automatically.
    config = _request_configs.get(req_id, {})

    _mistral_throttle = get_throttler("mistral")
    last_error: Optional[str] = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            async with _mistral_throttle.throttle():
                ai_msg = await llm.ainvoke(lc_messages, config=config)

                content = ai_msg.content or ""

            # Extract tool_calls in OpenAI format for backward compat
            tool_calls_out = None
            if use_xml:
                tool_calls_out = parse_xml_tool_calls(content)
            elif ai_msg.tool_calls:
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
            retryable = (
                bool(re.search(rf"\b({_codes_pattern})\b", err_str))
                or isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout))
                or _is_cloudflare_challenge(err_str)
            )

            if _is_cloudflare_challenge(err_str):
                log.warning(
                    f"[{req_id}] Cloudflare challenge detected — "
                    f"Mistral CDN rate-limit (attempt {attempt + 1}/{MAX_LLM_RETRIES + 1})"
                )

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

