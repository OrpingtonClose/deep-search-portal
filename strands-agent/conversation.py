# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""ChatML ↔ Strands message conversion utilities.

Provides helpers for converting between LibreChat's ChatML format and the
Strands internal message format, including conversation history loading.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in ChatML format (from LibreChat)."""

    role: str
    content: str | list = ""


def extract_user_message(messages: list[ChatMessage]) -> str:
    """Extract the latest user message from a ChatML message list.

    Args:
        messages: List of ChatML messages.

    Returns:
        The text content of the last user message, or empty string.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


def chatml_to_strands(messages: list[ChatMessage]) -> list[dict]:
    """Convert LibreChat ChatML messages to Strands Converse format.

    LibreChat sends the full conversation history on each request.  We
    convert user/assistant turns into the Strands internal format so the
    agent has full conversational context.

    Strands format:  ``{"role": "user"|"assistant", "content": [{"text": "..."}]}``

    Args:
        messages: List of ChatML messages from LibreChat.

    Returns:
        List of Strands-format message dicts.
    """
    result = []
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            continue
        content = msg.content
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content) if content else ""
        if text.strip():
            result.append({"role": msg.role, "content": [{"text": text}]})
    return result


def load_conversation_history(
    agent: object,
    messages: list[ChatMessage],
    researcher: object | None = None,
) -> str:
    """Load conversation history into the agent and return the latest user message.

    Converts all previous turns from the ChatML request into Strands
    format and injects them into ``agent.messages``.  The latest user
    message is extracted and returned (it will be passed to ``agent()``
    which adds it to messages internally).

    For multi-agent, the researcher's messages are always cleared (it
    starts fresh for each delegation from the planner).

    Args:
        agent: Strands Agent instance with a ``.messages`` attribute.
        messages: Full ChatML message list from the request.
        researcher: Optional researcher agent whose messages should be cleared.

    Returns:
        The latest user message text.
    """
    strands_messages = chatml_to_strands(messages)

    # The last message should be the new user query — exclude it from
    # history since agent() will add it.
    if strands_messages and strands_messages[-1]["role"] == "user":
        history = strands_messages[:-1]
        user_message = strands_messages[-1]["content"][0]["text"]
    else:
        # Last strands message is not a user message (unusual — e.g.
        # trailing assistant message).  Find the last user message by
        # index and slice: history = everything before it.
        user_message = extract_user_message(messages)
        last_user_idx = None
        for i in range(len(strands_messages) - 1, -1, -1):
            if strands_messages[i]["role"] == "user" and strands_messages[i]["content"][0]["text"] == user_message:
                last_user_idx = i
                break
        history = strands_messages[:last_user_idx] if last_user_idx is not None else strands_messages

    agent.messages.clear()  # type: ignore[attr-defined]
    agent.messages.extend(history)  # type: ignore[attr-defined]

    if researcher is not None:
        researcher.messages.clear()  # type: ignore[attr-defined]

    return user_message
