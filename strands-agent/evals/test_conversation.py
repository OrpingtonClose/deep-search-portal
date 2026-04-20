# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for conversation helpers (ChatML ↔ Strands conversion).

Verifies message extraction, format conversion, and history loading
without requiring a live agent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from conversation import (
    ChatMessage,
    chatml_to_strands,
    extract_user_message,
    load_conversation_history,
)


class TestExtractUserMessage:
    """Verify extraction of the latest user message."""

    def test_single_user_message(self) -> None:
        messages = [ChatMessage(role="user", content="hello")]
        assert extract_user_message(messages) == "hello"

    def test_multiple_messages_returns_last_user(self) -> None:
        messages = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="response"),
            ChatMessage(role="user", content="second"),
        ]
        assert extract_user_message(messages) == "second"

    def test_no_user_message_returns_empty(self) -> None:
        messages = [ChatMessage(role="assistant", content="hi")]
        assert extract_user_message(messages) == ""

    def test_empty_list_returns_empty(self) -> None:
        assert extract_user_message([]) == ""

    def test_list_content_extracts_text_parts(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": "..."},
                    {"type": "text", "text": "world"},
                ],
            )
        ]
        assert extract_user_message(messages) == "hello world"

    def test_system_messages_ignored(self) -> None:
        messages = [
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="query"),
        ]
        assert extract_user_message(messages) == "query"


class TestChatmlToStrands:
    """Verify ChatML → Strands Converse format conversion."""

    def test_basic_conversation(self) -> None:
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
        ]
        result = chatml_to_strands(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": [{"text": "hello"}]}
        assert result[1] == {"role": "assistant", "content": [{"text": "hi there"}]}

    def test_system_messages_excluded(self) -> None:
        messages = [
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hello"),
        ]
        result = chatml_to_strands(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_content_excluded(self) -> None:
        messages = [
            ChatMessage(role="user", content=""),
            ChatMessage(role="user", content="real query"),
        ]
        result = chatml_to_strands(messages)
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "real query"

    def test_list_content_joined(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            )
        ]
        result = chatml_to_strands(messages)
        assert result[0]["content"][0]["text"] == "part1 part2"


class TestLoadConversationHistory:
    """Verify conversation history loading into agent."""

    def test_loads_history_and_returns_last_user_message(self) -> None:
        agent = MagicMock()
        agent.messages = []

        messages = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="response"),
            ChatMessage(role="user", content="second"),
        ]
        result = load_conversation_history(agent, messages)
        assert result == "second"
        # History should have 2 messages (first user + assistant response)
        assert len(agent.messages) == 2

    def test_clears_researcher_messages(self) -> None:
        agent = MagicMock()
        agent.messages = []
        researcher = MagicMock()
        researcher.messages = ["old_msg"]

        messages = [ChatMessage(role="user", content="query")]
        load_conversation_history(agent, messages, researcher=researcher)
        assert researcher.messages == []

    def test_single_message_no_history(self) -> None:
        agent = MagicMock()
        agent.messages = []

        messages = [ChatMessage(role="user", content="only query")]
        result = load_conversation_history(agent, messages)
        assert result == "only query"
        assert len(agent.messages) == 0
