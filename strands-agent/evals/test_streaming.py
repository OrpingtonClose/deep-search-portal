# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for streaming presentation layer.

Verifies SSE chunk formatting, non-streaming response building,
and the openai_chunk helper — without requiring a live agent.
"""

from __future__ import annotations

import json

from streaming import build_non_streaming_response, openai_chunk


class TestOpenaiChunk:
    """Verify OpenAI-compatible SSE chunk formatting."""

    def test_content_chunk(self) -> None:
        raw = openai_chunk("req-1", "venice-model", "hello")
        assert raw.startswith("data: ")
        payload = json.loads(raw.removeprefix("data: ").strip())
        assert payload["id"] == "req-1"
        assert payload["model"] == "venice-model"
        assert payload["choices"][0]["delta"]["content"] == "hello"
        assert payload["choices"][0]["finish_reason"] is None

    def test_finish_chunk(self) -> None:
        raw = openai_chunk("req-1", "model", "", finish=True)
        payload = json.loads(raw.removeprefix("data: ").strip())
        assert payload["choices"][0]["delta"] == {}
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_sse_format_double_newline(self) -> None:
        raw = openai_chunk("r", "m", "x")
        assert raw.endswith("\n\n")

    def test_object_field(self) -> None:
        raw = openai_chunk("r", "m", "x")
        payload = json.loads(raw.removeprefix("data: ").strip())
        assert payload["object"] == "chat.completion.chunk"


class TestBuildNonStreamingResponse:
    """Verify non-streaming response formatting."""

    def test_answer_only(self) -> None:
        result = build_non_streaming_response(
            answer="42",
            captured_tool_events=[],
            captured_reasoning="",
            inline_log="",
        )
        assert result == "42"

    def test_with_tool_events(self) -> None:
        events = [
            {"tool": "brave_web_search", "input": '{"query": "test"}'},
        ]
        result = build_non_streaming_response(
            answer="result",
            captured_tool_events=events,
            captured_reasoning="",
            inline_log="",
        )
        assert "🔧" in result
        assert "**Answer:**" in result
        assert "result" in result

    def test_with_reasoning(self) -> None:
        result = build_non_streaming_response(
            answer="final answer",
            captured_tool_events=[],
            captured_reasoning="I need to think about this carefully",
            inline_log="",
        )
        assert "💭" in result
        assert "final answer" in result

    def test_reasoning_same_as_answer_suppressed(self) -> None:
        result = build_non_streaming_response(
            answer="the answer",
            captured_tool_events=[],
            captured_reasoning="the answer",
            inline_log="",
        )
        assert "💭" not in result

    def test_long_reasoning_truncated(self) -> None:
        long_reasoning = "x" * 600
        result = build_non_streaming_response(
            answer="done",
            captured_tool_events=[],
            captured_reasoning=long_reasoning,
            inline_log="",
        )
        assert "…" in result

    def test_refined_reasoning_used_when_provided(self) -> None:
        result = build_non_streaming_response(
            answer="final answer",
            captured_tool_events=[],
            captured_reasoning="raw chain of thought that is very long and messy",
            inline_log="",
            refined_reasoning="Agent is analyzing the problem",
        )
        assert "Agent is analyzing the problem" in result
        assert "raw chain of thought" not in result

    def test_inline_log_appended(self) -> None:
        result = build_non_streaming_response(
            answer="answer",
            captured_tool_events=[],
            captured_reasoning="",
            inline_log="\n*Researched using 3 sources in 12s*",
        )
        assert "*Researched using 3 sources in 12s*" in result

    def test_full_response_ordering(self) -> None:
        events = [{"tool": "web_search", "input": '{"q": "test"}'}]
        result = build_non_streaming_response(
            answer="The answer is 42.",
            captured_tool_events=events,
            captured_reasoning="Let me think about this.",
            inline_log="\n*footer*",
        )
        # Ordering: thinking → tools → answer separator → answer → footer
        think_pos = result.index("💭")
        tool_pos = result.index("🔧")
        answer_pos = result.index("**Answer:**")
        content_pos = result.index("The answer is 42.")
        footer_pos = result.index("*footer*")
        assert think_pos < tool_pos < answer_pos < content_pos < footer_pos
