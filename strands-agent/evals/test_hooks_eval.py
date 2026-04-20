# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Hooks-based eval assertions using EvalCollectorPlugin.

These tests verify that the EvalCollectorPlugin correctly captures
structured data from agent runs via SDK hooks — replacing fragile
regex-based SSE parsing with typed records.
"""

from __future__ import annotations

from strands import Agent, tool

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    multi_tool_then_answer,
    reasoning_then_text,
    simple_text_response,
    tool_call_response,
)


@tool
def echo_tool(text: str) -> str:
    """Echo the input text back."""
    return f"Echo: {text}"


class TestHooksBasicCapture:
    """Verify hook-based capture of tool calls, model calls, and invocations."""

    def test_captures_simple_invocation(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([simple_text_response("Hello!")])
        agent = Agent(model=model, plugins=[collector], callback_handler=None)
        result = agent("Hi")

        assert collector.total_invocations == 1
        assert collector.total_tool_calls == 0
        assert collector.total_model_calls == 1
        assert collector.invocations[0].stop_reason == "end_turn"

    def test_captures_tool_call(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("echo_tool", "tool-001", {"text": "hello"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        result = agent("Echo hello")

        assert collector.total_tool_calls == 1
        assert collector.tool_names == ["echo_tool"]
        assert collector.tool_calls[0].success is True
        assert collector.tool_calls[0].duration >= 0

    def test_captures_multiple_tools(self) -> None:
        collector = EvalCollectorPlugin()
        messages = [
            tool_call_response("echo_tool", "tool-001", {"text": "first"}),
            tool_call_response("echo_tool", "tool-002", {"text": "second"}),
            simple_text_response("All done"),
        ]
        model = MockModel(messages)
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        result = agent("Echo both")

        assert collector.total_tool_calls == 2
        assert collector.total_model_calls == 3  # one per response
        assert collector.invocations[0].stop_reason == "end_turn"

    def test_captures_reasoning_content(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([reasoning_then_text(
            reasoning="Thinking step by step about the problem...",
            answer="42",
        )])
        agent = Agent(model=model, plugins=[collector], callback_handler=None)
        result = agent("What is the meaning of life?")

        assert collector.total_invocations == 1
        assert collector.total_model_calls == 1
        assert "42" in str(result)


class TestHooksToolDetails:
    """Verify hook-captured tool call details match expectations."""

    def test_tool_input_captured(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("echo_tool", "tool-100", {"text": "specific input"}),
            simple_text_response("Result"),
        ])
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Test input capture")

        assert collector.tool_calls[0].tool_use_id == "tool-100"
        assert collector.tool_calls[0].tool_input == {"text": "specific input"}

    def test_unique_tool_names(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("echo_tool", "tool-001", {"text": "a"}),
            tool_call_response("echo_tool", "tool-002", {"text": "b"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Echo twice")

        assert collector.unique_tool_names == {"echo_tool"}
        assert collector.total_tool_calls == 2


class TestHooksInvocationTracking:
    """Verify invocation-level tracking across multiple agent calls."""

    def test_multiple_invocations_tracked(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            simple_text_response("First"),
            simple_text_response("Second"),
        ])
        agent = Agent(model=model, plugins=[collector], callback_handler=None)
        agent("First call")
        agent("Second call")

        assert collector.total_invocations == 2
        assert collector.total_model_calls == 2
        for inv in collector.invocations:
            assert inv.duration >= 0
            assert inv.stop_reason == "end_turn"

    def test_reset_clears_all_data(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("echo_tool", "t1", {"text": "x"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Collect some data")

        assert collector.total_tool_calls > 0
        assert collector.total_invocations > 0

        collector.reset()

        assert collector.total_tool_calls == 0
        assert collector.total_model_calls == 0
        assert collector.total_invocations == 0

    def test_tool_calls_grouped_by_invocation(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("echo_tool", "t1", {"text": "a"}),
            simple_text_response("First done"),
            tool_call_response("echo_tool", "t2", {"text": "b"}),
            tool_call_response("echo_tool", "t3", {"text": "c"}),
            simple_text_response("Second done"),
        ])
        agent = Agent(
            model=model,
            tools=[echo_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("One tool")
        agent("Two tools")

        assert len(collector.invocations) == 2
        assert len(collector.invocations[0].tool_calls) == 1
        assert len(collector.invocations[1].tool_calls) == 2
