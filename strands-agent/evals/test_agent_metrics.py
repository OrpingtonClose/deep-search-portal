# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Agent metrics eval assertions.

These tests verify that ``AgentResult.metrics`` (cycle counts, tool
metrics, token usage, traces) are correctly populated after agent runs.
Uses Strands-native ``EventLoopMetrics`` instead of custom counters.
"""

from __future__ import annotations

from strands import Agent, tool

from evals.mock_model import (
    MockModel,
    simple_text_response,
    tool_call_response,
)


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


class TestCycleMetrics:
    """Verify event loop cycle counting via AgentResult.metrics."""

    def test_simple_query_one_cycle(self) -> None:
        model = MockModel([simple_text_response("4")])
        agent = Agent(model=model, callback_handler=None)
        result = agent("What is 2+2?")

        assert result.metrics.cycle_count == 1
        assert len(result.metrics.cycle_durations) == 1
        assert result.metrics.cycle_durations[0] >= 0

    def test_tool_use_adds_cycles(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 2, "b": 3}),
            simple_text_response("The sum is 5."),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers],
            callback_handler=None,
        )
        result = agent("Add 2 and 3")

        # One cycle for tool call, one for final answer
        assert result.metrics.cycle_count == 2
        assert len(result.metrics.cycle_durations) == 2

    def test_multiple_tools_multiple_cycles(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 1, "b": 2}),
            tool_call_response("multiply_numbers", "t2", {"a": 3, "b": 4}),
            simple_text_response("Sum is 3, product is 12."),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers, multiply_numbers],
            callback_handler=None,
        )
        result = agent("Add 1+2 and multiply 3*4")

        assert result.metrics.cycle_count == 3


class TestToolMetrics:
    """Verify per-tool metrics tracking via AgentResult.metrics."""

    def test_tool_metrics_populated(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 5, "b": 3}),
            simple_text_response("8"),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers],
            callback_handler=None,
        )
        result = agent("5 + 3?")

        assert "add_numbers" in result.metrics.tool_metrics
        tm = result.metrics.tool_metrics["add_numbers"]
        assert tm.call_count == 1
        assert tm.success_count == 1
        assert tm.error_count == 0
        assert tm.total_time >= 0

    def test_multiple_tool_types_tracked(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 1, "b": 1}),
            tool_call_response("multiply_numbers", "t2", {"a": 2, "b": 2}),
            simple_text_response("2 and 4"),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers, multiply_numbers],
            callback_handler=None,
        )
        result = agent("Compute both")

        assert "add_numbers" in result.metrics.tool_metrics
        assert "multiply_numbers" in result.metrics.tool_metrics
        assert result.metrics.tool_metrics["add_numbers"].call_count == 1
        assert result.metrics.tool_metrics["multiply_numbers"].call_count == 1


class TestInvocationMetrics:
    """Verify agent invocation tracking via metrics."""

    def test_invocation_recorded(self) -> None:
        model = MockModel([simple_text_response("Hi")])
        agent = Agent(model=model, callback_handler=None)
        result = agent("Hello")

        assert len(result.metrics.agent_invocations) >= 1
        inv = result.metrics.agent_invocations[-1]
        assert len(inv.cycles) == 1

    def test_invocation_cycle_ids_unique(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 1, "b": 1}),
            simple_text_response("2"),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers],
            callback_handler=None,
        )
        result = agent("Add")

        inv = result.metrics.agent_invocations[-1]
        cycle_ids = [c.event_loop_cycle_id for c in inv.cycles]
        assert len(cycle_ids) == len(set(cycle_ids))  # all unique


class TestTraceMetrics:
    """Verify execution trace data via AgentResult.metrics."""

    def test_traces_populated(self) -> None:
        model = MockModel([simple_text_response("Done")])
        agent = Agent(model=model, callback_handler=None)
        result = agent("Simple query")

        assert len(result.metrics.traces) >= 1
        trace = result.metrics.traces[0]
        assert trace.name.startswith("Cycle")
        assert trace.duration() is not None
        assert trace.duration() >= 0

    def test_tool_traces_nested(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "t1", {"a": 1, "b": 2}),
            simple_text_response("3"),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers],
            callback_handler=None,
        )
        result = agent("Add 1+2")

        # First cycle trace should have tool children
        cycle_trace = result.metrics.traces[0]
        assert len(cycle_trace.children) >= 1
        tool_trace = cycle_trace.children[-1]
        assert tool_trace.duration() is not None


class TestStopReason:
    """Verify stop_reason is correctly captured."""

    def test_end_turn_stop_reason(self) -> None:
        model = MockModel([simple_text_response("Final answer")])
        agent = Agent(model=model, callback_handler=None)
        result = agent("Question")

        assert result.stop_reason == "end_turn"
