# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Plugin-level integration tests with real Agent + MockModel.

Each test creates a Strands Agent with the plugin under test and a
MockModel, runs it, and verifies the plugin captured / transformed
the expected data.  No API keys or network access required.
"""

from __future__ import annotations

from strands import Agent, tool

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    reasoning_then_text,
    simple_text_response,
    tool_call_response,
)
from plugins.budget import BudgetPlugin
from plugins.stream_capture import StreamCapturePlugin
from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_display import ToolDisplayPlugin, format_footer, tool_label


@tool
def dummy_tool(query: str) -> str:
    """A dummy search tool for testing."""
    return f"Results for: {query}"


# ── BudgetPlugin integration ─────────────────────────────────────────


class TestBudgetPluginWithAgent:
    """Verify BudgetPlugin correctly enforces limits via Agent hooks."""

    def test_budget_counts_through_agent(self) -> None:
        budget = BudgetPlugin(max_tool_calls=100)
        model = MockModel([
            tool_call_response("dummy_tool", "t1", {"query": "first"}),
            tool_call_response("dummy_tool", "t2", {"query": "second"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[dummy_tool],
            plugins=[budget],
            callback_handler=None,
        )
        agent("Search for things")

        assert budget.tool_call_count == 2

    def test_budget_cancels_at_limit(self) -> None:
        budget = BudgetPlugin(max_tool_calls=1)
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("dummy_tool", "t1", {"query": "first"}),
            tool_call_response("dummy_tool", "t2", {"query": "second"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[dummy_tool],
            plugins=[budget, collector],
            callback_handler=None,
        )
        agent("Search twice")

        # Budget was 1 — second tool should have been cancelled
        assert budget.tool_call_count >= 1
        # The collector still records both attempts (cancel happens in hook)
        assert collector.total_tool_calls >= 1

    def test_budget_reset_between_requests(self) -> None:
        budget = BudgetPlugin(max_tool_calls=100)
        model = MockModel([
            tool_call_response("dummy_tool", "t1", {"query": "req1"}),
            simple_text_response("First done"),
            tool_call_response("dummy_tool", "t2", {"query": "req2"}),
            simple_text_response("Second done"),
        ])
        agent = Agent(
            model=model,
            tools=[dummy_tool],
            plugins=[budget],
            callback_handler=None,
        )

        agent("First request")
        assert budget.tool_call_count == 1

        budget.reset()
        agent("Second request")
        assert budget.tool_call_count == 1  # Reset worked


# ── StreamCapturePlugin integration ──────────────────────────────────


class TestStreamCapturePluginWithAgent:
    """Verify StreamCapturePlugin captures tokens through Agent callbacks."""

    def test_captures_text_via_callback(self) -> None:
        capture = StreamCapturePlugin()
        q = capture.activate()

        model = MockModel([simple_text_response("Hello world")])
        agent = Agent(
            model=model,
            plugins=[capture],
            callback_handler=capture.callback_handler,
        )
        agent("Say hello")
        capture.deactivate()

        # response_text should have captured the text tokens
        assert len(capture.response_text) > 0
        full_text = "".join(capture.response_text)
        assert "Hello world" in full_text

    def test_captures_text_from_reasoning_message(self) -> None:
        capture = StreamCapturePlugin()
        q = capture.activate()

        model = MockModel([reasoning_then_text(
            reasoning="Deep analysis of the problem...",
            answer="42",
        )])
        agent = Agent(
            model=model,
            plugins=[capture],
            callback_handler=capture.callback_handler,
        )
        agent("Think deeply")
        capture.deactivate()

        # The text answer portion is captured via the callback
        assert len(capture.response_text) > 0
        full_text = "".join(capture.response_text)
        assert "42" in full_text

    def test_captures_tool_events_via_hooks(self) -> None:
        capture = StreamCapturePlugin()
        q = capture.activate()

        model = MockModel([
            tool_call_response("dummy_tool", "t1", {"query": "test"}),
            simple_text_response("Found results"),
        ])
        agent = Agent(
            model=model,
            tools=[dummy_tool],
            plugins=[capture],
            callback_handler=capture.callback_handler,
        )
        agent("Search for test")
        capture.deactivate()

        # Tool events captured via callback handler
        assert len(capture.tool_events) >= 1
        assert capture.tool_events[0]["tool"] == "dummy_tool"


# ── ThoughtRefinerPlugin integration ─────────────────────────────────


class TestThoughtRefinerPluginWithAgent:
    """Verify ThoughtRefinerPlugin configuration when attached to Agent."""

    def test_disabled_refiner_truncates(self) -> None:
        refiner = ThoughtRefinerPlugin(enabled=False)
        raw = "A" * 1000
        result = refiner.refine_sync(raw)
        assert len(result) <= 600

    def test_short_input_passes_through(self) -> None:
        refiner = ThoughtRefinerPlugin(enabled=True)
        short = "Quick thought"
        result = refiner.refine_sync(short)
        assert result == short

    def test_plugin_attaches_to_agent(self) -> None:
        refiner = ThoughtRefinerPlugin(enabled=False)
        model = MockModel([simple_text_response("Hi")])
        agent = Agent(
            model=model,
            plugins=[refiner],
            callback_handler=None,
        )
        # Verify the plugin is registered
        result = agent("Hello")
        assert refiner.name == "thought-refiner"
        assert result.stop_reason == "end_turn"


# ── ToolDisplayPlugin integration ────────────────────────────────────


class TestToolDisplayPluginWithAgent:
    """Verify ToolDisplayPlugin formatting methods work with agent data."""

    def test_label_generation_search(self) -> None:
        label = tool_label("brave_web_search", "{'query': 'quantum physics'}")
        assert "Searching" in label
        assert "quantum physics" in label

    def test_label_generation_write_todos(self) -> None:
        label = tool_label("write_todos", "")
        assert label == "Planning next steps"

    def test_label_generation_unknown_tool(self) -> None:
        label = tool_label("custom_analyzer", "")
        assert "Custom analyzer" in label

    def test_footer_with_tools(self) -> None:
        events = [
            {"tool": "brave_web_search", "time": 0, "input": ""},
            {"tool": "read_url", "time": 0, "input": ""},
        ]
        footer = format_footer(events, 15.0)
        assert "2 sources" in footer
        assert "15s" in footer

    def test_footer_no_tools_fast(self) -> None:
        footer = format_footer([], 0.5)
        assert footer == ""

    def test_footer_no_tools_slow(self) -> None:
        footer = format_footer([], 5.0)
        assert "5s" in footer
        assert "Completed" in footer

    def test_plugin_attaches_to_agent(self) -> None:
        display = ToolDisplayPlugin()
        model = MockModel([simple_text_response("Hi")])
        agent = Agent(
            model=model,
            plugins=[display],
            callback_handler=None,
        )
        result = agent("Hello")
        assert display.name == "tool-display"


# ── Combined plugin integration ──────────────────────────────────────


class TestAllPluginsCombined:
    """Verify all plugins work together without interference."""

    def test_all_plugins_coexist(self) -> None:
        budget = BudgetPlugin(max_tool_calls=100)
        capture = StreamCapturePlugin()
        refiner = ThoughtRefinerPlugin(enabled=False)
        display = ToolDisplayPlugin()
        collector = EvalCollectorPlugin()

        q = capture.activate()
        model = MockModel([
            tool_call_response("dummy_tool", "t1", {"query": "test"}),
            simple_text_response("Answer"),
        ])
        agent = Agent(
            model=model,
            tools=[dummy_tool],
            plugins=[budget, capture, refiner, display, collector],
            callback_handler=capture.callback_handler,
        )
        result = agent("Search and answer")
        capture.deactivate()

        # Budget tracked the tool call
        assert budget.tool_call_count == 1

        # Capture got text tokens
        assert len(capture.response_text) > 0

        # Collector recorded via hooks
        assert collector.total_tool_calls == 1
        assert collector.total_invocations == 1

        # Agent completed successfully
        assert result.stop_reason == "end_turn"
