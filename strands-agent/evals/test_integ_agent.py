# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""End-to-end integration evals with real Venice model.

These evals create a Strands Agent with the real Venice model provider
and verify response quality, formatting, tool usage, and multi-turn
conversation coherence.

Requires: VENICE_API_KEY

Usage::

    pytest evals/test_integ_agent.py -v
"""

from __future__ import annotations

import os
import re

import pytest
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models.openai import OpenAIModel

from evals.eval_collector import EvalCollectorPlugin
from plugins.budget import BudgetPlugin
from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_display import ToolDisplayPlugin

pytestmark = pytest.mark.integ


# ── Helpers ───────────────────────────────────────────────────────────


def _build_venice_model() -> OpenAIModel:
    """Build a Venice model for integration evals."""
    api_key = os.environ.get("VENICE_API_KEY", "")
    if not api_key:
        pytest.skip("VENICE_API_KEY not set")

    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1"),
        },
        model_id=os.environ.get("VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic"),
        params={
            "extra_body": {
                "venice_parameters": {"include_venice_system_prompt": False},
                "reasoning": {"effort": "high"},
            }
        },
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python math expression to evaluate.

    Returns:
        The result as a string.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 — eval is safe here with no builtins
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def fact_lookup(topic: str) -> str:
    """Look up a fact about a topic.

    Args:
        topic: The topic to look up.

    Returns:
        A factual statement about the topic.
    """
    facts = {
        "earth": "Earth is the third planet from the Sun, with a mean radius of 6,371 km.",
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "quantum": "Quantum entanglement allows particles to be correlated regardless of distance.",
    }
    key = topic.lower().strip()
    for k, v in facts.items():
        if k in key:
            return v
    return f"No facts available for: {topic}"


def _build_agent(
    tools: list | None = None,
    system_prompt: str = "You are a helpful research assistant. Answer questions accurately and concisely.",
) -> tuple[Agent, EvalCollectorPlugin]:
    """Build an agent with eval collector for assertions."""
    model = _build_venice_model()
    collector = EvalCollectorPlugin()
    plugins = [
        BudgetPlugin(max_tool_calls=10),
        ThoughtRefinerPlugin(enabled=False),
        ToolDisplayPlugin(),
        collector,
    ]
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
        plugins=plugins,
        callback_handler=None,
        conversation_manager=SlidingWindowConversationManager(window_size=10),
    )
    return agent, collector


# ── Response quality evals ────────────────────────────────────────────


class TestAgentResponseQuality:
    """Verify agent produces coherent, well-structured responses."""

    def test_simple_factual_question(self, venice_api_key: str) -> None:
        agent, collector = _build_agent()
        result = agent("What is the capital of France?")

        text = str(result).lower()
        assert "paris" in text
        assert collector.total_invocations == 1
        assert result.stop_reason == "end_turn"

    def test_reasoning_question(self, venice_api_key: str) -> None:
        agent, collector = _build_agent()
        result = agent("If a train travels 120km in 2 hours, what is its average speed?")

        text = str(result).lower()
        assert "60" in text
        assert collector.total_invocations == 1

    def test_response_not_empty(self, venice_api_key: str) -> None:
        agent, collector = _build_agent()
        result = agent("Explain photosynthesis in one sentence.")

        text = str(result).strip()
        assert len(text) > 20, f"response too short: {text}"
        assert collector.total_model_calls >= 1


# ── Tool usage evals ─────────────────────────────────────────────────


class TestAgentToolUsage:
    """Verify agent correctly uses tools when available."""

    def test_uses_calculator_for_math(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[calculator])
        result = agent("What is 347 * 29? Use the calculator tool.")

        assert collector.total_tool_calls >= 1
        assert "calculator" in collector.tool_names
        text = str(result)
        # Model may format with commas (10,063) or plain (10063)
        assert "10063" in text.replace(",", "")

    def test_uses_fact_lookup(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[fact_lookup])
        result = agent("Look up a fact about Python programming language.")

        assert collector.total_tool_calls >= 1
        assert "fact_lookup" in collector.tool_names
        text = str(result).lower()
        assert "python" in text

    def test_tool_metrics_populated(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[calculator])
        result = agent("Calculate 100 / 4 using the calculator.")

        assert result.metrics.cycle_count >= 2  # tool call + answer
        tool_names_in_metrics = list(result.metrics.tool_metrics.keys())
        assert len(tool_names_in_metrics) >= 1
        assert result.metrics.tool_metrics[tool_names_in_metrics[0]].success_count >= 1


# ── Multi-turn conversation evals ────────────────────────────────────


class TestAgentMultiTurn:
    """Verify context preservation across conversation turns."""

    def test_remembers_previous_context(self, venice_api_key: str) -> None:
        agent, collector = _build_agent()

        # Turn 1: establish context
        result1 = agent("My name is Alice and I work at CERN. Please remember both of these facts.")
        assert result1.stop_reason == "end_turn"

        # Turn 2: reference previous context
        result2 = agent("Repeat back to me: what is my name and where do I work?")
        text = str(result2).lower()
        assert "alice" in text, f"model forgot name 'Alice': {text[:200]}"
        # CERN may appear as "cern" or "CERN" — model should recall it
        assert "cern" in text, f"model forgot workplace 'CERN': {text[:200]}"
        assert collector.total_invocations == 2

    def test_multi_turn_with_tools(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[calculator])

        # Turn 1: first calculation
        result1 = agent("What is 15 * 8? Use the calculator.")
        assert "120" in str(result1)

        # Turn 2: reference previous result
        result2 = agent("Now add 30 to the previous result using the calculator.")
        text = str(result2)
        assert "150" in text
        assert collector.total_tool_calls >= 2


# ── Budget enforcement evals ─────────────────────────────────────────


class TestAgentBudgetEnforcement:
    """Verify budget plugin enforces limits with real model."""

    def test_budget_limits_tool_calls(self, venice_api_key: str) -> None:
        model = _build_venice_model()
        collector = EvalCollectorPlugin()
        budget = BudgetPlugin(max_tool_calls=1)

        agent = Agent(
            model=model,
            system_prompt="You are a calculator assistant. Always use the calculator tool.",
            tools=[calculator],
            plugins=[budget, collector],
            callback_handler=None,
        )
        # Ask for multiple calculations — budget should cap at 1
        result = agent("Calculate 2+2, then 3+3, then 4+4. Use the calculator for each.")

        # Budget cancels tools after the limit, but the model may attempt
        # several before stopping.  The key assertion is that cancellation fired
        # (tool_call_count exceeded the max of 1).
        assert budget.tool_call_count > budget.max_tool_calls


# ── Metrics completeness evals ───────────────────────────────────────


class TestAgentMetricsIntegrity:
    """Verify SDK metrics are complete and consistent with real model."""

    def test_cycle_durations_positive(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[calculator])
        result = agent("What is 7 * 6? Use the calculator.")

        for duration in result.metrics.cycle_durations:
            assert duration > 0, "cycle duration should be positive with real model"

    def test_traces_have_model_children(self, venice_api_key: str) -> None:
        agent, collector = _build_agent()
        result = agent("Say hello.")

        assert len(result.metrics.traces) >= 1
        cycle_trace = result.metrics.traces[0]
        assert cycle_trace.duration() > 0

    def test_collector_and_metrics_agree(self, venice_api_key: str) -> None:
        agent, collector = _build_agent(tools=[fact_lookup])
        result = agent("Look up a fact about Earth.")

        if collector.total_tool_calls > 0:
            # Both should report the same tool
            collector_tools = collector.unique_tool_names
            metrics_tools = set(result.metrics.tool_metrics.keys())
            assert collector_tools == metrics_tools, (
                f"collector={collector_tools} vs metrics={metrics_tools}"
            )
