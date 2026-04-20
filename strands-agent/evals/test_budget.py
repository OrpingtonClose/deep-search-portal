# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for budget enforcement (BudgetPlugin).

These tests verify that the BudgetPlugin correctly counts tool calls,
enforces limits, and resets between requests — without requiring a live
agent or API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from plugins.budget import BudgetPlugin


class TestBudgetCounting:
    """Verify tool call counting and deduplication."""

    def test_counts_unique_tool_calls(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=100)
        for i in range(5):
            event = MagicMock()
            event.tool_use = {"toolUseId": f"id-{i}", "name": "brave_web_search"}
            event.cancel_tool = False
            plugin.on_before_tool_call(event)
        assert plugin.tool_call_count == 5

    def test_deduplicates_same_tool_use_id(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=100)
        event = MagicMock()
        event.tool_use = {"toolUseId": "same-id", "name": "brave_web_search"}
        event.cancel_tool = False
        plugin.on_before_tool_call(event)
        plugin.on_before_tool_call(event)
        plugin.on_before_tool_call(event)
        assert plugin.tool_call_count == 1

    def test_reset_clears_counters(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=100)
        event = MagicMock()
        event.tool_use = {"toolUseId": "id-1", "name": "task"}
        event.cancel_tool = False
        plugin.on_before_tool_call(event)
        assert plugin.tool_call_count == 1

        plugin.reset()
        assert plugin.tool_call_count == 0
        assert plugin.elapsed < 1.0


class TestBudgetEnforcement:
    """Verify budget limits cancel tool calls."""

    def test_cancels_when_budget_exceeded(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=3)
        for i in range(4):
            event = MagicMock()
            event.tool_use = {"toolUseId": f"id-{i}", "name": "task"}
            event.cancel_tool = False
            plugin.on_before_tool_call(event)

        # The 4th call should have been cancelled
        assert event.cancel_tool
        assert "budget exceeded" in str(event.cancel_tool).lower()

    def test_does_not_cancel_within_budget(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=5)
        for i in range(5):
            event = MagicMock()
            event.tool_use = {"toolUseId": f"id-{i}", "name": "task"}
            event.cancel_tool = False
            plugin.on_before_tool_call(event)
            if i < 4:
                assert not event.cancel_tool


class TestBudgetExplicitReset:
    """Verify explicit reset() clears state (no auto-reset hook)."""

    def test_explicit_reset(self) -> None:
        plugin = BudgetPlugin(max_tool_calls=100)

        # Simulate some tool calls
        for i in range(3):
            event = MagicMock()
            event.tool_use = {"toolUseId": f"id-{i}", "name": "task"}
            event.cancel_tool = False
            plugin.on_before_tool_call(event)

        assert plugin.tool_call_count == 3

        # Explicit reset (called by reset_plugins() in main.py)
        plugin.reset()

        assert plugin.tool_call_count == 0
