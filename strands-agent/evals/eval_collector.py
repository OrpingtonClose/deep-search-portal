# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Hooks-based eval data collector plugin.

Uses Strands SDK ``@hook`` decorators to capture structured data during
agent runs — tool calls, model responses, timing, and invocation results.
This replaces regex-based SSE parsing for eval assertions with typed,
structured data captured at the SDK lifecycle level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.plugins import Plugin, hook


@dataclass
class ToolCallRecord:
    """Structured record of a single tool invocation captured via hooks."""

    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]
    duration: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class ModelCallRecord:
    """Structured record of a single model inference captured via hooks."""

    duration: float = 0.0


@dataclass
class InvocationRecord:
    """Structured record of a complete agent invocation."""

    duration: float = 0.0
    stop_reason: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_calls: list[ModelCallRecord] = field(default_factory=list)


class EvalCollectorPlugin(Plugin):
    """Captures structured eval data via SDK hooks.

    Attach this plugin to an Agent during test runs to collect typed
    records of every tool call, model call, and invocation — without
    parsing SSE streams or scraping text output.

    Usage::

        collector = EvalCollectorPlugin()
        agent = Agent(model=mock, plugins=[collector], callback_handler=None)
        result = agent("What is 2+2?")

        assert len(collector.tool_calls) == 0
        assert collector.invocations[0].stop_reason == "end_turn"
    """

    name: str = "eval-collector"

    def __init__(self) -> None:
        super().__init__()
        self.tool_calls: list[ToolCallRecord] = []
        self.model_calls: list[ModelCallRecord] = []
        self.invocations: list[InvocationRecord] = []
        self._current_invocation: InvocationRecord | None = None
        self._invocation_start: float = 0.0
        self._tool_starts: dict[str, float] = {}
        self._model_call_start: float = 0.0

    def reset(self) -> None:
        """Clear all collected data for a new test run."""
        self.tool_calls.clear()
        self.model_calls.clear()
        self.invocations.clear()
        self._current_invocation = None
        self._invocation_start = 0.0
        self._tool_starts.clear()
        self._model_call_start = 0.0

    @hook
    def on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Start tracking a new agent invocation."""
        self._invocation_start = time.time()
        self._current_invocation = InvocationRecord()

    @hook
    def on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Finalise the invocation record with result data."""
        if self._current_invocation is None:
            return
        self._current_invocation.duration = time.time() - self._invocation_start
        if event.result is not None:
            self._current_invocation.stop_reason = event.result.stop_reason
        self.invocations.append(self._current_invocation)
        self._current_invocation = None

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        """Record tool call start time."""
        tool_id = event.tool_use.get("toolUseId", "")
        self._tool_starts[tool_id] = time.time()

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        """Record completed tool call with timing and result."""
        tool_id = event.tool_use.get("toolUseId", "")
        start = self._tool_starts.pop(tool_id, time.time())
        duration = time.time() - start

        record = ToolCallRecord(
            tool_name=event.tool_use.get("name", "unknown"),
            tool_use_id=tool_id,
            tool_input=event.tool_use.get("input", {}),
            duration=duration,
            success=event.exception is None,
            error=str(event.exception) if event.exception else None,
        )
        self.tool_calls.append(record)
        if self._current_invocation is not None:
            self._current_invocation.tool_calls.append(record)

    @hook
    def on_before_model(self, event: BeforeModelCallEvent) -> None:
        """Record model call start time."""
        self._model_call_start = time.time()

    @hook
    def on_after_model(self, event: AfterModelCallEvent) -> None:
        """Record completed model call with timing."""
        duration = time.time() - self._model_call_start
        record = ModelCallRecord(duration=duration)
        self.model_calls.append(record)
        if self._current_invocation is not None:
            self._current_invocation.model_calls.append(record)

    # ── Convenience properties for assertions ─────────────────────────

    @property
    def tool_names(self) -> list[str]:
        """List of all tool names called, in order."""
        return [tc.tool_name for tc in self.tool_calls]

    @property
    def unique_tool_names(self) -> set[str]:
        """Set of unique tool names called."""
        return {tc.tool_name for tc in self.tool_calls}

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls across all invocations."""
        return len(self.tool_calls)

    @property
    def failed_tool_calls(self) -> list[ToolCallRecord]:
        """Tool calls that resulted in an error."""
        return [tc for tc in self.tool_calls if not tc.success]

    @property
    def total_model_calls(self) -> int:
        """Total number of model inference calls."""
        return len(self.model_calls)

    @property
    def total_invocations(self) -> int:
        """Total number of agent invocations."""
        return len(self.invocations)
