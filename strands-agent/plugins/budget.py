"""Budget guardrail plugin — replaces the global budget_callback.

Ported from strands-agent/agent.py lines 48-107 but using the
Strands Plugin system instead of callback handlers.
"""

from __future__ import annotations

import logging
import os
import time

from strands.plugins import Plugin, hook
from strands.hooks import BeforeToolCallEvent, BeforeInvocationEvent

log = logging.getLogger("budget-plugin")

MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "200"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))


class BudgetPlugin(Plugin):
    """Enforce tool call budget and session timeout.

    Unlike the old callback-based approach, this uses BeforeToolCallEvent
    to actually CANCEL tool calls when budget is exceeded (via event.cancel_tool),
    rather than just logging warnings.
    """

    name = "budget-guardrail"

    def __init__(self, max_calls: int = MAX_TOOL_CALLS, timeout: int = SESSION_TIMEOUT):
        super().__init__()
        self._max_calls = max_calls
        self._timeout = timeout
        self._call_count = 0
        self._start_time = time.time()

    def reset(self):
        self._call_count = 0
        self._start_time = time.time()

    @hook
    def on_new_invocation(self, event: BeforeInvocationEvent) -> None:
        """Reset budget at the start of each planner invocation."""
        agent_name = getattr(event.agent, "name", "") if hasattr(event, "agent") else ""
        if agent_name == "planner":
            self.reset()

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        """Count tool calls and enforce budget."""
        self._call_count += 1
        elapsed = time.time() - self._start_time

        if self._call_count % 10 == 0:
            log.info(
                "Budget: %d/%d tool calls, %.0fs elapsed",
                self._call_count,
                self._max_calls,
                elapsed,
            )

        if elapsed > self._timeout:
            event.cancel_tool = (
                f"[SESSION TIMEOUT] {elapsed:.0f}s > {self._timeout}s limit. "
                "Synthesize your answer from the data you have NOW."
            )
            return

        if self._call_count > self._max_calls:
            event.cancel_tool = (
                f"[BUDGET EXCEEDED] {self._call_count} > {self._max_calls} tool calls. "
                "Synthesize your answer from the data you have NOW."
            )
