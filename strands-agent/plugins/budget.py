# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native budget guardrail plugin.

Replaces the legacy ``budget_callback`` global-state function with a proper
Strands SDK Plugin that uses ``@hook`` on ``BeforeToolCallEvent`` to count
tool invocations and enforce per-request limits.
"""

from __future__ import annotations

import logging
import os
import time

from strands.hooks.events import BeforeToolCallEvent
from strands.plugins import Plugin, hook

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "200"))
_SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))


class BudgetPlugin(Plugin):
    """Enforces tool-call and session-timeout budgets.

    Tracks unique tool invocations per request via ``BeforeToolCallEvent``.
    When the budget is exceeded the tool call is cancelled with a descriptive
    message so the model can wrap up gracefully.

    The plugin is reset explicitly via ``reset()`` before each HTTP request
    in ``main.py``.  No auto-reset hook is used because the singleton
    instance is shared across planner and researcher agents in multi-agent
    mode — an auto-reset on ``BeforeInvocationEvent`` would zero the counter
    when a sub-agent is invoked mid-request.
    """

    name: str = "budget"

    def __init__(
        self,
        max_tool_calls: int = _MAX_TOOL_CALLS,
        session_timeout: int = _SESSION_TIMEOUT,
    ) -> None:
        """Initialise the budget plugin.

        Args:
            max_tool_calls: Maximum tool invocations allowed per request.
            session_timeout: Maximum wall-clock seconds before warning.
        """
        super().__init__()
        self.max_tool_calls = max_tool_calls
        self.session_timeout = session_timeout
        self._tool_call_count = 0
        self._seen_tool_ids: set[str] = set()
        self._start_time = time.time()

    def reset(self) -> None:
        """Reset counters for a new request."""
        self._tool_call_count = 0
        self._seen_tool_ids = set()
        self._start_time = time.time()

    @hook
    def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Count tool calls and enforce limits.

        Each unique ``toolUseId`` is counted once.  When the budget is
        exceeded, ``cancel_tool`` is set so the SDK injects an error
        result and the model can decide to stop.
        """
        tool_use_id = event.tool_use.get("toolUseId", "")
        if tool_use_id in self._seen_tool_ids:
            return
        self._seen_tool_ids.add(tool_use_id)

        self._tool_call_count += 1

        elapsed = time.time() - self._start_time
        if elapsed > self.session_timeout:
            logger.warning(
                "tool_calls=<%d>, elapsed=<%.0fs> | session timeout exceeded",
                self._tool_call_count,
                elapsed,
            )

        if self._tool_call_count > self.max_tool_calls:
            logger.warning(
                "tool_calls=<%d>, max=<%d> | budget exceeded, cancelling tool",
                self._tool_call_count,
                self.max_tool_calls,
            )
            event.cancel_tool = (
                f"Tool call budget exceeded ({self._tool_call_count}/{self.max_tool_calls}). "
                "Please provide your final answer with the information gathered so far."
            )

        if self._tool_call_count % 10 == 0:
            logger.info(
                "tool_calls=<%d>, elapsed=<%.0fs> | budget checkpoint",
                self._tool_call_count,
                elapsed,
            )

    @property
    def tool_call_count(self) -> int:
        """Current tool call count for the active request."""
        return self._tool_call_count

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the current request started."""
        return time.time() - self._start_time
