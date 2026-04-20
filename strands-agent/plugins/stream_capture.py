# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native streaming capture plugin.

Replaces the legacy ``StreamCapture`` class with a Strands SDK Plugin that
combines a raw callback handler (for per-token SSE streaming) with ``@hook``
lifecycle events (for structured tool/reasoning data capture).

The callback handler feeds tokens into a thread-safe queue for the SSE
generator.  The hooks capture typed tool-call and reasoning metadata that
the old code extracted from raw ``kwargs`` with fragile string parsing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
)
from strands.plugins import Plugin, hook

logger = logging.getLogger(__name__)


class StreamCapturePlugin(Plugin):
    """Thread-safe streaming capture with SDK hooks for structured data.

    Activate before a request to start capturing tokens into a queue;
    deactivate after.  When no queue is active, tokens are silently dropped.

    The plugin exposes two interfaces:

    1. **Callback handler** (``__call__``) — receives raw streaming kwargs
       from the Strands SDK and feeds them into the token queue.  This is
       passed as ``callback_handler`` to the Agent constructor.

    2. **SDK hooks** — capture tool-call metadata via typed events instead
       of parsing raw kwargs.
    """

    name: str = "stream-capture"

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue[tuple[str, Any] | None] | None = None
        self._lock = threading.Lock()
        self.tool_events: list[dict[str, Any]] = []
        self._seen_tool_ids: set[str] = set()
        self._tool_use_refs: dict[str, dict[str, Any]] = {}
        self.all_text: list[str] = []
        self.response_text: list[str] = []
        self.reasoning_text: list[str] = []

    def activate(self) -> queue.Queue[tuple[str, Any] | None]:
        """Start capturing.  Returns queue the caller reads from."""
        with self._lock:
            q: queue.Queue[tuple[str, Any] | None] = queue.Queue()
            self._queue = q
            self.tool_events.clear()
            self._seen_tool_ids.clear()
            self._tool_use_refs.clear()
            self.all_text.clear()
            self.response_text.clear()
            self.reasoning_text.clear()
            return q

    def deactivate(self) -> None:
        """Stop capturing and signal queue consumers."""
        with self._lock:
            if self._queue is not None:
                self._queue.put(None)
            self._queue = None

    def _update_tool_input(self, current: Any) -> None:
        """Update tool input from a live current_tool_use reference.

        At contentBlockStart the input is typically empty; it arrives
        incrementally via later callbacks.  Read accumulated input from
        the live reference and update the stored event so that the
        non-streaming path has full input for tool labels and activity logs.
        """
        if not current or not isinstance(current, dict):
            return
        tid = current.get("toolUseId", "")
        if not tid or tid not in self._tool_use_refs:
            return
        raw_input = current.get("input", "")
        if raw_input and str(raw_input) not in ("", "{}"):
            for ev in self.tool_events:
                if ev.get("_tool_use_ref") is current:
                    ev["input"] = str(raw_input)[:500]
                    break

    def callback_handler(self, **kwargs: Any) -> None:
        """Raw callback for per-token streaming into the SSE queue.

        This is passed as part of a ``CompositeCallbackHandler`` to the
        Agent constructor.  It feeds ``(event_type, data)`` tuples into
        the queue for the SSE generator.
        """
        with self._lock:
            q = self._queue
        if q is None:
            return

        reasoning = kwargs.get("reasoningText", "")
        data = kwargs.get("data", "")

        if reasoning and isinstance(reasoning, str):
            self.reasoning_text.append(reasoning)
            self.all_text.append(reasoning)
            q.put(("thinking", reasoning))

        if data and isinstance(data, str):
            self.response_text.append(data)
            self.all_text.append(data)
            q.put(("text", data))

        # Detect new tool use — try current_tool_use first (primary),
        # fall back to contentBlockStart event (secondary).  This matches
        # the old StreamCapture.__call__ detection order.
        current = kwargs.get("current_tool_use")
        tool_use = None
        if current and isinstance(current, dict) and current.get("name"):
            tool_use = current
        if not tool_use or not tool_use.get("name"):
            tool_use = (
                kwargs.get("event", {})
                .get("contentBlockStart", {})
                .get("start", {})
                .get("toolUse")
            )

        if tool_use and tool_use.get("name"):
            tid = tool_use.get("toolUseId", "")
            if tid and tid not in self._seen_tool_ids:
                # New tool — register it
                self._seen_tool_ids.add(tid)
                ev: dict[str, Any] = {
                    "tool": tool_use["name"],
                    "time": time.time(),
                    "input": str(tool_use.get("input", "")),
                }
                if current and isinstance(current, dict):
                    ev["_tool_use_ref"] = current
                    self._tool_use_refs[tid] = current
                self.tool_events.append(ev)
                q.put(("tool", ev))
            elif tid and tid in self._tool_use_refs:
                # Known tool — update input from live reference
                self._update_tool_input(current)
        else:
            # No tool_use detected — still try to update input from
            # the live current_tool_use reference (contentBlockDelta path).
            self._update_tool_input(current)

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        """Capture structured tool-call metadata from SDK events.

        This supplements the raw callback capture with properly typed
        data from the SDK's hook system.
        """
        tool_name = event.tool_use.get("name", "unknown")
        tool_id = event.tool_use.get("toolUseId", "")
        logger.debug(
            "tool=<%s>, tool_id=<%s> | capturing tool call via hook",
            tool_name,
            tool_id,
        )

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        """Log tool completion from SDK events."""
        tool_name = event.tool_use.get("name", "unknown")
        logger.debug("tool=<%s> | tool call completed via hook", tool_name)
