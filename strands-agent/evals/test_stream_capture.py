# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for streaming capture (StreamCapturePlugin).

Verifies thread-safe queue capture, token deduplication, and
activate/deactivate lifecycle — without requiring a live agent.
"""

from __future__ import annotations

import queue
import threading

import pytest

from plugins.stream_capture import StreamCapturePlugin


class TestActivateDeactivate:
    """Verify activate/deactivate lifecycle."""

    def test_activate_returns_queue(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        assert isinstance(q, queue.Queue)
        plugin.deactivate()

    def test_deactivate_sends_sentinel(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        plugin.deactivate()
        assert q.get(timeout=1) is None

    def test_drops_tokens_when_inactive(self) -> None:
        plugin = StreamCapturePlugin()
        # No activate — tokens should be silently dropped
        plugin.callback_handler(data="hello")
        assert len(plugin.response_text) == 0


class TestTokenCapture:
    """Verify text and reasoning token capture."""

    def test_captures_text_tokens(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        plugin.callback_handler(data="hello ")
        plugin.callback_handler(data="world")
        plugin.deactivate()

        items = _drain_queue(q)
        text_items = [d for t, d in items if t == "text"]
        assert text_items == ["hello ", "world"]
        assert plugin.response_text == ["hello ", "world"]

    def test_captures_reasoning_tokens(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        plugin.callback_handler(reasoningText="thinking about ")
        plugin.callback_handler(reasoningText="the answer")
        plugin.deactivate()

        items = _drain_queue(q)
        thinking_items = [d for t, d in items if t == "thinking"]
        assert thinking_items == ["thinking about ", "the answer"]
        assert plugin.reasoning_text == ["thinking about ", "the answer"]

    def test_captures_both_text_and_reasoning(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        plugin.callback_handler(reasoningText="thinking")
        plugin.callback_handler(data="answer")
        plugin.deactivate()

        assert plugin.reasoning_text == ["thinking"]
        assert plugin.response_text == ["answer"]
        assert plugin.all_text == ["thinking", "answer"]


class TestToolCapture:
    """Verify tool event capture and deduplication."""

    def test_captures_tool_from_content_block_start(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        plugin.callback_handler(
            event={
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "name": "brave_web_search",
                            "toolUseId": "tool-1",
                            "input": "{'query': 'test'}",
                        }
                    }
                }
            }
        )
        plugin.deactivate()

        assert len(plugin.tool_events) == 1
        assert plugin.tool_events[0]["tool"] == "brave_web_search"

    def test_deduplicates_tool_by_id(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        tool_use = {
            "name": "brave_web_search",
            "toolUseId": "tool-1",
            "input": "{}",
        }
        plugin.callback_handler(
            event={"contentBlockStart": {"start": {"toolUse": tool_use}}}
        )
        plugin.callback_handler(
            event={"contentBlockStart": {"start": {"toolUse": tool_use}}}
        )
        plugin.deactivate()

        assert len(plugin.tool_events) == 1


class TestThreadSafety:
    """Verify thread-safe access to the queue."""

    def test_concurrent_writes(self) -> None:
        plugin = StreamCapturePlugin()
        q = plugin.activate()
        errors = []

        def writer(n: int) -> None:
            try:
                for i in range(100):
                    plugin.callback_handler(data=f"thread-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        plugin.deactivate()

        assert not errors
        assert len(plugin.response_text) == 400  # 4 threads × 100 tokens


def _drain_queue(q: queue.Queue) -> list[tuple[str, str]]:
    """Drain all items from a queue until sentinel (None)."""
    items = []
    while True:
        item = q.get(timeout=5)
        if item is None:
            break
        items.append(item)
    return items
