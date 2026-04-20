# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Mock model provider for Strands-native eval tests.

Adapted from the SDK's ``tests/fixtures/mocked_model_provider.py`` to
provide deterministic agent responses for eval assertions without
requiring API keys or network access.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Message, Messages
from strands.types.event_loop import StopReason
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

T = TypeVar("T", bound=BaseModel)


class MockModel(Model):
    """Deterministic mock that replays pre-defined messages as streaming events.

    Use ``MockModel([msg1, msg2, ...])`` where each message is a Strands
    ``Message`` dict.  On each ``stream()`` call the next message is
    yielded as a sequence of streaming events (matching the SDK wire format).
    """

    def __init__(self, responses: Sequence[Message]) -> None:
        self.responses = [*responses]
        self.index = 0

    def format_chunk(self, event: Any) -> StreamEvent:
        return event

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> Any:
        return None

    def get_config(self) -> Any:
        return {}

    def update_config(self, **model_config: Any) -> None:
        pass

    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        yield None  # pragma: no cover

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        msg = self.responses[self.index]
        for event in self._message_to_events(msg):
            yield event
        self.index += 1

    @staticmethod
    def _message_to_events(msg: Message) -> list[dict[str, Any]]:
        """Convert a Message dict into SDK streaming events."""
        events: list[dict[str, Any]] = []
        stop_reason: StopReason = "end_turn"
        events.append({"messageStart": {"role": "assistant"}})

        for block in msg.get("content", []):
            if "reasoningContent" in block:
                events.append({"contentBlockStart": {"start": {}}})
                events.append(
                    {"contentBlockDelta": {"delta": {"reasoningContent": block["reasoningContent"]}}}
                )
                events.append({"contentBlockStop": {}})

            if "text" in block:
                events.append({"contentBlockStart": {"start": {}}})
                events.append({"contentBlockDelta": {"delta": {"text": block["text"]}}})
                events.append({"contentBlockStop": {}})

            if "toolUse" in block:
                stop_reason = "tool_use"
                tu = block["toolUse"]
                events.append(
                    {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": tu["name"],
                                    "toolUseId": tu["toolUseId"],
                                }
                            }
                        }
                    }
                )
                events.append(
                    {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tu["input"])}}}}
                )
                events.append({"contentBlockStop": {}})

        events.append({"messageStop": {"stopReason": stop_reason}})
        return events


# ── Pre-built response fixtures ──────────────────────────────────────


def simple_text_response(text: str = "The answer is 4.") -> Message:
    """A single text-only assistant message."""
    return {"role": "assistant", "content": [{"text": text}]}


def reasoning_then_text(
    reasoning: str = "Let me think about this step by step...",
    answer: str = "The answer is 42.",
) -> Message:
    """A message with reasoning content followed by a text answer."""
    return {
        "role": "assistant",
        "content": [
            {"reasoningContent": {"reasoningText": {"text": reasoning}}},
            {"text": answer},
        ],
    }


def tool_call_response(
    tool_name: str = "brave_web_search",
    tool_use_id: str = "tool-001",
    tool_input: dict[str, Any] | None = None,
) -> Message:
    """A message requesting a tool call."""
    return {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": tool_name,
                    "toolUseId": tool_use_id,
                    "input": tool_input or {"query": "test search"},
                }
            }
        ],
    }


def multi_tool_then_answer(
    tools: list[tuple[str, str, dict[str, Any]]] | None = None,
    answer: str = "Based on my research, here are the findings.",
) -> list[Message]:
    """A sequence: tool call messages followed by a final text answer.

    Args:
        tools: List of (tool_name, tool_use_id, input) tuples.
        answer: Final answer text.

    Returns:
        List of Messages to pass to MockModel.
    """
    if tools is None:
        tools = [
            ("brave_web_search", "tool-001", {"query": "quantum entanglement"}),
            ("read_url", "tool-002", {"url": "https://example.com/paper.pdf"}),
        ]

    messages: list[Message] = []
    for name, tid, inp in tools:
        messages.append(tool_call_response(name, tid, inp))
    messages.append(simple_text_response(answer))
    return messages
