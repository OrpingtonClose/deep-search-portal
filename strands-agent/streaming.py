# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SSE streaming presentation layer.

Encapsulates the OpenAI-compatible SSE chunk formatting, thinking-block
refinement/flush logic, and tool-label rendering for the streaming path.
Extracted from ``main.py`` to keep that file focused on routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from typing import Any, AsyncGenerator

from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_display import sanitize_for_italic, tool_label

logger = logging.getLogger(__name__)


def openai_chunk(req_id: str, model: str, content: str, finish: bool = False) -> str:
    """Format a single SSE chunk in OpenAI streaming format.

    Args:
        req_id: The request ID for the SSE stream.
        model: The model name.
        content: Text content for the chunk.
        finish: If True, emit a finish_reason="stop" chunk.

    Returns:
        SSE-formatted string ready to yield.
    """
    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def generate_sse(
    req_id: str,
    model: str,
    token_queue: queue.Queue[tuple[str, Any] | None],
    thought_refiner: ThoughtRefinerPlugin | None,
    result_holder: dict[str, Any],
    start_time: float,
    format_inline_log: Any,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Async generator that yields OpenAI-compatible SSE chunks.

    Reads from the token queue populated by the agent thread, refines
    thinking blocks via the thought refiner plugin, and formats tool
    labels as descriptive italic lines.

    Layout::

        *💭 refined summary*
        🔧 *Researching topic X*
        🔧 *Reading URL*
        ---
        **Answer:**
        <final answer text>
        *Researched using N sources in Xs*

    Args:
        req_id: Request identifier for SSE chunks.
        model: Model name for SSE chunks.
        token_queue: Thread-safe queue of (event_type, data) tuples.
        thought_refiner: Plugin for refining thinking blocks (or None).
        result_holder: Shared dict for error/text/metrics from agent thread.
        start_time: Request start timestamp for elapsed calculation.
        format_inline_log: Callable for formatting the activity log footer.
        user_message: The user's query (for the activity log).

    Yields:
        SSE-formatted strings.
    """
    loop = asyncio.get_event_loop()

    # ── Streaming presentation state ──
    thinking_buffer: list[str] = []
    has_answer_text = False
    tool_count = 0

    async def _flush_thinking(is_final: bool = False) -> AsyncGenerator[str, None]:
        """Refine and emit buffered thinking tokens.

        When the refiner is available, the raw thinking is sent to
        a fast LLM for summarisation.  The refined version is emitted
        as italic text.  If refinement fails or is disabled, the raw
        thinking is emitted truncated.

        Args:
            is_final: True when this is the last thinking block
                (agent finished with only thinking, no answer text).
        """
        if not thinking_buffer:
            return

        raw_thinking = "".join(thinking_buffer)
        thinking_buffer.clear()

        if is_final:
            # Thinking IS the answer — emit full text unmodified.
            yield openai_chunk(req_id, model, raw_thinking)
            return

        # Normal case: thinking followed by tools/answer.
        if thought_refiner and thought_refiner.is_available:
            yield openai_chunk(req_id, model, "*💭 ")
            refined = await thought_refiner.refine_async(raw_thinking)
            safe = sanitize_for_italic(refined)
            yield openai_chunk(req_id, model, f"{safe}*\n\n")
        else:
            truncated = raw_thinking[:500]
            if len(raw_thinking) > 500:
                truncated += "…"
            safe = sanitize_for_italic(truncated)
            yield openai_chunk(req_id, model, f"*💭 {safe}*\n\n")

    while True:
        try:
            item = await loop.run_in_executor(
                None, lambda: token_queue.get(timeout=5)
            )
        except queue.Empty:
            yield ": heartbeat\n\n"
            continue

        if item is None:
            # Agent finished — flush remaining thinking.
            async for chunk in _flush_thinking(is_final=not has_answer_text):
                yield chunk
            break

        event_type, data = item

        if event_type == "thinking":
            thinking_buffer.append(data)

        elif event_type == "tool":
            async for chunk in _flush_thinking():
                yield chunk
            tool_count += 1
            tool_name = data.get("tool", "unknown")
            tool_input = data.get("input", "")
            ref = data.get("_tool_use_ref")
            if ref and isinstance(ref, dict):
                await asyncio.sleep(0.15)
                live_input = ref.get("input", "")
                if live_input and str(live_input) not in ("", "{}"):
                    tool_input = str(live_input)[:500]
            label = tool_label(tool_name, tool_input)
            yield openai_chunk(req_id, model, f"🔧 *{label}*\n\n")

        elif event_type == "text":
            async for chunk in _flush_thinking():
                yield chunk
            if not has_answer_text and tool_count > 0:
                yield openai_chunk(req_id, model, "\n---\n\n**Answer:**\n\n")
            has_answer_text = True
            yield openai_chunk(req_id, model, data)

    # Error fallback
    has_streamed = result_holder.get("streamed_text") or result_holder.get("reasoning_text", "")
    if result_holder.get("error") and not has_streamed:
        yield openai_chunk(req_id, model, f"\n\nError: {result_holder['error']}")

    # Footer
    elapsed = round(time.time() - start_time, 2)
    reasoning_for_log = ""
    inline_log = format_inline_log(
        result_holder.get("tool_events", []), elapsed,
        query=user_message, model=model,
        reasoning=reasoning_for_log,
        metrics=result_holder.get("metrics"),
    )
    yield openai_chunk(req_id, model, inline_log)
    yield openai_chunk(req_id, model, "", finish=True)
    yield "data: [DONE]\n\n"


def build_non_streaming_response(
    answer: str,
    captured_tool_events: list[dict[str, Any]],
    captured_reasoning: str,
    thought_refiner: ThoughtRefinerPlugin | None,
    inline_log: str,
) -> str:
    """Build formatted non-streaming response with thinking, tools, and answer.

    Args:
        answer: The agent's final answer text.
        captured_tool_events: List of captured tool event dicts.
        captured_reasoning: Raw reasoning text from the agent.
        thought_refiner: Plugin for refining thinking (or None).
        inline_log: Pre-formatted activity log footer.

    Returns:
        Formatted response string.
    """
    parts: list[str] = []

    # Wrap reasoning if present AND distinct from the answer
    reasoning_is_answer = (
        captured_reasoning.strip()
        and answer.strip() == captured_reasoning.strip()
    )
    if captured_reasoning.strip() and not reasoning_is_answer:
        if thought_refiner and thought_refiner.is_available:
            # Note: this is sync context — caller should await refine_async
            # before calling this function and pass the refined text.
            refined = captured_reasoning[:500]
            if len(captured_reasoning) > 500:
                refined += "…"
        else:
            refined = captured_reasoning[:500]
            if len(captured_reasoning) > 500:
                refined += "…"
        parts.append(f"*💭 {sanitize_for_italic(refined)}*\n\n")

    # Tool call labels
    if captured_tool_events:
        for ev in captured_tool_events:
            label = tool_label(ev.get("tool", "unknown"), ev.get("input", ""))
            parts.append(f"🔧 *{label}*\n\n")
        parts.append("\n---\n\n**Answer:**\n\n")

    parts.append(answer)
    parts.append(inline_log)
    return "".join(parts)
