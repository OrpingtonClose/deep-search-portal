# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Session replay eval tests using Strands FileSessionManager.

These tests save agent sessions to disk via the SDK's native session
management, then replay / inspect them to verify formatting and
conversation structure without hitting an LLM again.
"""

from __future__ import annotations

import json
import os
import tempfile

from strands import Agent, tool
from strands.agent.agent_result import AgentResult
from strands.session.file_session_manager import FileSessionManager

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    reasoning_then_text,
    simple_text_response,
    tool_call_response,
)


@tool
def lookup_tool(query: str) -> str:
    """Look up information about a topic."""
    return f"Found: {query} is a well-known concept in computer science."


class TestSessionSaveAndReplay:
    """Verify sessions can be saved and replayed for eval assertions."""

    def test_save_simple_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-simple-001",
                storage_dir=tmpdir,
            )
            model = MockModel([simple_text_response("The answer is 42.")])
            agent = Agent(
                model=model,
                callback_handler=None,
                session_manager=session_mgr,
            )
            result = agent("What is the meaning of life?")

            assert result.stop_reason == "end_turn"
            assert "42" in str(result)

            # Verify session was persisted
            session_dir = os.path.join(tmpdir, "session_eval-simple-001")
            assert os.path.isdir(session_dir)

    def test_session_preserves_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-messages-001",
                storage_dir=tmpdir,
            )
            model = MockModel([simple_text_response("Hello!")])
            agent = Agent(
                model=model,
                callback_handler=None,
                session_manager=session_mgr,
            )
            agent("Hi there")

            # Messages should include user input and assistant response
            messages = agent.messages
            assert len(messages) >= 2
            # First message is user, second is assistant
            user_msg = messages[0]
            assert user_msg["role"] == "user"
            assistant_msg = messages[1]
            assert assistant_msg["role"] == "assistant"

    def test_session_with_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-tools-001",
                storage_dir=tmpdir,
            )
            collector = EvalCollectorPlugin()
            model = MockModel([
                tool_call_response("lookup_tool", "t1", {"query": "binary search"}),
                simple_text_response("Binary search is O(log n)."),
            ])
            agent = Agent(
                model=model,
                tools=[lookup_tool],
                plugins=[collector],
                callback_handler=None,
                session_manager=session_mgr,
            )
            result = agent("Explain binary search")

            # Collector confirms tool was called
            assert collector.total_tool_calls == 1
            assert collector.tool_names == ["lookup_tool"]

            # Session has more messages (user + tool_call + tool_result + answer)
            assert len(agent.messages) >= 4

    def test_multi_turn_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-multi-001",
                storage_dir=tmpdir,
            )
            model = MockModel([
                simple_text_response("Binary search divides the list in half."),
                simple_text_response("It runs in O(log n) time."),
            ])
            agent = Agent(
                model=model,
                callback_handler=None,
                session_manager=session_mgr,
            )

            # Turn 1
            result1 = agent("What is binary search?")
            assert "divides" in str(result1).lower() or "half" in str(result1).lower()

            # Turn 2 — same agent, session preserved
            result2 = agent("What is its time complexity?")
            assert "log" in str(result2).lower()

            # Session should have 4 messages (2 user + 2 assistant)
            assert len(agent.messages) == 4


class TestSessionReplayFormatting:
    """Replay saved sessions to verify response formatting."""

    def test_replay_verifies_text_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-replay-001",
                storage_dir=tmpdir,
            )
            model = MockModel([simple_text_response("Quantum entanglement is spooky action.")])
            agent = Agent(
                model=model,
                callback_handler=None,
                session_manager=session_mgr,
            )
            original_result = agent("Explain quantum entanglement")

            # Now inspect the persisted session files
            session_dir = os.path.join(tmpdir, "session_eval-replay-001")
            agent_dirs = os.listdir(os.path.join(session_dir, "agents"))
            assert len(agent_dirs) >= 1

            # Read message files to verify content was persisted
            agent_path = os.path.join(session_dir, "agents", agent_dirs[0], "messages")
            message_files = sorted(os.listdir(agent_path))
            assert len(message_files) >= 2

            # Verify assistant message content
            for mf in message_files:
                with open(os.path.join(agent_path, mf)) as f:
                    msg_data = json.load(f)
                if msg_data.get("role") == "assistant":
                    content = msg_data.get("content", [])
                    text_blocks = [b for b in content if "text" in b]
                    assert len(text_blocks) >= 1
                    assert "entanglement" in text_blocks[0]["text"].lower()
                    break

    def test_replay_verifies_answer_from_reasoning_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-reasoning-001",
                storage_dir=tmpdir,
            )
            model = MockModel([reasoning_then_text(
                reasoning="Considering the quantum mechanical implications...",
                answer="Entanglement allows instant correlation.",
            )])
            agent = Agent(
                model=model,
                callback_handler=None,
                session_manager=session_mgr,
            )
            result = agent("Explain entanglement")

            # The text answer is persisted in the session message
            messages = agent.messages
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            assert len(assistant_msgs) >= 1

            content = assistant_msgs[0].get("content", [])
            text_blocks = [b for b in content if "text" in b]
            assert len(text_blocks) >= 1
            assert "correlation" in text_blocks[0]["text"].lower()

    def test_replay_verifies_tool_result_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-tool-replay-001",
                storage_dir=tmpdir,
            )
            model = MockModel([
                tool_call_response("lookup_tool", "t1", {"query": "hash table"}),
                simple_text_response("Hash tables provide O(1) lookup."),
            ])
            agent = Agent(
                model=model,
                tools=[lookup_tool],
                callback_handler=None,
                session_manager=session_mgr,
            )
            agent("Explain hash tables")

            # Verify tool result is in the message history
            messages = agent.messages
            tool_result_msgs = [
                m for m in messages
                if m["role"] == "user"
                and any("toolResult" in b for b in m.get("content", []))
            ]
            assert len(tool_result_msgs) >= 1

            # Verify the tool result contains the lookup output
            tr_content = tool_result_msgs[0]["content"]
            tool_results = [b for b in tr_content if "toolResult" in b]
            assert len(tool_results) >= 1


class TestSessionMetricsPersistence:
    """Verify that metrics from sessions can be used for eval scoring."""

    def test_metrics_available_after_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = FileSessionManager(
                session_id="eval-metrics-001",
                storage_dir=tmpdir,
            )
            collector = EvalCollectorPlugin()
            model = MockModel([
                tool_call_response("lookup_tool", "t1", {"query": "sorting"}),
                simple_text_response("Quicksort is O(n log n) average."),
            ])
            agent = Agent(
                model=model,
                tools=[lookup_tool],
                plugins=[collector],
                callback_handler=None,
                session_manager=session_mgr,
            )
            result = agent("Explain quicksort")

            # SDK metrics
            assert result.metrics.cycle_count == 2
            assert "lookup_tool" in result.metrics.tool_metrics
            assert result.metrics.tool_metrics["lookup_tool"].success_count == 1

            # Collector metrics (hooks-based)
            assert collector.total_tool_calls == 1
            assert collector.invocations[0].stop_reason == "end_turn"

            # Both agree
            assert (
                result.metrics.tool_metrics["lookup_tool"].call_count
                == collector.total_tool_calls
            )
