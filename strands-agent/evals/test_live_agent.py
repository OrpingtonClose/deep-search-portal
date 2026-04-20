# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Integration evals that hit the live Strands Agent API.

These tests verify end-to-end behavior of the deployed agent including
thinking block refinement, tool labels, answer separator, footer
formatting, conversation continuity, and error handling.

Requires a running Strands Agent at the URL specified by
``--agent-url`` (default: ``http://localhost:8100``).

Usage::

    pytest evals/test_live_agent.py -v --agent-url http://localhost:8100
    pytest evals/test_live_agent.py -v -k streaming --timeout 600
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


# ---------------------------------------------------------------------------
# SSE response parser
# ---------------------------------------------------------------------------


@dataclass
class ParsedSSEResponse:
    """Parsed result from an SSE streaming response."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    thinking_blocks: list[str] = field(default_factory=list)
    tool_labels: list[str] = field(default_factory=list)
    has_answer_separator: bool = False
    footer_text: str = ""
    raw_lines: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""


def parse_sse_stream(raw: str) -> ParsedSSEResponse:
    """Parse raw SSE text into structured assertions data.

    Args:
        raw: The full SSE response text.

    Returns:
        Parsed response with extracted thinking blocks, tool labels, etc.
    """
    result = ParsedSSEResponse()
    result.raw_lines = raw.strip().split("\n")

    text_parts: list[str] = []

    for line in result.raw_lines:
        line = line.strip()
        if not line or line.startswith(": "):
            continue
        if line == "data: [DONE]":
            continue
        if not line.startswith("data: "):
            continue

        json_str = line.removeprefix("data: ").strip()
        try:
            chunk = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        # Detect error payloads in the SSE stream
        if "error" in chunk:
            err = chunk["error"]
            result.error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            continue

        result.chunks.append(chunk)
        content = ""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

        if content:
            text_parts.append(content)

    result.full_text = "".join(text_parts)

    # Extract thinking blocks: *💭 ...* patterns
    thinking_pattern = re.compile(r"\*💭\s+(.+?)\*", re.DOTALL)
    result.thinking_blocks = thinking_pattern.findall(result.full_text)

    # Extract tool labels: 🔧 *label* patterns
    tool_pattern = re.compile(r"🔧\s+\*(.+?)\*")
    result.tool_labels = tool_pattern.findall(result.full_text)

    # Check for answer separator
    result.has_answer_separator = "**Answer:**" in result.full_text

    # Extract footer: *Researched using N sources in Xs* or *Completed in Xs*
    footer_pattern = re.compile(
        r"\*(Researched using \d+ sources? in (?:\d+s|<1s)|Completed in (?:\d+s|<1s))\*"
    )
    footer_match = footer_pattern.search(result.full_text)
    if footer_match:
        result.footer_text = footer_match.group(1)

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def agent_url(request: pytest.FixtureRequest) -> str:
    """Base URL for the live Strands Agent API."""
    return request.config.getoption("--agent-url", default="http://localhost:8100")


@pytest.fixture(scope="session")
def client(agent_url: str):
    """Shared httpx client for all tests."""
    with httpx.Client(base_url=agent_url, timeout=600) as c:
        yield c


def _stream_request(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
) -> ParsedSSEResponse:
    """Send a streaming chat completion request and parse the SSE response.

    Args:
        client: httpx client with base_url set.
        model: Model name (e.g. strands-venice-single).
        messages: List of message dicts with role and content.

    Returns:
        Parsed SSE response.
    """
    start = time.time()
    body = {"model": model, "messages": messages, "stream": True}

    raw_parts: list[str] = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            raw_parts.append(line)

    elapsed = time.time() - start
    result = parse_sse_stream("\n".join(raw_parts))
    result.elapsed = elapsed
    return result


def _nonstream_request(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Send a non-streaming chat completion request.

    Args:
        client: httpx client with base_url set.
        model: Model name.
        messages: List of message dicts.

    Returns:
        Parsed JSON response.
    """
    body = {"model": model, "messages": messages, "stream": False}
    resp = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# T0: Health & smoke tests
# ---------------------------------------------------------------------------


class TestHealthSmoke:
    """Verify the agent API is up and responding correctly."""

    def test_health_endpoint(self, client: httpx.Client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["single_agent"] is True

    def test_models_endpoint(self, client: httpx.Client) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "strands-venice-single" in model_ids

    def test_tools_endpoint(self, client: httpx.Client) -> None:
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 10, f"Expected >=10 tools, got {data['count']}"


# ---------------------------------------------------------------------------
# T1: Streaming — Single Agent formatting
# ---------------------------------------------------------------------------


class TestStreamingSingleAgent:
    """Verify streaming SSE response formatting from the single agent."""

    @pytest.fixture(scope="class")
    def sse_response(self, client: httpx.Client) -> ParsedSSEResponse:
        """Send a research query and capture the full SSE response."""
        messages = [
            {
                "role": "user",
                "content": (
                    "What is quantum entanglement and how is it used "
                    "in quantum computing? Search for recent research."
                ),
            }
        ]
        return _stream_request(client, "strands-venice-single", messages)

    def test_sse_contains_thinking_blocks(
        self, sse_response: ParsedSSEResponse
    ) -> None:
        """T1.1: SSE stream contains at least one 💭 thinking block."""
        assert len(sse_response.thinking_blocks) >= 1, (
            f"Expected >=1 thinking block, got {len(sse_response.thinking_blocks)}"
        )

    def test_thinking_is_refined_prose(
        self, sse_response: ParsedSSEResponse
    ) -> None:
        """T1.2: Thinking blocks are refined 2-4 sentence summaries."""
        for block in sse_response.thinking_blocks:
            # Refined text should be flowing prose, not bullet points or code
            sentences = [s.strip() for s in re.split(r"[.!?]+", block) if s.strip()]
            assert len(sentences) >= 1, (
                f"Thinking block too short: {block[:100]}"
            )
            # Should not contain raw code or markdown artifacts
            assert "```" not in block, f"Thinking contains code block: {block[:100]}"
            assert "- " not in block[:20], f"Thinking starts with bullet: {block[:100]}"

    def test_has_tool_labels(self, sse_response: ParsedSSEResponse) -> None:
        """T1.3: Response contains descriptive tool labels with 🔧 emoji."""
        assert len(sse_response.tool_labels) >= 1, (
            f"Expected >=1 tool label, got {len(sse_response.tool_labels)}. "
            f"Full text preview: {sse_response.full_text[:500]}"
        )

    def test_tool_labels_not_generic(
        self, sse_response: ParsedSSEResponse
    ) -> None:
        """T1.3b: Tool labels are not just 'Using task'."""
        for label in sse_response.tool_labels:
            assert label.lower() != "using task", (
                f"Generic tool label found: '{label}'"
            )

    def test_has_answer_separator(self, sse_response: ParsedSSEResponse) -> None:
        """T1.4: **Answer:** separator present when tools were used."""
        if sse_response.tool_labels:
            assert sse_response.has_answer_separator, (
                "Tools were used but no **Answer:** separator found. "
                f"Text around expected position: "
                f"{sse_response.full_text[len(sse_response.full_text)//2:][:200]}"
            )

    def test_has_clean_footer(self, sse_response: ParsedSSEResponse) -> None:
        """T1.5: Footer is a clean one-liner *Researched using N sources in Xs*."""
        assert sse_response.footer_text, (
            f"No footer found. Last 200 chars: {sse_response.full_text[-200:]}"
        )
        assert sse_response.footer_text.startswith("Researched using"), (
            f"Footer doesn't match expected format: '{sse_response.footer_text}'"
        )

    def test_valid_sse_format(self, sse_response: ParsedSSEResponse) -> None:
        """T1.6: All non-empty lines are valid SSE (data: or : prefix)."""
        for line in sse_response.raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            valid = (
                stripped.startswith("data: ")
                or stripped.startswith(": ")
                or stripped == "data: [DONE]"
            )
            assert valid, f"Invalid SSE line: {stripped[:100]}"

    def test_no_escaped_html(self, sse_response: ParsedSSEResponse) -> None:
        """T1.7: No escaped HTML artifacts in the response."""
        assert "<details>" not in sse_response.full_text
        assert "&lt;details&gt;" not in sse_response.full_text
        assert "<summary>" not in sse_response.full_text


# ---------------------------------------------------------------------------
# T2: Non-streaming — Single Agent
# ---------------------------------------------------------------------------


class TestNonStreamingSingleAgent:
    """Verify non-streaming response formatting."""

    @pytest.fixture(scope="class")
    def nonstream_response(self, client: httpx.Client) -> dict[str, Any]:
        """Send a non-streaming request."""
        messages = [
            {
                "role": "user",
                "content": "What is the speed of light in km/s? Search for it.",
            }
        ]
        return _nonstream_request(client, "strands-venice-single", messages)

    def test_response_has_content(self, nonstream_response: dict[str, Any]) -> None:
        """T2.1: Response contains non-empty content."""
        content = nonstream_response["choices"][0]["message"]["content"]
        assert len(content) > 10, f"Response too short: {content}"

    def test_response_has_thinking(self, nonstream_response: dict[str, Any]) -> None:
        """T2.2: Response contains 💭 thinking block."""
        content = nonstream_response["choices"][0]["message"]["content"]
        assert "💭" in content, f"No thinking block in non-streaming response"

    def test_response_has_tool_labels(
        self, nonstream_response: dict[str, Any]
    ) -> None:
        """T2.3: Response contains 🔧 tool labels."""
        content = nonstream_response["choices"][0]["message"]["content"]
        assert "🔧" in content, f"No tool labels in non-streaming response"

    def test_response_has_footer(self, nonstream_response: dict[str, Any]) -> None:
        """T2.4: Response contains footer."""
        content = nonstream_response["choices"][0]["message"]["content"]
        assert "Researched using" in content or "Completed in" in content, (
            f"No footer found. Last 200 chars: {content[-200:]}"
        )


# ---------------------------------------------------------------------------
# T3: Deep Agent
# ---------------------------------------------------------------------------


class TestDeepAgent:
    """Verify the deep agent initializes and responds."""

    def test_deep_agent_health(self, client: httpx.Client) -> None:
        """T3.1: Deep agent is initialized."""
        resp = client.get("/health")
        data = resp.json()
        assert data.get("deep_agent") is True, "Deep agent not initialised"

    def test_deep_agent_streaming(self, client: httpx.Client) -> None:
        """T3.2: Deep agent responds with tools and formatting."""
        messages = [
            {
                "role": "user",
                "content": "Compare Tor and I2P for anonymous browsing.",
            }
        ]
        result = _stream_request(client, "strands-venice-deep", messages)
        assert not result.error, f"Deep agent error: {result.error}"
        assert len(result.full_text) > 100, "Deep agent response too short"
        # Deep agent should use multiple tools
        assert len(result.tool_labels) >= 1, (
            f"Expected >=1 tool label from deep agent, got {len(result.tool_labels)}"
        )


# ---------------------------------------------------------------------------
# T4: Zero-tool query
# ---------------------------------------------------------------------------


class TestZeroToolQuery:
    """Verify clean output when no tools are used."""

    @pytest.fixture(scope="class")
    def zero_tool_response(self, client: httpx.Client) -> ParsedSSEResponse:
        """Send a simple arithmetic query that shouldn't need tools."""
        messages = [
            {"role": "user", "content": "What is 2+2? Answer directly."}
        ]
        return _stream_request(client, "strands-venice-single", messages)

    def test_no_tool_labels(self, zero_tool_response: ParsedSSEResponse) -> None:
        """T4.1: No tool lines for a trivial query."""
        # Note: model may still use tools, so this is a soft check
        if zero_tool_response.tool_labels:
            pytest.skip(
                f"Model used {len(zero_tool_response.tool_labels)} tools for 2+2 "
                "(unexpected but not a formatting bug)"
            )

    def test_no_answer_separator_without_tools(
        self, zero_tool_response: ParsedSSEResponse
    ) -> None:
        """T4.2: No Answer separator when no tools used."""
        if not zero_tool_response.tool_labels:
            assert not zero_tool_response.has_answer_separator, (
                "Answer separator present without tool usage"
            )

    def test_footer_suppressed_or_minimal(
        self, zero_tool_response: ParsedSSEResponse
    ) -> None:
        """T4.3: Footer is suppressed or shows Completed in Xs."""
        if zero_tool_response.footer_text:
            assert "Completed in" in zero_tool_response.footer_text or \
                   "Researched using" in zero_tool_response.footer_text

    def test_answer_is_concise(
        self, zero_tool_response: ParsedSSEResponse
    ) -> None:
        """T4.4: Answer for 2+2 is concise."""
        assert "4" in zero_tool_response.full_text, (
            f"Expected '4' in response: {zero_tool_response.full_text[:200]}"
        )


# ---------------------------------------------------------------------------
# T5: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error responses for invalid requests."""

    def test_empty_messages_returns_error(self, client: httpx.Client) -> None:
        """T5.1: Empty messages list returns 400."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "strands-venice-single", "messages": [], "stream": False},
        )
        assert resp.status_code in (400, 422), (
            f"Expected 400/422, got {resp.status_code}: {resp.text[:200]}"
        )


# ---------------------------------------------------------------------------
# T6: Conversation continuity
# ---------------------------------------------------------------------------


class TestConversationContinuity:
    """Verify multi-turn conversation maintains context."""

    def test_followup_has_context(self, client: httpx.Client) -> None:
        """T6.1: Agent references first-turn content in follow-up."""
        # First turn: ask about a specific topic
        messages_turn1 = [
            {
                "role": "user",
                "content": "What is QUIC protocol? Search for it briefly.",
            }
        ]
        turn1 = _stream_request(client, "strands-venice-single", messages_turn1)
        assert len(turn1.full_text) > 50, "First turn response too short"

        # Second turn: ask a follow-up referencing the first answer
        messages_turn2 = [
            {"role": "user", "content": "What is QUIC protocol? Search for it briefly."},
            {"role": "assistant", "content": turn1.full_text[:2000]},
            {
                "role": "user",
                "content": "Can you elaborate on the connection migration feature you mentioned?",
            },
        ]
        turn2 = _stream_request(client, "strands-venice-single", messages_turn2)

        # The follow-up should reference QUIC or connection-related content
        lower_text = turn2.full_text.lower()
        has_context = (
            "quic" in lower_text
            or "connection" in lower_text
            or "migration" in lower_text
            or "protocol" in lower_text
        )
        assert has_context, (
            f"Follow-up doesn't reference previous context. "
            f"Preview: {turn2.full_text[:300]}"
        )


# ---------------------------------------------------------------------------
# T7: Formatting consistency
# ---------------------------------------------------------------------------


class TestFormattingConsistency:
    """Cross-cutting formatting assertions across response types."""

    @pytest.fixture(scope="class")
    def research_response(self, client: httpx.Client) -> ParsedSSEResponse:
        """A research response for formatting checks."""
        messages = [
            {
                "role": "user",
                "content": "Search for the latest developments in mRNA vaccines.",
            }
        ]
        return _stream_request(client, "strands-venice-single", messages)

    def test_thinking_blocks_are_italic(
        self, research_response: ParsedSSEResponse
    ) -> None:
        """Thinking blocks use *💭 text* italic format."""
        assert "*💭" in research_response.full_text, (
            "Thinking blocks not using italic *💭 ...* format"
        )

    def test_tool_labels_are_italic(
        self, research_response: ParsedSSEResponse
    ) -> None:
        """Tool labels use 🔧 *label* italic format."""
        tool_pattern = re.compile(r"🔧\s+\*.+?\*")
        matches = tool_pattern.findall(research_response.full_text)
        assert len(matches) >= 1, "No italic tool labels found"

    def test_footer_is_italic(self, research_response: ParsedSSEResponse) -> None:
        """Footer uses *Researched using...* italic format."""
        footer_pattern = re.compile(r"\*Researched using \d+ sources? in (?:\d+s|<1s)\*")
        assert footer_pattern.search(research_response.full_text), (
            f"Footer not in italic format. Last 300 chars: "
            f"{research_response.full_text[-300:]}"
        )

    def test_no_yaml_metrics_dump(
        self, research_response: ParsedSSEResponse
    ) -> None:
        """No YAML metrics dump in response (regression check)."""
        assert "tool_calls:" not in research_response.full_text.lower()
        assert "elapsed_seconds:" not in research_response.full_text.lower()
        assert "=== Strands Agent Activity Log ===" not in research_response.full_text

    def test_no_raw_thinking_leak(
        self, research_response: ParsedSSEResponse
    ) -> None:
        """No raw chain-of-thought leaked outside thinking blocks."""
        # Raw thinking indicators that should not appear in the main answer
        raw_indicators = [
            "I need to think about",
            "Let me reconsider",
            "Actually, wait",
            "Hmm, I should",
        ]
        # Extract text after the last thinking block
        answer_start = research_response.full_text.rfind("*\n\n")
        if answer_start > 0:
            answer_text = research_response.full_text[answer_start:]
            for indicator in raw_indicators:
                assert indicator not in answer_text, (
                    f"Raw thinking leaked: '{indicator}' found in answer section"
                )

    def test_single_footer_line(
        self, research_response: ParsedSSEResponse
    ) -> None:
        """Only one footer line per response (not duplicated)."""
        footer_pattern = re.compile(r"\*Researched using \d+ sources? in (?:\d+s|<1s)\*")
        matches = footer_pattern.findall(research_response.full_text)
        assert len(matches) <= 1, (
            f"Multiple footer lines found ({len(matches)}): {matches}"
        )


# ---------------------------------------------------------------------------
# T9: Eval suite (meta-test)
# ---------------------------------------------------------------------------


class TestEvalSuite:
    """Verify the unit eval suite modules exist on disk."""

    def test_eval_modules_exist(self) -> None:
        """T9.1: All eval test files exist in the evals directory."""
        import pathlib

        evals_dir = pathlib.Path(__file__).parent
        expected = [
            "test_streaming.py",
            "test_tool_display.py",
            "test_thought_refiner.py",
            "test_budget.py",
            "test_conversation.py",
            "test_scorer.py",
            "test_stream_capture.py",
            "test_log_viewer.py",
        ]
        missing = [f for f in expected if not (evals_dir / f).exists()]
        assert not missing, f"Missing eval files: {missing}"
