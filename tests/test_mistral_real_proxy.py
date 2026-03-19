"""Tests for the Mistral (Real) proxy — verified LLM responses via inline Veritas.

Covers:
- LangGraph state reducers
- Draft generation node (with THINKING/ANSWER parsing)
- Verification node (Veritas integration)
- Revision node (hallucination correction)
- Output formatting node
- Conditional routing (clean vs needs_revision)
- Streaming response generator
- FastAPI endpoints (models, chat completions, verify)
- Utility request passthrough
"""

import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add proxies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

# Mock shared module before import (require_env called at import time)
_mock_shared = MagicMock()
_mock_shared.setup_logging.return_value = MagicMock()
_mock_shared.require_env.return_value = "test-key"
_mock_shared.env_int.side_effect = lambda name, default, **kw: default
_mock_shared.http_client.return_value = MagicMock()
_mock_shared.create_app.return_value = MagicMock()
_mock_shared.register_standard_routes = MagicMock()
_mock_shared.make_sse_chunk = MagicMock(
    side_effect=lambda content, **kw: f"data: {json.dumps({'content': content})}\n\n"
)
_mock_shared.ConcurrencyLimiter = MagicMock
_mock_shared.RequestTracker = MagicMock
_mock_shared.is_utility_request = MagicMock(return_value=False)
_mock_shared.stream_passthrough = MagicMock()
sys.modules["shared"] = _mock_shared

# veritas_inquisitor is imported lazily inside functions in mistral_real_proxy,
# so no module-level mock is needed.  Individual tests use patch() to mock
# specific functions (run_reactor, verify_output) when called.

import mistral_real_proxy as mrp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides):
    """Build a minimal MistralRealState dict for testing."""
    state = {
        "req_id": "test-req",
        "messages": [{"role": "user", "content": "What is quantum computing?"}],
        "original_body": {"model": "mistral-real"},
        "draft_text": "",
        "draft_thinking": "",
        "veritas_report": {},
        "veritas_iterations": 0,
        "veritas_artifact_count": 0,
        "hallucination_probability": 0.0,
        "overall_score": 1.0,
        "claims_summary": [],
        "revision_round": 0,
        "needs_revision": False,
        "revised_text": "",
        "final_answer": "",
        "progress_log": [],
        "phase": "generate_draft",
        "elapsed": 0.0,
        "error": "",
    }
    state.update(overrides)
    return state


def _make_veritas_report(
    overall_score=0.85,
    halluc_prob=0.15,
    claims=None,
    revised_output="",
):
    """Build a mock Veritas report."""
    if claims is None:
        claims = [
            {
                "claim_text": "Quantum computers use qubits",
                "status": "verified",
                "confidence": 0.95,
                "evidence_summary": "Well-established physics",
            },
            {
                "claim_text": "Quantum computers are 1000x faster",
                "status": "overconfident",
                "confidence": 0.4,
                "evidence_summary": "Only for specific algorithms",
            },
        ]
    return {
        "report": {
            "overall_score": overall_score,
            "overall_hallucination_probability": halluc_prob,
            "claims": claims,
            "revised_output": revised_output,
            "evidence_links": [],
        },
        "artifact_count": 12,
        "iterations": 5,
        "dag": [],
    }


# ============================================================================
# Test: State reducer
# ============================================================================

class TestReducers:
    def test_append_log_merges(self):
        left = ["msg1"]
        right = ["msg2", "msg3"]
        result = mrp._append_log(left, right)
        assert result == ["msg1", "msg2", "msg3"]

    def test_append_log_empty_left(self):
        assert mrp._append_log([], ["msg1"]) == ["msg1"]

    def test_append_log_empty_right(self):
        assert mrp._append_log(["msg1"], []) == ["msg1"]

    def test_append_log_both_empty(self):
        assert mrp._append_log([], []) == []


# ============================================================================
# Test: node_generate_draft
# ============================================================================

class TestNodeGenerateDraft:
    @pytest.mark.asyncio
    async def test_parses_thinking_and_answer(self):
        """Draft with both THINKING and ANSWER tags should be parsed correctly."""
        llm_response = {
            "content": (
                "<THINKING>\nStep 1: Consider quantum mechanics\n"
                "Step 2: Evaluate qubit properties\n</THINKING>\n\n"
                "<ANSWER>\nQuantum computing uses qubits to perform calculations.\n</ANSWER>"
            ),
            "finish_reason": "stop",
            "usage": {},
        }

        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await mrp.node_generate_draft(_base_state())

        assert "quantum" in result["draft_text"].lower()
        assert "Step 1" in result["draft_thinking"]
        assert result["phase"] == "verify"

    @pytest.mark.asyncio
    async def test_handles_no_tags(self):
        """Draft without THINKING/ANSWER tags should use entire content."""
        llm_response = {
            "content": "Quantum computing is a field of study.",
            "finish_reason": "stop",
            "usage": {},
        }

        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await mrp.node_generate_draft(_base_state())

        assert "Quantum computing" in result["draft_text"]
        assert result["phase"] == "verify"

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        """LLM error should set error phase."""
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value={"error": "timeout"}):
            result = await mrp.node_generate_draft(_base_state())

        assert result["phase"] == "error"
        assert "timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_injects_system_prompt(self):
        """Should inject DRAFT_SYSTEM_PROMPT into messages."""
        captured_messages = []

        async def capture_call(messages, *args, **kwargs):
            captured_messages.extend(messages)
            return {"content": "<THINKING>ok</THINKING>\n<ANSWER>answer</ANSWER>", "finish_reason": "stop", "usage": {}}

        with patch.object(mrp, "call_llm", side_effect=capture_call):
            await mrp.node_generate_draft(_base_state())

        system_msgs = [m for m in captured_messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Mistral (Real)" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_appends_to_existing_system_prompt(self):
        """If messages already have a system prompt, should append to it."""
        state = _base_state(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "test"},
        ])

        async def capture_call(messages, *args, **kwargs):
            return {"content": "<THINKING>ok</THINKING>\n<ANSWER>answer</ANSWER>", "finish_reason": "stop", "usage": {}}

        with patch.object(mrp, "call_llm", side_effect=capture_call) as mock:
            await mrp.node_generate_draft(state)

        # The system prompt should have been augmented, not replaced
        call_messages = mock.call_args[0][0]
        system_content = call_messages[0]["content"]
        assert "You are helpful." in system_content
        assert "Mistral (Real)" in system_content

    @pytest.mark.asyncio
    async def test_answer_after_thinking_without_answer_tags(self):
        """Content after </THINKING> should be treated as answer if no <ANSWER> tags."""
        llm_response = {
            "content": "<THINKING>reasoning here</THINKING>\nThe actual answer is 42.",
            "finish_reason": "stop",
            "usage": {},
        }

        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await mrp.node_generate_draft(_base_state())

        assert "42" in result["draft_text"]


# ============================================================================
# Test: node_verify_draft
# ============================================================================

class TestNodeVerifyDraft:
    @pytest.mark.asyncio
    async def test_clean_draft_no_revision(self):
        """Clean draft should not trigger revision."""
        report = _make_veritas_report(overall_score=0.9, halluc_prob=0.1)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(
                draft_text="Quantum computing uses qubits.",
            ))

        assert result["needs_revision"] is False
        assert result["phase"] == "format_output"
        assert result["overall_score"] == 0.9

    @pytest.mark.asyncio
    async def test_hallucinated_draft_triggers_revision(self):
        """High hallucination probability should trigger revision."""
        report = _make_veritas_report(
            overall_score=0.3,
            halluc_prob=0.7,
            claims=[
                {"claim_text": "Fake claim", "status": "hallucinated", "confidence": 0.1, "evidence_summary": "No evidence"},
            ],
        )

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(
                draft_text="Some hallucinated text.",
            ))

        assert result["needs_revision"] is True
        assert result["phase"] == "revise"
        assert result["hallucination_probability"] == 0.7

    @pytest.mark.asyncio
    async def test_empty_draft_skips_verification(self):
        """Empty draft should skip verification entirely."""
        result = await mrp.node_verify_draft(_base_state(draft_text=""))

        assert result["needs_revision"] is False
        assert result["phase"] == "format_output"

    @pytest.mark.asyncio
    async def test_max_revisions_prevents_further_revision(self):
        """Should not revise if already at max revision rounds."""
        report = _make_veritas_report(overall_score=0.3, halluc_prob=0.7)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(
                draft_text="Some text.",
                revision_round=1,  # already at MAX_REVISION_ROUNDS default of 1
            ))

        assert result["needs_revision"] is False

    @pytest.mark.asyncio
    async def test_negative_scores_normalised(self):
        """Veritas returning -1 (error) should be normalised to 0.5."""
        report = _make_veritas_report(overall_score=-1, halluc_prob=-1)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(
                draft_text="Some text.",
            ))

        assert result["hallucination_probability"] == 0.5
        assert result["overall_score"] == 0.5

    @pytest.mark.asyncio
    async def test_claims_summary_populated(self):
        """Claims should be summarised with status and evidence."""
        report = _make_veritas_report()

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(
                draft_text="Quantum computing uses qubits.",
            ))

        assert len(result["claims_summary"]) == 2
        statuses = [c["status"] for c in result["claims_summary"]]
        assert "verified" in statuses
        assert "overconfident" in statuses

    @pytest.mark.asyncio
    async def test_extracts_user_query_from_messages(self):
        """Should extract the last user message as original_query for Veritas."""
        report = _make_veritas_report()
        captured_args = {}

        async def capture_reactor(text, query, req_id, *args, **kwargs):
            captured_args["query"] = query
            return report

        with patch("veritas_inquisitor.run_reactor", side_effect=capture_reactor):
            await mrp.node_verify_draft(_base_state(
                draft_text="Some text.",
                messages=[
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "What about quantum computing?"},
                ],
            ))

        assert captured_args["query"] == "What about quantum computing?"


# ============================================================================
# Test: node_revise_response
# ============================================================================

class TestNodeReviseResponse:
    @pytest.mark.asyncio
    async def test_produces_revised_text(self):
        """Revision should produce corrected text."""
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value={
            "content": "Corrected: Quantum computers use qubits for specific types of calculations.",
            "finish_reason": "stop",
            "usage": {},
        }):
            result = await mrp.node_revise_response(_base_state(
                draft_text="Quantum computers are 1000x faster than classical.",
                veritas_report={
                    "revised_output": "Quantum computers can be faster for specific problems.",
                },
                claims_summary=[
                    {"claim": "1000x faster", "status": "hallucinated", "confidence": 0.1, "evidence": "No evidence"},
                ],
                overall_score=0.3,
                hallucination_probability=0.7,
            ))

        assert "Corrected" in result["revised_text"]
        assert result["revision_round"] == 1
        assert result["phase"] == "format_output"

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self):
        """If revision LLM call fails, should fall back to Veritas revised output."""
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value={"error": "timeout"}):
            result = await mrp.node_revise_response(_base_state(
                draft_text="Original draft.",
                veritas_report={"revised_output": "Veritas corrected version."},
                claims_summary=[],
                overall_score=0.3,
                hallucination_probability=0.7,
            ))

        assert result["revised_text"] == "Veritas corrected version."

    @pytest.mark.asyncio
    async def test_falls_back_to_original_draft(self):
        """If both revision and Veritas output fail, should use original draft."""
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value={"error": "timeout"}):
            result = await mrp.node_revise_response(_base_state(
                draft_text="Original draft.",
                veritas_report={"revised_output": ""},
                claims_summary=[],
                overall_score=0.3,
                hallucination_probability=0.7,
            ))

        assert result["revised_text"] == "Original draft."

    @pytest.mark.asyncio
    async def test_increments_revision_round(self):
        """Revision round counter should increment."""
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value={
            "content": "revised", "finish_reason": "stop", "usage": {},
        }):
            result = await mrp.node_revise_response(_base_state(
                draft_text="draft",
                veritas_report={},
                claims_summary=[],
                revision_round=0,
            ))

        assert result["revision_round"] == 1


# ============================================================================
# Test: node_format_output
# ============================================================================

class TestNodeFormatOutput:
    @pytest.mark.asyncio
    async def test_uses_revised_text_when_available(self):
        result = await mrp.node_format_output(_base_state(
            revised_text="Revised answer",
            draft_text="Original draft",
        ))
        assert result["final_answer"] == "Revised answer"

    @pytest.mark.asyncio
    async def test_uses_draft_text_when_no_revision(self):
        result = await mrp.node_format_output(_base_state(
            revised_text="",
            draft_text="Original draft",
        ))
        assert result["final_answer"] == "Original draft"

    @pytest.mark.asyncio
    async def test_sets_done_phase(self):
        result = await mrp.node_format_output(_base_state(draft_text="answer"))
        assert result["phase"] == "done"


# ============================================================================
# Test: Conditional routing
# ============================================================================

class TestRouting:
    def test_needs_revision_routes_to_revise(self):
        state = _base_state(needs_revision=True)
        assert mrp.route_after_verify(state) == "revise_response"

    def test_clean_routes_to_format(self):
        state = _base_state(needs_revision=False)
        assert mrp.route_after_verify(state) == "format_output"

    def test_missing_needs_revision_defaults_clean(self):
        state = _base_state()
        del state["needs_revision"]
        assert mrp.route_after_verify(state) == "format_output"


# ============================================================================
# Test: Graph construction
# ============================================================================

class TestGraphConstruction:
    def test_graph_compiles(self):
        """The LangGraph should compile without errors."""
        graph = mrp.build_mistral_real_graph()
        assert graph is not None

    def test_singleton_graph_exists(self):
        """Module-level graph singleton should be available."""
        assert mrp._mistral_real_graph is not None


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    def test_default_hallucination_threshold(self):
        assert mrp.HALLUCINATION_THRESHOLD == 0.3

    def test_default_max_revisions(self):
        assert mrp.MAX_REVISION_ROUNDS == 1

    def test_default_port(self):
        assert mrp.LISTEN_PORT == 9600

    def test_default_max_concurrent(self):
        assert mrp.MAX_CONCURRENT == 2


# ============================================================================
# Test: Draft system prompt
# ============================================================================

class TestPrompts:
    def test_draft_prompt_mentions_thinking(self):
        assert "<THINKING>" in mrp.DRAFT_SYSTEM_PROMPT
        assert "<ANSWER>" in mrp.DRAFT_SYSTEM_PROMPT

    def test_revision_prompt_mentions_hallucinated(self):
        assert "hallucinated" in mrp.REVISION_PROMPT.lower()

    def test_revision_prompt_mentions_verified(self):
        assert "verified" in mrp.REVISION_PROMPT.lower()


# ============================================================================
# Test: End-to-end flow (mocked LLM + Veritas)
# ============================================================================

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_clean_draft_flow(self):
        """Clean draft: generate -> verify -> format (no revision)."""
        draft_response = {
            "content": "<THINKING>I'll answer about quantum computing.</THINKING>\n<ANSWER>Quantum computing uses qubits.</ANSWER>",
            "finish_reason": "stop",
            "usage": {},
        }
        clean_report = _make_veritas_report(overall_score=0.95, halluc_prob=0.05)

        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=draft_response):
            draft_result = await mrp.node_generate_draft(_base_state())

        state_after_draft = _base_state(**draft_result)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=clean_report):
            verify_result = await mrp.node_verify_draft(state_after_draft)

        assert verify_result["needs_revision"] is False

        state_after_verify = {**state_after_draft, **verify_result}
        format_result = await mrp.node_format_output(state_after_verify)
        assert "qubits" in format_result["final_answer"].lower()

    @pytest.mark.asyncio
    async def test_hallucinated_draft_flow(self):
        """Hallucinated draft: generate -> verify -> revise -> format."""
        draft_response = {
            "content": "<THINKING>reasoning</THINKING>\n<ANSWER>Quantum computers are 1000x faster.</ANSWER>",
            "finish_reason": "stop",
            "usage": {},
        }
        bad_report = _make_veritas_report(
            overall_score=0.3,
            halluc_prob=0.7,
            claims=[
                {"claim_text": "1000x faster", "status": "hallucinated", "confidence": 0.1, "evidence_summary": "Only for specific problems"},
            ],
            revised_output="Quantum computers can be faster for specific algorithmic problems.",
        )

        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=draft_response):
            draft_result = await mrp.node_generate_draft(_base_state())

        state_after_draft = _base_state(**draft_result)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=bad_report):
            verify_result = await mrp.node_verify_draft(state_after_draft)

        assert verify_result["needs_revision"] is True

        state_after_verify = {**state_after_draft, **verify_result}

        revision_response = {
            "content": "Quantum computers can outperform classical computers for specific types of calculations, such as factoring large numbers.",
            "finish_reason": "stop",
            "usage": {},
        }
        with patch.object(mrp, "call_llm", new_callable=AsyncMock, return_value=revision_response):
            revise_result = await mrp.node_revise_response(state_after_verify)

        state_after_revise = {**state_after_verify, **revise_result}
        format_result = await mrp.node_format_output(state_after_revise)

        assert "specific" in format_result["final_answer"].lower()
        assert "1000x" not in format_result["final_answer"]


# ============================================================================
# Test: call_llm
# ============================================================================

class TestCallLlm:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "hello"
        mock_ai_msg.response_metadata = {
            "finish_reason": "stop",
            "token_usage": {"total_tokens": 10},
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_msg)

        with patch.object(mrp, "_get_real_llm", return_value=mock_llm):
            result = await mrp.call_llm(
                [{"role": "user", "content": "hi"}], "test-req"
            )

        assert result["content"] == "hello"
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_error_response(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("status_code: 400 Bad request"))

        with patch.object(mrp, "_get_real_llm", return_value=mock_llm):
            result = await mrp.call_llm(
                [{"role": "user", "content": "hi"}], "test-req"
            )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.object(mrp, "_get_real_llm", return_value=mock_llm):
            result = await mrp.call_llm(
                [{"role": "user", "content": "hi"}], "test-req"
            )

        assert "error" in result


# ============================================================================
# Test: Claim counting in verification
# ============================================================================

class TestClaimCounting:
    @pytest.mark.asyncio
    async def test_counts_hallucinated_claims(self):
        report = _make_veritas_report(claims=[
            {"claim_text": "A", "status": "hallucinated", "confidence": 0.1, "evidence_summary": "fake"},
            {"claim_text": "B", "status": "hallucinated", "confidence": 0.2, "evidence_summary": "fake"},
            {"claim_text": "C", "status": "verified", "confidence": 0.9, "evidence_summary": "real"},
        ])

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(draft_text="test"))

        progress = " ".join(result["progress_log"])
        assert "2 hallucinated" in progress
        assert "1 verified" in progress

    @pytest.mark.asyncio
    async def test_counts_overconfident_claims(self):
        report = _make_veritas_report(claims=[
            {"claim_text": "A", "status": "overconfident", "confidence": 0.4, "evidence_summary": "weak"},
        ])

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(draft_text="test"))

        progress = " ".join(result["progress_log"])
        assert "1 overconfident" in progress


# ============================================================================
# Test: Threshold boundary conditions
# ============================================================================

class TestThresholdBoundary:
    @pytest.mark.asyncio
    async def test_exactly_at_threshold_no_revision(self):
        """Hallucination probability exactly at threshold should NOT trigger revision."""
        report = _make_veritas_report(halluc_prob=0.3)  # exactly at default threshold

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(draft_text="test"))

        assert result["needs_revision"] is False

    @pytest.mark.asyncio
    async def test_just_above_threshold_triggers_revision(self):
        """Hallucination probability just above threshold should trigger revision."""
        report = _make_veritas_report(halluc_prob=0.31)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(draft_text="test"))

        assert result["needs_revision"] is True

    @pytest.mark.asyncio
    async def test_zero_hallucination_no_revision(self):
        report = _make_veritas_report(halluc_prob=0.0)

        with patch("veritas_inquisitor.run_reactor", new_callable=AsyncMock, return_value=report):
            result = await mrp.node_verify_draft(_base_state(draft_text="test"))

        assert result["needs_revision"] is False
