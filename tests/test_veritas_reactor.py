"""Critical tests for the Veritas Inquisitor LangGraph reactor.

Covers the bugs found during review:
- _open_needs correctly filters by closed_need_ids
- node_verify triggers debate when last verify need completes
- node_debate triggers final judge when debate converges
- node_dispatch closes all low-pressure needs when report exists
- State reducers merge correctly
"""

import asyncio
import sys
import os
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add proxies to path so we can import veritas_inquisitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

# We need to mock shared module before importing veritas_inquisitor
# because it calls require_env("UPSTREAM_KEY") at import time.
_mock_shared = MagicMock()
_mock_shared.setup_logging.return_value = MagicMock()
_mock_shared.require_env.return_value = "test-key"
_mock_shared.env_int.side_effect = lambda name, default, **kw: default
_mock_shared.http_client.return_value = MagicMock()
_mock_shared.create_app.return_value = MagicMock()
_mock_shared.register_standard_routes = MagicMock()
_mock_shared.make_sse_chunk = MagicMock(side_effect=lambda *a, **kw: "data: {}\n\n")
_mock_shared.ConcurrencyLimiter = MagicMock
_mock_shared.RequestTracker = MagicMock
_mock_shared.is_utility_request = MagicMock(return_value=False)
_mock_shared.stream_passthrough = MagicMock()
sys.modules["shared"] = _mock_shared

import veritas_inquisitor as vi


# ---------------------------------------------------------------------------
# Helpers to build test state
# ---------------------------------------------------------------------------

def _make_need(need_id, need_type, pressure=0.5, target_artifact_id="art-root"):
    """Create a serialised need dict."""
    return {
        "id": need_id,
        "need_type": need_type,
        "target_artifact_id": target_artifact_id,
        "target_claim_id": "",
        "pressure_score": pressure,
        "required_skill": "",
        "context": {},
        "is_open": True,
    }


def _make_artifact(art_id, art_type, content=None):
    """Create a serialised artifact dict."""
    return {
        "id": art_id,
        "artifact_type": art_type,
        "content": content or {},
        "epistemic_tag": "",
        "tool_receipts": [],
        "parent_artifact_ids": [],
        "pressure_score": 0.0,
        "created_at": "",
        "created_by": "",
    }


def _base_state(**overrides):
    """Build a minimal ReactorState dict for testing."""
    state = {
        "target_text": "Test text",
        "original_query": "Test query",
        "req_id": "test-req",
        "artifacts": [],
        "needs": [],
        "closed_need_ids": [],
        "progress": [],
        "phase": "init",
        "iteration": 0,
        "reactor_start": time.monotonic(),
        "current_need": {},
        "claims": [],
        "debate_round": 0,
        "report": {},
    }
    state.update(overrides)
    return state


# ============================================================================
# Test: State reducers
# ============================================================================

class TestReducers:
    def test_append_artifacts_merges(self):
        left = [{"id": "a1"}]
        right = [{"id": "a2"}, {"id": "a3"}]
        result = vi._append_artifacts(left, right)
        assert len(result) == 3
        assert result[0]["id"] == "a1"
        assert result[2]["id"] == "a3"

    def test_append_needs_merges(self):
        left = [{"id": "n1"}]
        right = [{"id": "n2"}]
        result = vi._append_needs(left, right)
        assert len(result) == 2

    def test_append_progress_merges(self):
        left = ["msg1"]
        right = ["msg2", "msg3"]
        result = vi._append_progress(left, right)
        assert result == ["msg1", "msg2", "msg3"]

    def test_append_empty_left(self):
        assert vi._append_artifacts([], [{"id": "a1"}]) == [{"id": "a1"}]

    def test_append_empty_right(self):
        assert vi._append_needs([{"id": "n1"}], []) == [{"id": "n1"}]


# ============================================================================
# Test: _close_need_ids
# ============================================================================

class TestCloseNeedIds:
    def test_returns_dict_with_closed_need_ids(self):
        result = vi._close_need_ids(["need-1", "need-2"])
        assert result == {"closed_need_ids": ["need-1", "need-2"]}

    def test_empty_list(self):
        result = vi._close_need_ids([])
        assert result == {"closed_need_ids": []}


# ============================================================================
# Test: _open_needs
# ============================================================================

class TestOpenNeeds:
    def test_excludes_closed_ids(self):
        state = _base_state(
            needs=[
                _make_need("n1", "verify_claim", pressure=0.9),
                _make_need("n2", "verify_claim", pressure=0.5),
                _make_need("n3", "verify_claim", pressure=0.3),
            ],
            closed_need_ids=["n2"],
        )
        result = vi._open_needs(state)
        ids = [n["id"] for n in result]
        assert "n2" not in ids
        assert ids == ["n1", "n3"]  # sorted by pressure desc

    def test_sorted_by_pressure_descending(self):
        state = _base_state(
            needs=[
                _make_need("n1", "verify_claim", pressure=0.3),
                _make_need("n2", "verify_claim", pressure=0.9),
                _make_need("n3", "verify_claim", pressure=0.6),
            ],
        )
        result = vi._open_needs(state)
        pressures = [n["pressure_score"] for n in result]
        assert pressures == [0.9, 0.6, 0.3]

    def test_all_closed_returns_empty(self):
        state = _base_state(
            needs=[_make_need("n1", "verify_claim")],
            closed_need_ids=["n1"],
        )
        assert vi._open_needs(state) == []

    def test_empty_needs(self):
        state = _base_state()
        assert vi._open_needs(state) == []

    def test_missing_closed_need_ids_key(self):
        """_open_needs should handle state without closed_need_ids (uses .get)."""
        state = _base_state()
        del state["closed_need_ids"]  # simulate legacy state
        state["needs"] = [_make_need("n1", "verify_claim")]
        result = vi._open_needs(state)
        assert len(result) == 1


# ============================================================================
# Test: node_dispatch
# ============================================================================

class TestNodeDispatch:
    @pytest.mark.asyncio
    async def test_no_open_needs_returns_done(self):
        state = _base_state(iteration=5)
        result = await vi.node_dispatch(state)
        assert result["phase"] == "done"
        assert result["iteration"] == 6

    @pytest.mark.asyncio
    async def test_routes_to_correct_phase(self):
        state = _base_state(
            needs=[_make_need("n1", "verify_claim", pressure=0.8)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "verify"
        assert result["current_need"]["id"] == "n1"

    @pytest.mark.asyncio
    async def test_routes_interrogate(self):
        state = _base_state(
            needs=[_make_need("n1", "interrogate", pressure=0.9)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "interrogate"

    @pytest.mark.asyncio
    async def test_routes_decompose(self):
        state = _base_state(
            needs=[_make_need("n1", "decompose_claims", pressure=0.8)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "decompose"

    @pytest.mark.asyncio
    async def test_routes_debate(self):
        state = _base_state(
            needs=[_make_need("n1", "debate_round", pressure=0.7)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "debate"

    @pytest.mark.asyncio
    async def test_routes_judge(self):
        state = _base_state(
            needs=[_make_need("n1", "final_judgement", pressure=1.0)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "judge"

    @pytest.mark.asyncio
    async def test_closes_all_low_pressure_when_report_exists(self):
        """When a report exists and all open needs are below threshold,
        node_dispatch should close ALL of them and return done."""
        state = _base_state(
            artifacts=[_make_artifact("art-report", "report", {"claims": []})],
            needs=[
                _make_need("n1", "verify_claim", pressure=0.2),
                _make_need("n2", "counter_evidence", pressure=0.1),
                _make_need("n3", "debate_round", pressure=0.05),
            ],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "done"
        # All 3 needs should be closed
        closed = result.get("closed_need_ids", [])
        assert set(closed) == {"n1", "n2", "n3"}

    @pytest.mark.asyncio
    async def test_does_not_close_high_pressure_with_report(self):
        """Needs above threshold should still be dispatched even with report."""
        state = _base_state(
            artifacts=[_make_artifact("art-report", "report", {"claims": []})],
            needs=[_make_need("n1", "verify_claim", pressure=0.8)],
        )
        result = await vi.node_dispatch(state)
        assert result["phase"] == "verify"
        assert "closed_need_ids" not in result

    @pytest.mark.asyncio
    async def test_picks_highest_pressure_need(self):
        state = _base_state(
            needs=[
                _make_need("n-low", "verify_claim", pressure=0.3),
                _make_need("n-high", "debate_round", pressure=0.9),
                _make_need("n-mid", "verify_claim", pressure=0.6),
            ],
        )
        result = await vi.node_dispatch(state)
        assert result["current_need"]["id"] == "n-high"


# ============================================================================
# Test: node_verify triggers debate (BUG_0001 regression test)
# ============================================================================

class TestNodeVerifyTriggersDebate:
    """Regression test for the self-counting bug in node_verify.

    The bug: _open_needs(state) included the current verify need because it
    hadn't been closed yet. remaining_verify was always >= 1, so the debate
    phase was never triggered.
    """

    @pytest.mark.asyncio
    async def test_last_verify_need_triggers_debate(self):
        """When this is the LAST verify need and all claims are verified,
        node_verify should post a debate NeedItem."""
        claim = {"id": "claim-001", "claim_text": "test claim", "tag": "inference"}
        claims_artifact = _make_artifact(
            "art-claims", "claims",
            {"claims": [claim], "total": 1},
        )

        # The current need is the only verify need
        current_need = _make_need(
            "n-verify-001", "verify_claim", pressure=0.5,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"claim": claim}

        state = _base_state(
            artifacts=[
                _make_artifact("art-root", "root"),
                claims_artifact,
            ],
            needs=[current_need],  # only one verify need, currently being processed
            current_need=current_need,
        )

        # Mock run_evidence_gatherer to return an evidence artifact
        mock_evidence = vi.Artifact(
            id="art-evidence-001",
            artifact_type="evidence",
            content={
                "claim_id": "claim-001",
                "conflicts_found": 0,
                "evidence_summary": "test",
            },
            created_by="EvidenceGatherer",
        )

        with patch.object(vi, "run_evidence_gatherer", new_callable=AsyncMock, return_value=mock_evidence):
            result = await vi.node_verify(state)

        # The current need should be closed
        assert "n-verify-001" in result.get("closed_need_ids", [])

        # A debate need should have been posted
        debate_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.DEBATE_ROUND.value
        ]
        assert len(debate_needs) == 1, (
            "node_verify should post a debate need when the last verify need completes"
        )

    @pytest.mark.asyncio
    async def test_not_last_verify_need_does_not_trigger_debate(self):
        """When other verify needs remain open, debate should NOT be triggered."""
        claim1 = {"id": "claim-001", "claim_text": "test claim 1", "tag": "inference"}
        claim2 = {"id": "claim-002", "claim_text": "test claim 2", "tag": "inference"}
        claims_artifact = _make_artifact(
            "art-claims", "claims",
            {"claims": [claim1, claim2], "total": 2},
        )

        current_need = _make_need(
            "n-verify-001", "verify_claim", pressure=0.5,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"claim": claim1}

        other_verify_need = _make_need(
            "n-verify-002", "verify_claim", pressure=0.5,
            target_artifact_id="art-claims",
        )
        other_verify_need["context"] = {"claim": claim2}

        state = _base_state(
            artifacts=[
                _make_artifact("art-root", "root"),
                claims_artifact,
            ],
            needs=[current_need, other_verify_need],
            current_need=current_need,
        )

        mock_evidence = vi.Artifact(
            id="art-evidence-001",
            artifact_type="evidence",
            content={
                "claim_id": "claim-001",
                "conflicts_found": 0,
                "evidence_summary": "test",
            },
            created_by="EvidenceGatherer",
        )

        with patch.object(vi, "run_evidence_gatherer", new_callable=AsyncMock, return_value=mock_evidence):
            result = await vi.node_verify(state)

        debate_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.DEBATE_ROUND.value
        ]
        assert len(debate_needs) == 0, (
            "node_verify should NOT post debate need when other verify needs remain"
        )

    @pytest.mark.asyncio
    async def test_conflicts_post_counter_evidence_need(self):
        """When conflicts are found, a counter-evidence need should be posted."""
        claim = {"id": "claim-001", "claim_text": "test claim", "tag": "citation"}
        current_need = _make_need(
            "n-verify-001", "verify_claim", pressure=0.5,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"claim": claim}

        state = _base_state(
            artifacts=[_make_artifact("art-root", "root")],
            needs=[current_need],
            current_need=current_need,
        )

        mock_evidence = vi.Artifact(
            id="art-evidence-001",
            artifact_type="evidence",
            content={
                "claim_id": "claim-001",
                "conflicts_found": 2,
                "evidence_summary": "conflicts found",
            },
            pressure_score=0.7,
            created_by="EvidenceGatherer",
        )

        with patch.object(vi, "run_evidence_gatherer", new_callable=AsyncMock, return_value=mock_evidence):
            result = await vi.node_verify(state)

        counter_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.COUNTER_EVIDENCE.value
        ]
        assert len(counter_needs) == 1


# ============================================================================
# Test: node_debate triggers final judge (BUG_0002 regression test)
# ============================================================================

class TestNodeDebateTriggersJudge:
    """Regression test for the self-counting bug in node_debate.

    The bug: _open_needs(state) included the current debate need because it
    hadn't been closed yet. pending_debate was always non-empty, so the
    FINAL_JUDGEMENT need was never posted.
    """

    @pytest.mark.asyncio
    async def test_final_debate_round_triggers_judge(self):
        """When this is the last debate round (no conflicts) and no other
        debate needs exist, node_debate should post a FINAL_JUDGEMENT need."""
        current_need = _make_need(
            "n-debate-001", "debate_round", pressure=0.7,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"round": 1}

        claims_artifact = _make_artifact(
            "art-claims", "claims",
            {"claims": [{"id": "c1", "claim_text": "test", "tag": "inference"}]},
        )
        evidence_artifact = _make_artifact(
            "art-evidence", "evidence",
            {"claim_id": "c1", "conflicts_found": 0, "evidence_summary": "ok"},
        )

        state = _base_state(
            artifacts=[
                _make_artifact("art-root", "root"),
                claims_artifact,
                evidence_artifact,
            ],
            needs=[current_need],  # only debate need
            current_need=current_need,
        )

        # Mock: debate returns no key_conflicts (converged)
        mock_debate = vi.Artifact(
            id="art-debate-001",
            artifact_type="debate",
            content={
                "messages": [{"speaker": "examiner", "message": "All resolved"}],
                "key_conflicts": [],  # no conflicts -> converged
                "resolved_points": ["claim verified"],
                "round": 1,
            },
            created_by="CriticDebater",
        )

        with patch.object(vi, "run_critic_debater", new_callable=AsyncMock, return_value=mock_debate):
            result = await vi.node_debate(state)

        # The current debate need should be closed
        assert "n-debate-001" in result.get("closed_need_ids", [])

        # A FINAL_JUDGEMENT need should have been posted
        judge_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.FINAL_JUDGEMENT.value
        ]
        assert len(judge_needs) == 1, (
            "node_debate should post FINAL_JUDGEMENT need when debate converges"
        )

    @pytest.mark.asyncio
    async def test_debate_with_conflicts_posts_another_round(self):
        """When conflicts remain, another debate round should be posted."""
        current_need = _make_need(
            "n-debate-001", "debate_round", pressure=0.7,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"round": 1}

        state = _base_state(
            artifacts=[_make_artifact("art-root", "root")],
            needs=[current_need],
            current_need=current_need,
        )

        mock_debate = vi.Artifact(
            id="art-debate-001",
            artifact_type="debate",
            content={
                "messages": [{"speaker": "prosecutor", "message": "I disagree"}],
                "key_conflicts": ["unresolved claim"],
                "resolved_points": [],
                "round": 1,
            },
            created_by="CriticDebater",
        )

        with patch.object(vi, "run_critic_debater", new_callable=AsyncMock, return_value=mock_debate):
            result = await vi.node_debate(state)

        # Should post another debate round, not final judgement
        debate_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.DEBATE_ROUND.value
        ]
        judge_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.FINAL_JUDGEMENT.value
        ]
        assert len(debate_needs) == 1, "Should post another debate round"
        # Should NOT post judge yet since another debate round is queued
        assert len(judge_needs) == 0, "Should not post FINAL_JUDGEMENT with pending debate"

    @pytest.mark.asyncio
    async def test_max_debate_rounds_triggers_judge(self):
        """When at MAX_DEBATE_ROUNDS, should not post another debate even with conflicts,
        and should trigger final judge."""
        current_need = _make_need(
            "n-debate-max", "debate_round", pressure=0.7,
            target_artifact_id="art-claims",
        )
        current_need["context"] = {"round": vi.MAX_DEBATE_ROUNDS}

        state = _base_state(
            artifacts=[_make_artifact("art-root", "root")],
            needs=[current_need],
            current_need=current_need,
        )

        mock_debate = vi.Artifact(
            id="art-debate-max",
            artifact_type="debate",
            content={
                "messages": [{"speaker": "prosecutor", "message": "Still disagree"}],
                "key_conflicts": ["unresolved"],
                "resolved_points": [],
                "round": vi.MAX_DEBATE_ROUNDS,
            },
            created_by="CriticDebater",
        )

        with patch.object(vi, "run_critic_debater", new_callable=AsyncMock, return_value=mock_debate):
            result = await vi.node_debate(state)

        # Should NOT post another debate round (at max)
        debate_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.DEBATE_ROUND.value
        ]
        assert len(debate_needs) == 0, "Should not post debate at max rounds"

        # Should post final judge since no more debate rounds possible
        judge_needs = [
            n for n in result.get("needs", [])
            if n.get("need_type") == vi.NeedType.FINAL_JUDGEMENT.value
        ]
        assert len(judge_needs) == 1, "Should post FINAL_JUDGEMENT at max debate rounds"


# ============================================================================
# Test: route_after_dispatch
# ============================================================================

class TestRouteAfterDispatch:
    def test_routes_to_end_on_done(self):
        from langgraph.graph import END
        state = _base_state(phase="done")
        assert vi.route_after_dispatch(state) == END

    def test_routes_to_phase(self):
        for phase in ["interrogate", "decompose", "verify", "debate", "judge"]:
            state = _base_state(phase=phase)
            assert vi.route_after_dispatch(state) == phase


# ============================================================================
# Test: _rebuild_index
# ============================================================================

class TestRebuildIndex:
    def test_rebuilds_from_dicts(self):
        state = _base_state(
            artifacts=[
                _make_artifact("a1", "root"),
                _make_artifact("a2", "claims"),
                _make_artifact("a3", "evidence"),
            ],
        )
        idx = vi._rebuild_index(state)
        assert idx.count == 3
        assert len(idx.by_type("root")) == 1
        assert len(idx.by_type("claims")) == 1
        assert len(idx.by_type("evidence")) == 1

    def test_empty_artifacts(self):
        state = _base_state()
        idx = vi._rebuild_index(state)
        assert idx.count == 0


# ============================================================================
# Test: compute_pressure
# ============================================================================

class TestComputePressure:
    def test_direct_tool_low_risk(self):
        score = vi.compute_pressure("direct-tool", 0, 0)
        assert 0.0 <= score <= 0.2

    def test_citation_high_risk(self):
        score = vi.compute_pressure("citation", 3, 5)
        assert score > 0.5

    def test_clamped_to_unit(self):
        score = vi.compute_pressure("citation", 100, 100)
        assert score <= 1.0

    def test_zero_everything(self):
        score = vi.compute_pressure("direct-tool", 0, 0)
        assert score >= 0.0


# ============================================================================
# Test: _parse_json_from_llm
# ============================================================================

class TestParseJsonFromLlm:
    def test_plain_json(self):
        result = vi._parse_json_from_llm('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced_json(self):
        result = vi._parse_json_from_llm('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_array(self):
        result = vi._parse_json_from_llm('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_embedded_in_text(self):
        # Note: _parse_json_from_llm scans for arrays before objects,
        # so use a text where the first JSON is unambiguously an object
        text = 'Here is the result: {"claims": "verified", "score": 0.9} done'
        result = vi._parse_json_from_llm(text)
        assert result == {"claims": "verified", "score": 0.9}

    def test_invalid_json_returns_none(self):
        result = vi._parse_json_from_llm("not json at all")
        assert result is None

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2]}}'
        result = vi._parse_json_from_llm(text)
        assert result["outer"]["inner"] == [1, 2]


# ============================================================================
# Test: node_judge closes its need
# ============================================================================

class TestNodeJudge:
    @pytest.mark.asyncio
    async def test_judge_closes_need_and_sets_report(self):
        current_need = _make_need(
            "n-judge-001", "final_judgement", pressure=1.0,
            target_artifact_id="art-debate",
        )

        state = _base_state(
            artifacts=[_make_artifact("art-root", "root")],
            needs=[current_need],
            current_need=current_need,
        )

        mock_report = vi.Artifact(
            id="art-report-001",
            artifact_type="report",
            content={
                "claims": [{"id": "c1", "status": "verified"}],
                "overall_score": 0.9,
                "overall_hallucination_probability": 0.1,
                "revised_output": "Revised text",
                "evidence_links": [],
            },
            created_by="FinalJudge",
        )

        with patch.object(vi, "run_final_judge", new_callable=AsyncMock, return_value=mock_report):
            result = await vi.node_judge(state)

        assert "n-judge-001" in result.get("closed_need_ids", [])
        assert result["report"]["overall_score"] == 0.9
        assert len(result["artifacts"]) == 1


# ============================================================================
# Test: node_interrogate closes its need
# ============================================================================

class TestNodeInterrogate:
    @pytest.mark.asyncio
    async def test_interrogate_closes_need(self):
        current_need = _make_need(
            "n-interr-001", "interrogate", pressure=0.9,
            target_artifact_id="art-root",
        )

        state = _base_state(
            needs=[current_need],
            current_need=current_need,
        )

        mock_probe = vi.Artifact(
            id="art-probe-001",
            artifact_type="probe",
            content={"new_probe_questions": ["Q1?", "Q2?"]},
            created_by="Interrogator",
        )

        with patch.object(vi, "run_interrogator", new_callable=AsyncMock, return_value=mock_probe):
            result = await vi.node_interrogate(state)

        assert "n-interr-001" in result.get("closed_need_ids", [])
        assert len(result["artifacts"]) == 1
