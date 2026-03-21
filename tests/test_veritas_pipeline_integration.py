"""Tests for the Veritas Inquisitor integration into the persistent research
pipeline's verify phase.

Covers:
- _fuzzy_match_claim_to_condition: token-overlap matching
- verify_conditions_with_veritas: Veritas reactor invocation + result mapping
- pdr_node_verify: two-stage verification (self-eval + Veritas)
- Config toggles (VERITAS_VERIFY_ENABLED, VERITAS_MIN_CONDITIONS)
"""

import asyncio
import sys
import os
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add proxies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

# Mock shared module before importing persistent_deep_research_proxy
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
if "shared" not in sys.modules:
    sys.modules["shared"] = _mock_shared

# Mock optional dependencies that may not be installed
for mod_name in ["knowledge_client", "research_report", "b2_publisher"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Mock research_metrics with real class-like objects
_mock_metrics = MagicMock()
_mock_metrics.MetricsCollector = MagicMock
_mock_metrics.ResearchMetricsCallback = MagicMock
_mock_metrics.SubagentMetrics = MagicMock
_mock_metrics.list_available_reports = MagicMock(return_value=[])
_mock_metrics.load_metrics = MagicMock(return_value=None)
_mock_metrics.save_metrics = MagicMock()
if "research_metrics" not in sys.modules:
    sys.modules["research_metrics"] = _mock_metrics

import persistent_deep_research_proxy as pdr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_condition(fact: str, source_url: str = "", confidence: float = 0.5) -> pdr.AtomicCondition:
    return pdr.AtomicCondition(
        fact=fact,
        source_url=source_url,
        confidence=confidence,
        angle="test",
        trust_score=0.5,
    )


def _pdr_state(**overrides) -> dict:
    """Build a minimal PersistentResearchState dict for testing."""
    state = {
        "req_id": "test-req-001",
        "user_query": "What is the price of used Technogym Biostrength?",
        "start_time": time.monotonic(),
        "prior_conditions": [],
        "graph_neighbors": [],
        "angles": [],
        "subagent_results": [],
        "all_conditions": [],
        "total_turns": 0,
        "total_tools": 0,
        "total_children": 0,
        "reflection": {},
        "final_answer": "",
        "progress_log": [],
        "phase": "verify",
        "report_url": "",
        "metrics_url": "",
    }
    state.update(overrides)
    return state


# ============================================================================
# Test: _fuzzy_match_claim_to_condition
# ============================================================================

class TestFuzzyMatchClaimToCondition:
    def test_exact_match(self):
        conditions = [
            _make_condition("Technogym Biostrength uses AI-driven resistance"),
            _make_condition("The price is approximately 15000 EUR"),
        ]
        idx = pdr._fuzzy_match_claim_to_condition(
            "Technogym Biostrength uses AI-driven resistance", conditions,
        )
        assert idx == 0

    def test_partial_overlap(self):
        conditions = [
            _make_condition("Sarah Hospitality specializes in hotel liquidations"),
            _make_condition("Technogym Biostrength costs around 15000 EUR"),
        ]
        idx = pdr._fuzzy_match_claim_to_condition(
            "Sarah Hospitality is a company that specializes in hotel liquidations and equipment resale",
            conditions,
        )
        assert idx == 0

    def test_no_match_returns_negative(self):
        conditions = [
            _make_condition("The sky is blue"),
            _make_condition("Water boils at 100 degrees Celsius"),
        ]
        idx = pdr._fuzzy_match_claim_to_condition(
            "Quantum computing uses qubits for parallel processing",
            conditions,
        )
        assert idx == -1

    def test_empty_claim(self):
        conditions = [_make_condition("Something")]
        idx = pdr._fuzzy_match_claim_to_condition("", conditions)
        assert idx == -1

    def test_empty_conditions(self):
        idx = pdr._fuzzy_match_claim_to_condition("Some claim", [])
        assert idx == -1

    def test_picks_best_match(self):
        conditions = [
            _make_condition("Used gym equipment prices vary widely"),
            _make_condition("Technogym Biostrength AI resistance machine costs 15000 EUR"),
            _make_condition("Commercial gym liquidation auctions"),
        ]
        idx = pdr._fuzzy_match_claim_to_condition(
            "Technogym Biostrength AI machine price is 15000 EUR",
            conditions,
        )
        assert idx == 1


# ============================================================================
# Test: verify_conditions_with_veritas
# ============================================================================

class TestVerifyConditionsWithVeritas:
    @pytest.mark.asyncio
    async def test_removes_fabricated_keeps_speculative(self):
        """Fabricated entities (zero grounding) are removed; speculative kept."""
        conditions = [
            _make_condition("Sarah Hospitality specializes in hotel liquidations", confidence=0.7),
            _make_condition("Technogym Biostrength uses AI-driven resistance adjustment", confidence=0.8),
            _make_condition("Used gym equipment resale market is growing", confidence=0.6),
        ]

        mock_veritas_result = {
            "report": {
                "claims": [
                    {
                        "claim_text": "Sarah Hospitality specializes in hotel liquidations",
                        "status": "fabricated",
                        "confidence": 0.05,
                        "evidence_summary": "No evidence of Sarah Hospitality existing — fabricated entity",
                    },
                    {
                        "claim_text": "Technogym Biostrength uses AI-driven resistance adjustment",
                        "status": "verified",
                        "confidence": 0.9,
                        "evidence_summary": "Confirmed via Technogym official website",
                    },
                    {
                        "claim_text": "Used gym equipment resale market is growing",
                        "status": "speculative",
                        "confidence": 0.4,
                        "evidence_summary": "Some anecdotal evidence but no market report",
                    },
                ],
                "overall_score": 0.5,
                "overall_hallucination_probability": 0.33,
            },
            "artifact_count": 12,
            "iterations": 8,
        }

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_veritas_result):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "used technogym biostrength", "test-req",
            )

        # Fabricated removed, verified + speculative kept
        assert len(filtered) == 2
        facts = [c.fact for c in filtered]
        assert "Sarah Hospitality" not in " ".join(facts)
        assert "AI-driven resistance" in filtered[0].fact
        assert filtered[0].confidence >= 0.8
        # Speculative claim kept with status set
        assert filtered[1].verification_status == "speculative"
        assert filtered[1].confidence <= 0.6  # capped downward

    @pytest.mark.asyncio
    async def test_legacy_hallucinated_treated_as_fabricated(self):
        """Legacy 'hallucinated' status should be treated as 'fabricated'."""
        conditions = [
            _make_condition("Fake Corp Ltd sells widgets", confidence=0.7),
            _make_condition("Real product costs 500 EUR", confidence=0.8),
            _make_condition("Another real claim", confidence=0.6),
        ]

        mock_veritas_result = {
            "report": {
                "claims": [
                    {
                        "claim_text": "Fake Corp Ltd sells widgets",
                        "status": "hallucinated",
                        "confidence": 0.02,
                    },
                    {
                        "claim_text": "Real product costs 500 EUR",
                        "status": "verified",
                        "confidence": 0.85,
                    },
                    {
                        "claim_text": "Another real claim",
                        "status": "verified",
                        "confidence": 0.8,
                    },
                ],
                "overall_score": 0.7,
            },
        }

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_veritas_result):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "query", "test-req",
            )

        # Legacy hallucinated treated as fabricated => removed
        assert len(filtered) == 2
        assert all("Fake Corp" not in c.fact for c in filtered)

    @pytest.mark.asyncio
    async def test_adjusts_overconfident_conditions(self):
        """Overconfident conditions should have confidence capped."""
        conditions = [
            _make_condition("Product X costs exactly $500", confidence=0.95),
            _make_condition("Product Y is available in 5 countries", confidence=0.8),
            _make_condition("Product Z has 99% satisfaction rate", confidence=0.9),
        ]

        mock_result = {
            "report": {
                "claims": [
                    {
                        "claim_text": "Product X costs exactly $500",
                        "status": "overconfident",
                        "confidence": 0.4,
                    },
                    {
                        "claim_text": "Product Y is available in 5 countries",
                        "status": "verified",
                        "confidence": 0.85,
                    },
                    {
                        "claim_text": "Product Z has 99% satisfaction rate",
                        "status": "plausible-unverified",
                        "confidence": 0.5,
                    },
                ],
                "overall_score": 0.5,
            },
        }

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_result):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "product comparison", "test-req",
            )

        assert len(filtered) == 3  # None hallucinated
        # Overconfident: capped
        assert filtered[0].confidence <= 0.5
        # Verified: boosted
        assert filtered[1].confidence >= 0.8
        # Plausible-unverified: downgraded
        assert filtered[2].confidence <= 0.9

    @pytest.mark.asyncio
    async def test_skips_below_min_conditions(self):
        """Should skip Veritas if fewer than VERITAS_MIN_CONDITIONS."""
        conditions = [
            _make_condition("Only one condition", confidence=0.7),
        ]

        # Should return unchanged without calling Veritas
        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock) as mock_verify:
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "query", "test-req",
            )
            mock_verify.assert_not_called()

        assert len(filtered) == 1
        assert report == {}

    @pytest.mark.asyncio
    async def test_handles_veritas_error_gracefully(self):
        """If Veritas throws an exception, conditions should be returned unchanged."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, side_effect=RuntimeError("LLM timeout")):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "query", "test-req",
            )

        assert len(filtered) == 3  # All retained
        assert "error" in report

    @pytest.mark.asyncio
    async def test_handles_empty_veritas_claims(self):
        """If Veritas produces no claims, conditions should be returned unchanged."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        mock_result = {
            "report": {"claims": [], "overall_score": -1},
            "artifact_count": 2,
            "iterations": 1,
        }

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_result):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "query", "test-req",
            )

        assert len(filtered) == 3

    @pytest.mark.asyncio
    async def test_unmatched_veritas_claims_are_ignored(self):
        """Veritas claims that don't match any condition should not crash."""
        conditions = [
            _make_condition("Technogym equipment is expensive", confidence=0.7),
            _make_condition("Used equipment market is growing", confidence=0.6),
            _make_condition("Online auctions are popular for resale", confidence=0.5),
        ]

        mock_result = {
            "report": {
                "claims": [
                    {
                        "claim_text": "Something completely unrelated to any condition",
                        "status": "hallucinated",
                        "confidence": 0.0,
                    },
                ],
                "overall_score": 0.8,
            },
        }

        with patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_result):
            filtered, report = await pdr.verify_conditions_with_veritas(
                conditions, "query", "test-req",
            )

        # No conditions removed because the claim didn't match any
        assert len(filtered) == 3


# ============================================================================
# Test: pdr_node_verify (two-stage)
# ============================================================================

class TestPdrNodeVerifyWithVeritas:
    @pytest.mark.asyncio
    async def test_runs_both_stages_when_forced(self):
        """When VERITAS_FORCE_POST_HOC is set, both self-eval and Veritas run."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B is hallucinated", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        state = _pdr_state(all_conditions=conditions)

        mock_veritas_result = {
            "report": {
                "claims": [
                    {"claim_text": "Claim A", "status": "verified", "confidence": 0.85},
                    {"claim_text": "Claim B is hallucinated", "status": "hallucinated", "confidence": 0.05},
                    {"claim_text": "Claim C", "status": "plausible-unverified", "confidence": 0.5},
                ],
                "overall_score": 0.5,
                "overall_hallucination_probability": 0.33,
            },
        }

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 3), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions), \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_veritas_result), \
             patch.object(pdr, "_metrics_collectors", {}), \
             patch.dict("os.environ", {"VERITAS_FORCE_POST_HOC": "true"}):

            result = await pdr.pdr_node_verify(state)

        # Hallucinated claim should be removed
        result_conditions = result["all_conditions"]
        assert len(result_conditions) == 2
        facts = [c.fact for c in result_conditions]
        assert "Claim B is hallucinated" not in facts

        # Progress should mention both stages
        progress_text = " ".join(result["progress_log"])
        assert "Phase 5a" in progress_text
        assert "Phase 5b" in progress_text or "Veritas" in progress_text

    @pytest.mark.asyncio
    async def test_skips_veritas_by_default_with_inline_verification(self):
        """By default, Veritas is skipped because inline verification runs
        during the tree phase. Progress log should mention this."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        state = _pdr_state(all_conditions=conditions)

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 3), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions), \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock) as mock_veritas, \
             patch.object(pdr, "_metrics_collectors", {}), \
             patch.dict("os.environ", {}, clear=False):
            # Ensure VERITAS_FORCE_POST_HOC is NOT set
            import os
            os.environ.pop("VERITAS_FORCE_POST_HOC", None)

            result = await pdr.pdr_node_verify(state)

        # Veritas should NOT have been called
        mock_veritas.assert_not_called()

        # All conditions preserved (only self-eval ran)
        assert len(result["all_conditions"]) == 3

        # Progress should mention inline verification
        progress_text = " ".join(result["progress_log"])
        assert "Inline Verification" in progress_text

    @pytest.mark.asyncio
    async def test_skips_veritas_when_disabled(self):
        """When VERITAS_VERIFY_ENABLED is False, only self-eval should run."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        state = _pdr_state(all_conditions=conditions)

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", False), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions) as mock_self_eval, \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock) as mock_veritas, \
             patch.object(pdr, "_metrics_collectors", {}):

            result = await pdr.pdr_node_verify(state)

        mock_self_eval.assert_called_once()
        mock_veritas.assert_not_called()
        assert len(result["all_conditions"]) == 3

    @pytest.mark.asyncio
    async def test_skips_veritas_with_few_conditions(self):
        """When fewer than VERITAS_MIN_CONDITIONS, only self-eval runs."""
        conditions = [
            _make_condition("Only two claims", confidence=0.7),
            _make_condition("Not enough for Veritas", confidence=0.6),
        ]

        state = _pdr_state(all_conditions=conditions)

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 5), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions), \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock) as mock_veritas, \
             patch.object(pdr, "_metrics_collectors", {}):

            result = await pdr.pdr_node_verify(state)

        mock_veritas.assert_not_called()

    @pytest.mark.asyncio
    async def test_progress_log_shows_removal_count(self):
        """Progress log should report how many conditions were removed
        when Veritas is forced."""
        conditions = [
            _make_condition("Real claim A", confidence=0.8),
            _make_condition("Fake company XYZ Ltd", confidence=0.6),
            _make_condition("Real claim B", confidence=0.7),
            _make_condition("Fake entity ABC Corp", confidence=0.5),
        ]

        state = _pdr_state(all_conditions=conditions)

        mock_result = {
            "report": {
                "claims": [
                    {"claim_text": "Fake company XYZ Ltd", "status": "hallucinated", "confidence": 0.02},
                    {"claim_text": "Fake entity ABC Corp", "status": "hallucinated", "confidence": 0.01},
                    {"claim_text": "Real claim A", "status": "verified", "confidence": 0.9},
                    {"claim_text": "Real claim B", "status": "verified", "confidence": 0.85},
                ],
                "overall_score": 0.5,
                "overall_hallucination_probability": 0.5,
            },
        }

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 3), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions), \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, return_value=mock_result), \
             patch.object(pdr, "_metrics_collectors", {}), \
             patch.dict("os.environ", {"VERITAS_FORCE_POST_HOC": "true"}):

            result = await pdr.pdr_node_verify(state)

        assert len(result["all_conditions"]) == 2
        progress_text = " ".join(result["progress_log"])
        assert "2 fabricated claims removed" in progress_text
        assert "2 conditions retained out of 4" in progress_text

    @pytest.mark.asyncio
    async def test_single_condition_skips_both_stages(self):
        """With only 1 condition, both verification stages should be skipped."""
        conditions = [_make_condition("Single claim", confidence=0.7)]
        state = _pdr_state(all_conditions=conditions)

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 3), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock) as mock_self_eval, \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock) as mock_veritas, \
             patch.object(pdr, "_metrics_collectors", {}):

            result = await pdr.pdr_node_verify(state)

        mock_self_eval.assert_not_called()
        mock_veritas.assert_not_called()
        assert len(result["all_conditions"]) == 1

    @pytest.mark.asyncio
    async def test_veritas_error_preserves_conditions(self):
        """If Veritas crashes (when forced), self-eval results should still be preserved."""
        conditions = [
            _make_condition("Claim A", confidence=0.7),
            _make_condition("Claim B", confidence=0.6),
            _make_condition("Claim C", confidence=0.5),
        ]

        state = _pdr_state(all_conditions=conditions)

        with patch.object(pdr, "VERITAS_VERIFY_ENABLED", True), \
             patch.object(pdr, "VERITAS_MIN_CONDITIONS", 3), \
             patch.object(pdr, "verify_conditions", new_callable=AsyncMock, return_value=conditions), \
             patch("veritas_inquisitor.verify_output", new_callable=AsyncMock, side_effect=RuntimeError("boom")), \
             patch.object(pdr, "_metrics_collectors", {}), \
             patch.dict("os.environ", {"VERITAS_FORCE_POST_HOC": "true"}):

            result = await pdr.pdr_node_verify(state)

        # All conditions preserved despite Veritas error
        assert len(result["all_conditions"]) == 3


# ============================================================================
# Test: Config toggles
# ============================================================================

class TestConfigToggles:
    def test_veritas_enabled_default(self):
        """VERITAS_VERIFY_ENABLED defaults to True."""
        # The module was imported with no env var override, so it should be True
        # (unless the test env sets it to something else)
        assert isinstance(pdr.VERITAS_VERIFY_ENABLED, bool)

    def test_min_conditions_default(self):
        """VERITAS_MIN_CONDITIONS defaults to 3."""
        assert pdr.VERITAS_MIN_CONDITIONS == 3

    def test_hallucination_threshold_default(self):
        """VERITAS_HALLUCINATION_THRESHOLD defaults to 0.3."""
        assert pdr.VERITAS_HALLUCINATION_THRESHOLD == 0.3

    def test_commercial_search_enabled_default(self):
        """COMMERCIAL_SEARCH_ENABLED defaults to True."""
        assert isinstance(pdr.COMMERCIAL_SEARCH_ENABLED, bool)

    def test_moderation_model_default(self):
        """MODERATION_MODEL defaults to mistral-small-latest."""
        assert pdr.MODERATION_MODEL == "mistral-small-latest"


# ============================================================================
# Test: Mistral Moderation Gate
# ============================================================================

class TestClassifyQuery:
    """Tests for classify_query (advisory content classifier).

    classify_query returns a list of category strings.  It does NOT block
    any search or tool — the categories are advisory only, used for model
    routing (e.g. choosing uncensored vs censored LLM).
    """

    @pytest.mark.asyncio
    async def test_clean_query_returns_empty(self):
        """Normal research query should return no categories."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = '{"categories": []}'

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_msg)

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            categories = await pdr.classify_query("used Technogym Biostrength price")

        assert categories == []

    @pytest.mark.asyncio
    async def test_sensitive_query_returns_categories(self):
        """Sensitive query should return matching categories (but NOT block anything)."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = '{"categories": ["violence_and_threats", "dangerous_and_criminal_content"]}'

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_msg)

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            categories = await pdr.classify_query("how to make explosives")

        assert "violence_and_threats" in categories
        assert "dangerous_and_criminal_content" in categories

    @pytest.mark.asyncio
    async def test_api_failure_returns_empty(self):
        """If classifier LLM call fails, return empty (fail open — never block)."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            categories = await pdr.classify_query("normal query")

        assert categories == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        """Without an API key, classifier should return empty (never block)."""
        with patch.object(pdr, "UPSTREAM_KEY", ""):
            categories = await pdr.classify_query("any query")

        assert categories == []

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_empty(self):
        """Unparseable LLM response should return empty (never block)."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "I cannot classify this query."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_msg)

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            categories = await pdr.classify_query("normal query")

        assert categories == []


class TestModerateQueryLegacy:
    """Tests for moderate_query legacy wrapper.

    moderate_query always returns is_safe=True now.  It exists only for
    backward compatibility.  The moderation gate no longer blocks searches.
    """

    @pytest.mark.asyncio
    async def test_always_returns_safe(self):
        """moderate_query should always return is_safe=True regardless of classification."""
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = '{"categories": ["dangerous_and_criminal_content"]}'

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_msg)

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            is_safe, details = await pdr.moderate_query("buy insulin without prescription")

        assert is_safe is True
        assert "categories" in details
        assert "dangerous_and_criminal_content" in details["categories"]

    @pytest.mark.asyncio
    async def test_safe_on_api_failure(self):
        """moderate_query should still return is_safe=True even on API failure."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        with patch.object(pdr, "_get_moderation_llm", return_value=mock_llm), \
             patch.object(pdr, "UPSTREAM_KEY", "test-key"):
            is_safe, details = await pdr.moderate_query("normal query")

        assert is_safe is True


# ============================================================================
# Test: Commercial SERP APIs
# ============================================================================

class TestBrightDataSerp:
    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        """Should parse Bright Data SERP organic results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "organic": [
                {
                    "title": "Technogym Biostrength Review",
                    "link": "https://example.com/review",
                    "description": "A comprehensive review of Technogym Biostrength.",
                },
                {
                    "title": "Buy Used Gym Equipment",
                    "link": "https://example.com/buy",
                    "description": "Best deals on used gym equipment.",
                },
            ],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(pdr, "http_client", return_value=mock_client), \
             patch.object(pdr, "BRIGHT_DATA_API_KEY", "test-key"):
            results = await pdr._search_bright_data_serp("technogym biostrength")

        assert len(results) == 2
        assert results[0]["title"] == "Technogym Biostrength Review"
        assert results[0]["source"] == "bright_data"

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self):
        """Should return empty list if no Bright Data API key."""
        with patch.object(pdr, "BRIGHT_DATA_API_KEY", ""):
            results = await pdr._search_bright_data_serp("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Should return empty list on API error."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.object(pdr, "http_client", return_value=mock_client), \
             patch.object(pdr, "BRIGHT_DATA_API_KEY", "test-key"):
            results = await pdr._search_bright_data_serp("test query")

        assert results == []


class TestOxylabsSerp:
    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        """Should parse Oxylabs SERP organic results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{
                "content": {
                    "results": {
                        "organic": [
                            {
                                "title": "Oxylabs Result 1",
                                "url": "https://example.com/oxy1",
                                "desc": "Description from Oxylabs.",
                            },
                        ],
                    },
                },
            }],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(pdr, "http_client", return_value=mock_client), \
             patch.object(pdr, "OXYLABS_USERNAME", "user"), \
             patch.object(pdr, "OXYLABS_PASSWORD", "pass"):
            results = await pdr._search_oxylabs_serp("test query")

        assert len(results) == 1
        assert results[0]["source"] == "oxylabs"

    @pytest.mark.asyncio
    async def test_returns_empty_without_credentials(self):
        """Should return empty list without Oxylabs credentials."""
        with patch.object(pdr, "OXYLABS_USERNAME", ""), \
             patch.object(pdr, "OXYLABS_PASSWORD", ""):
            results = await pdr._search_oxylabs_serp("test query")

        assert results == []


class TestCommercialSearch:
    @pytest.mark.asyncio
    async def test_falls_back_to_oxylabs(self):
        """Should try Oxylabs when Bright Data returns empty."""
        oxy_results = [{"title": "Oxy Result", "url": "https://oxy.com", "snippet": "test", "source": "oxylabs"}]

        with patch.object(pdr, "_search_bright_data_serp", new_callable=AsyncMock, return_value=[]), \
             patch.object(pdr, "_search_oxylabs_serp", new_callable=AsyncMock, return_value=oxy_results):
            results = await pdr._commercial_search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "oxylabs"

    @pytest.mark.asyncio
    async def test_prefers_bright_data(self):
        """Should use Bright Data results when available."""
        bd_results = [{"title": "BD Result", "url": "https://bd.com", "snippet": "test", "source": "bright_data"}]

        with patch.object(pdr, "_search_bright_data_serp", new_callable=AsyncMock, return_value=bd_results), \
             patch.object(pdr, "_search_oxylabs_serp", new_callable=AsyncMock) as mock_oxy:
            results = await pdr._commercial_search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "bright_data"
        mock_oxy.assert_not_called()


# ============================================================================
# Test: Unified Web Search (tool_web_search)
# ============================================================================

class TestToolWebSearch:
    @pytest.mark.asyncio
    async def test_merges_commercial_results(self):
        """Commercial results should always be merged with SearXNG (no moderation gate)."""
        searxng_output = (
            "1. **SearXNG Result** [trust: 0.5]\n"
            "   URL: https://searx.example.com\n   SearXNG snippet"
        )
        commercial_results = [
            {"title": "Commercial Result", "url": "https://commercial.example.com", "snippet": "Commercial snippet", "source": "bright_data"},
        ]

        with patch.object(pdr, "tool_searxng_search", new_callable=AsyncMock, return_value=searxng_output), \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", True), \
             patch.object(pdr, "_commercial_search", new_callable=AsyncMock, return_value=commercial_results):
            result = await pdr.tool_web_search("test query")

        assert "SearXNG Result" in result
        assert "Commercial Result" in result
        assert "bright_data" in result

    @pytest.mark.asyncio
    async def test_commercial_runs_for_sensitive_queries(self):
        """Commercial search should run even for sensitive/flagged queries (no moderation gate)."""
        searxng_output = "1. **SearXNG Result** [trust: 0.5]\n   URL: https://searx.example.com\n   snippet"
        commercial_results = [
            {"title": "Vendor Info", "url": "https://vendor.example.com", "snippet": "vendor details", "source": "bright_data"},
        ]

        with patch.object(pdr, "tool_searxng_search", new_callable=AsyncMock, return_value=searxng_output), \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", True), \
             patch.object(pdr, "_commercial_search", new_callable=AsyncMock, return_value=commercial_results) as mock_commercial:
            result = await pdr.tool_web_search("buy insulin without prescription poland")

        mock_commercial.assert_called_once()
        assert "Vendor Info" in result

    @pytest.mark.asyncio
    async def test_skips_commercial_when_disabled(self):
        """When COMMERCIAL_SEARCH_ENABLED is False, only SearXNG runs."""
        searxng_output = "1. **SearXNG Result** [trust: 0.5]\n   URL: https://searx.example.com\n   snippet"

        with patch.object(pdr, "tool_searxng_search", new_callable=AsyncMock, return_value=searxng_output), \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", False), \
             patch.object(pdr, "_commercial_search", new_callable=AsyncMock) as mock_commercial:
            result = await pdr.tool_web_search("test query")

        mock_commercial.assert_not_called()
        assert result == searxng_output

    @pytest.mark.asyncio
    async def test_deduplicates_urls(self):
        """Commercial results with URLs already in SearXNG should be skipped."""
        searxng_output = (
            "1. **Duplicate Result** [trust: 0.5]\n"
            "   URL: https://example.com/page\n   SearXNG snippet"
        )
        commercial_results = [
            {"title": "Same Page", "url": "https://example.com/page", "snippet": "duplicate", "source": "bright_data"},
            {"title": "New Page", "url": "https://example.com/new", "snippet": "fresh content", "source": "bright_data"},
        ]

        with patch.object(pdr, "tool_searxng_search", new_callable=AsyncMock, return_value=searxng_output), \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", True), \
             patch.object(pdr, "_commercial_search", new_callable=AsyncMock, return_value=commercial_results):
            result = await pdr.tool_web_search("test query")

        # Only the new page should be added, not the duplicate
        assert result.count("example.com/page") == 1
        assert "New Page" in result

    @pytest.mark.asyncio
    async def test_routes_through_execute_tool(self):
        """execute_tool('searxng_search', ...) should call tool_web_search."""
        with patch.object(pdr, "tool_web_search", new_callable=AsyncMock, return_value="search results") as mock_ws:
            result = await pdr.execute_tool("searxng_search", {"query": "test"})

        mock_ws.assert_called_once_with("test")
        assert result == "search results"


# ============================================================================
# Test: News Intent Detection
# ============================================================================

class TestNewsIntentDetection:
    def test_detects_news_keywords(self):
        assert pdr._has_news_intent("blockchain news today") is True
        assert pdr._has_news_intent("latest market movements") is True
        assert pdr._has_news_intent("breaking crypto developments") is True
        assert pdr._has_news_intent("stocks today performance") is True
        assert pdr._has_news_intent("what was announced yesterday") is True

    def test_detects_date_references(self):
        assert pdr._has_news_intent("events march 2026") is True
        assert pdr._has_news_intent("financial update this week") is True
        assert pdr._has_news_intent("bitcoin today price") is True

    def test_no_news_intent_for_general_queries(self):
        assert pdr._has_news_intent("how does photosynthesis work") is False
        assert pdr._has_news_intent("python programming tutorial") is False
        assert pdr._has_news_intent("best pizza recipe") is False


# ============================================================================
# Test: News Search Tool
# ============================================================================

class TestToolNewsSearch:
    @pytest.mark.asyncio
    async def test_queries_news_category(self):
        """news_search should query SearXNG with categories=news."""
        news_results = [
            {"title": "Blockchain Rally", "url": "https://news.example.com/1", "content": "BTC hits new highs"},
        ]

        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, side_effect=[news_results, []]):
            result = await pdr.tool_news_search("blockchain news", time_range="day")

        assert "Blockchain Rally" in result
        assert "(news)" in result

    @pytest.mark.asyncio
    async def test_merges_general_fallback(self):
        """news_search should merge general results with news results."""
        news_results = [
            {"title": "News Article", "url": "https://news.example.com/1", "content": "News content"},
        ]
        general_results = [
            {"title": "Blog Post", "url": "https://blog.example.com/1", "content": "Blog content"},
        ]

        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, side_effect=[news_results, general_results]):
            result = await pdr.tool_news_search("blockchain", time_range="week")

        assert "News Article" in result
        assert "Blog Post" in result

    @pytest.mark.asyncio
    async def test_deduplicates_across_categories(self):
        """Same URL in both news and general should appear only once."""
        shared_url = "https://news.example.com/shared"
        news_results = [
            {"title": "Same Article", "url": shared_url, "content": "Content"},
        ]
        general_results = [
            {"title": "Same Article Duplicate", "url": shared_url, "content": "Content"},
        ]

        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, side_effect=[news_results, general_results]):
            result = await pdr.tool_news_search("test", time_range="week")

        assert result.count(shared_url) == 1

    @pytest.mark.asyncio
    async def test_invalid_time_range_defaults_to_week(self):
        """Invalid time_range should default to 'week'."""
        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, return_value=[]) as mock_q:
            await pdr.tool_news_search("test", time_range="invalid")

        # Both calls should use "week"
        for call in mock_q.call_args_list:
            assert call.kwargs.get("time_range", call.args[2] if len(call.args) > 2 else "") == "week"

    @pytest.mark.asyncio
    async def test_routes_through_execute_tool(self):
        """execute_tool('news_search', ...) should call tool_news_search."""
        with patch.object(pdr, "tool_news_search", new_callable=AsyncMock, return_value="news results") as mock_ns:
            result = await pdr.execute_tool("news_search", {"query": "blockchain", "time_range": "day"})

        mock_ns.assert_called_once_with("blockchain", "day")
        assert result == "news results"


class TestSearxngSearchNewsAutoDetect:
    @pytest.mark.asyncio
    async def test_auto_queries_news_for_news_intent(self):
        """searxng_search should auto-query news category for news-like queries."""
        general_results = [
            {"title": "General Result", "url": "https://example.com/1", "content": "General"},
        ]
        news_results = [
            {"title": "News Result", "url": "https://news.example.com/1", "content": "Fresh news"},
        ]

        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, side_effect=[general_results, news_results]), \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", False):
            result = await pdr.tool_web_search("blockchain news today")

        assert "General Result" in result
        assert "News Result" in result

    @pytest.mark.asyncio
    async def test_skips_news_for_general_queries(self):
        """searxng_search should NOT auto-query news for non-news queries."""
        general_results = [
            {"title": "General Result", "url": "https://example.com/1", "content": "General"},
        ]

        with patch.object(pdr, "_searxng_query", new_callable=AsyncMock, return_value=general_results) as mock_q, \
             patch.object(pdr, "COMMERCIAL_SEARCH_ENABLED", False):
            result = await pdr.tool_web_search("how does photosynthesis work")

        # Should only call _searxng_query once (general), not twice
        assert mock_q.call_count == 1
