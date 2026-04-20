# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for the TrajectoryScorer and assertion classes.

Verifies the two-tier assertion system (success/expect) works correctly
with StrandsTrajectory data — without requiring a live agent.
"""

from __future__ import annotations

import pytest

from evals.utils import (
    FinalTextContains,
    FinalTextMatchesPattern,
    MaxElapsed,
    MaxToolCalls,
    RefinedTextIsShort,
    StrandsTrajectory,
    ThinkingWasRefined,
    ToolEvent,
    ToolNotCalled,
    ToolWasCalled,
    TrajectoryScorer,
)


def _make_trajectory(**kwargs) -> StrandsTrajectory:
    """Create a StrandsTrajectory with sensible defaults."""
    defaults = {
        "query": "test query",
        "response_text": "The answer is 42.",
        "elapsed": 5.0,
        "tool_call_count": 2,
        "model": "test-model",
        "tool_events": [
            ToolEvent(name="brave_web_search", input="test", timestamp=0.0),
            ToolEvent(name="firecrawl_scrape", input="url", timestamp=1.0),
        ],
    }
    defaults.update(kwargs)
    return StrandsTrajectory(**defaults)


# ---------------------------------------------------------------------------
# Success assertions
# ---------------------------------------------------------------------------


class TestFinalTextContains:
    def test_passes_when_substring_present(self) -> None:
        t = _make_trajectory(response_text="The answer is 42.")
        assert FinalTextContains("42").check(t)

    def test_fails_when_substring_missing(self) -> None:
        t = _make_trajectory(response_text="The answer is 42.")
        assert not FinalTextContains("99").check(t)

    def test_case_insensitive_by_default(self) -> None:
        t = _make_trajectory(response_text="Hello World")
        assert FinalTextContains("hello world").check(t)

    def test_case_sensitive_when_specified(self) -> None:
        t = _make_trajectory(response_text="Hello World")
        assert not FinalTextContains("hello world", case_sensitive=True).check(t)


class TestFinalTextMatchesPattern:
    def test_passes_on_match(self) -> None:
        t = _make_trajectory(response_text="Found 42 results")
        assert FinalTextMatchesPattern(r"\d+ results").check(t)

    def test_fails_on_no_match(self) -> None:
        t = _make_trajectory(response_text="No results found")
        assert not FinalTextMatchesPattern(r"\d+ results").check(t)


class TestToolWasCalled:
    def test_passes_when_tool_used(self) -> None:
        t = _make_trajectory()
        assert ToolWasCalled("brave_web_search").check(t)

    def test_fails_when_tool_not_used(self) -> None:
        t = _make_trajectory()
        assert not ToolWasCalled("nonexistent_tool").check(t)


class TestToolNotCalled:
    def test_passes_when_tool_absent(self) -> None:
        t = _make_trajectory()
        assert ToolNotCalled("nonexistent_tool").check(t)

    def test_fails_when_tool_present(self) -> None:
        t = _make_trajectory()
        assert not ToolNotCalled("brave_web_search").check(t)


class TestThinkingWasRefined:
    def test_passes_when_no_reasoning(self) -> None:
        t = _make_trajectory(reasoning_text="")
        assert ThinkingWasRefined().check(t)

    def test_passes_when_refined_differs(self) -> None:
        t = _make_trajectory(
            reasoning_text="long raw thinking...",
            refined_text="Concise summary.",
        )
        assert ThinkingWasRefined().check(t)

    def test_fails_when_not_refined(self) -> None:
        t = _make_trajectory(
            reasoning_text="long raw thinking...",
            refined_text="",
        )
        assert not ThinkingWasRefined().check(t)


# ---------------------------------------------------------------------------
# Efficiency assertions
# ---------------------------------------------------------------------------


class TestMaxToolCalls:
    def test_passes_within_limit(self) -> None:
        t = _make_trajectory(tool_call_count=5)
        assert MaxToolCalls(10).check(t)

    def test_fails_over_limit(self) -> None:
        t = _make_trajectory(tool_call_count=15)
        assert not MaxToolCalls(10).check(t)


class TestMaxElapsed:
    def test_passes_within_limit(self) -> None:
        t = _make_trajectory(elapsed=5.0)
        assert MaxElapsed(10.0).check(t)

    def test_fails_over_limit(self) -> None:
        t = _make_trajectory(elapsed=15.0)
        assert not MaxElapsed(10.0).check(t)


class TestRefinedTextIsShort:
    def test_passes_when_short(self) -> None:
        t = _make_trajectory(refined_text="Short summary.")
        assert RefinedTextIsShort(max_chars=500).check(t)

    def test_fails_when_long(self) -> None:
        t = _make_trajectory(refined_text="x" * 600)
        assert not RefinedTextIsShort(max_chars=500).check(t)


# ---------------------------------------------------------------------------
# TrajectoryScorer
# ---------------------------------------------------------------------------


class TestTrajectoryScorer:
    def test_all_pass(self) -> None:
        t = _make_trajectory()
        scorer = TrajectoryScorer()
        scorer.success(FinalTextContains("42"))
        scorer.success(ToolWasCalled("brave_web_search"))
        scorer.expect(MaxToolCalls(10))
        result = scorer.evaluate(t)
        assert len(result["failed"]) == 0
        assert len(result["passed"]) == 3

    def test_hard_fail_raises(self) -> None:
        t = _make_trajectory()
        scorer = TrajectoryScorer()
        scorer.success(FinalTextContains("nonexistent"))
        with pytest.raises(AssertionError, match="Trajectory evaluation failed"):
            scorer.evaluate(t)

    def test_soft_fail_warns_only(self) -> None:
        t = _make_trajectory(tool_call_count=20)
        scorer = TrajectoryScorer()
        scorer.expect(MaxToolCalls(5))
        result = scorer.evaluate(t)
        assert len(result["warnings"]) == 1
        assert len(result["failed"]) == 0

    def test_mixed_assertions(self) -> None:
        t = _make_trajectory(elapsed=100.0)
        scorer = TrajectoryScorer()
        scorer.success(FinalTextContains("42"))
        scorer.expect(MaxElapsed(10.0))
        result = scorer.evaluate(t)
        assert len(result["passed"]) == 1
        assert len(result["warnings"]) == 1

    def test_chaining(self) -> None:
        scorer = (
            TrajectoryScorer()
            .success(FinalTextContains("42"))
            .success(ToolWasCalled("brave_web_search"))
            .expect(MaxToolCalls(10))
        )
        t = _make_trajectory()
        result = scorer.evaluate(t)
        assert "3 passed" in result["summary"]
