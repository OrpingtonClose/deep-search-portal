# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Evaluation utilities for the Strands Venice agent.

Provides trajectory capture, assertion classes, and a TrajectoryScorer
adapted from the deepagents evals framework.  Uses the SDK-native plugins
(BudgetPlugin, StreamCapturePlugin) for data capture instead of LangChain
message types.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolEvent:
    """A single tool invocation captured during agent execution."""

    name: str
    input: str
    timestamp: float
    duration: float = 0.0


@dataclass
class StrandsTrajectory:
    """Captured trajectory of a Strands agent execution.

    Attributes:
        query: The user query that triggered the execution.
        tool_events: Ordered list of tool invocations.
        reasoning_text: Raw chain-of-thought text (before refinement).
        refined_text: Refined thinking text (after ThoughtRefinerPlugin).
        response_text: Final answer text from the agent.
        elapsed: Total wall-clock seconds.
        tool_call_count: Number of unique tool calls (from BudgetPlugin).
        model: Model identifier used for the execution.
    """

    query: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)
    reasoning_text: str = ""
    refined_text: str = ""
    response_text: str = ""
    elapsed: float = 0.0
    tool_call_count: int = 0
    model: str = ""

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names invoked."""
        return [e.name for e in self.tool_events]

    @property
    def unique_tools(self) -> set[str]:
        """Set of unique tool names invoked."""
        return {e.name for e in self.tool_events}

    def pretty(self) -> str:
        """Return a human-readable summary of the trajectory."""
        lines = [f"query: {self.query}", f"model: {self.model}"]
        for i, ev in enumerate(self.tool_events, 1):
            lines.append(f"  tool {i}: {ev.name} — {ev.input[:80]}")
        if self.reasoning_text:
            preview = self.reasoning_text[:200].replace("\n", "\\n")
            lines.append(f"  reasoning: {preview}")
        if self.response_text:
            preview = self.response_text[:200].replace("\n", "\\n")
            lines.append(f"  response: {preview}")
        lines.append(f"  elapsed: {self.elapsed:.1f}s, tools: {self.tool_call_count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assertion base classes
# ---------------------------------------------------------------------------


class SuccessAssertion:
    """Base for correctness assertions that fail the test when violated."""

    def check(self, trajectory: StrandsTrajectory) -> bool:
        """Return True when the assertion holds."""
        raise NotImplementedError

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        """Return a human-readable explanation of why the check failed."""
        raise NotImplementedError


@dataclass(frozen=True)
class EfficiencyAssertion:
    """Base for trajectory-shape assertions that are logged but never fail."""

    def check(self, trajectory: StrandsTrajectory) -> bool:
        """Return True when the assertion holds."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return a human-readable description of this assertion."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in assertions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalTextContains(SuccessAssertion):
    """Assert that the final response contains specific text."""

    substring: str
    case_sensitive: bool = False

    def check(self, trajectory: StrandsTrajectory) -> bool:
        text = trajectory.response_text
        target = self.substring
        if not self.case_sensitive:
            text = text.lower()
            target = target.lower()
        return target in text

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        return f"Response does not contain '{self.substring}'"


@dataclass(frozen=True)
class FinalTextMatchesPattern(SuccessAssertion):
    """Assert that the final response matches a regex pattern."""

    pattern: str
    flags: int = re.IGNORECASE

    def check(self, trajectory: StrandsTrajectory) -> bool:
        return bool(re.search(self.pattern, trajectory.response_text, self.flags))

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        return f"Response does not match pattern '{self.pattern}'"


@dataclass(frozen=True)
class ToolWasCalled(SuccessAssertion):
    """Assert that a specific tool was invoked."""

    tool_name: str

    def check(self, trajectory: StrandsTrajectory) -> bool:
        return self.tool_name in trajectory.unique_tools

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        return (
            f"Tool '{self.tool_name}' was not called. "
            f"Tools used: {sorted(trajectory.unique_tools)}"
        )


@dataclass(frozen=True)
class ToolNotCalled(SuccessAssertion):
    """Assert that a specific tool was NOT invoked."""

    tool_name: str

    def check(self, trajectory: StrandsTrajectory) -> bool:
        return self.tool_name not in trajectory.unique_tools

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        return f"Tool '{self.tool_name}' was unexpectedly called"


@dataclass(frozen=True)
class MaxToolCalls(EfficiencyAssertion):
    """Soft assertion: tool count should not exceed a threshold."""

    max_calls: int

    def check(self, trajectory: StrandsTrajectory) -> bool:
        return trajectory.tool_call_count <= self.max_calls

    def describe(self) -> str:
        return f"Expected at most {self.max_calls} tool calls"


@dataclass(frozen=True)
class MaxElapsed(EfficiencyAssertion):
    """Soft assertion: execution should complete within a time limit."""

    max_seconds: float

    def check(self, trajectory: StrandsTrajectory) -> bool:
        return trajectory.elapsed <= self.max_seconds

    def describe(self) -> str:
        return f"Expected completion within {self.max_seconds}s"


@dataclass(frozen=True)
class ThinkingWasRefined(SuccessAssertion):
    """Assert that raw thinking was refined (not passed through raw)."""

    def check(self, trajectory: StrandsTrajectory) -> bool:
        if not trajectory.reasoning_text:
            return True  # No thinking to refine
        return bool(trajectory.refined_text) and (
            trajectory.refined_text != trajectory.reasoning_text
        )

    def describe_failure(self, trajectory: StrandsTrajectory) -> str:
        return "Thinking was not refined — raw chain-of-thought passed through"


@dataclass(frozen=True)
class RefinedTextIsShort(EfficiencyAssertion):
    """Soft assertion: refined thinking should be concise."""

    max_chars: int = 500

    def check(self, trajectory: StrandsTrajectory) -> bool:
        if not trajectory.refined_text:
            return True
        return len(trajectory.refined_text) <= self.max_chars

    def describe(self) -> str:
        return f"Refined thinking should be under {self.max_chars} chars"


# ---------------------------------------------------------------------------
# TrajectoryScorer
# ---------------------------------------------------------------------------


class TrajectoryScorer:
    """Two-tier assertion scorer adapted from the deepagents pattern.

    Usage::

        scorer = TrajectoryScorer()
        scorer.success(FinalTextContains("Tor"))
        scorer.success(ToolWasCalled("brave_web_search"))
        scorer.expect(MaxToolCalls(10))
        scorer.expect(MaxElapsed(60))
        scorer.evaluate(trajectory)
    """

    def __init__(self) -> None:
        self._success: list[SuccessAssertion] = []
        self._expect: list[EfficiencyAssertion] = []

    def success(self, assertion: SuccessAssertion) -> "TrajectoryScorer":
        """Add a hard-fail assertion.  Test fails if this doesn't pass."""
        self._success.append(assertion)
        return self

    def expect(self, assertion: EfficiencyAssertion) -> "TrajectoryScorer":
        """Add a soft assertion.  Logged but does not fail the test."""
        self._expect.append(assertion)
        return self

    def evaluate(self, trajectory: StrandsTrajectory) -> dict[str, Any]:
        """Run all assertions against a trajectory.

        Args:
            trajectory: The captured agent trajectory to evaluate.

        Returns:
            Dict with 'passed', 'failed', 'warnings', and 'summary' keys.

        Raises:
            AssertionError: If any success assertion fails.
        """
        results: dict[str, Any] = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "summary": "",
        }

        # Hard-fail assertions
        for assertion in self._success:
            if assertion.check(trajectory):
                results["passed"].append(type(assertion).__name__)
                logger.info(
                    "assertion=<%s> | PASSED",
                    type(assertion).__name__,
                )
            else:
                failure = assertion.describe_failure(trajectory)
                results["failed"].append(failure)
                logger.error(
                    "assertion=<%s>, reason=<%s> | FAILED",
                    type(assertion).__name__,
                    failure,
                )

        # Soft assertions (warnings only)
        for assertion in self._expect:
            if assertion.check(trajectory):
                results["passed"].append(type(assertion).__name__)
            else:
                desc = assertion.describe()
                results["warnings"].append(desc)
                logger.warning(
                    "assertion=<%s>, reason=<%s> | SOFT FAIL",
                    type(assertion).__name__,
                    desc,
                )

        n_passed = len(results["passed"])
        n_failed = len(results["failed"])
        n_warnings = len(results["warnings"])
        results["summary"] = (
            f"{n_passed} passed, {n_failed} failed, {n_warnings} warnings"
        )

        if results["failed"]:
            failures = "\n".join(f"  - {f}" for f in results["failed"])
            raise AssertionError(
                f"Trajectory evaluation failed:\n{failures}\n\n"
                f"Trajectory:\n{trajectory.pretty()}"
            )

        return results


# ---------------------------------------------------------------------------
# Trajectory capture helper
# ---------------------------------------------------------------------------


def capture_trajectory(
    agent: Any,
    query: str,
    model: str = "",
    stream_capture: Any = None,
    budget_plugin: Any = None,
    thought_refiner: Any = None,
) -> StrandsTrajectory:
    """Run an agent and capture its trajectory using SDK plugins.

    Args:
        agent: A Strands Agent instance.
        query: The user query to send.
        model: Model identifier for logging.
        stream_capture: StreamCapturePlugin instance for token capture.
        budget_plugin: BudgetPlugin instance for tool count.
        thought_refiner: ThoughtRefinerPlugin instance for refinement.

    Returns:
        A StrandsTrajectory with all captured data.
    """
    start = time.time()

    # Activate stream capture if available
    if stream_capture is not None:
        stream_capture.activate()

    try:
        result = agent(query)
        response_text = str(result)
    except Exception as exc:
        response_text = f"Error: {exc}"
    finally:
        if stream_capture is not None:
            stream_capture.deactivate()

    elapsed = time.time() - start

    # Build trajectory from plugin state
    tool_events = []
    reasoning_text = ""
    if stream_capture is not None:
        for ev in stream_capture.tool_events:
            tool_events.append(
                ToolEvent(
                    name=ev.get("tool", "unknown"),
                    input=ev.get("input", ""),
                    timestamp=ev.get("time", 0.0),
                )
            )
        reasoning_text = "".join(stream_capture.reasoning_text)

    tool_call_count = 0
    if budget_plugin is not None:
        tool_call_count = budget_plugin.tool_call_count

    return StrandsTrajectory(
        query=query,
        tool_events=tool_events,
        reasoning_text=reasoning_text,
        response_text=response_text,
        elapsed=elapsed,
        tool_call_count=tool_call_count,
        model=model,
    )
