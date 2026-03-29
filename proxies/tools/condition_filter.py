"""
Pre-synthesis condition filter — categorize and prioritize conditions
so actionable findings always surface first in the synthesis prompt.

Problem addressed: The synthesis LLM reads 800+ conditions where 700 are
"no results" / errors / negative findings, drowning out the 7 actual
vendor leads.  By categorizing conditions before feeding them to synthesis,
we ensure actionable findings are placed first and dominate the LLM's
attention window.

Categories (from Kimi's solution, adapted to our AtomicCondition model):
  ACTIONABLE  — concrete entities, vendors, URLs, prices, contacts, methods
  CONTEXT     — legal/regulatory, background knowledge, general information
  NEGATIVE    — "not found", "no evidence", "no results" conclusions
  ERROR       — [TOOL_ERROR], [ACCESS BLOCKED], [CENSORSHIP DETECTED], tool failures

Usage in the synthesis pipeline:
    from .condition_filter import categorize_and_prioritize
    prioritized = categorize_and_prioritize(subagent_results)
    # prioritized.to_synthesis_text() → ACTIONABLE first, then CONTEXT, then brief NEGATIVE summary
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import AtomicCondition, SubagentResult


# ---------------------------------------------------------------------------
# Category detection patterns
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = re.compile(
    r"(?i)"
    r"\[TOOL_ERROR\]"
    r"|\[ACCESS.BLOCKED\]"
    r"|\[CENSORSHIP.DETECTED\]"
    r"|Tool error"
    r"|Tool failed"
    r"|Unknown tool:"
    r"|HTTP\s+\d{3}"
    r"|timed?\s*out"
    r"|access tiers exhausted"
    r"|credentials.*missing"
    r"|No credentials"
    r"|search error:"
)

_NEGATIVE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:^|\.\s+)no\s+(?:publicly\s+)?(?:documented|verified|confirmed|available|evidence|results?|data|information|records?|cases?|sources?|vendors?|reports?|listings?)\s+(?:were\s+)?(?:found|identified|available|exist|located|detected|discovered)"
    r"|does\s+not\s+(?:ship|sell|offer|provide|confirm|list|stock|carry)"
    r"|not\s+(?:publicly\s+)?available"
    r"|no\s+public\s+(?:evidence|information|data|records)"
    r"|could\s+not\s+(?:find|locate|identify|verify|confirm)"
    r"|(?:zero|0)\s+results"
    r"|nothing\s+(?:was\s+)?found"
    r"|no\s+(?:relevant\s+)?(?:matches|hits|listings)"
)

_ACTIONABLE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:https?://\S+)"                    # contains a URL
    r"|(?:\$\d+|\€\d+|PLN\s*\d+|\d+\s*(?:PLN|EUR|USD|GBP))"  # contains pricing
    r"|(?:ships?\s+(?:to|via|through))"     # shipping information
    r"|(?:(?:contact|order|buy|purchase)\s+(?:via|through|at|from))"  # purchase instructions
    r"|(?:Telegram|Discord|Signal|Threema|WhatsApp)\s+(?:group|channel|contact|@)"  # messaging contacts
    r"|(?:vendor|supplier|seller|shop|store|pharmacy|source)\s*(?::|named|called)"  # vendor names
    r"|(?:ships?\s+DHL|ships?\s+FedEx|ships?\s+EMS)"  # specific shipping methods
)


# ---------------------------------------------------------------------------
# Categorized result
# ---------------------------------------------------------------------------

@dataclass
class CategorizedConditions:
    """Conditions split into priority categories for synthesis."""
    actionable: list[AtomicCondition] = field(default_factory=list)
    context: list[AtomicCondition] = field(default_factory=list)
    negative: list[AtomicCondition] = field(default_factory=list)
    errors: list[AtomicCondition] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.actionable) + len(self.context) + len(self.negative) + len(self.errors)

    def summary_line(self) -> str:
        return (
            f"{len(self.actionable)} actionable, "
            f"{len(self.context)} context, "
            f"{len(self.negative)} negative, "
            f"{len(self.errors)} errors"
        )

    def to_synthesis_text(self, max_negative: int = 5, max_errors: int = 3) -> str:
        """Format conditions for the synthesis prompt.

        ACTIONABLE conditions come first (full detail).
        CONTEXT conditions come second (full detail).
        NEGATIVE conditions are summarized (only top N, rest as count).
        ERROR conditions are briefly noted (only count + sample).
        """
        parts: list[str] = []

        if self.actionable:
            parts.append("### ACTIONABLE FINDINGS (highest priority — these are concrete leads)")
            for c in self.actionable:
                parts.append(c.to_text())

        if self.context:
            parts.append("\n### CONTEXTUAL FINDINGS (background/regulatory information)")
            for c in self.context:
                parts.append(c.to_text())

        if self.negative:
            parts.append(f"\n### NEGATIVE FINDINGS ({len(self.negative)} total — topics where nothing was found)")
            for c in self.negative[:max_negative]:
                parts.append(c.to_text())
            remaining = len(self.negative) - max_negative
            if remaining > 0:
                parts.append(f"  ... and {remaining} more negative findings omitted")

        if self.errors:
            parts.append(f"\n### TOOL ERRORS ({len(self.errors)} total — access barriers encountered)")
            for c in self.errors[:max_errors]:
                fact_short = c.fact[:150]
                parts.append(f"- {fact_short}")
            remaining = len(self.errors) - max_errors
            if remaining > 0:
                parts.append(f"  ... and {remaining} more tool errors omitted")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Categorization logic
# ---------------------------------------------------------------------------

def categorize_condition(c: AtomicCondition) -> str:
    """Categorize a single condition.

    Returns one of: "actionable", "context", "negative", "error".
    """
    fact = c.fact

    # Errors first — tool failures, access blocks
    if _ERROR_PATTERNS.search(fact):
        return "error"

    # Negative findings — "nothing found" conclusions
    if _NEGATIVE_PATTERNS.search(fact):
        return "negative"

    # Actionable — concrete entities, URLs, prices, contacts
    if _ACTIONABLE_PATTERNS.search(fact):
        return "actionable"

    # High-confidence findings with real source URLs are likely actionable
    if c.confidence >= 0.7 and c.source_url and c.source_url.startswith("http"):
        return "actionable"

    # Everything else is context
    return "context"


def categorize_and_prioritize(
    subagent_results: list[SubagentResult],
) -> CategorizedConditions:
    """Categorize all conditions from subagent results.

    Returns a CategorizedConditions object with conditions sorted by category
    and ready for synthesis.
    """
    result = CategorizedConditions()

    for sr in subagent_results:
        for c in (sr.conditions or []):
            if not c.fact or not c.fact.strip():
                continue

            category = categorize_condition(c)
            if category == "actionable":
                result.actionable.append(c)
            elif category == "negative":
                result.negative.append(c)
            elif category == "error":
                result.errors.append(c)
            else:
                result.context.append(c)

    # Sort actionable by confidence (highest first)
    result.actionable.sort(key=lambda c: c.confidence, reverse=True)
    # Sort context by confidence
    result.context.sort(key=lambda c: c.confidence, reverse=True)

    return result
