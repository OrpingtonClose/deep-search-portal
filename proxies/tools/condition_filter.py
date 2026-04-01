"""
Pre-synthesis condition filter — simple pass-through that preserves all
findings for synthesis without lossy regex categorization.

Previous version had 239 lines of fragile regex patterns that misclassified
natural-language findings (e.g., a verified vendor lead containing the word
"not" would be classified as NEGATIVE).  This version passes all conditions
through to synthesis and lets the LLM handle prioritization.

The CategorizedConditions dataclass is kept for backward compatibility with
synthesis.py's `categorize_and_prioritize()` call, but all conditions are
placed in a single flat list ordered by confidence (highest first).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import AtomicCondition, SubagentResult


# ---------------------------------------------------------------------------
# Categorized result (simplified — no regex categorization)
# ---------------------------------------------------------------------------

@dataclass
class CategorizedConditions:
    """All conditions in a single flat list, ordered by confidence."""
    all_findings: list[AtomicCondition] = field(default_factory=list)

    # Legacy fields kept for backward compat — always empty
    procurement_verified: list[AtomicCondition] = field(default_factory=list)
    actionable: list[AtomicCondition] = field(default_factory=list)
    context: list[AtomicCondition] = field(default_factory=list)
    negative: list[AtomicCondition] = field(default_factory=list)
    errors: list[AtomicCondition] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.all_findings)

    def summary_line(self) -> str:
        return f"{len(self.all_findings)} total findings"

    def to_synthesis_text(self, max_negative: int = 5, max_errors: int = 3) -> str:
        """Format all conditions for the synthesis prompt.

        Conditions are listed in confidence order (highest first).
        No regex-based categorization — synthesis LLM handles prioritization.
        """
        if not self.all_findings:
            return ""

        parts: list[str] = []
        parts.append("### RESEARCH FINDINGS (ordered by confidence, highest first)")
        parts.append("")
        for c in self.all_findings:
            parts.append(c.to_text())

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simple pass-through (replaces regex categorization)
# ---------------------------------------------------------------------------

def categorize_and_prioritize(
    subagent_results: list[SubagentResult],
) -> CategorizedConditions:
    """Collect all conditions from subagent results, sorted by confidence.

    No regex categorization — all findings are preserved and passed to
    synthesis in confidence order.  The synthesis LLM is responsible for
    determining which findings are actionable vs. contextual.
    """
    result = CategorizedConditions()

    for sr in subagent_results:
        for c in (sr.conditions or []):
            if not c.fact or not c.fact.strip():
                continue
            result.all_findings.append(c)

    # Sort by confidence (highest first)
    result.all_findings.sort(key=lambda c: c.confidence, reverse=True)

    return result
