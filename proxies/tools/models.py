"""
Data models: CrossRef, AtomicCondition, SubagentResult, ResearchNode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class CrossRef:
    """A directional link between two conditions in the knowledge net.

    relation is one of: "confirms", "contradicts", "related"
    target_idx is the index of the linked condition in the ConditionStore.
    similarity is the Jaccard similarity score that triggered the link.
    """
    relation: str   # "confirms" | "contradicts" | "related"
    target_idx: int
    similarity: float = 0.0


@dataclass
class AtomicCondition:
    """A single compressed research finding (Atom of Thoughts)."""
    fact: str
    source_url: str = ""
    confidence: float = 0.5
    angle: str = ""
    domain: str = ""
    is_serendipitous: bool = False
    trust_score: float = 0.5
    serendipity_score_val: float = 0.0
    entities: list[str] = field(default_factory=list)
    # Verification status set by Veritas: "verified", "speculative",
    # "fabricated", "overconfident", or "" (not yet checked).
    verification_status: str = ""
    # Enrichment metadata
    publication_date: str = ""   # ISO date string when available
    author: str = ""             # Author or creator name
    content_type: str = ""       # e.g. "academic_paper", "news", "forum_post", "video"
    source_type: str = ""        # e.g. "pubmed", "arxiv", "hackernews", "substack"
    # Cross-reference links to other conditions — forms the knowledge net
    cross_refs: list[CrossRef] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"- {self.fact}"]
        if self.source_url:
            parts[0] += f" [source: {self.source_url}]"
        if self.confidence != 0.5:
            parts[0] += f" (confidence: {self.confidence:.1f})"
        if self.trust_score != 0.5:
            parts[0] += f" (trust: {self.trust_score:.1f})"
        if self.verification_status == "speculative":
            parts[0] += " [SPECULATIVE]"
        elif self.verification_status == "verified":
            parts[0] += " [VERIFIED]"
        elif self.verification_status == "fabricated":
            parts[0] += " [FABRICATED]"
        if self.is_serendipitous:
            parts[0] += " [SERENDIPITOUS]"
        if self.serendipity_score_val > 0.3:
            parts[0] += f" [serendipity: {self.serendipity_score_val:.2f}]"
        if self.source_type:
            parts[0] += f" [via: {self.source_type}]"
        if self.author:
            parts[0] += f" [author: {self.author}]"
        if self.publication_date:
            parts[0] += f" [date: {self.publication_date}]"
        return parts[0]


@dataclass
class SubagentResult:
    """Result from a single subagent's research."""
    angle: str
    conditions: list[AtomicCondition] = field(default_factory=list)
    turns_used: int = 0
    tool_calls_made: int = 0
    error: str = ""
    novelty_history: list[float] = field(default_factory=list)
    spawned_children: int = 0


@dataclass
class ResearchNode:
    """A single node in the research exploration tree.

    Each node represents a question/claim to investigate.  Workers pull
    nodes from a priority queue, research them, and push child nodes
    back.  Only the active LLM+tool research phase occupies a
    semaphore slot.
    """
    id: str
    question: str
    context: str  # why this node was spawned
    depth: int
    pressure: float  # 0-1, higher = explore first
    parent_id: Optional[str] = None
    status: str = "pending"  # pending | researching | done | pruned

    def __lt__(self, other: "ResearchNode") -> bool:
        """PriorityQueue needs ordering; higher pressure = higher priority."""
        return self.pressure > other.pressure


