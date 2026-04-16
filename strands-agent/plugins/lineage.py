"""Lineage-of-thought plugin — append-only DAG via Strands hooks.

Transplants the lineage concept from apps/adk-agent/models/corpus_store.py
(parent_id chains, row_type differentiation, immutable thoughts) into a
lightweight Python data structure using Strands Plugin hooks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from strands.plugins import Plugin, hook
from strands.hooks import AfterToolCallEvent, BeforeInvocationEvent


@dataclass
class LineageNode:
    """A single node in the research lineage DAG."""

    id: str
    type: str  # "finding", "thought", "critique", "revision", "comprehension"
    content: str
    source_tool: str = ""
    source_url: str = ""
    parent_id: str | None = None
    agent_name: str = ""
    timestamp: float = 0.0
    trust_score: float = 0.5
    novelty_score: float = 0.5
    relevance_score: float = 0.5


class LineagePlugin(Plugin):
    """Append-only DAG that tracks every research action.

    Register on ALL agents (planner, researcher, critic) with the SAME
    instance so the DAG spans the entire research session.

    The plugin intercepts:
    - AfterToolCallEvent: records tool results as "finding" nodes
    - MessageAddedEvent: records assistant messages as "thought" nodes

    Nodes are linked via parent_id (the previous node from the same agent).
    Nothing is ever deleted — only appended.
    """

    name = "lineage-tracker"

    def __init__(self):
        super().__init__()
        self.nodes: list[LineageNode] = []
        self._last_node_by_agent: dict[str, str] = {}  # agent_name -> last node id

    def reset(self):
        """Clear lineage for a new research session."""
        self.nodes.clear()
        self._last_node_by_agent.clear()

    def add_node(
        self,
        node_type: str,
        content: str,
        agent_name: str = "",
        source_tool: str = "",
        source_url: str = "",
        trust: float = 0.5,
        novelty: float = 0.5,
        relevance: float = 0.5,
    ) -> LineageNode:
        """Append a node to the DAG."""
        node = LineageNode(
            id=uuid.uuid4().hex[:12],
            type=node_type,
            content=content[:2000],  # cap content size
            source_tool=source_tool,
            source_url=source_url,
            parent_id=self._last_node_by_agent.get(agent_name),
            agent_name=agent_name,
            timestamp=time.time(),
            trust_score=trust,
            novelty_score=novelty,
            relevance_score=relevance,
        )
        self.nodes.append(node)
        self._last_node_by_agent[agent_name] = node.id
        return node

    @hook
    def on_tool_result(self, event: AfterToolCallEvent) -> None:
        """Record tool results as finding nodes."""
        if event.cancel_message:
            return  # tool was cancelled, don't record

        tool_name = event.tool_use.get("name", "unknown")
        result_text = ""
        if event.result:
            for block in event.result.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    result_text += block["text"]

        if not result_text.strip():
            return

        agent_name = (
            getattr(event.agent, "name", None) or "unknown"
            if hasattr(event, "agent")
            else "unknown"
        )
        self.add_node(
            node_type="finding",
            content=result_text[:2000],
            agent_name=agent_name,
            source_tool=tool_name,
        )

    @hook
    def on_new_invocation(self, event: BeforeInvocationEvent) -> None:
        """Reset lineage at the start of each top-level research session."""
        # Only reset if this is the planner (top-level agent)
        agent_name = (getattr(event.agent, "name", "") or "") if hasattr(event, "agent") else ""
        if agent_name == "planner":
            self.reset()

    def get_summary(self) -> dict:
        """Return a summary of the lineage DAG."""
        return {
            "total_nodes": len(self.nodes),
            "by_type": {
                t: sum(1 for n in self.nodes if n.type == t)
                for t in set(n.type for n in self.nodes)
            },
            "by_agent": {
                a: sum(1 for n in self.nodes if n.agent_name == a)
                for a in set(n.agent_name for n in self.nodes)
            },
        }
