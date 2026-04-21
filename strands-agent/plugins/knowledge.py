# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""KnowledgePlugin — persistent cross-conversation knowledge accumulation.

A pure Strands SDK plugin that hooks into the agent lifecycle to:

1. ``BeforeInvocationEvent``: retrieve relevant past knowledge for the
   current query and inject it as context.
2. ``AfterToolCallEvent``: extract facts from individual tool results
   as they complete (granular, per-tool capture).
3. ``AfterInvocationEvent``: extract entities and any remaining facts
   from the final assistant response.

Tools are declared directly on the plugin via ``@tool`` so they are
auto-discovered by the SDK when the plugin is attached to an agent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from strands import tool
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
)
from strands.plugins import Plugin, hook
from strands.types.content import Message

from knowledge_store import Entity, Insight, KnowledgeStore, get_knowledge_store

logger = logging.getLogger(__name__)

_KNOWLEDGE_MARKER = "[PRIOR KNOWLEDGE]"

# Maximum number of insights to inject per query
_MAX_INJECT = 8

# Minimum confidence for injected insights
_MIN_INJECT_CONFIDENCE = 0.3

# Minimum word count for a fact to be worth storing
_MIN_FACT_WORDS = 5


class KnowledgePlugin(Plugin):
    """Accumulate and retrieve persistent knowledge across conversations.

    On each invocation:
    - Before: searches past knowledge for the current query and injects
      relevant findings as context (if any exist)
    - After each tool call: extracts factual claims from tool results
    - After invocation: extracts entities from the full response

    Tools (``recall_knowledge``, ``store_insight``, ``recall_entities``,
    ``knowledge_stats``) are auto-registered on the agent via the SDK's
    plugin tool discovery.
    """

    name: str = "knowledge"

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        max_inject: int = _MAX_INJECT,
        min_inject_confidence: float = _MIN_INJECT_CONFIDENCE,
    ) -> None:
        """Initialize the knowledge plugin.

        Args:
            store: KnowledgeStore instance. Uses the global singleton
                if not provided.
            max_inject: Maximum past insights to inject per query.
            min_inject_confidence: Minimum confidence for injection.
        """
        super().__init__()
        self._store = store or get_knowledge_store()
        self._max_inject = max_inject
        self._min_inject_confidence = min_inject_confidence
        self._current_query: str = ""

    # ── Hooks ─────────────────────────────────────────────────────────

    @hook
    def inject_knowledge(self, event: BeforeInvocationEvent) -> None:
        """Retrieve and inject relevant past knowledge before invocation."""
        if event.messages is None:
            self._current_query = ""
            return

        query = self._extract_query(event.messages)
        self._current_query = query
        if not query:
            return

        # Strip any stale knowledge markers from previous turns
        msgs = [
            msg for msg in event.messages
            if not self._is_knowledge_message(msg)
        ]

        # Search for relevant past knowledge
        insights = self._store.search_insights(
            query=query,
            limit=self._max_inject,
            min_confidence=self._min_inject_confidence,
        )

        if not insights:
            event.messages = msgs
            return

        knowledge_text = self._format_knowledge_block(insights)
        knowledge_msg: Message = {
            "role": "user",
            "content": [{"text": f"{_KNOWLEDGE_MARKER}\n{knowledge_text}"}],
        }

        # Insert before the last user message
        insert_idx = len(msgs) - 1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                insert_idx = i
                break
        msgs.insert(insert_idx, knowledge_msg)
        event.messages = msgs

        logger.info(
            "injected=<%d>, query=<%s> | prior knowledge injected",
            len(insights),
            query[:80],
        )

    @hook
    def capture_tool_knowledge(self, event: AfterToolCallEvent) -> None:
        """Extract facts from individual tool results as they complete."""
        if not self._current_query:
            return

        # Skip results from our own tools to avoid feedback loops
        tool_name = event.tool_use.get("name", "") if event.tool_use else ""
        if tool_name in {"recall_knowledge", "store_insight", "recall_entities", "knowledge_stats"}:
            return

        # Get the text content from the tool result
        result = event.result
        if not result:
            return

        result_text = self._extract_tool_result_text(result)
        if not result_text or len(result_text) < 20:
            return

        # Extract facts from this tool's output
        facts = self._extract_facts(result_text)
        stored = 0
        for fact_text, source_url in facts:
            if self._store.has_similar_insight(fact_text):
                continue
            insight = Insight(
                fact=fact_text,
                source_url=source_url,
                source_type=self._infer_source_type(source_url),
                topic=self._infer_topic(self._current_query),
                confidence=0.7,
                query_context=self._current_query[:500],
            )
            self._store.store_insight(insight)
            stored += 1

        if stored > 0:
            logger.info(
                "tool=<%s>, insights=<%d> | knowledge captured from tool",
                tool_name,
                stored,
            )

    @hook
    def extract_entities(self, event: AfterInvocationEvent) -> None:
        """Extract entities from the final response."""
        if not self._current_query:
            return

        messages = event.result.messages if event.result else []
        if not messages:
            return

        assistant_text = self._extract_assistant_response(messages)
        if not assistant_text:
            return

        entities = self._extract_entities_from_text(assistant_text)
        for name, etype in entities:
            self._store.store_entity(Entity(name=name, entity_type=etype))

        if entities:
            logger.info(
                "entities=<%d> | entities extracted from response",
                len(entities),
            )

    # ── Tools (auto-discovered by SDK) ────────────────────────────────

    @tool
    def recall_knowledge(
        self,
        query: str,
        max_results: int = 10,
        min_confidence: float = 0.0,
        topic: str = "",
    ) -> str:
        """Search past research knowledge for relevant facts and insights.

        Use this BEFORE starting a new research task to check what you
        already know. This saves time and avoids duplicate work.

        Args:
            query: Search query — describe what you're looking for.
            max_results: Maximum number of results to return.
            min_confidence: Minimum confidence threshold (0.0-1.0).
            topic: Optional topic filter.

        Returns:
            Formatted list of relevant past insights with metadata.
        """
        insights = self._store.search_insights(
            query=query, limit=max_results,
            min_confidence=min_confidence, topic=topic,
        )
        if not insights:
            return f"No prior knowledge found for: {query}"

        lines = [f"Found {len(insights)} relevant prior insight(s):\n"]
        for i, ins in enumerate(insights, 1):
            line = f"{i}. {ins.get('fact', '')}"
            meta = []
            if ins.get("topic"):
                meta.append(f"topic: {ins['topic']}")
            if ins.get("source_url"):
                meta.append(f"source: {ins['source_url']}")
            meta.append(f"confidence: {ins.get('confidence', 0):.2f}")
            meta.append(f"stored: {ins.get('created_at', '')[:10]}")
            access = ins.get("access_count", 0)
            if access > 1:
                meta.append(f"accessed {access}x")
            line += f"\n   [{', '.join(meta)}]"
            lines.append(line)
        return "\n".join(lines)

    @tool
    def store_insight(
        self,
        fact: str,
        source_url: str = "",
        topic: str = "",
        confidence: float = 0.7,
    ) -> str:
        """Store an important finding for future reference.

        Use this when you discover a key fact that would be valuable for
        future research queries. The insight persists across conversations.

        Args:
            fact: The factual claim to store (one clear sentence).
            source_url: URL where this fact was found.
            topic: Topic tag for categorization.
            confidence: How confident you are in this fact (0.0-1.0).

        Returns:
            Confirmation message with the stored insight ID.
        """
        if self._store.has_similar_insight(fact):
            return f"Similar insight already exists, skipping: {fact[:80]}..."

        insight = Insight(
            fact=fact,
            source_url=source_url,
            topic=topic,
            confidence=max(0.0, min(1.0, confidence)),
        )
        insight_id = self._store.store_insight(insight)
        logger.info("id=<%d>, topic=<%s> | insight manually stored", insight_id, topic)
        return f"Stored insight #{insight_id}: {fact[:80]}..."

    @tool
    def recall_entities(self, query: str = "", limit: int = 20) -> str:
        """Look up named entities from past research.

        Entities (people, compounds, organizations) are automatically
        tracked across all conversations.

        Args:
            query: Entity name to search for. Empty returns top entities.
            limit: Maximum results to return.

        Returns:
            Formatted list of entities with metadata.
        """
        if query:
            entities = self._store.search_entities(query, limit=limit)
        else:
            entities = self._store.get_top_entities(limit=limit)

        if not entities:
            return f"No entities found matching: {query}" if query else "No entities tracked yet"

        lines = [f"{'Matching' if query else 'Top'} entities ({len(entities)}):\n"]
        for e in entities:
            line = f"- {e.get('name', '')}"
            if e.get("entity_type"):
                line += f" ({e['entity_type']})"
            line += f" — {e.get('mention_count', 0)} mention(s)"
            if e.get("description"):
                line += f"\n  {e['description'][:120]}"
            lines.append(line)
        return "\n".join(lines)

    @tool
    def knowledge_stats(self) -> str:
        """Get summary statistics about accumulated knowledge.

        Shows total insights, entities, top topics, average confidence,
        and most frequently accessed facts.

        Returns:
            Formatted knowledge statistics.
        """
        stats = self._store.get_stats()
        lines = [
            "=== KNOWLEDGE BASE STATISTICS ===",
            f"Total insights: {stats['total_insights']}",
            f"Total entities: {stats['total_entities']}",
            f"Average confidence: {stats['avg_confidence']:.2f}",
        ]
        if stats["top_topics"]:
            lines.append("\nTop topics:")
            for t in stats["top_topics"]:
                lines.append(f"  - {t['topic']} ({t['count']} insights)")
        if stats["most_accessed"]:
            lines.append("\nMost accessed facts:")
            for m in stats["most_accessed"]:
                lines.append(f"  - [{m['access_count']}x] {m['fact']}")
        return "\n".join(lines)

    # ── Formatting ────────────────────────────────────────────────────

    @staticmethod
    def _format_knowledge_block(insights: list[dict[str, Any]]) -> str:
        """Format insights into a concise context block for the agent."""
        lines = [
            "PRIOR KNOWLEDGE (from previous research conversations):",
            "The following facts were gathered in earlier conversations.",
            "Use them as a starting point — verify if needed, don't repeat the search.\n",
        ]
        for i, ins in enumerate(insights, 1):
            line = f"{i}. {ins.get('fact', '')}"
            meta_parts = []
            if ins.get("topic"):
                meta_parts.append(f"topic: {ins['topic']}")
            if ins.get("source_url"):
                meta_parts.append(f"source: {ins['source_url']}")
            if ins.get("confidence", 1.0) < 0.5:
                meta_parts.append("low confidence — verify")
            if meta_parts:
                line += f" [{', '.join(meta_parts)}]"
            lines.append(line)
        return "\n".join(lines)

    # ── Extraction helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_query(messages: list[Any] | None) -> str:
        """Extract the user's query from the last user message."""
        if not messages:
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text and _KNOWLEDGE_MARKER not in text:
                            return text
            elif isinstance(content, str) and _KNOWLEDGE_MARKER not in content:
                return content
        return ""

    @staticmethod
    def _extract_assistant_response(messages: list[Any]) -> str:
        """Extract text from the last assistant message."""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                if texts:
                    return " ".join(texts)
            elif isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _extract_tool_result_text(result: dict[str, Any]) -> str:
        """Extract readable text from a ToolResult dict."""
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        texts.append(block["text"])
                    elif "json" in block:
                        texts.append(str(block["json"]))
            return " ".join(texts)
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _extract_facts(text: str) -> list[tuple[str, str]]:
        """Extract factual statements from text.

        Uses heuristics: sentences with numbers, dates, proper nouns,
        or citations are more likely to be worth storing.

        Args:
            text: The text to extract facts from.

        Returns:
            List of (fact_text, source_url) tuples.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        facts: list[tuple[str, str]] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < _MIN_FACT_WORDS:
                continue

            lower = sentence.lower()
            if any(skip in lower for skip in [
                "let me", "i'll ", "i will", "based on",
                "here are", "here is", "searching for",
                "looking for", "found the following",
                "in summary", "to summarize", "in conclusion",
                "i think", "i believe", "it seems",
            ]):
                continue

            # Check for factual indicators
            has_number = bool(re.search(r'\d+', sentence))
            has_year = bool(re.search(r'\b(19|20)\d{2}\b', sentence))
            # Skip first word (sentence-initial capitalization) when checking for proper nouns
            words = sentence.split()
            rest = " ".join(words[1:]) if len(words) > 1 else ""
            has_proper_noun = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', rest))
            has_unit = bool(re.search(
                r'\d+\s*(?:mg|mcg|kg|ml|%|percent|billion|million|thousand)',
                sentence, re.IGNORECASE,
            ))

            if not any([has_number, has_year, has_proper_noun, has_unit]):
                continue

            # Extract URL if present
            url_match = re.search(r'https?://\S+', sentence)
            source_url = url_match.group(0).rstrip('.,;)') if url_match else ""

            clean = re.sub(r'https?://\S+', '', sentence).strip()
            clean = re.sub(r'\s+', ' ', clean)

            if len(clean.split()) >= _MIN_FACT_WORDS:
                facts.append((clean, source_url))

        return facts

    @staticmethod
    def _extract_entities_from_text(text: str) -> list[tuple[str, str]]:
        """Extract named entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            List of (entity_name, entity_type) tuples.
        """
        entities: list[tuple[str, str]] = []
        seen: set[str] = set()

        skip = {"THE", "AND", "BUT", "FOR", "NOT", "ARE", "WAS", "HAS",
                "THIS", "THAT", "WITH", "FROM", "WILL", "CAN", "ALL",
                "ITS", "MAY", "USE", "HOW", "NEW", "ONE", "TWO", "OUR",
                "LET", "SEE", "YES", "ALSO", "JUST", "BEEN", "ONLY",
                "EACH", "VERY", "THAN", "HERE", "THEN", "SOME"}

        # Hyphenated compounds (e.g., BPC-157, GLP-1)
        for m in re.finditer(r'\b([A-Z][A-Za-z0-9]*-\d+[A-Za-z]*)\b', text):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                entities.append((name, "compound"))

        # Uppercase abbreviations (3+ letters)
        for m in re.finditer(r'\b([A-Z]{3,})\b', text):
            name = m.group(1)
            if name not in skip and name not in seen:
                seen.add(name)
                entities.append((name, "organization"))

        return entities

    @staticmethod
    def _infer_source_type(url: str) -> str:
        """Infer the source type from a URL."""
        if not url:
            return "research"
        lower = url.lower()
        if "pubmed" in lower or ".edu" in lower:
            return "academic"
        if "arxiv" in lower:
            return "preprint"
        if "clinicaltrials.gov" in lower or ".gov" in lower:
            return "government"
        if "reddit" in lower:
            return "forum"
        if "youtube" in lower:
            return "video"
        return "research"

    @staticmethod
    def _infer_topic(query: str) -> str:
        """Extract a short topic from the user query."""
        stop_words = {
            "what", "how", "why", "when", "where", "who", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "would", "should", "will",
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "not", "with", "from", "by",
            "find", "search", "look", "tell", "me", "about",
            "give", "show", "explain", "describe",
            "papers", "research", "info", "information", "data",
            "latest", "recent", "new", "current", "results",
        }
        words = [
            w for w in query.lower().split()
            if w not in stop_words and len(w) > 2
        ]
        return " ".join(words[:4]) if words else ""

    @staticmethod
    def _is_knowledge_message(msg: Any) -> bool:
        """Check if a message is a knowledge injection marker."""
        if not isinstance(msg, dict) or msg.get("role") != "user":
            return False
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and _KNOWLEDGE_MARKER in block.get("text", ""):
                    return True
        elif isinstance(content, str) and _KNOWLEDGE_MARKER in content:
            return True
        return False
