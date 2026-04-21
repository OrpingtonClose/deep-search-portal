# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Simple JSON-backed persistent knowledge store for cross-conversation learning.

Stores facts, insights, and entities extracted from research conversations
so new queries can leverage accumulated knowledge instead of starting from
scratch every time.

Uses keyword matching for retrieval — no external dependencies.

Location: ~/.mirothinker/knowledge/knowledge.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(
    os.environ.get(
        "MIROTHINKER_KNOWLEDGE_DIR",
        os.path.expanduser("~/.mirothinker/knowledge"),
    )
)
KNOWLEDGE_FILE = KNOWLEDGE_DIR / "knowledge.json"


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class Insight:
    """A single knowledge entry extracted from a research conversation."""

    fact: str
    source_url: str = ""
    source_type: str = ""
    topic: str = ""
    confidence: float = 0.7
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = 0
    query_context: str = ""


@dataclass
class Entity:
    """A named entity seen across multiple conversations."""

    name: str
    entity_type: str = ""
    description: str = ""
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    mention_count: int = 1


# ── KnowledgeStore ────────────────────────────────────────────────────


class KnowledgeStore:
    """Persistent knowledge store backed by a JSON file.

    Thread-safe via RLock. The store keeps data in memory and flushes
    to disk on every write so nothing is lost across restarts.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the knowledge store.

        Args:
            path: Path to the JSON file. Defaults to
                ``~/.mirothinker/knowledge/knowledge.json``.
                Pass ``:memory:`` to skip file persistence (testing).
        """
        self._path: str | None = None if str(path) == ":memory:" else str(path or KNOWLEDGE_FILE)
        self._lock = threading.RLock()
        self._insights: list[dict[str, Any]] = []
        self._entities: list[dict[str, Any]] = []
        self._next_id = 1
        self._load()

        logger.info(
            "insights=<%d>, entities=<%d> | knowledge store opened",
            len(self._insights),
            len(self._entities),
        )

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load knowledge from disk if the file exists."""
        if self._path is None:
            return
        p = Path(self._path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._insights = data.get("insights", [])
            self._entities = data.get("entities", [])
            self._next_id = data.get("next_id", len(self._insights) + 1)
        except (json.JSONDecodeError, OSError):
            logger.warning("failed to load knowledge file, starting fresh")

    def _flush(self) -> None:
        """Write current state to disk."""
        if self._path is None:
            return
        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "insights": self._insights,
                    "entities": self._entities,
                    "next_id": self._next_id,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(tmp, str(p))

    # ── Write operations ──────────────────────────────────────────────

    def store_insight(self, insight: Insight) -> int:
        """Store a new insight and return its ID.

        Args:
            insight: The insight to store.

        Returns:
            The auto-generated ID of the stored insight.
        """
        with self._lock:
            row = asdict(insight)
            row["id"] = self._next_id
            self._next_id += 1
            self._insights.append(row)
            self._flush()

        logger.debug("id=<%d>, topic=<%s> | insight stored", row["id"], insight.topic)
        return row["id"]

    def store_entity(self, entity: Entity) -> int:
        """Store or update a named entity.

        If an entity with the same name and type already exists, its
        mention_count is incremented instead of creating a duplicate.

        Args:
            entity: The entity to store or update.

        Returns:
            The index of the stored/updated entity.
        """
        with self._lock:
            for existing in self._entities:
                if (
                    existing["name"] == entity.name
                    and existing["entity_type"] == entity.entity_type
                ):
                    existing["mention_count"] = existing.get("mention_count", 1) + 1
                    if entity.description:
                        existing["description"] = entity.description
                    self._flush()
                    return self._entities.index(existing)

            row = asdict(entity)
            self._entities.append(row)
            self._flush()
            return len(self._entities) - 1

    # ── Read operations ───────────────────────────────────────────────

    def search_insights(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        topic: str = "",
    ) -> list[dict[str, Any]]:
        """Search insights using keyword matching.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            topic: Filter by topic (empty = all topics).

        Returns:
            List of insight dicts ordered by relevance.
        """
        keywords = set(query.lower().split())
        if not keywords:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        with self._lock:
            for ins in self._insights:
                if min_confidence > 0 and ins.get("confidence", 0) < min_confidence:
                    continue
                if topic and topic.lower() not in ins.get("topic", "").lower():
                    continue

                fact_words = set(ins.get("fact", "").lower().split())
                topic_words = set(ins.get("topic", "").lower().split())
                context_words = set(ins.get("query_context", "").lower().split())
                all_words = fact_words | topic_words | context_words

                hits = len(keywords & all_words)
                if hits > 0:
                    scored.append((hits, ins))

            # Update access counts for returned results
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for _, ins in scored[:limit]:
                ins["access_count"] = ins.get("access_count", 0) + 1
                results.append(dict(ins))
            if scored:
                self._flush()

        return results

    def get_recent_insights(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recently stored insights.

        Args:
            limit: Maximum results to return.

        Returns:
            List of insight dicts ordered by creation time (newest first).
        """
        with self._lock:
            return list(reversed(self._insights[-limit:]))

    def get_top_entities(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most frequently mentioned entities.

        Args:
            limit: Maximum results to return.

        Returns:
            List of entity dicts ordered by mention count.
        """
        with self._lock:
            return sorted(
                self._entities,
                key=lambda e: e.get("mention_count", 0),
                reverse=True,
            )[:limit]

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search entities by name.

        Args:
            query: Entity name to search for.
            limit: Maximum results.

        Returns:
            List of matching entity dicts.
        """
        q = query.lower()
        with self._lock:
            matches = [e for e in self._entities if q in e.get("name", "").lower()]
            return sorted(
                matches,
                key=lambda e: e.get("mention_count", 0),
                reverse=True,
            )[:limit]

    # ── Stats ─────────────────────────────────────────────────────────

    def count_insights(self) -> int:
        """Return total number of stored insights."""
        with self._lock:
            return len(self._insights)

    def count_entities(self) -> int:
        """Return total number of stored entities."""
        with self._lock:
            return len(self._entities)

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about accumulated knowledge."""
        with self._lock:
            insights = list(self._insights)
            entities = list(self._entities)

            # Topic counts
            topic_counts: dict[str, int] = {}
            total_conf = 0.0
            for ins in insights:
                t = ins.get("topic", "")
                if t:
                    topic_counts[t] = topic_counts.get(t, 0) + 1
                total_conf += ins.get("confidence", 0.0)

            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Most accessed
            by_access = sorted(insights, key=lambda x: x.get("access_count", 0), reverse=True)[:5]

        return {
            "total_insights": len(insights),
            "total_entities": len(entities),
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "avg_confidence": round(total_conf / len(insights), 3) if insights else 0.0,
            "most_accessed": [
                {"fact": m["fact"][:100], "access_count": m.get("access_count", 0)}
                for m in by_access
            ],
        }

    # ── Deduplication ─────────────────────────────────────────────────

    def has_similar_insight(self, fact: str, threshold: float = 0.8) -> bool:
        """Check if a substantially similar insight already exists.

        Uses word overlap ratio as a lightweight dedup check.

        Args:
            fact: The fact text to check for duplicates.
            threshold: Minimum word overlap ratio (0.0-1.0).

        Returns:
            True if a similar insight already exists.
        """
        fact_words = set(fact.lower().split())
        if not fact_words:
            return False

        with self._lock:
            # Check recent insights (last 500)
            for ins in self._insights[-500:]:
                existing_words = set(ins.get("fact", "").lower().split())
                if not existing_words:
                    continue
                overlap = len(fact_words & existing_words)
                ratio = overlap / max(len(fact_words), len(existing_words))
                if ratio >= threshold:
                    return True

        return False

    def close(self) -> None:
        """Flush and release resources."""
        with self._lock:
            self._flush()
        logger.info("knowledge store closed")


# ── Singleton ─────────────────────────────────────────────────────────

_store: KnowledgeStore | None = None
_store_lock = threading.Lock()


def get_knowledge_store(path: str | Path | None = None) -> KnowledgeStore:
    """Get or create the singleton KnowledgeStore.

    Args:
        path: Override path for the JSON file. Only used on
            first call (when the singleton is created).

    Returns:
        The global KnowledgeStore instance.
    """
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        _store = KnowledgeStore(path)
        return _store


def reset_knowledge_store() -> None:
    """Close and reset the singleton (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
            _store = None
