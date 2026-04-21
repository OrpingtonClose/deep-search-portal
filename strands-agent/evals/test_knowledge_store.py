# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the persistent KnowledgeStore (JSON-backed)."""

from __future__ import annotations

import pytest

from knowledge_store import (
    Entity,
    Insight,
    KnowledgeStore,
    get_knowledge_store,
    reset_knowledge_store,
)


@pytest.fixture
def store():
    """Create an in-memory KnowledgeStore for testing."""
    s = KnowledgeStore(path=":memory:")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton is reset between tests."""
    reset_knowledge_store()
    yield
    reset_knowledge_store()


class TestKnowledgeStoreBasics:
    """Basic store operations."""

    def test_store_and_retrieve_insight(self, store: KnowledgeStore) -> None:
        """Store an insight and retrieve it by search."""
        insight = Insight(
            fact="GLP-1 receptor agonists reduce HbA1c by 1.0-1.5%",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
            source_type="academic",
            topic="GLP-1 pharmacology",
            confidence=0.9,
        )
        insight_id = store.store_insight(insight)
        assert insight_id > 0

        results = store.search_insights("GLP-1 receptor HbA1c")
        assert len(results) >= 1
        assert any("GLP-1" in r["fact"] for r in results)

    def test_store_multiple_insights(self, store: KnowledgeStore) -> None:
        """Store multiple insights and verify count."""
        for i in range(5):
            store.store_insight(Insight(
                fact=f"Research finding number {i} about topic {i}",
                topic=f"topic-{i}",
            ))
        assert store.count_insights() == 5

    def test_get_recent_insights(self, store: KnowledgeStore) -> None:
        """Recent insights returned in reverse chronological order."""
        store.store_insight(Insight(fact="First finding about alpha"))
        store.store_insight(Insight(fact="Second finding about beta"))
        store.store_insight(Insight(fact="Third finding about gamma"))

        recent = store.get_recent_insights(limit=2)
        assert len(recent) == 2
        assert "gamma" in recent[0]["fact"]
        assert "beta" in recent[1]["fact"]

    def test_empty_search_returns_empty(self, store: KnowledgeStore) -> None:
        """Searching an empty store returns no results."""
        results = store.search_insights("anything")
        assert results == []

    def test_count_on_empty_store(self, store: KnowledgeStore) -> None:
        """Counts are zero on fresh store."""
        assert store.count_insights() == 0
        assert store.count_entities() == 0


class TestEntityTracking:
    """Entity storage and deduplication."""

    def test_store_entity(self, store: KnowledgeStore) -> None:
        """Store and retrieve a named entity."""
        entity = Entity(
            name="Tirzepatide",
            entity_type="compound",
            description="GIP/GLP-1 dual agonist",
        )
        eid = store.store_entity(entity)
        assert eid >= 0
        assert store.count_entities() == 1

    def test_entity_deduplication(self, store: KnowledgeStore) -> None:
        """Same entity name+type increments mention count."""
        store.store_entity(Entity(name="FDA", entity_type="organization"))
        store.store_entity(Entity(name="FDA", entity_type="organization"))
        store.store_entity(Entity(name="FDA", entity_type="organization"))

        entities = store.get_top_entities()
        fda = [e for e in entities if e["name"] == "FDA"]
        assert len(fda) == 1
        assert fda[0]["mention_count"] == 3

    def test_search_entities(self, store: KnowledgeStore) -> None:
        """Search entities by name substring."""
        store.store_entity(Entity(name="Semaglutide", entity_type="compound"))
        store.store_entity(Entity(name="Tirzepatide", entity_type="compound"))
        store.store_entity(Entity(name="WHO", entity_type="organization"))

        results = store.search_entities("tide")
        assert len(results) == 2

    def test_different_types_not_deduplicated(self, store: KnowledgeStore) -> None:
        """Same name but different type creates separate entries."""
        store.store_entity(Entity(name="Mercury", entity_type="compound"))
        store.store_entity(Entity(name="Mercury", entity_type="organization"))
        assert store.count_entities() == 2


class TestDeduplication:
    """Insight deduplication via word overlap."""

    def test_similar_insight_detected(self, store: KnowledgeStore) -> None:
        """Near-duplicate insights are detected."""
        store.store_insight(Insight(
            fact="GLP-1 receptor agonists reduce HbA1c by 1.0-1.5 percent in trials",
        ))
        assert store.has_similar_insight(
            "GLP-1 receptor agonists reduce HbA1c by 1.0-1.5 percent in clinical trials"
        )

    def test_different_insight_not_flagged(self, store: KnowledgeStore) -> None:
        """Genuinely different insights pass the dedup check."""
        store.store_insight(Insight(
            fact="GLP-1 receptor agonists reduce HbA1c by 1.0-1.5%",
        ))
        assert not store.has_similar_insight(
            "The SEC filed charges against three major cryptocurrency exchanges"
        )


class TestKeywordSearch:
    """Keyword-based search."""

    def test_relevance_ranking(self, store: KnowledgeStore) -> None:
        """More relevant results should rank higher."""
        store.store_insight(Insight(
            fact="BPC-157 promotes angiogenesis in rat tendon healing models",
            topic="peptides",
        ))
        store.store_insight(Insight(
            fact="The S&P 500 reached an all-time high in 2024",
            topic="finance",
        ))
        store.store_insight(Insight(
            fact="BPC-157 shows gastric protection at 10mcg/kg in rodent studies",
            topic="peptides",
        ))

        results = store.search_insights("BPC-157 healing peptide")
        assert len(results) >= 2
        # BPC-157 results should come first
        assert "BPC-157" in results[0]["fact"]

    def test_search_with_topic_filter(self, store: KnowledgeStore) -> None:
        """Topic filter narrows results."""
        store.store_insight(Insight(fact="Finding about finance", topic="finance"))
        store.store_insight(Insight(fact="Finding about science", topic="science"))

        results = store.search_insights("finding", topic="finance")
        assert len(results) == 1
        assert results[0]["topic"] == "finance"

    def test_search_with_confidence_filter(self, store: KnowledgeStore) -> None:
        """Confidence filter excludes low-confidence results."""
        store.store_insight(Insight(fact="High confidence fact about proteins", confidence=0.9))
        store.store_insight(Insight(fact="Low confidence guess about proteins", confidence=0.2))

        results = store.search_insights("proteins", min_confidence=0.5)
        assert len(results) == 1
        assert results[0]["confidence"] >= 0.5

    def test_access_count_incremented_on_search(self, store: KnowledgeStore) -> None:
        """Searching increments the access_count of returned results."""
        store.store_insight(Insight(fact="Unique testable fact about quantum"))

        store.search_insights("quantum")
        results = store.search_insights("quantum")
        assert results[0]["access_count"] >= 2


class TestStats:
    """Knowledge statistics."""

    def test_stats_structure(self, store: KnowledgeStore) -> None:
        """Stats returns expected keys."""
        store.store_insight(Insight(fact="Test fact", topic="test"))
        stats = store.get_stats()

        assert "total_insights" in stats
        assert "total_entities" in stats
        assert "top_topics" in stats
        assert "avg_confidence" in stats
        assert "most_accessed" in stats
        assert stats["total_insights"] == 1

    def test_top_topics(self, store: KnowledgeStore) -> None:
        """Top topics aggregated correctly."""
        for _ in range(3):
            store.store_insight(Insight(fact="Peptide finding", topic="peptides"))
        store.store_insight(Insight(fact="Finance finding", topic="finance"))

        stats = store.get_stats()
        topics = {t["topic"]: t["count"] for t in stats["top_topics"]}
        assert topics.get("peptides") == 3
        assert topics.get("finance") == 1


class TestSingleton:
    """Singleton pattern for global store."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_knowledge_store returns the same instance."""
        s1 = get_knowledge_store(":memory:")
        s2 = get_knowledge_store()
        assert s1 is s2

    def test_reset_clears_singleton(self) -> None:
        """reset_knowledge_store allows creating a new instance."""
        s1 = get_knowledge_store(":memory:")
        reset_knowledge_store()
        s2 = get_knowledge_store(":memory:")
        assert s1 is not s2


class TestLifecycle:
    """Store open/close lifecycle."""

    def test_close_and_reopen(self, tmp_path) -> None:
        """Data persists after close and reopen."""
        db = str(tmp_path / "test.json")

        store1 = KnowledgeStore(path=db)
        store1.store_insight(Insight(fact="Persistent fact about durability"))
        store1.close()

        store2 = KnowledgeStore(path=db)
        assert store2.count_insights() == 1
        results = store2.search_insights("durability")
        assert len(results) >= 1
        store2.close()
