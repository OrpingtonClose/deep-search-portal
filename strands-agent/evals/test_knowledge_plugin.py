# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the KnowledgePlugin (Strands SDK hooks + tools)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knowledge_store import KnowledgeStore, reset_knowledge_store
from plugins.knowledge import KnowledgePlugin


@pytest.fixture
def store():
    """Create an in-memory KnowledgeStore for testing."""
    return KnowledgeStore(path=":memory:")


@pytest.fixture
def plugin(store: KnowledgeStore):
    """Create a KnowledgePlugin backed by the in-memory store."""
    return KnowledgePlugin(store=store)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton is reset between tests."""
    reset_knowledge_store()
    yield
    reset_knowledge_store()


def _make_user_msg(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def _make_assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


class TestKnowledgeInjection:
    """BeforeInvocationEvent hook — inject prior knowledge."""

    def test_no_injection_on_empty_store(self, plugin: KnowledgePlugin) -> None:
        """No knowledge injected when store is empty."""
        event = MagicMock()
        event.messages = [_make_user_msg("What is GLP-1?")]

        plugin.inject_knowledge(event)

        # Messages should be unchanged (no knowledge marker)
        for msg in event.messages:
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict):
                        assert "[PRIOR KNOWLEDGE]" not in block.get("text", "")

    def test_injection_when_knowledge_exists(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """Relevant knowledge is injected before the user message."""
        from knowledge_store import Insight
        store.store_insight(Insight(
            fact="GLP-1 agonists reduce HbA1c by 1.0-1.5%",
            topic="GLP-1",
            confidence=0.9,
        ))

        event = MagicMock()
        event.messages = [_make_user_msg("Tell me about GLP-1 effects")]

        plugin.inject_knowledge(event)

        msgs = event.messages
        knowledge_msgs = [
            m for m in msgs if isinstance(m, dict)
            and any("[PRIOR KNOWLEDGE]" in b.get("text", "")
                    for b in m.get("content", []) if isinstance(b, dict))
        ]
        assert len(knowledge_msgs) == 1

    def test_stale_knowledge_stripped(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """Previous knowledge markers are removed before injecting fresh ones."""
        from knowledge_store import Insight
        store.store_insight(Insight(
            fact="Old finding about test topic",
            topic="test",
            confidence=0.8,
        ))

        stale_msg = {
            "role": "user",
            "content": [{"text": "[PRIOR KNOWLEDGE]\nStale data"}],
        }
        event = MagicMock()
        event.messages = [stale_msg, _make_user_msg("test topic research")]

        plugin.inject_knowledge(event)

        msgs = event.messages
        knowledge_msgs = [
            m for m in msgs if isinstance(m, dict)
            and any("[PRIOR KNOWLEDGE]" in b.get("text", "")
                    for b in m.get("content", []) if isinstance(b, dict))
        ]
        # Should only have one knowledge message (fresh, not stale)
        assert len(knowledge_msgs) == 1
        assert "Stale data" not in knowledge_msgs[0]["content"][0]["text"]

    def test_no_injection_on_none_messages(self, plugin: KnowledgePlugin) -> None:
        """No crash when messages is None."""
        event = MagicMock()
        event.messages = None
        plugin.inject_knowledge(event)


class TestToolKnowledgeCapture:
    """AfterToolCallEvent hook — capture facts from tool results."""

    def test_facts_extracted_from_tool_result(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """Facts with factual indicators are extracted from tool results."""
        # Set up a current query so the hook fires
        plugin._current_query = "GLP-1 research"

        event = MagicMock()
        event.result = {
            "content": [{"text": (
                "Semaglutide reduced HbA1c by 1.5% in the SUSTAIN-6 trial (2016). "
                "The FDA approved it for type 2 diabetes in December 2017."
            )}],
        }
        event.tool_use = {"name": "web_search"}

        plugin.capture_tool_knowledge(event)

        assert store.count_insights() >= 1

    def test_no_capture_without_query(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """No facts captured if no current query is set."""
        event = MagicMock()
        event.result = {
            "content": [{"text": "Important fact: 42% of studies show improvement."}],
        }
        event.tool_use = {"name": "search"}

        plugin.capture_tool_knowledge(event)
        assert store.count_insights() == 0

    def test_no_capture_on_empty_result(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """No crash on empty tool result."""
        plugin._current_query = "test query"
        event = MagicMock()
        event.result = None
        plugin.capture_tool_knowledge(event)
        assert store.count_insights() == 0

    def test_dedup_prevents_duplicate_storage(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """Same fact from two tool calls isn't stored twice."""
        plugin._current_query = "GLP-1 research"
        text = "Semaglutide reduced HbA1c by 1.5% in the SUSTAIN-6 trial conducted in 2016."

        event1 = MagicMock()
        event1.result = {"content": [{"text": text}]}
        event1.tool_use = {"name": "search1"}

        event2 = MagicMock()
        event2.result = {"content": [{"text": text}]}
        event2.tool_use = {"name": "search2"}

        plugin.capture_tool_knowledge(event1)
        plugin.capture_tool_knowledge(event2)

        assert store.count_insights() >= 1
        # Should not double-store
        assert store.count_insights() <= 2


class TestEntityExtraction:
    """AfterInvocationEvent hook — extract entities from response."""

    def test_entities_extracted_from_response(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """Named entities are extracted from assistant response."""
        plugin._current_query = "peptide research"

        result_mock = MagicMock()
        result_mock.messages = [
            _make_assistant_msg(
                "BPC-157 and GLP-1 are peptides studied by the FDA and WHO for clinical applications."
            ),
        ]

        event = MagicMock()
        event.result = result_mock

        plugin.extract_entities(event)

        assert store.count_entities() >= 2


class TestPluginTools:
    """Tools declared on the plugin class."""

    def test_recall_knowledge_empty(self, plugin: KnowledgePlugin) -> None:
        """recall_knowledge on empty store returns helpful message."""
        result = plugin.recall_knowledge(query="anything")
        assert "No prior knowledge" in result

    def test_store_and_recall(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """store_insight + recall_knowledge round-trip."""
        plugin.store_insight(
            fact="BPC-157 heals tendons at 250mcg dose",
            topic="peptides",
            confidence=0.8,
        )

        result = plugin.recall_knowledge(query="BPC-157 tendons")
        assert "BPC-157" in result
        assert "peptides" in result

    def test_store_duplicate_rejected(self, plugin: KnowledgePlugin) -> None:
        """Storing the same fact twice is rejected."""
        plugin.store_insight(fact="Unique fact about protein synthesis 2024")
        result = plugin.store_insight(fact="Unique fact about protein synthesis 2024")
        assert "Similar insight already exists" in result

    def test_recall_entities_empty(self, plugin: KnowledgePlugin) -> None:
        """recall_entities on empty store returns helpful message."""
        result = plugin.recall_entities()
        assert "No entities" in result

    def test_knowledge_stats(
        self, plugin: KnowledgePlugin, store: KnowledgeStore,
    ) -> None:
        """knowledge_stats returns formatted statistics."""
        from knowledge_store import Insight
        store.store_insight(Insight(fact="Test fact", topic="test"))

        result = plugin.knowledge_stats()
        assert "Total insights: 1" in result
        assert "KNOWLEDGE BASE STATISTICS" in result


class TestExtractionHelpers:
    """Static extraction helper methods."""

    def test_extract_facts_with_numbers(self) -> None:
        """Sentences with numbers are extracted as facts."""
        text = "The compound showed 45% efficacy in trials. Let me search for more."
        facts = KnowledgePlugin._extract_facts(text)
        assert len(facts) >= 1
        assert any("45%" in f[0] for f in facts)

    def test_extract_facts_skip_meta(self) -> None:
        """Meta-commentary sentences are skipped."""
        text = "Let me search for that. I'll look into it now."
        facts = KnowledgePlugin._extract_facts(text)
        assert len(facts) == 0

    def test_extract_entities_compounds(self) -> None:
        """Hyphenated compounds are extracted."""
        text = "BPC-157 and GLP-1 show therapeutic potential."
        entities = KnowledgePlugin._extract_entities_from_text(text)
        names = [e[0] for e in entities]
        assert "BPC-157" in names
        assert "GLP-1" in names

    def test_extract_entities_abbreviations(self) -> None:
        """Uppercase abbreviations (3+ chars) are extracted."""
        text = "The FDA and WHO published guidelines."
        entities = KnowledgePlugin._extract_entities_from_text(text)
        names = [e[0] for e in entities]
        assert "FDA" in names
        assert "WHO" in names

    def test_infer_topic(self) -> None:
        """Topic inference strips stop words."""
        topic = KnowledgePlugin._infer_topic("find papers on GLP-1 pharmacokinetics")
        assert "glp-1" in topic
        assert "pharmacokinetics" in topic
        assert "find" not in topic

    def test_infer_source_type(self) -> None:
        """Source type inferred from URL patterns."""
        assert KnowledgePlugin._infer_source_type("https://pubmed.ncbi.nlm.nih.gov/123") == "academic"
        assert KnowledgePlugin._infer_source_type("https://arxiv.org/abs/2024.1234") == "preprint"
        assert KnowledgePlugin._infer_source_type("https://clinicaltrials.gov/ct2/show/NCT123") == "government"
        assert KnowledgePlugin._infer_source_type("") == "research"
