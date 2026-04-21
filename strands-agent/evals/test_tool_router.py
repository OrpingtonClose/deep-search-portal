# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the ToolRouterPlugin."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.domains import ACADEMIC, FORUM, GENERAL, PRACTITIONER
from plugins.tool_router import ToolRouterPlugin


class TestToolRouterPlugin:
    """Test ToolRouterPlugin query classification and guidance injection."""

    def setup_method(self) -> None:
        self.plugin = ToolRouterPlugin()

    def test_name(self) -> None:
        assert self.plugin.name == "tool-router"

    def test_extract_query_from_text_content(self) -> None:
        messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "find papers on GLP-1"

    def test_extract_query_from_string_content(self) -> None:
        messages = [
            {"role": "user", "content": "find papers on GLP-1"},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "find papers on GLP-1"

    def test_extract_query_picks_last_user_message(self) -> None:
        messages = [
            {"role": "user", "content": [{"text": "old query"}]},
            {"role": "assistant", "content": [{"text": "response"}]},
            {"role": "user", "content": [{"text": "new query"}]},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "new query"

    def test_extract_query_empty_messages(self) -> None:
        assert ToolRouterPlugin._extract_query(None) == ""
        assert ToolRouterPlugin._extract_query([]) == ""

    def test_get_recommended_tools_empty_before_classification(self) -> None:
        assert self.plugin.get_recommended_tools() == set()

    def test_get_recommended_tools_after_classification(self) -> None:
        """Simulate what route_tools does: classify and store the match."""
        from plugins.domains import classify_query

        self.plugin.last_match = classify_query("find PubMed papers on insulin")
        tools = self.plugin.get_recommended_tools()
        assert len(tools) > 0
        assert "search_pubmed" in tools or "openalex_search" in tools

    def test_route_tools_injects_guidance_before_last_user_message(self) -> None:
        """Test that route_tools inserts guidance right before the last user message."""
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1 pharmacokinetics"}]},
        ]

        self.plugin.route_tools(event)

        # Should have inserted a routing message before the user message
        assert len(event.messages) == 2
        guidance_msg = event.messages[0]
        user_msg = event.messages[1]
        assert guidance_msg["role"] == "user"
        text = guidance_msg["content"][0]["text"]
        assert "TOOL ROUTING" in text
        assert "ACADEMIC" in text.upper() or "openalex" in text.lower()
        # Original user message is last
        assert user_msg["content"][0]["text"] == "find papers on GLP-1 pharmacokinetics"

    def test_route_tools_multi_turn_inserts_before_last_user(self) -> None:
        """In multi-turn, guidance goes right before the last user message."""
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "old question"}]},
            {"role": "assistant", "content": [{"text": "old response"}]},
            {"role": "user", "content": [{"text": "find PubMed papers on insulin"}]},
        ]

        self.plugin.route_tools(event)

        # Guidance should be at index 2, last user message at index 3
        assert len(event.messages) == 4
        assert event.messages[0]["content"][0]["text"] == "old question"
        assert event.messages[1]["content"][0]["text"] == "old response"
        guidance = event.messages[2]["content"][0]["text"]
        assert "TOOL ROUTING" in guidance
        assert event.messages[3]["content"][0]["text"] == "find PubMed papers on insulin"

    def test_route_tools_sets_last_match(self) -> None:
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "MesoRx forum thread about tren"}]},
        ]

        self.plugin.route_tools(event)

        assert self.plugin.last_match is not None
        assert FORUM in self.plugin.last_match.domains or PRACTITIONER in self.plugin.last_match.domains

    def test_route_tools_no_messages_is_noop(self) -> None:
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = None

        # Should not raise
        self.plugin.route_tools(event)
        assert self.plugin.last_match is None

    def test_route_tools_general_query_still_injects(self) -> None:
        """Even general queries get routing guidance."""
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "what is the weather in London"}]},
        ]

        self.plugin.route_tools(event)
        assert self.plugin.last_match is not None
        # General domain still has guidance
        assert len(event.messages) == 2

    def test_route_tools_skips_reclassification_during_resume(self) -> None:
        """During resume, route_tools should skip to preserve last_match."""
        from unittest.mock import MagicMock
        from plugins.domains import classify_query

        # First invocation: academic query
        event1 = MagicMock()
        event1.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1 pharmacokinetics"}]},
        ]
        self.plugin.route_tools(event1)
        assert ACADEMIC in self.plugin.last_match.domains

        # Audit sets _is_resuming on router before resume fires
        self.plugin._is_resuming = True

        # Resume: last user message is the nudge (mentions "Forum", "YouTube" etc)
        event2 = MagicMock()
        event2.messages = list(event1.messages) + [
            {"role": "user", "content": [{"text": "You have NOT used specialized tools like Forum search, YouTube..."}]},
        ]
        self.plugin.route_tools(event2)

        # last_match should still be academic, NOT reclassified to forum
        assert ACADEMIC in self.plugin.last_match.domains
        # Flag consumed
        assert self.plugin._is_resuming is False

    def test_route_tools_replaces_stale_guidance_on_resume(self) -> None:
        """On resume, old guidance is stripped and fresh guidance is injected."""
        from unittest.mock import MagicMock

        # First invocation: guidance injected
        event1 = MagicMock()
        event1.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        self.plugin.route_tools(event1)
        assert len(event1.messages) == 2

        # Resume: messages contain the guidance marker from first pass
        event2 = MagicMock()
        event2.messages = list(event1.messages)
        self.plugin.route_tools(event2)
        # Old marker stripped, new one injected — still exactly 2
        assert len(event2.messages) == 2
        # last_match is refreshed (not stale)
        assert self.plugin.last_match is not None

    def test_route_tools_strips_stale_guidance_in_multi_turn(self) -> None:
        """In multi-turn, old guidance from turn 1 is replaced with
        domain-appropriate guidance for turn 2."""
        from unittest.mock import MagicMock

        # Turn 1: academic query
        event1 = MagicMock()
        event1.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1 pharmacokinetics"}]},
        ]
        self.plugin.route_tools(event1)
        assert len(event1.messages) == 2
        assert ACADEMIC in self.plugin.last_match.domains

        # Turn 2: multi-turn with history — different domain query
        event2 = MagicMock()
        event2.messages = list(event1.messages) + [
            {"role": "assistant", "content": [{"text": "Here are the papers..."}]},
            {"role": "user", "content": [{"text": "MesoRx forum thread about tren"}]},
        ]
        self.plugin.route_tools(event2)
        # Old academic guidance stripped, new guidance injected, total = 4
        # (user1, assistant, guidance, user2)
        assert len(event2.messages) == 4
        # last_match updated to the new domain
        assert FORUM in self.plugin.last_match.domains or PRACTITIONER in self.plugin.last_match.domains

    def test_route_tools_reinjects_after_messages_cleared(self) -> None:
        """When messages are cleared between requests, guidance is re-injected."""
        from unittest.mock import MagicMock

        # First query
        event1 = MagicMock()
        event1.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        self.plugin.route_tools(event1)
        assert len(event1.messages) == 2

        # Messages cleared (as main.py does between requests), same query again
        event2 = MagicMock()
        event2.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        self.plugin.route_tools(event2)
        assert len(event2.messages) == 2
        guidance = event2.messages[0]["content"][0]["text"]
        assert "TOOL ROUTING" in guidance

    def test_route_tools_injects_for_new_query_after_previous(self) -> None:
        """A genuinely new query on a reused agent gets fresh guidance."""
        from unittest.mock import MagicMock

        # First query
        event1 = MagicMock()
        event1.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        self.plugin.route_tools(event1)
        assert len(event1.messages) == 2

        # Different query, messages cleared — should inject new guidance
        event2 = MagicMock()
        event2.messages = [
            {"role": "user", "content": [{"text": "find SEC filings for Tesla"}]},
        ]
        self.plugin.route_tools(event2)
        assert len(event2.messages) == 2
        guidance = event2.messages[0]["content"][0]["text"]
        assert "TOOL ROUTING" in guidance
