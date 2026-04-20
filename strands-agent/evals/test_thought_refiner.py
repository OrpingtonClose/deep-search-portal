# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for thought refinement (ThoughtRefinerPlugin).

Unit-level tests verify plugin configuration, availability checks, and
short-input passthrough.  Integration tests (marked with ``integ``)
require VENICE_API_KEY and call the real Venice API.
"""

from __future__ import annotations

import pytest

from plugins.thought_refiner import ThoughtRefinerPlugin


# ---------------------------------------------------------------------------
# Unit tests (no API key required)
# ---------------------------------------------------------------------------


class TestThoughtRefinerConfig:
    """Verify plugin configuration and availability."""

    def test_disabled_plugin_returns_truncated(self) -> None:
        plugin = ThoughtRefinerPlugin(enabled=False)
        assert not plugin.enabled
        result = plugin.refine_sync("A" * 1000)
        assert len(result) <= 600  # 500 + truncation marker

    def test_short_input_passthrough(self) -> None:
        plugin = ThoughtRefinerPlugin(enabled=True)
        short = "Quick thought"
        result = plugin.refine_sync(short)
        assert result == short

    def test_is_available_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "plugins.thought_refiner.REFINER_API_KEY", ""
        )
        plugin = ThoughtRefinerPlugin(enabled=True)
        assert not plugin.is_available

    def test_is_available_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "plugins.thought_refiner.REFINER_API_KEY", "test-key"
        )
        plugin = ThoughtRefinerPlugin(enabled=True)
        assert plugin.is_available

    def test_name_property(self) -> None:
        plugin = ThoughtRefinerPlugin()
        assert plugin.name == "thought-refiner"


class TestThoughtRefinerSync:
    """Verify sync refinement edge cases."""

    def test_empty_input(self) -> None:
        plugin = ThoughtRefinerPlugin(enabled=True)
        result = plugin.refine_sync("")
        assert result == ""

    def test_whitespace_only_input(self) -> None:
        plugin = ThoughtRefinerPlugin(enabled=True)
        result = plugin.refine_sync("   \n\n  ")
        assert result == "   \n\n  "  # Under MIN_THINKING_LENGTH

    def test_no_api_key_returns_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "plugins.thought_refiner.REFINER_API_KEY", ""
        )
        plugin = ThoughtRefinerPlugin(enabled=True)
        long_input = "Thinking about " * 100  # Well over MIN_THINKING_LENGTH
        result = plugin.refine_sync(long_input)
        assert len(result) <= 600  # Truncated fallback


# ---------------------------------------------------------------------------
# Integration tests (require VENICE_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integ
class TestThoughtRefinerInteg:
    """Integration tests that call the real Venice API."""

    def test_refines_long_thinking(self, venice_api_key: str) -> None:
        plugin = ThoughtRefinerPlugin(enabled=True)
        raw = (
            "I need to think about this carefully. First, let me consider the "
            "user's question about Tor protocols. I should search for information "
            "about onion routing, exit nodes, and bridge relays. Actually, wait, "
            "maybe I should also look into I2P as an alternative. Let me reconsider "
            "my approach. The user seems interested in censorship circumvention "
            "specifically, so I should focus on that angle. Let me also think about "
            "Nym mixnet and its advantages over Tor. This is a complex topic with "
            "many facets. I'll need to be thorough but concise in my response."
        )
        refined = plugin.refine_sync(raw, timeout=30.0)
        assert refined != raw
        assert len(refined) < len(raw)
        assert len(refined) > 20  # Not empty

    @pytest.mark.asyncio
    async def test_refines_async(self, venice_api_key: str) -> None:
        plugin = ThoughtRefinerPlugin(enabled=True)
        raw = (
            "Let me analyze this step by step. The question asks about internet "
            "censorship techniques. I should look at DNS-based blocking, deep packet "
            "inspection, IP blacklisting, and more sophisticated methods like SNI "
            "filtering. I need to consider both the technical and political dimensions."
        )
        refined = await plugin.refine_async(raw, timeout=30.0)
        assert refined != raw
        assert len(refined) < len(raw)
