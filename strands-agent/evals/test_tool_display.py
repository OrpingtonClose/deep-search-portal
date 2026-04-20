# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for tool display formatting (ToolDisplayPlugin).

These tests verify that tool labels, sanitization, and footer formatting
produce user-friendly output without requiring a live agent or API keys.
"""

from __future__ import annotations

import pytest

from plugins.tool_display import format_footer, sanitize_for_italic, tool_label


# ---------------------------------------------------------------------------
# Tool label generation
# ---------------------------------------------------------------------------

class TestToolLabel:
    """Verify tool_label produces human-friendly descriptions."""

    def test_task_with_description(self) -> None:
        label = tool_label("task", "{'description': 'Research Tor protocols'}")
        assert "Researching" in label
        assert "Tor protocols" in label

    def test_task_without_input(self) -> None:
        label = tool_label("task", "{}")
        assert label == "Researching"

    def test_write_todos(self) -> None:
        label = tool_label("write_todos", "{'items': ['step 1']}")
        assert label == "Planning next steps"

    def test_brave_search_with_query(self) -> None:
        label = tool_label("brave_web_search", "{'query': 'mesh networking'}")
        assert "Searching" in label
        assert "mesh networking" in label

    def test_firecrawl_scrape_with_url(self) -> None:
        label = tool_label("firecrawl_scrape", "{'url': 'https://example.com/doc'}")
        assert "Reading" in label

    def test_unknown_tool_fallback(self) -> None:
        label = tool_label("my_custom_tool", "{}")
        assert "My custom tool" in label

    def test_truncation_at_50_chars(self) -> None:
        long_desc = "a" * 100
        label = tool_label("task", f"{{'description': '{long_desc}'}}")
        assert len(label) < 70  # verb + truncated subject

    def test_firecrawl_search(self) -> None:
        label = tool_label("firecrawl_search", "{'query': 'I2P vs Tor'}")
        assert "Searching" in label


# ---------------------------------------------------------------------------
# Italic sanitization
# ---------------------------------------------------------------------------

class TestSanitizeForItalic:
    """Verify sanitize_for_italic produces markdown-safe text."""

    def test_removes_asterisks(self) -> None:
        assert "*" not in sanitize_for_italic("hello *world* test")

    def test_collapses_newlines(self) -> None:
        result = sanitize_for_italic("line1\n\nline2\nline3")
        assert "\n" not in result
        assert "line1 line2 line3" == result

    def test_collapses_whitespace(self) -> None:
        result = sanitize_for_italic("hello    world")
        assert result == "hello world"

    def test_strips_leading_trailing(self) -> None:
        result = sanitize_for_italic("  hello  ")
        assert result == "hello"

    def test_empty_string(self) -> None:
        assert sanitize_for_italic("") == ""


# ---------------------------------------------------------------------------
# Footer formatting
# ---------------------------------------------------------------------------

class TestFormatFooter:
    """Verify format_footer produces clean one-line summaries."""

    def test_zero_tools_short_elapsed(self) -> None:
        result = format_footer([], 0.5)
        assert result == ""

    def test_zero_tools_long_elapsed(self) -> None:
        result = format_footer([], 5.0)
        assert "Completed in 5s" in result

    def test_single_tool(self) -> None:
        events = [{"tool": "brave_web_search"}]
        result = format_footer(events, 10.0)
        assert "1 source" in result
        assert "10s" in result

    def test_multiple_tools(self) -> None:
        events = [
            {"tool": "brave_web_search"},
            {"tool": "firecrawl_scrape"},
            {"tool": "exa_search"},
        ]
        result = format_footer(events, 30.0)
        assert "3 sources" in result
        assert "30s" in result

    def test_duplicate_tools_deduplicated(self) -> None:
        events = [
            {"tool": "brave_web_search"},
            {"tool": "brave_web_search"},
        ]
        result = format_footer(events, 10.0)
        assert "1 source" in result

    def test_empty_tool_name_excluded(self) -> None:
        events = [{"tool": ""}, {"tool": "brave_web_search"}]
        result = format_footer(events, 10.0)
        assert "1 source" in result
