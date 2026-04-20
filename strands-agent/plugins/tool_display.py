# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native tool display plugin.

Replaces the scattered ``_tool_label()``, ``_sanitize_for_italic()``, and
``_format_inline_log()`` functions from ``main.py`` with a Strands SDK
Plugin.  Centralises all user-facing formatting logic for tool calls and
the response footer.
"""

from __future__ import annotations

import re
from typing import Any

from strands.plugins import Plugin


# ── Verb map for well-known tools ─────────────────────────────────────

_VERB_MAP: dict[str, str] = {
    "task": "Researching",
    "brave_web_search": "Searching",
    "exa_search": "Searching",
    "duckduckgo_search": "Searching",
    "firecrawl_search": "Searching",
    "firecrawl_scrape": "Reading",
    "read_url": "Reading",
    "http_request": "Fetching",
}

# Tools whose label is a fixed phrase — no subject extraction needed.
_FIXED_LABELS: dict[str, str] = {
    "write_todos": "Planning next steps",
}


def sanitize_for_italic(text: str) -> str:
    """Collapse newlines/whitespace so text can be wrapped in ``*...*`` italic markdown.

    Markdown emphasis spans cannot cross blank lines.  Literal ``*``
    characters inside the text also break surrounding italic markers.
    This helper removes ``*``, replaces all newlines with spaces, and
    collapses consecutive whitespace into a single space.

    Args:
        text: Raw text to sanitize.

    Returns:
        Sanitized text safe for italic markdown wrapping.
    """
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def tool_label(tool_name: str, tool_input: str) -> str:
    """Build a short, human-friendly label for a tool call.

    Instead of ``Using task``, produces something like ``Researching Tor
    protocols`` by extracting the first meaningful phrase from the tool
    input.  Keeps the label under ~60 chars so it stays on one line.

    Args:
        tool_name: The SDK tool name (e.g. ``brave_web_search``).
        tool_input: String representation of the tool input dict.

    Returns:
        Human-friendly label like ``Searching Tor protocols``.
    """
    # Fixed labels return as-is (no subject extraction)
    fixed = _FIXED_LABELS.get(tool_name)
    if fixed:
        return fixed

    verb = _VERB_MAP.get(tool_name)
    if verb:
        subject = _extract_subject(tool_input)
        if subject:
            if len(subject) > 50:
                subject = subject[:47] + "..."
            safe = sanitize_for_italic(subject)
            return f"{verb} {safe}"
        return verb

    # Fallback: capitalise the tool name
    display = tool_name.replace("_", " ").capitalize()
    return sanitize_for_italic(display)


def _extract_subject(tool_input: str) -> str:
    """Extract a brief subject phrase from tool input for display.

    Tries dict-style key extraction first, then falls back to cleaning
    the raw input string.

    Args:
        tool_input: String representation of the tool input dict.

    Returns:
        Short subject string, or empty string if nothing useful found.
    """
    # Try dict-style 'key': 'value' — grab first string value
    m = re.search(
        r"'(?:description|query|url|topic|search_query)':\s*'([^']{3,})'",
        tool_input,
    )
    if not m:
        m = re.search(
            r'"(?:description|query|url|topic|search_query)":\s*"([^"]{3,})"',
            tool_input,
        )
    if m:
        return m.group(1).strip()

    # Fallback: grab first chunk of alphanumeric text
    cleaned = re.sub(r"[{}'\"\[\]]", " ", tool_input)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 5:
        return cleaned

    return ""


def format_footer(tool_events: list[dict[str, Any]], elapsed: float) -> str:
    """Format a user-friendly footer summarising the agent's work.

    Returns a clean one-liner like ``*Researched using 3 sources in 12s*``
    instead of the raw tool list or YAML metrics dump.

    Args:
        tool_events: List of tool event dicts from StreamCapturePlugin.
        elapsed: Total wall-clock seconds for the request.

    Returns:
        Markdown footer string, or empty string for zero-tool queries
        with fast responses.
    """
    unique_tools = {e.get("tool", "") for e in tool_events}
    unique_tools.discard("")
    n_sources = len(unique_tools)

    if n_sources == 0:
        if elapsed > 1.0:
            return f"\n\n---\n*Completed in {elapsed:.0f}s*\n"
        return ""

    time_str = f"{elapsed:.0f}s" if elapsed >= 1 else "<1s"
    if n_sources == 1:
        return f"\n\n---\n*Researched using 1 source in {time_str}*\n"
    return f"\n\n---\n*Researched using {n_sources} sources in {time_str}*\n"


class ToolDisplayPlugin(Plugin):
    """Centralises tool-call display formatting and response footers.

    Provides static methods for tool label generation, italic sanitization,
    and footer formatting.  These are called by the SSE streaming handler
    in ``main.py`` rather than being scattered as module-level functions.

    The plugin does not use ``@hook`` decorators because the formatting
    happens in the SSE presentation layer (after tool events are captured
    by ``StreamCapturePlugin``).  Instead, it provides a clean API that
    ``main.py`` calls when rendering tool events and footers.
    """

    name: str = "tool-display"

    @staticmethod
    def label(tool_name: str, tool_input: str) -> str:
        """Build a human-friendly tool label.

        Args:
            tool_name: The SDK tool name.
            tool_input: String representation of the tool input.

        Returns:
            Human-friendly label.
        """
        return tool_label(tool_name, tool_input)

    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize text for italic markdown wrapping.

        Args:
            text: Raw text to sanitize.

        Returns:
            Sanitized text.
        """
        return sanitize_for_italic(text)

    @staticmethod
    def footer(tool_events: list[dict[str, Any]], elapsed: float) -> str:
        """Format the response footer.

        Args:
            tool_events: Tool event dicts.
            elapsed: Request duration in seconds.

        Returns:
            Markdown footer string.
        """
        return format_footer(tool_events, elapsed)
