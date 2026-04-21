# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""ToolAuditPlugin — post-invocation tool usage verification.

Hooks into AfterToolCallEvent and AfterInvocationEvent to:
1. Track which tools were actually called during the invocation
2. Compare against what *should* have been called (from ToolRouterPlugin)
3. If critical tools were skipped, set event.resume to re-invoke the
   agent with a message listing the missed tools

This is the safety net that catches cases where the agent ignored the
routing guidance and stuck with DuckDuckGo for an academic query.

The plugin uses a configurable threshold: if fewer than N% of the
recommended tool *categories* were used, it triggers a resume.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands.hooks.events import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
)
from strands.plugins import Plugin, hook

from plugins.domains import DOMAIN_TOOLS, classify_query

if TYPE_CHECKING:
    from plugins.tool_router import ToolRouterPlugin

logger = logging.getLogger(__name__)

# Default: require at least 1 tool from the recommended set to be used.
# If zero recommended tools were used, trigger a resume.
_MIN_RECOMMENDED_RATIO = float(0.0)

# Maximum number of resume attempts to prevent infinite loops
_MAX_RESUMES = 1


class ToolAuditPlugin(Plugin):
    """Track tool usage and nudge the agent toward underused tools.

    Works in tandem with ToolRouterPlugin: the router classifies the
    query and recommends tools; the auditor checks if those tools were
    actually used and re-invokes the agent if they weren't.
    """

    name: str = "tool-audit"

    def __init__(
        self,
        router: "ToolRouterPlugin | None" = None,
        max_resumes: int = _MAX_RESUMES,
    ) -> None:
        """Initialize the audit plugin.

        Args:
            router: Reference to the ToolRouterPlugin for getting
                recommended tools. If None, the plugin classifies
                queries independently.
            max_resumes: Maximum resume attempts per invocation.
        """
        super().__init__()
        self._router = router
        self._max_resumes = max_resumes
        self._tools_called: set[str] = set()
        self._resume_count: int = 0
        self._current_query: str = ""
        self._is_resuming: bool = False

    def reset(self) -> None:
        """Reset per-request state."""
        self._tools_called.clear()
        self._resume_count = 0
        self._current_query = ""
        self._is_resuming = False

    @hook
    def track_invocation_start(self, event: BeforeInvocationEvent) -> None:
        """Reset tracking at the start of each invocation."""
        if self._is_resuming:
            # Resume cycle — preserve _tools_called and _resume_count
            self._is_resuming = False
            return

        # Fresh request — reset everything
        self._tools_called.clear()
        self._resume_count = 0
        self._current_query = self._extract_query(event.messages)

    @hook
    def track_tool_call(self, event: AfterToolCallEvent) -> None:
        """Record each tool that was called."""
        tool_name = event.tool_use.get("name", "")
        if tool_name:
            self._tools_called.add(tool_name)

    @hook
    def audit_tool_usage(self, event: AfterInvocationEvent) -> None:
        """Check tool usage against recommendations and resume if needed."""
        if self._resume_count >= self._max_resumes:
            logger.debug(
                "resume_count=<%d>, max=<%d> | skipping audit, max resumes reached",
                self._resume_count,
                self._max_resumes,
            )
            return

        # Get recommended tools
        recommended = self._get_recommended_tools()
        if not recommended:
            logger.debug("no recommended tools for this query, skipping audit")
            return

        # Check overlap
        used_recommended = self._tools_called & recommended
        missed = recommended - self._tools_called

        logger.info(
            "tools_called=<%d>, recommended=<%d>, used_recommended=<%d>, missed=<%d> | audit result",
            len(self._tools_called),
            len(recommended),
            len(used_recommended),
            len(missed),
        )

        # Only trigger resume if ZERO recommended tools were used
        # (the agent completely ignored the routing guidance)
        if used_recommended:
            logger.debug("agent used %d recommended tools, no resume needed", len(used_recommended))
            return

        if not self._tools_called:
            logger.debug("no tools called at all, skipping resume")
            return

        # Build a targeted nudge message
        # Group missed tools by category for readability
        nudge = self._build_nudge_message(missed)
        if not nudge:
            return

        logger.info(
            "resume_count=<%d> | triggering resume with tool nudge",
            self._resume_count,
        )
        self._resume_count += 1
        self._is_resuming = True
        # Signal the router to skip reclassification on the resume cycle
        # so it doesn't overwrite last_match with nudge-text classification.
        if self._router is not None:
            self._router._is_resuming = True
        event.resume = nudge

    def _get_recommended_tools(self) -> set[str]:
        """Get the set of recommended tools for the current query."""
        if self._router is not None:
            return self._router.get_recommended_tools()

        # Fall back to independent classification
        if not self._current_query:
            return set()

        match = classify_query(self._current_query)
        tools: set[str] = set()
        for domain in match.domains:
            domain_tools = DOMAIN_TOOLS.get(domain, [])
            tools.update(domain_tools)
        return tools

    def _build_nudge_message(self, missed_tools: set[str]) -> str:
        """Build a concise message listing missed tool categories.

        Groups tools by functional area rather than listing all 30+
        individually, which would overwhelm the context.

        Args:
            missed_tools: Set of tool names that should have been used.

        Returns:
            Nudge message string, or empty string if no actionable nudge.
        """
        # Group by prefix/category
        categories: dict[str, list[str]] = {}
        for tool_name in sorted(missed_tools):
            # Extract category from tool name prefix
            if tool_name.startswith(("openalex_", "ss_", "semantic_scholar_")):
                cat = "Academic databases (OpenAlex, Semantic Scholar)"
            elif tool_name.startswith(("arxiv_", "search_google_scholar")):
                cat = "Academic literature (arXiv, Google Scholar)"
            elif tool_name.startswith(("search_pubmed", "pubmed_")):
                cat = "PubMed"
            elif tool_name.startswith("forum_"):
                cat = "Forum search"
            elif tool_name.startswith(("youtube_", "search_youtube", "search_channel", "get_channel", "list_channel")):
                cat = "YouTube/video"
            elif tool_name.startswith("reddit_"):
                cat = "Reddit"
            elif tool_name.startswith(("search_clinical", "get_trial", "search_fda", "search_court")):
                cat = "Government databases"
            elif tool_name.startswith(("search_sec", "search_offshore")):
                cat = "SEC filings & corporate intelligence"
            elif tool_name.startswith(("search_biorxiv", "search_chemrxiv", "search_ssrn", "search_osf")):
                cat = "Preprint servers"
            elif tool_name.startswith(("wayback_", "archive_", "ipfs_", "search_common")):
                cat = "Web archives & OSINT"
            elif tool_name.startswith(("check_retraction", "batch_check", "search_retraction")):
                cat = "Research integrity"
            elif tool_name.startswith(("download_paper", "search_open_access", "resolve_doi", "search_core", "search_springer", "search_zenodo")):
                cat = "Document acquisition"
            elif tool_name.startswith("wikidata_"):
                cat = "Entity disambiguation (Wikidata)"
            else:
                cat = "Other specialized tools"

            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool_name)

        if not categories:
            return ""

        # Pick the top 3 most important missed categories
        # (prioritize by number of missed tools — more tools = bigger gap)
        top_cats = sorted(categories.keys(), key=lambda c: len(categories[c]), reverse=True)[:3]

        lines = [
            "You have NOT used any of the specialized tools recommended for this query. "
            "Before finalizing your answer, search these sources:"
        ]
        for cat in top_cats:
            tools = categories[cat][:3]  # Show at most 3 tool names per category
            lines.append(f"- {cat}: {', '.join(tools)}")

        lines.append(
            "\nUse at least one tool from each category above, then incorporate "
            "the findings into your answer."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_query(messages: list[Any] | None) -> str:
        """Extract the user's query text from the messages list."""
        if not messages:
            return ""
        for msg in reversed(messages):
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            if role != "user":
                continue
            content = msg.get("content", [])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                    elif isinstance(block, str):
                        texts.append(block)
                return " ".join(texts)
        return ""
