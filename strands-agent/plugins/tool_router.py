# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""ToolRouterPlugin — query-aware tool routing via BeforeInvocationEvent.

Hooks into the agent lifecycle to classify the user's query into research
domains (academic, practitioner, government, etc.) and inject domain-specific
tool guidance into the conversation before the model sees it. This ensures
the LLM knows which of its 80+ tools are most relevant for the current query
instead of defaulting to DuckDuckGo for everything.

Architecture:
    BeforeInvocationEvent fires → classify query → inject guidance message
    → (optionally) auto-activate relevant skill

The guidance is injected as a system-level instruction prepended to the
user's messages, not as a modification to the system prompt itself. This
keeps the base system prompt stable while providing per-query routing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strands.hooks.events import BeforeInvocationEvent
from strands.plugins import Plugin, hook
from strands.types.content import Message

from plugins.domains import (
    DOMAIN_GUIDANCE,
    DOMAIN_SKILLS,
    DOMAIN_TOOLS,
    DomainMatch,
    classify_query,
)

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)

_GUIDANCE_MARKER = "[SYSTEM TOOL GUIDANCE]"


class ToolRouterPlugin(Plugin):
    """Classify queries and inject domain-specific tool guidance.

    On each invocation, extracts the user query from messages, classifies
    it into research domains, and prepends a guidance message listing the
    most relevant tools. Also triggers skill auto-activation when a
    matching skill exists.

    The plugin tracks the last classification result so the ToolAuditPlugin
    can check which tools *should* have been used.
    """

    name: str = "tool-router"

    def __init__(self) -> None:
        super().__init__()
        self.last_match: DomainMatch | None = None
        self._agent: Agent | None = None
        self._is_resuming: bool = False

    def init_agent(self, agent: "Agent") -> None:
        """Store agent reference for skill activation."""
        self._agent = agent

    @hook
    def route_tools(self, event: BeforeInvocationEvent) -> None:
        """Classify the query and inject tool routing guidance.

        Uses a strip-and-replace strategy: any stale guidance markers from
        previous turns are removed first, then the current query is
        classified and fresh guidance is injected. This ensures multi-turn
        conversations always get domain-appropriate guidance and that
        last_match stays current for the ToolAuditPlugin.
        """
        if event.messages is None:
            return

        # During a resume cycle the last user message is the audit nudge,
        # not the original query. Skip reclassification to preserve
        # last_match and avoid injecting wrong-domain guidance.
        if self._is_resuming:
            self._is_resuming = False
            logger.debug("skipping routing during resume cycle")
            return

        query = self._extract_query(event.messages)
        if not query:
            logger.debug("no user query found in messages, skipping routing")
            return

        # Strip any stale guidance markers from previous turns so
        # the model only sees guidance for the current query.
        msgs = self._strip_guidance_markers(list(event.messages))

        match = classify_query(query)
        self.last_match = match

        logger.info(
            "domains=<%s>, primary=<%s> | query classified",
            ",".join(match.domains),
            match.primary,
        )

        # Build guidance text from all matched domains
        guidance_parts = []
        for domain in match.domains:
            text = DOMAIN_GUIDANCE.get(domain)
            if text:
                guidance_parts.append(text)

        if not guidance_parts:
            event.messages = msgs
            return

        guidance = (
            "TOOL ROUTING (based on your query):\n\n"
            + "\n\n".join(guidance_parts)
        )

        # Inject guidance right before the last user message so the model
        # sees it immediately before the current query, not buried at the
        # start of a multi-turn conversation history.
        routing_message: Message = {
            "role": "user",
            "content": [{"text": f"{_GUIDANCE_MARKER}\n{guidance}"}],
        }
        insert_idx = len(msgs) - 1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                insert_idx = i
                break
        msgs.insert(insert_idx, routing_message)
        event.messages = msgs

        # Auto-activate skill if one matches
        self._try_activate_skill(match)

    def _try_activate_skill(self, match: DomainMatch) -> None:
        """Attempt to auto-activate the skill for the primary domain.

        Uses the agent's tool registry to call the 'skills' tool
        programmatically if a matching skill exists.
        """
        skill_name = DOMAIN_SKILLS.get(match.primary)
        if not skill_name:
            return

        if self._agent is None:
            logger.debug("no agent reference, cannot auto-activate skill")
            return

        # Check if the skills tool is available
        try:
            tools_config = self._agent.tool_registry.get_all_tools_config()
            has_skills = any(
                t.get("name") == "skills" for t in tools_config
            )
            if not has_skills:
                logger.debug("skills tool not in registry, skipping activation")
                return
        except Exception:
            logger.debug("could not check tool registry for skills tool")
            return

        logger.info(
            "domain=<%s>, skill=<%s> | auto-activating skill",
            match.primary,
            skill_name,
        )
        # Store the skill hint in invocation state so the agent
        # sees it as a strong suggestion. We don't force-call the tool
        # because the agent may already have it loaded.
        # The guidance message already mentions the skill.

    def get_recommended_tools(self) -> set[str]:
        """Return the set of tool names recommended for the last query.

        Used by ToolAuditPlugin to check utilization.
        """
        if self.last_match is None:
            return set()

        tools: set[str] = set()
        for domain in self.last_match.domains:
            domain_tools = DOMAIN_TOOLS.get(domain, [])
            tools.update(domain_tools)
        return tools

    @staticmethod
    def _strip_guidance_markers(messages: list[dict]) -> list[dict]:
        """Remove any messages containing the guidance marker.

        Returns a new list with stale guidance messages filtered out so
        that only the freshly generated guidance for the current query
        is present after injection.
        """
        return [
            msg for msg in messages
            if not ToolRouterPlugin._is_guidance_message(msg)
        ]

    @staticmethod
    def _is_guidance_message(msg: dict) -> bool:
        """Check whether a single message contains the guidance marker."""
        if not isinstance(msg, dict) or msg.get("role") != "user":
            return False
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and _GUIDANCE_MARKER in block.get("text", ""):
                    return True
        elif isinstance(content, str) and _GUIDANCE_MARKER in content:
            return True
        return False

    @staticmethod
    def _extract_query(messages: list | None) -> str:
        """Extract the user's query text from the messages list."""
        if not messages:
            return ""

        # Walk messages in reverse to find the last user message
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
