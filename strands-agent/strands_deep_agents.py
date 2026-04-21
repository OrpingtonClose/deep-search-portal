# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""SDK-native deep agent implementation using Strands Agent.as_tool().

Replaces the missing ``strands_deep_agents`` package with a local
implementation that uses the Strands SDK's native multi-agent pattern:
each SubAgent becomes a Strands Agent wrapped via ``Agent.as_tool()``,
and the lead agent orchestrates them as regular tool calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler

logger = logging.getLogger(__name__)


@dataclass
class SubAgent:
    """Configuration for a sub-agent in the deep agent system.

    Args:
        name: Tool name exposed to the lead agent.
        description: Tool description the lead agent sees.
        tools: MCP or native tools the sub-agent can use.
        prompt: System prompt for the sub-agent.
        model: Model instance for the sub-agent.
    """

    name: str
    description: str
    prompt: str
    model: Any
    tools: list = field(default_factory=list)


def create_deep_agent(
    instructions: str,
    model: Any,
    subagents: list[SubAgent],
    tools: list | None = None,
    callback_handler: Any = None,
) -> Agent:
    """Create a lead agent that orchestrates sub-agents via Agent.as_tool().

    Each SubAgent is instantiated as a full Strands Agent with its own
    tools and system prompt, then exposed to the lead agent as a callable
    tool using the SDK's native ``Agent.as_tool()`` method.

    Args:
        instructions: System prompt for the lead agent.
        model: Model instance for the lead agent.
        subagents: List of SubAgent configurations.
        tools: Additional tools available directly to the lead agent.
        callback_handler: Callback handler for the lead agent.

    Returns:
        Configured lead Agent with sub-agent tools.
    """
    subagent_tools = []

    for spec in subagents:
        agent = Agent(
            model=spec.model,
            system_prompt=spec.prompt,
            tools=spec.tools,
            conversation_manager=SlidingWindowConversationManager(
                window_size=15,
                should_truncate_results=True,
            ),
            callback_handler=PrintingCallbackHandler(),
        )

        tool = agent.as_tool(
            name=spec.name,
            description=spec.description,
        )
        subagent_tools.append(tool)
        logger.info(
            "subagent=<%s>, tools=<%d> | sub-agent registered as tool",
            spec.name,
            len(spec.tools),
        )

    # Lead agent gets sub-agent tools + any direct tools
    all_tools = list(subagent_tools)
    if tools:
        all_tools.extend(tools)

    lead_agent = Agent(
        model=model,
        system_prompt=instructions,
        tools=all_tools,
        conversation_manager=SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True,
        ),
        callback_handler=callback_handler or PrintingCallbackHandler(),
    )

    logger.info(
        "subagents=<%d>, total_tools=<%d> | deep agent created",
        len(subagents),
        len(all_tools),
    )
    return lead_agent
