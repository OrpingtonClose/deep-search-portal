# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Venice GLM-4.7 uncensored research agent — Strands Agents SDK.

Features used:
- OpenAI-compatible model provider (Venice AI)
- MCP tool integration (Brave, Firecrawl, Exa, Kagi)
- SDK-native plugins (BudgetPlugin, StreamCapturePlugin, ThoughtRefinerPlugin, ToolDisplayPlugin,
  KnowledgePlugin, ToolRouterPlugin, ToolAuditPlugin)
- Streaming responses (PrintingCallbackHandler + StreamCapturePlugin)
- Conversation memory (SlidingWindowConversationManager)
- Agent loop with automatic tool dispatch
- Multi-agent orchestration (planner + researcher via agent-as-tool)
- Adaptive loop prevention (Plugin + @hook — temperature escalation)
- OpenTelemetry observability (OTEL_EXPORTER_OTLP_ENDPOINT)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from strands import Agent, Plugin
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)

from config import build_model
from plugins.budget import BudgetPlugin
from plugins.knowledge import KnowledgePlugin
from plugins.stream_capture import StreamCapturePlugin
from plugins.thought_refiner import ThoughtRefinerPlugin
from plugins.tool_audit import ToolAuditPlugin
from plugins.tool_display import ToolDisplayPlugin
from plugins.tool_router import ToolRouterPlugin
from prompts import (
    DEEP_AGENT_INSTRUCTIONS,
    DEEP_CITATIONS_PROMPT,
    DEEP_RESEARCHER_PROMPT,
    PLANNER_PROMPT,
    RAG_SYSTEM_PROMPT,
    RESEARCHER_PROMPT,
    SYSTEM_PROMPT,
)
from tools import get_all_mcp_clients

logger = logging.getLogger(__name__)

# ── Shared plugin instances ──────────────────────────────────────────
# Created once at module level so the FastAPI server can access them
# from main.py (e.g. stream_capture.activate() in the SSE handler).

budget_plugin = BudgetPlugin()
knowledge_plugin = KnowledgePlugin()
stream_capture = StreamCapturePlugin()
thought_refiner = ThoughtRefinerPlugin()
tool_audit = ToolAuditPlugin()
tool_display = ToolDisplayPlugin()
tool_router = ToolRouterPlugin()

# Module-level reference to the AdaptiveLoopPlugin instance (if created).
# Set by create_multi_agent() so reset_plugins() can call plugin.reset().
_adaptive_plugin: Any = None


def get_default_plugins() -> list[Plugin]:
    """Return the standard plugin set for all agents.

    Returns:
        List of SDK Plugin instances.
    """
    return [
        knowledge_plugin,
        tool_router,
        budget_plugin,
        stream_capture,
        thought_refiner,
        tool_display,
        tool_audit,
    ]


def reset_plugins() -> None:
    """Reset per-request state on all plugins.

    Call this before each HTTP request so that plugin state doesn't
    accumulate across requests in a long-running server process.
    """
    budget_plugin.reset()
    if _adaptive_plugin is not None:
        _adaptive_plugin.reset()


def _build_callback_handler() -> CompositeCallbackHandler:
    """Build a composite callback handler.

    Combines PrintingCallbackHandler (stdout for REPL) with
    StreamCapturePlugin's raw callback (per-token SSE streaming).

    Returns:
        CompositeCallbackHandler with printing and stream capture.
    """
    return CompositeCallbackHandler(
        PrintingCallbackHandler(),
        stream_capture.callback_handler,
    )


# ── OpenTelemetry setup ──────────────────────────────────────────────


def _setup_otel() -> None:
    """Configure OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("endpoint=<%s> | OTEL tracing enabled", endpoint)
    except ImportError:
        logger.warning(
            "OTEL packages not installed — run: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )


# ── Agent factories ──────────────────────────────────────────────────


def _enter_mcp_clients(mcp_clients: list) -> list:
    """Enter MCP client contexts and collect tools, skipping failures.

    Each client is entered independently.  If a client's ``__enter__()``
    or ``list_tools_sync()`` raises, it is logged and skipped — the
    remaining clients still initialise.

    Args:
        mcp_clients: List of MCP client instances to enter.

    Returns:
        List of MCP tools from all successfully entered clients.
    """
    tool_list: list = []
    for client in mcp_clients:
        try:
            client.__enter__()
            tools = client.list_tools_sync()
            tool_list.extend(tools)
            logger.info("tools=<%d> | MCP client ready", len(tools))
        except Exception:
            logger.exception("MCP client failed to initialise — skipping")
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass
    return tool_list


def create_single_agent(
    tool_list: list | None = None,
    mcp_clients: list | None = None,
    plugins: list[Plugin] | None = None,
) -> tuple[Agent, list]:
    """Create a single-agent setup with all tools directly available.

    Args:
        tool_list: Pre-built list of MCP tools.  When None the function
            enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce tool_list.
        plugins: Override the default plugin set.  When None, uses
            get_default_plugins().

    Returns:
        Tuple of (agent, mcp_clients).
    """
    model = build_model()
    if tool_list is None:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
        plugins=plugins if plugins is not None else get_default_plugins(),
    )
    return agent, mcp_clients or []


def create_rag_agent(
    tool_list: list | None = None,
    mcp_clients: list | None = None,
    plugins: list[Plugin] | None = None,
) -> tuple[Agent, list]:
    """Create a single-agent setup with the RAG system prompt.

    Identical to :func:`create_single_agent` but uses
    :data:`RAG_SYSTEM_PROMPT`.  The RAG system prompt contains a
    ``{context}`` placeholder that is replaced at dispatch time with
    results retrieved from the Knowledge Engine.
    """
    model = build_model()
    if tool_list is None:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    agent = Agent(
        model=model,
        system_prompt=RAG_SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
        plugins=plugins if plugins is not None else get_default_plugins(),
    )
    return agent, mcp_clients or []


def create_multi_agent(
    tool_list: list | None = None,
    mcp_clients: list | None = None,
    plugins: list[Plugin] | None = None,
) -> tuple[Agent, Agent, list]:
    """Create a planner + researcher multi-agent setup.

    The researcher agent has direct access to all MCP tools.  The planner
    delegates to the researcher via the agent-as-tool pattern.

    Args:
        tool_list: Pre-built list of MCP tools.
        mcp_clients: MCP clients that were entered to produce tool_list.
        plugins: Override the default plugin set for the planner.

    Returns:
        Tuple of (planner_agent, researcher_agent, mcp_clients).
    """
    global _adaptive_plugin

    planner_model = build_model()
    researcher_model = build_model()

    if tool_list is None:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    # Researcher: tool-capable agent that does the actual searching
    researcher = Agent(
        model=researcher_model,
        system_prompt=RESEARCHER_PROMPT,
        tools=tool_list,
        conversation_manager=SlidingWindowConversationManager(
            window_size=15,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
        plugins=plugins if plugins is not None else get_default_plugins(),
    )

    # Planner plugins: default set + optional adaptive loop prevention
    planner_plugins: list[Plugin] = list(plugins if plugins is not None else get_default_plugins())
    try:
        from strands_adaptive import AdaptiveLoopPlugin

        adaptive = AdaptiveLoopPlugin(researcher_model)
        planner_plugins.append(adaptive)
        _adaptive_plugin = adaptive
        logger.info("AdaptiveLoopPlugin registered on planner")
    except ImportError:
        logger.warning(
            "strands_adaptive not available — loop prevention disabled"
        )

    # Planner: strategic agent that delegates to the researcher
    planner = Agent(
        model=planner_model,
        system_prompt=PLANNER_PROMPT,
        tools=[
            researcher.as_tool(
                name="researcher",
                description=(
                    "Deep web research agent with access to Brave Search, "
                    "Firecrawl, Exa, and Kagi. Delegate any web search, "
                    "scrape, or data retrieval task to this tool."
                ),
            ),
        ],
        conversation_manager=SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
        plugins=planner_plugins,
    )
    return planner, researcher, mcp_clients or []


def create_deep_agent_instance(
    tool_list: list | None = None,
    mcp_clients: list | None = None,
) -> tuple[Agent, list]:
    """Create a Deep Agent using the strands-deep-agents package.

    The Deep Agent uses the DeepAgents pattern: a lead agent with strategic
    planning (TODO lists), file-based context management, and sub-agent
    orchestration.

    Args:
        tool_list: Pre-built list of MCP tools.
        mcp_clients: MCP clients that were entered to produce tool_list.

    Returns:
        Tuple of (deep_agent, mcp_clients).
    """
    from strands_deep_agents import SubAgent, create_deep_agent

    model = build_model()

    if tool_list is None:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    research_subagent = SubAgent(
        name="research_subagent",
        description=(
            "Specialized research agent for focused web investigations. "
            "Searches specific questions, gathers facts, analyzes sources. "
            "Has access to Brave Search, Firecrawl, Exa, Kagi, and "
            "Qualitative Research tools. Writes findings to files."
        ),
        tools=tool_list,
        prompt=DEEP_RESEARCHER_PROMPT,
        model=build_model(),
    )

    citations_agent = SubAgent(
        name="citations_agent",
        description=(
            "Adds proper citations and source references to research "
            "reports. Call after research is complete to add inline "
            "citations with URLs."
        ),
        prompt=DEEP_CITATIONS_PROMPT,
        model=build_model(),
    )

    agent = create_deep_agent(
        instructions=DEEP_AGENT_INSTRUCTIONS,
        model=model,
        subagents=[research_subagent, citations_agent],
        tools=tool_list,
        callback_handler=_build_callback_handler(),
    )

    logger.info(
        "tools=<%d> | deep agent ready",
        len(agent.tool_registry.get_all_tools_config()),
    )
    return agent, mcp_clients or []


def _cleanup_mcp(mcp_clients: list) -> None:
    """Gracefully close all MCP client connections."""
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass


def main() -> None:
    """Interactive REPL for the Venice GLM-4.7 agent."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _setup_otel()

    multi_agent = "--multi" in sys.argv

    if multi_agent:
        print("Venice GLM-4.7 Uncensored Research Agent (Strands — Multi-Agent)")
        agent, _researcher, mcp_clients = create_multi_agent()
    else:
        print("Venice GLM-4.7 Uncensored Research Agent (Strands)")
        agent, mcp_clients = create_single_agent()

    tool_count = len(agent.tool_registry.get_all_tools_config())
    print(f"Tools loaded: {tool_count}")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            try:
                query = input("You: ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            response = agent(query)
            print(f"\nAgent: {response}\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _cleanup_mcp(mcp_clients)
        print("MCP connections closed.")


if __name__ == "__main__":
    main()
