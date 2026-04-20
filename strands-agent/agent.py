# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""
Venice GLM-4.7 uncensored research agent — Strands Agents SDK.

Features used:
- OpenAI-compatible model provider (Venice AI)
- MCP tool integration (Brave, Firecrawl, Exa, Kagi)
- Streaming responses (PrintingCallbackHandler)
- Conversation memory (SlidingWindowConversationManager)
- Agent loop with automatic tool dispatch
- Multi-agent orchestration (planner + researcher via agent-as-tool)
- Guardrails (callback-based pre/post processing with budget limits)
- Adaptive loop prevention (Plugin + @hook — temperature escalation)
- OpenTelemetry observability (OTEL_EXPORTER_OTLP_ENDPOINT)
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time

from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)
from strands.agent.conversation_manager import SlidingWindowConversationManager

from config import build_model
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

# ── Guardrails: budget tracking callback ─────────────────────────────
# Tracks actual tool invocations (not user queries) via Strands' callback
# system.  The callback fires for every streaming event; we only increment
# when ``current_tool_use`` is present with a tool ``name`` and we haven't
# already counted that specific invocation (keyed by ``toolUseId``).

_session_start = time.time()
_tool_call_count = 0
_seen_tool_use_ids: set[str] = set()
_MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "200"))
_SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))

# Module-level reference to the AdaptiveLoopPlugin instance (if created).
# Set by create_multi_agent() so reset_budget() can call plugin.reset().
_adaptive_plugin = None


def reset_budget() -> None:
    """Reset per-request budget counters and adaptive plugin state.

    Call this before each HTTP request so that budget globals don't
    accumulate across requests in a long-running server process.
    Also resets the AdaptiveLoopPlugin (query history + temperature)
    if one is registered.
    """
    global _session_start, _tool_call_count, _seen_tool_use_ids
    _session_start = time.time()
    _tool_call_count = 0
    _seen_tool_use_ids = set()
    if _adaptive_plugin is not None:
        _adaptive_plugin.reset()


def budget_callback(**kwargs) -> None:
    """Callback-handler guardrail that counts actual tool invocations.

    Strands fires this callback for every streaming event.  We detect new
    tool calls by checking for the ``current_tool_use`` kwarg with a tool
    ``name`` and a unique ``toolUseId``.  Each unique ID is counted once.
    """
    global _tool_call_count

    tool_use = kwargs.get("current_tool_use")
    if not tool_use or not tool_use.get("name"):
        return

    tool_use_id = tool_use.get("toolUseId", "")
    if tool_use_id in _seen_tool_use_ids:
        return
    _seen_tool_use_ids.add(tool_use_id)

    _tool_call_count += 1

    elapsed = time.time() - _session_start
    if elapsed > _SESSION_TIMEOUT:
        logger.warning(
            "Session timeout reached (%.0fs > %ds). Consider wrapping up.",
            elapsed,
            _SESSION_TIMEOUT,
        )

    if _tool_call_count > _MAX_TOOL_CALLS:
        logger.warning(
            "Tool call budget exceeded (%d > %d). Consider wrapping up.",
            _tool_call_count,
            _MAX_TOOL_CALLS,
        )

    if _tool_call_count % 10 == 0:
        logger.info(
            "Budget: %d tool calls, %.0fs elapsed",
            _tool_call_count,
            elapsed,
        )


class StreamCapture:
    """Thread-safe callback that captures streaming tokens to a queue.

    Activate before a request to start capturing; deactivate after.
    When no queue is active, tokens are silently dropped.
    ``PrintingCallbackHandler`` is included separately in the composite
    handler so REPL users still see real-time stdout output.
    """

    def __init__(self):
        self._queue: queue.Queue | None = None
        self._lock = threading.Lock()
        self.tool_events: list[dict] = []
        self._seen_tool_ids: set[str] = set()
        self.all_text: list[str] = []
        self.response_text: list[str] = []
        self.reasoning_text: list[str] = []

    def activate(self) -> queue.Queue:
        """Start capturing. Returns queue the caller reads from."""
        with self._lock:
            q: queue.Queue = queue.Queue()
            self._queue = q
            self.tool_events.clear()
            self._seen_tool_ids.clear()
            self.all_text.clear()
            self.response_text.clear()
            self.reasoning_text.clear()
            return q

    def deactivate(self):
        """Stop capturing and send sentinel so readers know we're done."""
        with self._lock:
            if self._queue is not None:
                self._queue.put(None)
            self._queue = None

    def __call__(self, **kwargs):
        # Only accumulate data when a consumer is actively capturing
        # (i.e. activate() has been called).  This prevents unbounded
        # memory growth from /query endpoints that never activate.
        with self._lock:
            active = self._queue is not None
        if not active:
            return

        # Capture streaming text tokens (both regular data and reasoning text)
        data = kwargs.get("data", "")
        reasoning = kwargs.get("reasoningText", "")

        # Track reasoning and response text separately.
        # all_text = everything (for logging); response_text = data only (for answer fallback)
        if reasoning and isinstance(reasoning, str):
            self.all_text.append(reasoning)
            self.reasoning_text.append(reasoning)
            with self._lock:
                if self._queue is not None:
                    self._queue.put(("thinking", reasoning))

        if data and isinstance(data, str):
            self.all_text.append(data)
            self.response_text.append(data)
            with self._lock:
                if self._queue is not None:
                    self._queue.put(("text", data))

        # Capture tool invocations (deduplicated by toolUseId)
        # Tools can come via either 'current_tool_use' or 'event.contentBlockStart'
        tool_use = kwargs.get("current_tool_use")
        if not tool_use or not tool_use.get("name"):
            tool_use = (
                kwargs.get("event", {})
                .get("contentBlockStart", {})
                .get("start", {})
                .get("toolUse")
            )
        if tool_use and tool_use.get("name"):
            tid = tool_use.get("toolUseId", "")
            if tid and tid not in self._seen_tool_ids:
                self._seen_tool_ids.add(tid)
                event = {
                    "tool": tool_use["name"],
                    "input": str(tool_use.get("input", {}))[:500],
                    "time": time.time(),
                }
                self.tool_events.append(event)
                with self._lock:
                    if self._queue is not None:
                        self._queue.put(("tool", event))


# Global stream-capture instance shared by all agents
stream_capture = StreamCapture()


def _build_callback_handler():
    """Build a composite callback handler: printing + streaming capture + budget guardrail."""
    return CompositeCallbackHandler(PrintingCallbackHandler(), stream_capture, budget_callback)


# ── OpenTelemetry setup ──────────────────────────────────────────────
# Strands has built-in OTEL support.  Set OTEL_EXPORTER_OTLP_ENDPOINT
# in .env to stream traces to Phoenix or any OTEL backend.


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
        logger.info("OTEL tracing enabled → %s", endpoint)
    except ImportError:
        logger.warning(
            "OTEL packages not installed. Run: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )


# ── Agent factories ──────────────────────────────────────────────────


def _enter_mcp_clients(mcp_clients):
    """Enter MCP client contexts and collect tools, skipping failures.

    Each client is entered independently.  If a client's ``__enter__()``
    or ``list_tools_sync()`` raises, it is logged and skipped — the
    remaining clients still initialise.  This prevents a single flaky
    MCP server (e.g. network timeout on npx download) from taking down
    all tools.
    """
    entered: list = []
    tool_list: list = []
    for client in mcp_clients:
        try:
            client.__enter__()
            tools = client.list_tools_sync()
            entered.append(client)
            tool_list.extend(tools)
            logger.info("MCP client ready — %d tools", len(tools))
        except Exception:
            logger.exception("MCP client failed to initialise — skipping")
            # Try to clean up the partially-entered client
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass
    return tool_list


def create_single_agent(tool_list=None, mcp_clients=None):
    """Create a single-agent setup with all tools directly available.

    Use this for simple interactive sessions where one agent handles
    both search and synthesis.

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
    )
    return agent, mcp_clients or []


def create_rag_agent(tool_list=None, mcp_clients=None):
    """Create a single-agent setup with the RAG system prompt.

    Identical to :func:`create_single_agent` but uses
    :data:`RAG_SYSTEM_PROMPT`.  The RAG system prompt contains a
    ``{context}`` placeholder that is replaced at dispatch time with
    results retrieved from the Knowledge Engine.
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    agent = Agent(
        model=model,
        system_prompt=RAG_SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
    )
    return agent, mcp_clients or []


def create_multi_agent(tool_list=None, mcp_clients=None):
    """Create a planner + researcher multi-agent setup.

    The researcher agent has direct access to all MCP tools and handles
    web search/scraping.  The planner agent delegates to the researcher
    via the agent-as-tool pattern and handles strategic decomposition
    and synthesis.

    The planner is equipped with the AdaptiveLoopPlugin which detects
    repeated/similar queries and escalates the researcher's temperature
    to force divergent thinking, preventing infinite loops.

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.

    Returns:
        Tuple of (planner_agent, researcher_agent, mcp_clients).
    """
    # Separate model instances so temperature escalation on the researcher
    # does not affect the planner's reasoning quality.
    planner_model = build_model()
    researcher_model = build_model()

    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

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
    )

    # ── Adaptive loop prevention plugin ──
    # Imported from deep-search-portal/proxies/strands_adaptive.py via PYTHONPATH.
    # If unavailable, the planner still works — just without loop prevention.
    plugins = []
    try:
        from strands_adaptive import AdaptiveLoopPlugin

        plugin = AdaptiveLoopPlugin(researcher_model)
        plugins.append(plugin)
        global _adaptive_plugin
        _adaptive_plugin = plugin
        logger.info("AdaptiveLoopPlugin registered on planner")
    except ImportError:
        logger.warning(
            "strands_adaptive not available — loop prevention disabled. "
            "Ensure deep-search-portal/proxies is on PYTHONPATH."
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
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
        plugins=plugins,
    )
    return planner, researcher, mcp_clients or []


def create_deep_agent_instance(tool_list=None, mcp_clients=None):
    """Create a Deep Agent using the strands-deep-agents package.

    The Deep Agent uses the DeepAgents pattern: a lead agent with strategic
    planning (TODO lists), file-based context management, and sub-agent
    orchestration.  Sub-agents are spawned ephemerally for each task and
    run in isolation to keep the lead agent's context lean.

    Architecture:
      - **Lead Agent** — plans research, delegates, synthesizes
      - **research_subagent** — focused web research with all MCP tools
      - **citations_agent** — adds source references to reports

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.

    Returns:
        Tuple of (deep_agent, mcp_clients).
    """
    from strands_deep_agents import SubAgent, create_deep_agent

    model = build_model()

    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    # Research sub-agent: has all MCP search tools
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

    # Citations sub-agent: adds source references to reports
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
        "Deep agent ready — %d tools",
        len(agent.tool_registry.get_all_tools_config()),
    )
    return agent, mcp_clients or []


def _cleanup_mcp(mcp_clients):
    """Gracefully close all MCP client connections."""
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass


def main():
    """Interactive REPL for the Venice GLM-4.7 agent."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _setup_otel()

    # Choose mode based on --multi flag
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
