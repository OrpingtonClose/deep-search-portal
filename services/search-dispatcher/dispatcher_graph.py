"""LangGraph StateGraph — the core intelligent agentic router.

Replaces:
- planning.py:route_research_question() (lines 72-135)
- search_gateway.py:gateway_search() (lines 89-223)
- tool_executor.py:_execute_tool_inner() if/elif chain (lines 254-424)

The dispatcher receives a tool request (tool_name + kwargs), looks up the
capability matrix, filters by health, and tries each MCP server in order
until one succeeds.  For fan-out queries it dispatches to multiple domain
categories concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langgraph.graph import StateGraph, END

from .capability_matrix import CAPABILITY_MATRIX, DOMAIN_CATEGORIES
from .config import DISPATCHER_TIMEOUT, MAX_FANOUT_CONCURRENCY, get_mcp_server_url
from .health_aware_router import (
    filter_healthy_servers,
    record_server_outcome,
)

log = logging.getLogger("search-dispatcher.graph")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class DispatcherState:
    """State flowing through the LangGraph dispatcher."""

    # Input
    tool_name: str = ""
    tool_kwargs: dict[str, Any] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)  # for fan-out

    # Internal
    candidates: list[tuple[str, str]] = field(default_factory=list)
    current_index: int = 0
    fanout_results: dict[str, str] = field(default_factory=dict)

    # Output
    result: str = ""
    error: str = ""
    provider_used: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def resolve_candidates(state: DispatcherState) -> DispatcherState:
    """Look up the capability matrix and filter by health."""
    candidates = CAPABILITY_MATRIX.get(state.tool_name, [])
    if not candidates:
        state.error = f"Unknown tool: {state.tool_name}"
        return state

    state.candidates = filter_healthy_servers(candidates)
    state.current_index = 0
    return state


async def try_next_server(state: DispatcherState) -> DispatcherState:
    """Try invoking the current candidate MCP server."""
    if state.current_index >= len(state.candidates):
        state.error = (
            f"All {len(state.candidates)} servers exhausted for {state.tool_name}"
        )
        return state

    server_name, tool_name = state.candidates[state.current_index]
    t0 = time.monotonic()

    try:
        result = await _invoke_mcp_tool(server_name, tool_name, state.tool_kwargs)
        elapsed = (time.monotonic() - t0) * 1000

        # Check for tool-level errors in the result
        if result and "[TOOL_ERROR]" in result:
            record_server_outcome(server_name, False, result[:200])
            state.current_index += 1
            return state

        record_server_outcome(server_name, True)
        state.result = result
        state.provider_used = server_name
        state.latency_ms = elapsed
        return state

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        log.warning(
            "MCP server %s/%s failed (%.0fms): %s",
            server_name, tool_name, elapsed, e,
        )
        record_server_outcome(server_name, False, str(e))
        state.current_index += 1
        return state


async def fanout_dispatch(state: DispatcherState) -> DispatcherState:
    """Dispatch to multiple domain categories concurrently."""
    if not state.categories:
        state.error = "No categories specified for fan-out"
        return state

    # Collect all tool names from requested categories
    tool_names: list[str] = []
    for cat in state.categories:
        tool_names.extend(DOMAIN_CATEGORIES.get(cat, []))

    if not tool_names:
        state.error = f"No tools found for categories: {state.categories}"
        return state

    sem = asyncio.Semaphore(MAX_FANOUT_CONCURRENCY)

    async def _dispatch_one(tn: str) -> tuple[str, str]:
        async with sem:
            sub_state = DispatcherState(
                tool_name=tn,
                tool_kwargs=state.tool_kwargs,
            )
            sub_state = await resolve_candidates(sub_state)
            if sub_state.error:
                return tn, f"[ERROR] {sub_state.error}"

            while sub_state.current_index < len(sub_state.candidates):
                sub_state = await try_next_server(sub_state)
                if sub_state.result:
                    return tn, sub_state.result
            return tn, f"[ERROR] {sub_state.error}"

    tasks = [_dispatch_one(tn) for tn in tool_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, tuple):
            tn, res = item
            state.fanout_results[tn] = res
        elif isinstance(item, Exception):
            log.warning("Fan-out task error: %s", item)

    # Combine results
    parts: list[str] = []
    for tn, res in state.fanout_results.items():
        if not res.startswith("[ERROR]"):
            parts.append(f"--- {tn} ---\n{res}")

    state.result = "\n\n".join(parts) if parts else "No results from any source"
    state.provider_used = "fanout"
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def should_continue(state: DispatcherState) -> str:
    """Decide the next node after try_next_server."""
    if state.result:
        return END
    if state.current_index >= len(state.candidates):
        return END
    return "try_next_server"


def route_entry(state: DispatcherState) -> str:
    """Route at graph entry: fan-out or single tool dispatch."""
    if state.categories:
        return "fanout_dispatch"
    return "resolve_candidates"


def after_resolve(state: DispatcherState) -> str:
    """Route after resolve_candidates: skip to END if no candidates found."""
    if state.error or not state.candidates:
        return END
    return "try_next_server"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_dispatcher_graph() -> StateGraph:
    """Build and compile the LangGraph dispatcher."""
    graph = StateGraph(DispatcherState)

    # Add nodes
    graph.add_node("resolve_candidates", resolve_candidates)
    graph.add_node("try_next_server", try_next_server)
    graph.add_node("fanout_dispatch", fanout_dispatch)

    # Entry routing
    graph.set_conditional_entry_point(route_entry)

    # Conditional edges from resolve_candidates (skip to END on error)
    graph.add_conditional_edges("resolve_candidates", after_resolve)

    # Conditional edges from try_next_server
    graph.add_conditional_edges("try_next_server", should_continue)

    # Fan-out goes directly to END
    graph.add_edge("fanout_dispatch", END)

    return graph.compile()


# Singleton compiled graph
_dispatcher = None


def get_dispatcher():
    """Return the compiled dispatcher graph (lazy singleton)."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = build_dispatcher_graph()
    return _dispatcher


# ---------------------------------------------------------------------------
# MCP tool invocation (placeholder — will use MCP client SDK)
# ---------------------------------------------------------------------------


async def _invoke_mcp_tool(
    server_name: str, tool_name: str, kwargs: dict[str, Any]
) -> str:
    """Invoke a tool on a remote MCP server via SSE transport.

    Uses httpx to call the MCP server's SSE endpoint directly.
    This is a simplified implementation; production would use the
    full MCP client SDK for session management and streaming.
    """
    import httpx

    url = get_mcp_server_url(server_name)
    # MCP JSON-RPC call over HTTP (simplified — real MCP uses SSE streams)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": kwargs,
        },
    }

    async with httpx.AsyncClient(timeout=DISPATCHER_TIMEOUT) as client:
        resp = await client.post(
            url.replace("/sse", "/messages"),  # MCP HTTP endpoint
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")

    # Extract text content from MCP response
    result = data.get("result", {})
    content = result.get("content", [])
    if isinstance(content, list):
        text_parts = [
            item.get("text", "") for item in content if item.get("type") == "text"
        ]
        return "\n".join(text_parts)

    return str(result)
