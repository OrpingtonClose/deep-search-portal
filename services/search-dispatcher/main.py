"""Search Dispatcher — FastAPI + MCP server entry point.

Provides both a REST API and an MCP server interface for the LangGraph
dispatcher.  The REST API is used by the proxy services; the MCP interface
allows the dispatcher itself to be consumed by other MCP clients.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MCP_DISPATCHER_PORT
from .dispatcher_graph import DispatcherState, get_dispatcher
from .health_aware_router import get_server_health_summary
from .capability_matrix import CAPABILITY_MATRIX, DOMAIN_CATEGORIES

log = logging.getLogger("search-dispatcher")

app = FastAPI(
    title="Search Dispatcher",
    description="LangGraph-based intelligent search router with health-aware MCP server dispatch",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DispatchRequest(BaseModel):
    """Request to dispatch a tool invocation."""

    tool_name: str = Field(..., description="Domain-level tool name from capability matrix")
    tool_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Domain categories for fan-out queries (e.g. ['social', 'academic'])",
    )


class DispatchResponse(BaseModel):
    """Response from a dispatch operation."""

    result: str = ""
    error: str = ""
    provider_used: str = ""
    latency_ms: float = 0.0
    fanout_results: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "server_health": get_server_health_summary(),
    }


@app.get("/tools")
async def list_tools():
    """List all available tools from the capability matrix."""
    return {
        "tools": list(CAPABILITY_MATRIX.keys()),
        "categories": list(DOMAIN_CATEGORIES.keys()),
    }


@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(request: DispatchRequest):
    """Dispatch a tool invocation through the LangGraph router.

    The dispatcher looks up the capability matrix, filters by server health,
    and tries each candidate MCP server in order until one succeeds.

    For fan-out queries (when categories are specified), it dispatches to
    all tools in the requested domain categories concurrently.
    """
    t0 = time.monotonic()

    dispatcher = get_dispatcher()
    initial_state = DispatcherState(
        tool_name=request.tool_name,
        tool_kwargs=request.tool_kwargs,
        categories=request.categories,
    )

    try:
        final_state = await dispatcher.ainvoke(initial_state)
    except Exception as e:
        log.error("Dispatcher error for %s: %s", request.tool_name, e)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.monotonic() - t0) * 1000

    return DispatchResponse(
        result=final_state.result,
        error=final_state.error,
        provider_used=final_state.provider_used,
        latency_ms=elapsed,
        fanout_results=final_state.fanout_results,
    )


@app.post("/fanout", response_model=DispatchResponse)
async def fanout(request: DispatchRequest):
    """Fan-out dispatch across domain categories.

    Convenience endpoint that always uses fan-out mode.
    Equivalent to /dispatch with categories specified.
    """
    if not request.categories:
        raise HTTPException(
            status_code=400,
            detail="categories must be specified for fan-out dispatch",
        )
    return await dispatch(request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.search-dispatcher.main:app",
        host="0.0.0.0",
        port=MCP_DISPATCHER_PORT,
        log_level="info",
    )
