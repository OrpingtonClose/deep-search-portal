"""Grok X/Twitter MCP Server — Grok Responses API for X search ONLY.

CRITICAL: search_type="x" is hardcoded. Grok is NEVER used for web search
in the new architecture — that role belongs to DuckDuckGo, Brave, etc.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("grok-x-search")


@mcp.tool()
async def grok_x_search(query: str, instructions: str = "") -> str:
    """Search X/Twitter via Grok 4.20 Responses API. X/Twitter ONLY — no web search."""
    from tools.grok_search import tool_grok_deep_search

    return await tool_grok_deep_search(query, search_type="x", instructions=instructions)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_GROK_X_PORT", "9811")))
