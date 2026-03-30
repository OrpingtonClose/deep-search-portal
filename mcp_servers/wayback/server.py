"""Wayback Machine MCP Server — wraps Wayback Machine API."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("wayback-fetch")


@mcp.tool()
async def wayback_fetch(url: str) -> str:
    """Fetch an archived version of a URL from the Wayback Machine."""
    from tools.search_tools2 import tool_wayback_fetch

    return await tool_wayback_fetch(url)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_WAYBACK_PORT", "9822")))
