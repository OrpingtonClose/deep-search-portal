"""StackExchange MCP Server — wraps SE API search."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("stackexchange-search")


@mcp.tool()
async def stackexchange_search(
    query: str, site: str = "stackoverflow", sort: str = "relevance"
) -> str:
    """Search StackExchange sites (StackOverflow, etc.) via the SE API."""
    from tools.search_tools2 import tool_stackexchange_search

    return await tool_stackexchange_search(query, site, sort)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_STACKEXCHANGE_PORT", "9816")))
