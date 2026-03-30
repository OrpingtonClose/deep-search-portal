"""Warosu MCP Server — wraps warosu imageboard archive search."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("warosu-search")


@mcp.tool()
async def search_warosu(query: str, board: str = "biz") -> str:
    """Search Warosu imageboard archive."""
    from tools.web_fetch import tool_warosu_search

    return await tool_warosu_search(query, board)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_WAROSU_PORT", "9826")))
