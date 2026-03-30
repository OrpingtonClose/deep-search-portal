"""b4k MCP Server — wraps b4k imageboard archive search."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("b4k-search")


@mcp.tool()
async def search_b4k(query: str, board: str = "pol") -> str:
    """Search b4k imageboard archive."""
    from tools.web_fetch import tool_b4k_search

    return await tool_b4k_search(query, board)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_B4K_PORT", "9825")))
