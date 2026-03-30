"""4plebs MCP Server — wraps 4plebs imageboard archive API."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("4plebs-search")


@mcp.tool()
async def search_4plebs(query: str, board: str = "pol") -> str:
    """Search 4plebs imageboard archive."""
    from tools.web_fetch import tool_4plebs_search

    return await tool_4plebs_search(query, board)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_4PLEBS_PORT", "9824")))
