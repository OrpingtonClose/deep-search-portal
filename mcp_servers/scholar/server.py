"""Google Scholar MCP Server — wraps scholar search tool."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("scholar-search")


@mcp.tool()
async def scholar_search(query: str, max_results: int = 10) -> str:
    """Search Google Scholar for academic papers."""
    from tools.search_tools2 import tool_scholar_search

    return await tool_scholar_search(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_SCHOLAR_PORT", "9821")))
