"""arXiv MCP Server — wraps arXiv API search."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("arxiv-search")


@mcp.tool()
async def arxiv_search(
    query: str, max_results: int = 10, sort_by: str = "relevance"
) -> str:
    """Search arXiv preprint repository."""
    from tools.search_tools2 import tool_arxiv_search

    return await tool_arxiv_search(query, max_results, sort_by)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_ARXIV_PORT", "9819")))
