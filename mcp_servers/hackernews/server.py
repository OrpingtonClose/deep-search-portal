"""HackerNews MCP Server — wraps Algolia HN Search API."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("hackernews-search")


@mcp.tool()
async def hackernews_search(
    query: str, sort_by: str = "relevance", time_range: str = ""
) -> str:
    """Search Hacker News via Algolia API."""
    from tools.search_tools2 import tool_hackernews_search

    return await tool_hackernews_search(query, sort_by, time_range)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_HACKERNEWS_PORT", "9815")))
