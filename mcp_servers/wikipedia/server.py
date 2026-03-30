"""Wikipedia MCP Server — wraps MediaWiki API search."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("wikipedia-search")


@mcp.tool()
async def wikipedia_search(query: str, language: str = "en") -> str:
    """Search Wikipedia via MediaWiki API."""
    from tools.search_tools2 import tool_wikipedia_search

    return await tool_wikipedia_search(query, language)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_WIKIPEDIA_PORT", "9817")))
