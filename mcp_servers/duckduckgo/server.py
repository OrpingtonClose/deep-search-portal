"""DuckDuckGo MCP Server — wraps search_providers._search_duckduckgo."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("duckduckgo-search")


@mcp.tool()
async def duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Search the web via DuckDuckGo."""
    from search_providers import _search_duckduckgo, results_to_text

    results = await _search_duckduckgo(query, max_results)
    return results_to_text(results, source_label="duckduckgo")


@mcp.tool()
async def duckduckgo_news(query: str, max_results: int = 10) -> str:
    """Search news via DuckDuckGo."""
    from search_providers import _search_duckduckgo_news, results_to_text

    results = await _search_duckduckgo_news(query, max_results)
    return results_to_text(results, source_label="duckduckgo-news")


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_DUCKDUCKGO_PORT", "9810")))
