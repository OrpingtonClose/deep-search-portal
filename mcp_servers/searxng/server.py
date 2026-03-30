"""SearXNG MCP Server — wraps search_providers._search_searxng."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("searxng-search")


@mcp.tool()
async def searxng_search(
    query: str, categories: str = "general", max_results: int = 10
) -> str:
    """Search via local SearXNG instance."""
    from search_providers import _search_searxng, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_searxng(query, categories=categories, max_results=max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="searxng") or "No results found."


@mcp.tool()
async def searxng_news(query: str, max_results: int = 10) -> str:
    """Search news via local SearXNG instance."""
    from search_providers import _search_searxng, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_searxng(query, categories="news", max_results=max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="searxng-news") or "No news results found."


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_SEARXNG_PORT", "9814")))
