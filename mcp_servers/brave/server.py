"""Brave Search MCP Server — wraps search_providers._search_brave."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("brave-search")


@mcp.tool()
async def brave_search(query: str, max_results: int = 10) -> str:
    """Search the web via Brave Search API."""
    from search_providers import _search_brave, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_brave(query, max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="brave") or "No results found."


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_BRAVE_PORT", "9812")))
