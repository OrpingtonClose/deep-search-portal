"""Mojeek MCP Server — wraps search_providers._search_mojeek."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("mojeek-search")


@mcp.tool()
async def mojeek_search(query: str, max_results: int = 10) -> str:
    """Search the web via Mojeek Search API."""
    from search_providers import _search_mojeek, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_mojeek(query, max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="mojeek") or "No results found."


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_MOJEEK_PORT", "9813")))
