"""Exa Semantic Search MCP Server — wraps search_providers._search_exa."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("exa-search")


@mcp.tool()
async def exa_search(query: str, max_results: int = 10) -> str:
    """Semantic AI search via Exa — finds conceptually relevant results using neural embeddings."""
    from search_providers import _search_exa, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_exa(query, max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="exa") or "No results found."


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_EXA_PORT", "9841")))
