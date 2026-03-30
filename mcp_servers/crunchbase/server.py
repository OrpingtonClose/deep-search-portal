"""Crunchbase MCP Server — searches company/startup information.

Rewired: uses multi_search_site() as primary (DDG + Brave + SearXNG in parallel),
with legacy _searxng_query as last-resort fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("crunchbase-search")


@mcp.tool()
async def crunchbase_search(query: str) -> str:
    """Search Crunchbase for company/startup information."""
    import search_providers
    from tools.search_tools import _format_search_results

    results_all: list[dict] = []

    # Primary: multi-source site search
    try:
        raw = await search_providers.multi_search_site(
            query, "crunchbase.com", max_results=10
        )
        results_all.extend(search_providers.results_to_raw_dicts(raw))
    except Exception:
        pass

    # Fallback: legacy _searxng_query (last resort)
    if not results_all:
        try:
            from tools.search_tools import _searxng_query

            results_all = await _searxng_query(
                f"site:crunchbase.com {query}", categories="general"
            )
        except Exception:
            pass

    if not results_all:
        return f"No Crunchbase results for: {query}"

    # Dedup by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results_all:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)

    return (
        _format_search_results(unique[:15], source_label="crunchbase")
        or f"No Crunchbase results for: {query}"
    )


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_CRUNCHBASE_PORT", "9836")))
