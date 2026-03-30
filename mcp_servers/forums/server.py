"""Forums MCP Server — searches forum content across multiple platforms.

Rewired: uses multi_search_site() as primary (DDG + Brave + SearXNG in parallel),
with legacy _searxng_query as last-resort fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("forum-search")


@mcp.tool()
async def forum_search(query: str, forum_url: str = "") -> str:
    """Search forum content across multiple platforms."""
    import search_providers
    from tools.search_tools import _format_search_results

    results_all: list[dict] = []

    if forum_url:
        # Search a specific forum
        try:
            raw = await search_providers.multi_search_site(
                query, forum_url, max_results=10
            )
            results_all.extend(search_providers.results_to_raw_dicts(raw))
        except Exception:
            pass
    else:
        # Search across common forum platforms
        forum_sites = [
            "reddit.com",
            "quora.com",
            "discourse.org",
            "phpbb.com",
            "vbulletin.com",
            "xenforo.com",
        ]
        for site in forum_sites:
            try:
                raw = await search_providers.multi_search_site(
                    query, site, max_results=5
                )
                results_all.extend(search_providers.results_to_raw_dicts(raw))
            except Exception:
                pass

    # Fallback: legacy _searxng_query (last resort)
    if not results_all:
        try:
            from tools.search_tools import _searxng_query

            fq = f"site:{forum_url} {query}" if forum_url else f"{query} forum discussion"
            results_all = await _searxng_query(fq, categories="general")
        except Exception:
            pass

    if not results_all:
        return f"No forum results for: {query}"

    # Dedup by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results_all:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)

    return (
        _format_search_results(unique[:15], source_label="forums")
        or f"No forum results for: {query}"
    )


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_FORUMS_PORT", "9830")))
