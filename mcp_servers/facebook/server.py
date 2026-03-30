"""Facebook MCP Server — searches public Facebook content.

Rewired: uses multi_search_site() as primary (DDG + Brave + SearXNG in parallel),
with legacy _searxng_query as last-resort fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("facebook-search")


@mcp.tool()
async def facebook_search(
    query: str, result_type: str = "posts"
) -> str:
    """Search public Facebook pages, groups, and posts via web indexes.

    Args:
        query: Search terms.
        result_type: One of 'posts' (default), 'groups', or 'pages'.
    """
    import search_providers
    from tools.search_tools import _format_search_results

    results_all: list[dict] = []

    # Primary: multi-source site search
    fb_sites = ["facebook.com", "fb.com"]
    for site in fb_sites:
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

            type_filter = ""
            if result_type == "groups":
                type_filter = "inurl:groups"
            elif result_type == "pages":
                type_filter = "inurl:pages"
            site_query = (
                f"({query}) (site:facebook.com OR site:fb.com) {type_filter}"
            ).strip()
            results_all = await _searxng_query(site_query, categories="general")
        except Exception:
            pass

    if not results_all:
        return f"No Facebook results for: {query}"

    # Dedup by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results_all:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)

    return (
        _format_search_results(unique[:15], source_label="facebook")
        or f"No Facebook results for: {query}"
    )


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_FACEBOOK_PORT", "9833")))
