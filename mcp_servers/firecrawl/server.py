"""Firecrawl MCP Server — wraps search_providers._search_firecrawl and adds scrape/crawl tools."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP
from shared import http_client

mcp = FastMCP("firecrawl")

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")


@mcp.tool()
async def firecrawl_search(query: str, max_results: int = 10) -> str:
    """Search the web via Firecrawl — returns clean markdown content from top results."""
    from search_providers import _search_firecrawl, results_to_raw_dicts
    from tools.search_tools import _format_search_results

    results = await _search_firecrawl(query, max_results)
    return _format_search_results(results_to_raw_dicts(results), source_label="firecrawl") or "No results found."


@mcp.tool()
async def firecrawl_scrape(url: str) -> str:
    """Scrape a single URL via Firecrawl — returns clean markdown with JS rendering and anti-bot handling."""
    if not FIRECRAWL_API_KEY:
        return "Error: FIRECRAWL_API_KEY not configured."

    client = http_client()
    resp = await client.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers={
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"url": url, "formats": ["markdown"]},
        timeout=30.0,
    )
    if resp.status_code != 200:
        return f"Firecrawl scrape error: HTTP {resp.status_code}"

    data = resp.json().get("data", {})
    title = data.get("metadata", {}).get("title", url)
    markdown = data.get("markdown", "")
    return f"# {title}\n\n{markdown[:10000]}" if markdown else "No content extracted."


@mcp.tool()
async def firecrawl_crawl(url: str, limit: int = 5) -> str:
    """Crawl a site starting from a URL — discovers and scrapes multiple linked pages."""
    if not FIRECRAWL_API_KEY:
        return "Error: FIRECRAWL_API_KEY not configured."

    client = http_client()
    resp = await client.post(
        "https://api.firecrawl.dev/v1/crawl",
        headers={
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"url": url, "limit": limit, "scrapeOptions": {"formats": ["markdown"]}},
        timeout=60.0,
    )
    if resp.status_code != 200:
        return f"Firecrawl crawl error: HTTP {resp.status_code}"

    data = resp.json()
    pages = data.get("data", [])
    parts = []
    for page in pages[:limit]:
        meta = page.get("metadata", {})
        title = meta.get("title", page.get("url", ""))
        md = page.get("markdown", "")[:3000]
        parts.append(f"## {title}\nURL: {page.get('url', '')}\n\n{md}")
    return "\n\n---\n\n".join(parts) if parts else "No pages crawled."


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_FIRECRAWL_PORT", "9815")))
