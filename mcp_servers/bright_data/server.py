"""Bright Data MCP Server — wraps Bright Data Web Scraper API.

Exposes one tool per platform: reddit, twitter, instagram, tiktok, linkedin, youtube.
Each tool calls social_media_scrapers.social_media_search() with the appropriate platform.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("bright-data-search")


@mcp.tool()
async def bright_data_reddit_search(
    query: str, subreddit: str = "", sort: str = "relevance"
) -> str:
    """Search Reddit via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("reddit", query, subreddit=subreddit, sort=sort)


@mcp.tool()
async def bright_data_twitter_search(query: str) -> str:
    """Search Twitter/X via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("twitter", query)


@mcp.tool()
async def bright_data_instagram_search(query: str) -> str:
    """Search Instagram via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("instagram", query)


@mcp.tool()
async def bright_data_tiktok_search(query: str) -> str:
    """Search TikTok via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("tiktok", query)


@mcp.tool()
async def bright_data_linkedin_search(query: str) -> str:
    """Search LinkedIn via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("linkedin", query)


@mcp.tool()
async def bright_data_youtube_search(query: str) -> str:
    """Search YouTube via Bright Data Web Scraper API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("youtube", query)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_BRIGHT_DATA_PORT", "9839")))
