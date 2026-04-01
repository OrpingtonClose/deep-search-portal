"""Apify MCP Server — wraps Apify Actor API.

Exposes one tool per platform: reddit, twitter, instagram, tiktok, youtube.
Each tool calls social_media_scrapers.social_media_search() with the appropriate platform.
Note: LinkedIn is NOT supported via Apify free actors.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("apify-search")


@mcp.tool()
async def apify_reddit_search(
    query: str, subreddit: str = "", sort: str = "relevance"
) -> str:
    """Search Reddit via Apify Actor API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("reddit", query, subreddit=subreddit, sort=sort)


@mcp.tool()
async def apify_twitter_search(query: str) -> str:
    """Search Twitter/X via Apify Actor API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("twitter", query)


@mcp.tool()
async def apify_instagram_search(query: str) -> str:
    """Search Instagram via Apify Actor API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("instagram", query)


@mcp.tool()
async def apify_tiktok_search(query: str) -> str:
    """Search TikTok via Apify Actor API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("tiktok", query)


@mcp.tool()
async def apify_youtube_search(query: str) -> str:
    """Search YouTube via Apify Actor API."""
    from social_media_scrapers import social_media_search

    return await social_media_search("youtube", query)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_APIFY_PORT", "9840")))
