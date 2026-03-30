"""YouTube MCP Server — wraps YouTube search, transcript, metadata, and analysis tools."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("youtube-tools")


@mcp.tool()
async def youtube_search(query: str) -> str:
    """Search YouTube for videos matching the query."""
    from tools.search_tools2 import tool_youtube_search

    return await tool_youtube_search(query)


@mcp.tool()
async def youtube_transcript(url: str, lang: str = "en") -> str:
    """Extract transcript from a YouTube video."""
    from tools.search_tools2 import tool_youtube_transcript

    return await tool_youtube_transcript(url, lang)


@mcp.tool()
async def youtube_video_metadata(url: str) -> str:
    """Get metadata (title, description, stats) for a YouTube video."""
    from tools.search_tools2 import tool_youtube_video_metadata

    return await tool_youtube_video_metadata(url)


@mcp.tool()
async def youtube_video_analyze(url: str, question: str = "") -> str:
    """Analyze a YouTube video's visual content using a vision-capable model."""
    from tools.search_tools2 import tool_youtube_video_analyze

    return await tool_youtube_video_analyze(url, question)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_YOUTUBE_PORT", "9827")))
