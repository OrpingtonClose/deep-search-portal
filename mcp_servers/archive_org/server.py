"""Archive.org MCP Server — wraps Archive.org search API."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("archiveorg-search")


@mcp.tool()
async def archiveorg_search(query: str, max_results: int = 10) -> str:
    """Search the Internet Archive (archive.org) collection."""
    from tools.search_tools2 import tool_archiveorg_search

    return await tool_archiveorg_search(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_ARCHIVE_ORG_PORT", "9823")))
