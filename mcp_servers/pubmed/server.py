"""PubMed MCP Server — wraps NCBI E-utilities API."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("pubmed-search")


@mcp.tool()
async def pubmed_search(query: str, max_results: int = 10) -> str:
    """Search PubMed biomedical literature via NCBI E-utilities."""
    from tools.search_tools2 import tool_pubmed_search

    return await tool_pubmed_search(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_PUBMED_PORT", "9820")))
