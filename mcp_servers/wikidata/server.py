"""Wikidata MCP Server — wraps SPARQL query interface."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("wikidata-query")


@mcp.tool()
async def wikidata_query(query: str) -> str:
    """Query Wikidata via SPARQL endpoint."""
    from tools.search_tools2 import tool_wikidata_query

    return await tool_wikidata_query(query)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_WIKIDATA_PORT", "9818")))
