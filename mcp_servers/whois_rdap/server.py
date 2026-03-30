"""WHOIS/RDAP MCP Server — wraps domain lookup tool."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

from fastmcp import FastMCP

mcp = FastMCP("whois-rdap-lookup")


@mcp.tool()
async def whois_lookup(domain: str = "", query: str = "") -> str:
    """Look up WHOIS/RDAP information for a domain."""
    from tools.search_tools2 import tool_whois_lookup

    return await tool_whois_lookup(domain=domain, query=query)


if __name__ == "__main__":
    mcp.run(transport="sse", port=int(os.getenv("MCP_WHOIS_RDAP_PORT", "9838")))
