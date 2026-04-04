"""Dispatcher configuration — environment variables and defaults."""

import os

# MCP server registry: maps server names to their SSE endpoints.
# Ports are assigned in the 9810-9849 range.
MCP_SERVER_PORTS: dict[str, int] = {
    "duckduckgo": int(os.getenv("MCP_DUCKDUCKGO_PORT", "9810")),
    "grok_x": int(os.getenv("MCP_GROK_X_PORT", "9811")),
    "brave": int(os.getenv("MCP_BRAVE_PORT", "9812")),
    "mojeek": int(os.getenv("MCP_MOJEEK_PORT", "9813")),
    "searxng": int(os.getenv("MCP_SEARXNG_PORT", "9814")),
    "hackernews": int(os.getenv("MCP_HACKERNEWS_PORT", "9815")),
    "stackexchange": int(os.getenv("MCP_STACKEXCHANGE_PORT", "9816")),
    "wikipedia": int(os.getenv("MCP_WIKIPEDIA_PORT", "9817")),
    "wikidata": int(os.getenv("MCP_WIKIDATA_PORT", "9818")),
    "arxiv": int(os.getenv("MCP_ARXIV_PORT", "9819")),
    "pubmed": int(os.getenv("MCP_PUBMED_PORT", "9820")),
    "scholar": int(os.getenv("MCP_SCHOLAR_PORT", "9821")),
    "wayback": int(os.getenv("MCP_WAYBACK_PORT", "9822")),
    "archive_org": int(os.getenv("MCP_ARCHIVE_ORG_PORT", "9823")),
    "4plebs": int(os.getenv("MCP_4PLEBS_PORT", "9824")),
    "b4k": int(os.getenv("MCP_B4K_PORT", "9825")),
    "warosu": int(os.getenv("MCP_WAROSU_PORT", "9826")),
    "youtube": int(os.getenv("MCP_YOUTUBE_PORT", "9827")),
    "telegram": int(os.getenv("MCP_TELEGRAM_PORT", "9828")),
    "darknet": int(os.getenv("MCP_DARKNET_PORT", "9829")),
    "forums": int(os.getenv("MCP_FORUMS_PORT", "9830")),
    "substack": int(os.getenv("MCP_SUBSTACK_PORT", "9831")),
    "discord": int(os.getenv("MCP_DISCORD_PORT", "9832")),
    "facebook": int(os.getenv("MCP_FACEBOOK_PORT", "9833")),
    "signal": int(os.getenv("MCP_SIGNAL_PORT", "9834")),
    "whatsapp": int(os.getenv("MCP_WHATSAPP_PORT", "9835")),
    "crunchbase": int(os.getenv("MCP_CRUNCHBASE_PORT", "9836")),
    "trustpilot": int(os.getenv("MCP_TRUSTPILOT_PORT", "9837")),
    "whois_rdap": int(os.getenv("MCP_WHOIS_RDAP_PORT", "9838")),
    "bright_data": int(os.getenv("MCP_BRIGHT_DATA_PORT", "9839")),
    "apify": int(os.getenv("MCP_APIFY_PORT", "9840")),
    "exa": int(os.getenv("MCP_EXA_PORT", "9841")),
    "firecrawl": int(os.getenv("MCP_FIRECRAWL_PORT", "9842")),
}

# Dispatcher settings
MCP_DISPATCHER_PORT = int(os.getenv("MCP_DISPATCHER_PORT", "9801"))
MCP_GATEWAY_PORT = int(os.getenv("MCP_GATEWAY_PORT", "9800"))
DISPATCHER_TIMEOUT = float(os.getenv("DISPATCHER_TIMEOUT", "60.0"))
MAX_FANOUT_CONCURRENCY = int(os.getenv("MAX_FANOUT_CONCURRENCY", "5"))


def get_mcp_server_url(server_name: str) -> str:
    """Return the SSE endpoint URL for a named MCP server."""
    port = MCP_SERVER_PORTS.get(server_name)
    if port is None:
        raise ValueError(f"Unknown MCP server: {server_name}")
    return f"http://localhost:{port}/sse"
