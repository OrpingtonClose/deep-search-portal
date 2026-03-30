"""Capability matrix — maps tool requests to ordered lists of capable MCP servers.

Each key is a domain-level tool name (what the proxy requests).
Each value is an ordered list of (mcp_server, mcp_tool) tuples to try.
The dispatcher tries them in order, returning the first successful result.

This replaces the hardcoded routing in:
- planning.py:route_research_question() (lines 72-135)
- search_gateway.py:gateway_search() (lines 89-223)
- tool_executor.py:_execute_tool_inner() if/elif chain (lines 254-424)
"""

from typing import TypeAlias

ServerTool: TypeAlias = tuple[str, str]

CAPABILITY_MATRIX: dict[str, list[ServerTool]] = {
    # --- Web Search ---
    "web_search": [
        ("duckduckgo", "duckduckgo_search"),
        ("brave", "brave_search"),
        ("mojeek", "mojeek_search"),
        ("searxng", "searxng_search"),
    ],
    "news_search": [
        ("duckduckgo", "duckduckgo_news"),
        ("searxng", "searxng_news"),
    ],
    # --- Social Media ---
    "twitter_search": [
        ("grok_x", "grok_x_search"),
        ("bright_data", "bright_data_twitter_search"),
        ("apify", "apify_twitter_search"),
    ],
    "reddit_search": [
        ("bright_data", "bright_data_reddit_search"),
        ("apify", "apify_reddit_search"),
        ("duckduckgo", "duckduckgo_search"),  # site:reddit.com fallback
    ],
    "instagram_search": [
        ("bright_data", "bright_data_instagram_search"),
        ("apify", "apify_instagram_search"),
    ],
    "tiktok_search": [
        ("bright_data", "bright_data_tiktok_search"),
        ("apify", "apify_tiktok_search"),
    ],
    "linkedin_search": [
        ("bright_data", "bright_data_linkedin_search"),
    ],
    # --- YouTube ---
    "youtube_search": [
        ("bright_data", "bright_data_youtube_search"),
        ("apify", "apify_youtube_search"),
        ("youtube", "youtube_search"),
    ],
    "youtube_transcript": [
        ("youtube", "youtube_transcript"),
    ],
    "youtube_video_metadata": [
        ("youtube", "youtube_video_metadata"),
    ],
    "youtube_video_analyze": [
        ("youtube", "youtube_video_analyze"),
    ],
    # --- Community / Forums ---
    "hackernews_search": [
        ("hackernews", "hackernews_search"),
    ],
    "stackexchange_search": [
        ("stackexchange", "stackexchange_search"),
    ],
    "forum_search": [
        ("forums", "forum_search"),
    ],
    # --- Academic ---
    "arxiv_search": [
        ("arxiv", "arxiv_search"),
    ],
    "pubmed_search": [
        ("pubmed", "pubmed_search"),
    ],
    "scholar_search": [
        ("scholar", "scholar_search"),
    ],
    # --- Knowledge Bases ---
    "wikipedia_search": [
        ("wikipedia", "wikipedia_search"),
    ],
    "wikidata_query": [
        ("wikidata", "wikidata_query"),
    ],
    # --- Archives ---
    "wayback_fetch": [
        ("wayback", "wayback_fetch"),
    ],
    "archiveorg_search": [
        ("archive_org", "archiveorg_search"),
    ],
    # --- Chan Archives ---
    "chan_4plebs_search": [
        ("4plebs", "search_4plebs"),
    ],
    "chan_b4k_search": [
        ("b4k", "search_b4k"),
    ],
    "chan_warosu_search": [
        ("warosu", "search_warosu"),
    ],
    # --- Messaging Platforms ---
    "telegram_search": [
        ("telegram", "telegram_search"),
    ],
    "discord_search": [
        ("discord", "discord_search"),
    ],
    "signal_search": [
        ("signal", "signal_search"),
    ],
    "whatsapp_search": [
        ("whatsapp", "whatsapp_search"),
    ],
    # --- OSINT ---
    "darknet_market_search": [
        ("darknet", "darknet_search"),
    ],
    "facebook_search": [
        ("facebook", "facebook_search"),
    ],
    "substack_search": [
        ("substack", "substack_search"),
    ],
    # --- Business Intelligence ---
    "crunchbase_search": [
        ("crunchbase", "crunchbase_search"),
    ],
    "trustpilot_search": [
        ("trustpilot", "trustpilot_search"),
    ],
    "whois_lookup": [
        ("whois_rdap", "whois_lookup"),
    ],
}

# Domain categories for fan-out queries (replaces search_gateway.py SOURCE_CATEGORIES)
DOMAIN_CATEGORIES: dict[str, list[str]] = {
    "social": [
        "twitter_search",
        "reddit_search",
        "instagram_search",
        "tiktok_search",
        "linkedin_search",
    ],
    "community": [
        "hackernews_search",
        "stackexchange_search",
        "forum_search",
        "chan_4plebs_search",
        "chan_b4k_search",
        "chan_warosu_search",
    ],
    "academic": [
        "arxiv_search",
        "pubmed_search",
        "scholar_search",
    ],
    "archive": [
        "wayback_fetch",
        "archiveorg_search",
    ],
    "video": [
        "youtube_search",
        "youtube_transcript",
    ],
    "messaging": [
        "telegram_search",
        "discord_search",
        "signal_search",
        "whatsapp_search",
    ],
    "osint": [
        "darknet_market_search",
        "facebook_search",
        "substack_search",
    ],
    "business": [
        "crunchbase_search",
        "trustpilot_search",
        "whois_lookup",
    ],
    "knowledge": [
        "wikipedia_search",
        "wikidata_query",
    ],
    "web": [
        "web_search",
        "news_search",
    ],
}
