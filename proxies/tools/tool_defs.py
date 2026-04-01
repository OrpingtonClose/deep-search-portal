"""
Native and LangChain tool definitions (OpenAI function-calling format).
"""
from __future__ import annotations


# ============================================================================
# Native Tool Definitions (OpenAI function-calling format)
# ============================================================================

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": (
                "Search the web using SearXNG. Returns top results with titles, "
                "URLs, and snippets. Use this to find information, verify facts, "
                "discover sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch a webpage and extract its readable text content. Uses a "
                "multi-tier fallback chain: fast HTTP fetch → headless browser "
                "(JS rendering) → Bright Data/Oxylabs proxy → Wayback Machine "
                "archive. Automatically retries with escalating methods if the "
                "page is blocked, requires JavaScript, or returns a 404."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "extract_info": {
                        "type": "string",
                        "description": "Optional: specific information to look for",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute Python code for calculations, data processing, or analysis. "
                "Code runs in a sandboxed subprocess with a 30-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv for academic papers. Returns paper titles, authors, "
                "abstracts, and links. Use this for academic and scientific research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for arXiv papers"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wayback_fetch",
            "description": (
                "Fetch an archived version of a webpage from the Wayback Machine "
                "(archive.org). Use this to recover dead links or see historical "
                "versions of pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to look up in the Wayback Machine",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikidata_query",
            "description": (
                "Query Wikidata for structured facts about entities. Returns "
                "entity properties and relationships. Use this for verifiable "
                "factual data about people, places, organizations, concepts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity to look up (e.g., 'Albert Einstein', 'Python programming language')",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_graph_search",
            "description": (
                "Search the Neo4j knowledge graph for relevant concepts, claims, evidence, "
                "anomalies, and text chunks. Supports hybrid search (keyword + graph traversal "
                "with reciprocal rank fusion). Use this FIRST when the user's question may relate "
                "to documents or knowledge that has been ingested into the knowledge engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace to search within (default: 'default')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "graph"],
                        "description": "Search mode (default: 'hybrid')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10, max 50)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_discover",
            "description": (
                "Run graph discovery algorithms on the knowledge graph to find hidden "
                "connections and serendipitous links. Supports: spreading_activation (multi-hop "
                "activation propagation from seed concepts), swanson_abc (find concepts connected "
                "through intermediaries but not directly \u2014 bisociation discovery), and "
                "information_gaps (find under-connected but important concepts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["spreading_activation", "swanson_abc", "information_gaps"],
                        "description": "The discovery algorithm to run",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The conversation/context namespace",
                    },
                    "seed_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting concept names (required for spreading_activation and swanson_abc)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 15)",
                    },
                },
                "required": ["algorithm", "namespace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chan_4plebs_search",
            "description": (
                "Search 4chan archives via 4plebs. Covers boards: /pol/, /sp/, /int/, "
                "/tv/, /k/, /vg/, and others. Returns archived posts with full text, "
                "timestamps, and thread links. Use for researching political discourse, "
                "anonymous intelligence, cultural signals, and early meme/narrative tracking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "board": {
                        "type": "string",
                        "description": "Board to search (default: 'pol'). Options: pol, sp, int, tv, k, vg, etc.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chan_b4k_search",
            "description": (
                "Search the arch.b4k.co archive for /biz/ (4chan's business & finance "
                "board). The only reliable /biz/ archive covering 2017-present. Use for "
                "cryptocurrency discussions, financial alpha, DeFi analysis, 'link marines', "
                "and early-stage project sentiment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for /biz/"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chan_warosu_search",
            "description": (
                "Search warosu.org archives for /g/ (technology), /sci/ (science), "
                "/lit/ (literature), /jp/, /vr/, /fa/. Use for technical discussions, "
                "scientific discourse, and niche hobbyist knowledge not found on mainstream "
                "search engines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "board": {
                        "type": "string",
                        "description": "Board to search (default: 'g'). Options: g, sci, lit, jp, vr, fa.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "twitter_search",
            "description": (
                "Search Twitter/X for tweets and discussions. Supports Twitter search "
                "operators: from:handle, since:YYYY-MM-DD, until:YYYY-MM-DD, \"exact phrase\". "
                "Use for real-time signals, financial market sentiment, geopolitical breaking "
                "news, expert commentary, and public discourse analysis. Results are routed "
                "through commercial proxies (Bright Data/Oxylabs) for reliable access."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Twitter search query. Supports operators like "
                            "'from:elonmusk since:2024-01-01 \"AI safety\"'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "social_media_search",
            "description": (
                "Search social media platforms via commercial scrapers (Bright Data + Apify). "
                "Supports: twitter, reddit, instagram, tiktok, linkedin, youtube. "
                "WARNING: These are censored commercial services — results may be filtered, "
                "truncated, or silently dropped. The tool will flag suspicious gaps. "
                "Cost is tracked per-call; budget limits are enforced. "
                "For Twitter specifically, prefer the dedicated twitter_search tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["twitter", "reddit", "instagram", "tiktok", "linkedin", "youtube"],
                        "description": "Social media platform to search",
                    },
                    "query": {"type": "string", "description": "Search query"},
                    "subreddit": {
                        "type": "string",
                        "description": "Reddit-only: subreddit to search within (e.g., 'wallstreetbets')",
                    },
                    "result_type": {
                        "type": "string",
                        "description": "Platform-specific result type (e.g., 'posts', 'videos')",
                    },
                },
                "required": ["platform", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reddit_search",
            "description": (
                "Search Reddit posts and comments via commercial scrapers (Bright Data/Apify). "
                "WARNING: Censored service — content moderation may filter results. "
                "Cross-validate thin results with chan archives or web search. "
                "Cost tracked and budget-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "subreddit": {
                        "type": "string",
                        "description": "Optional: subreddit to search within (e.g., 'wallstreetbets')",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "hot", "top", "new"],
                        "description": "Sort order (default: relevance)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "instagram_search",
            "description": (
                "Search Instagram posts by hashtag or keyword via commercial scrapers. "
                "WARNING: Heavily censored platform — NSFW, political, and controversial "
                "content is aggressively filtered. Treat thin results with skepticism. "
                "Cost tracked and budget-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Hashtag or keyword to search"},
                    "result_type": {
                        "type": "string",
                        "description": "'posts' (default) or 'profiles'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tiktok_search",
            "description": (
                "Search TikTok videos by keyword via commercial scrapers. "
                "WARNING: Censored platform — content moderation and geo-restrictions "
                "may limit results. Cost tracked and budget-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "result_type": {
                        "type": "string",
                        "description": "'posts' (default)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "linkedin_search",
            "description": (
                "Search LinkedIn posts by keyword via Bright Data (no Apify fallback). "
                "WARNING: Heavily restricted platform — LinkedIn aggressively blocks scrapers "
                "and filters content. Only available via Bright Data. "
                "Cost tracked and budget-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "result_type": {
                        "type": "string",
                        "description": "'posts' (default)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": (
                "Search YouTube videos by keyword via commercial scrapers. "
                "Returns video titles, channels, view counts, and descriptions. "
                "WARNING: Censored service — content moderation may filter results. "
                "Cost tracked and budget-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "result_type": {
                        "type": "string",
                        "description": "'videos' (default)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": (
                "Search for recent news articles using news-specific search engines "
                "(Google News, Bing News, etc.). Use this for any query about current "
                "events, recent developments, breaking news, market movements, or "
                "anything that happened within the last days/weeks/months. Supports "
                "time_range filtering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The news search query"},
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter results to this time range (default: week)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hackernews_search",
            "description": (
                "Search Hacker News (news.ycombinator.com) via the Algolia API. "
                "Covers stories, comments, Ask HN, and Show HN posts. Excellent for "
                "tech industry discourse, startup culture, programming debates, "
                "security incidents, and expert opinions from engineers/founders. "
                "Free API, no authentication required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "description": "Sort by relevance (default) or date (newest first)",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter to posts within this time range (optional)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stackexchange_search",
            "description": (
                "Search Stack Exchange Q&A sites for expert answers. Covers hundreds "
                "of niche communities: stackoverflow, superuser, serverfault, askubuntu, "
                "math, physics, chemistry, biology, electronics, diy, cooking, gaming, "
                "rpg, worldbuilding, law, money, academia, etc. Returns questions with "
                "body text, scores, and answer counts. Free API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "site": {
                        "type": "string",
                        "description": (
                            "Stack Exchange site to search (default: stackoverflow). "
                            "Examples: superuser, serverfault, math, physics, chemistry, "
                            "biology, electronics, diy, cooking, gaming, rpg, worldbuilding, "
                            "law, money, academia, security, unix, apple, etc."
                        ),
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "activity", "votes", "creation"],
                        "description": "Sort order (default: relevance)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": (
                "Search PubMed for biomedical and life science research papers. "
                "Covers medical journals, clinical trials, pharmacology, biochemistry, "
                "genetics, epidemiology, public health, toxicology, and more. Returns "
                "paper titles, authors, journals, and DOIs. Free NCBI API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query (supports MeSH terms and boolean operators)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 10, max 15)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia for encyclopedic knowledge. Returns article extracts "
                "with text snippets, timestamps, and word counts. Use for background "
                "context, definitions, historical facts, and general reference. "
                "Free MediaWiki API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 8, max 15)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archiveorg_search",
            "description": (
                "Search the Internet Archive's full-text index across all collections. "
                "Covers books, magazines, government documents, academic papers, audio, "
                "video, software, and web archives. NOT the Wayback Machine URL lookup — "
                "this searches actual content of archived materials. Use for rare "
                "historical documents, out-of-print books, government reports, and "
                "primary sources. Free API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "media_type": {
                        "type": "string",
                        "description": "Filter by media type: texts, audio, movies, software, image, etc.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 10, max 15)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forum_search",
            "description": (
                "Search niche internet forums for first-hand experiences, hobbyist "
                "knowledge, and discussions not found on mainstream platforms. Searches "
                "across SomethingAwful, Bodybuilding.com, XDA-Developers, Head-Fi, "
                "AVSForum, Overclock.net, ResetEra, KiwiFarms, HardwareZone, and more. "
                "Optionally target a specific forum URL. Use for niche expertise, "
                "product reviews, and underground knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "forum_url": {
                        "type": "string",
                        "description": (
                            "Optional: specific forum URL to search within "
                            "(e.g., 'forums.somethingawful.com', 'forum.bodybuilding.com')"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scholar_search",
            "description": (
                "Search academic literature via SearXNG's science/scholar engines. "
                "Broader than arXiv alone — covers Google Scholar, Semantic Scholar, "
                "ResearchGate, Academia.edu, SSRN, JSTOR, and more. Use for journal "
                "articles, conference papers, theses, patents, and court opinions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Academic search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_search",
            "description": (
                "Search Substack newsletters for long-form independent analysis, "
                "investigative journalism, and expert commentary. Covers niche topics "
                "not found in mainstream media — geopolitics, finance, science, tech, "
                "health, culture. Use fetch_webpage on results to get full article text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_transcript",
            "description": (
                "Extract the full transcript/subtitles from a YouTube video. Returns "
                "timestamped spoken content — the actual knowledge: practitioner "
                "explanations, lecture content, interview dialogue, tutorial steps. "
                "No API key needed. Works with auto-generated and manual captions. "
                "This is the PRIMARY way to extract knowledge from YouTube videos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "YouTube video URL or video ID"},
                    "lang": {
                        "type": "string",
                        "description": "Language code for transcript (default: en). Falls back to any available.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_video_metadata",
            "description": (
                "Extract rich metadata from a YouTube video: title, channel, upload date, "
                "view/like counts, full description, chapter markers, tags, categories, "
                "and top comments. Comments contain corrections, additional knowledge, "
                "and community reactions. Chapter markers help navigate long videos. "
                "Description often has links, timestamps, and context not in spoken content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "YouTube video URL or video ID"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_video_analyze",
            "description": (
                "Analyze a YouTube video's VISUAL content using Qwen Omni vision model. "
                "Extracts key frames and sends to a vision-language model for in-depth "
                "analysis. Use when the video contains diagrams, charts, code on screen, "
                "product teardowns, demonstrations, or visual evidence that the transcript "
                "alone cannot capture. Requires QWEN_OMNI_BASE_URL to be configured."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "YouTube video URL or video ID"},
                    "question": {
                        "type": "string",
                        "description": "Specific question about the video visuals (optional). If empty, does general visual analysis.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    # -----------------------------------------------------------------
    # Site-filtered search tools (SearXNG proxy for platforms without
    # direct APIs).  These exist so the LLM can explicitly target
    # specific platforms rather than relying on generic search.
    # -----------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "telegram_search",
            "description": (
                "Search for publicly indexed Telegram channel and group content. "
                "Queries t.me links, Telegram aggregator sites (tgstat.com, telemetr.io), "
                "and general web results mentioning Telegram channels. "
                "Does NOT access Telegram's private API — only publicly indexed content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms to find Telegram content"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "darknet_market_search",
            "description": (
                "Search for darknet market intelligence via publicly indexed OSINT sources. "
                "Queries clearnet darknet-market discussion sites (darknetlive.com, dark.fail, "
                "darknetmarkets.org, dread.support) and research databases (gwern.net). "
                "Does NOT access .onion sites directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms for darknet market OSINT"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "facebook_search",
            "description": (
                "Search public Facebook pages, groups, and posts via web search. "
                "Only finds content indexed by search engines — private groups and "
                "personal profiles are not accessible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "result_type": {
                        "type": "string",
                        "description": "Type of results: 'posts', 'groups', 'pages' (default: posts)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discord_search",
            "description": (
                "Search public Discord server content, server listings, and "
                "archived messages via web search. Private server content is "
                "not accessible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signal_search",
            "description": (
                "Search for Signal group links and public references. "
                "Signal is end-to-end encrypted; this only finds publicly-shared "
                "group invite links and references on indexable websites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatsapp_search",
            "description": (
                "Search for WhatsApp group invite links and public references. "
                "WhatsApp is end-to-end encrypted; this only finds publicly-shared "
                "group links and community references on indexable websites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crunchbase_search",
            "description": (
                "Search Crunchbase for company profiles, funding rounds, "
                "and organizational data. Uses web search targeting crunchbase.com."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company or search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trustpilot_search",
            "description": (
                "Search Trustpilot for business reviews, customer feedback, "
                "and trust scores. Uses web search targeting trustpilot.com."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Business name or search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whois_lookup",
            "description": (
                "Look up WHOIS/RDAP registration data for a domain. "
                "Returns registrar, creation date, nameservers, and registrant info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain name to look up (e.g. example.com)"},
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "onion_fetch",
            "description": (
                "Fetch a .onion page or any webpage through the Tor network. "
                "Routes through a local Tor SOCKS proxy for anonymous access. "
                "Use this for: (1) accessing .onion darknet sites directly, "
                "(2) accessing clearnet sites that block datacenter IPs, "
                "(3) accessing geo-restricted content. Tor connections are "
                "slow (10-30s) — use sparingly and only when other fetch "
                "methods fail or when .onion access is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The URL to fetch. Can be a .onion address "
                            "(e.g. http://example.onion/path) or a regular "
                            "clearnet URL (e.g. https://blocked-site.com)"
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grok_deep_search",
            "description": (
                "Perform a deep web + X/Twitter search using Grok 4.20's built-in "
                "search capabilities. This is a HIGH-QUALITY search tool that "
                "autonomously performs multiple web searches and X/Twitter searches, "
                "returning cited results with URLs. Use this as the PRIMARY search "
                "tool for any query — it often finds results that SearXNG misses, "
                "especially for recent events, social media discussions, and "
                "controversial/sensitive topics. Returns structured results with "
                "citations and source URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — be specific and detailed",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["web", "x", "both"],
                        "description": (
                            "Type of search: 'web' for web only, 'x' for "
                            "X/Twitter only, 'both' for both (default: both)"
                        ),
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Optional instructions for the search agent — e.g. "
                            "'focus on forum discussions', 'find vendor reviews', "
                            "'look for recent news from the last week'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gateway",
            "description": (
                "Unified search API gateway that fans out to ALL available search "
                "backends simultaneously: Grok deep search (web + X/Twitter), "
                "Apify (Reddit, Twitter, Telegram, Discord), SearXNG, forums, "
                "academic sources (PubMed, arXiv, Scholar), and archives "
                "(Archive.org, Wayback Machine). Results are deduplicated by URL "
                "and merged with trust scores. Use this for comprehensive multi-source "
                "research that requires breadth across many platforms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "sources": {
                        "type": "string",
                        "description": (
                            "Comma-separated source categories to query. Options: "
                            "'all' (default), 'grok', 'searxng', 'social', "
                            "'community', 'academic', 'archive', 'video'"
                        ),
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["web", "x", "both"],
                        "description": "Search type for Grok backend (default: both)",
                    },
                    "max_results_per_source": {
                        "type": "integer",
                        "description": "Max results per source category (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # -----------------------------------------------------------------------
    # Sicry Dark Web Search Tools (Tor/.onion access via Sicry MCP)
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "sicry_search",
            "description": (
                "Search the dark web using 18 Tor search engines simultaneously "
                "(Ahmia, OnionLand, Tor66, Torgle, etc.) via the Sicry OSINT layer. "
                "Returns deduplicated results with titles, .onion URLs, and source "
                "engine names. Use this for researching hidden services, darknet "
                "markets, underground forums, leaked databases, and any information "
                "only available on the Tor network. This is a REAL dark web search — "
                "not a clearnet proxy. Requires Tor daemon running."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords work best)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 20)",
                    },
                    "engines": {
                        "type": "string",
                        "description": (
                            "Comma-separated engine names to use (optional, default: all 18). "
                            "Options include: ahmia, torch, onionland, tor66, torgle, etc."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sicry_fetch",
            "description": (
                "Fetch the full text content of any .onion hidden service page "
                "or clearnet URL routed anonymously through Tor. Returns extracted "
                "text, title, status code, and links found on the page. Use after "
                "sicry_search to read the actual content of dark web pages. Also "
                "useful for fetching clearnet pages anonymously when direct access "
                "is blocked by geo-restrictions or censorship."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "URL to fetch — supports both .onion addresses "
                            "(e.g. http://example.onion/page) and clearnet URLs"
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sicry_check_tor",
            "description": (
                "Check if the Tor network is accessible and return the current "
                "exit node IP address. Use this to diagnose connectivity issues "
                "before attempting dark web searches. If Tor is not active, other "
                "sicry_* tools will fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sicry_renew_identity",
            "description": (
                "Rotate the Tor circuit to get a new exit node IP address. "
                "Use this when a hidden service blocks the current exit node "
                "or when you want to appear as a different user. Takes ~5-10 "
                "seconds for the new circuit to establish."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ============================================================================
# LangChain Tool Definitions (for bind_tools + callback tracking)
# ============================================================================
# Convert NATIVE_TOOLS (OpenAI function-calling format) into the format
# that ChatOpenAI.bind_tools() expects.  We also build a registry so
# execute_tool can fire on_tool_start / on_tool_end callbacks.

LANGCHAIN_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": t["function"],
    }
    for t in NATIVE_TOOLS
]


