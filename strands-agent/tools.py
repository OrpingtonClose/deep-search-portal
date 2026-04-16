# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""
MCP tool wiring for the Strands Venice agent.

Wires up all 4 search tools as MCP servers using Strands' native MCP
support.  Each tool family is conditionally loaded based on whether
its API key is configured.
"""

import os

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from strands.tools.mcp import MCPClient


def _full_env(**overrides):
    """Return a copy of the current environment with *overrides* applied.

    MCP server subprocesses inherit PATH, HOME, etc. so that ``npx`` and
    other tools resolve correctly.
    """
    env = dict(os.environ)
    env.update(overrides)
    return env


# ── Brave Search MCP ─────────────────────────────────────────────────
# npm: @brave/brave-search-mcp-server  (MIT, brave/brave-search-mcp-server)
# Tools: brave_web_search, brave_local_search, brave_image_search,
#   brave_video_search, brave_news_search, brave_summarizer
# Increase startup_timeout for slow npx/uvx downloads on staging VMs.
_MCP_STARTUP_TIMEOUT = int(os.environ.get("MCP_STARTUP_TIMEOUT", "120"))

brave_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@brave/brave-search-mcp-server"],
            env=_full_env(BRAVE_API_KEY=os.environ.get("BRAVE_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Firecrawl MCP ────────────────────────────────────────────────────
# npm: firecrawl-mcp  (MIT, firecrawl/firecrawl-mcp-server)
# Tools: firecrawl_scrape, firecrawl_crawl, firecrawl_map,
#   firecrawl_search, firecrawl_extract
firecrawl_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env=_full_env(FIRECRAWL_API_KEY=os.environ.get("FIRECRAWL_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Exa MCP ──────────────────────────────────────────────────────────
# npm: exa-mcp-server  (MIT, exa-labs/exa-mcp-server)
# Tools: web_search_exa, web_search_advanced_exa, crawling_exa,
#   get_code_context_exa
# Requires: npm install -g exa-mcp-server
exa_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="node",
            args=[
                "-e",
                # Bootstrap Smithery entry-point with config that enables
                # ALL non-deprecated Exa tools.  Smithery reads config from
                # process.argv.slice(2) as key=value pairs.
                "process.argv[2]='enabledTools=web_search_exa,web_search_advanced_exa,crawling_exa,get_code_context_exa';"
                "const r=require('child_process').execSync('npm root -g',{encoding:'utf8'}).trim();"
                "require(r+'/exa-mcp-server/.smithery/stdio/index.cjs');",
            ],
            env=_full_env(EXA_API_KEY=os.environ.get("EXA_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Kagi MCP ─────────────────────────────────────────────────────────
# uvx: kagimcp  (MIT, kagisearch/kagimcp)
# Tools: kagi_search, kagi_summarize, kagi_fastgpt, kagi_enrich_web,
#   kagi_enrich_news
kagi_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["kagimcp"],
            env=_full_env(KAGI_API_KEY=os.environ.get("KAGI_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Qualitative Research MCP ──────────────────────────────────────────
# GitHub: tejpalvirk/qualitativeresearch  (TypeScript, knowledge-graph MCP)
# Tools: startsession, loadcontext, endsession, buildcontext,
#   deletecontext, advancedcontext, getProjectOverview, getThematicAnalysis,
#   getCodedData, getMemos, getMethodology, getChronologicalData,
#   getCodeCoOccurrence + CRUD for projects/participants/interviews/
#   observations/codes/themes/findings/memos/researchQuestions
# No API key required — runs as a local Node.js process with file-based
# knowledge graph storage.
_QUAL_RESEARCH_DATA_DIR = os.environ.get(
    "QUAL_RESEARCH_DATA_DIR", "/opt/qualitative-research"
)
_QUAL_RESEARCH_DIR = os.environ.get(
    "QUAL_RESEARCH_DIR", "/opt/qualitativeresearch"
)
_QUAL_RESEARCH_INDEX = os.path.join(_QUAL_RESEARCH_DIR, "index.js")

qualitative_research_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="node",
            args=[_QUAL_RESEARCH_INDEX],
            env=_full_env(
                MEMORY_FILE_PATH=os.path.join(_QUAL_RESEARCH_DATA_DIR, "memory.json"),
                SESSIONS_FILE_PATH=os.path.join(_QUAL_RESEARCH_DATA_DIR, "sessions.json"),
            ),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Registry mapping ─────────────────────────────────────────────────
# API-key-gated tools: only loaded when the corresponding key is set.
_MCP_REGISTRY = {
    "BRAVE_API_KEY": brave_mcp,
    "FIRECRAWL_API_KEY": firecrawl_mcp,
    "EXA_API_KEY": exa_mcp,
    "KAGI_API_KEY": kagi_mcp,
}

# Always-on tools: loaded unconditionally (no API key required).
_ALWAYS_ON_MCP = [qualitative_research_mcp]


def get_all_mcp_clients():
    """Return list of MCP clients whose API keys are configured, plus always-on tools."""
    clients = list(_ALWAYS_ON_MCP)
    for env_var, client in _MCP_REGISTRY.items():
        if os.environ.get(env_var):
            clients.append(client)
    return clients
