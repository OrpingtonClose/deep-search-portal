"""Decomposed tools package for the persistent deep research proxy.

This package breaks the monolith persistent_deep_research_proxy.py into
testable, independently-importable modules:

  models       - AtomicCondition, SubagentResult, ResearchNode dataclasses
  scoring      - trust_score_url, serendipity_score
  config       - Environment variables and configuration constants
  llm          - LLM factories, call_llm, message conversion
  persistence  - JSONL logging, Neo4j storage/retrieval, document ingestion
  tool_defs    - NATIVE_TOOLS, LANGCHAIN_TOOLS definitions
  moderation   - Mistral moderation gate, commercial SERP APIs
  web_fetch    - Multi-tier web fetch (httpx, Playwright, Selenium, proxies, Wayback)
  search_tools - All search tool implementations (30+ tools)
  tool_executor- execute_tool, execute_tools_parallel, retry, PDF extraction
  verification - verify_conditions, Veritas Inquisitor integration
  planning     - plan_research, reflect_on_conditions
  subagent     - run_subagent, _parse_conditions
  tree_reactor - tree_research_reactor with pressure-based spawning
  heartbeat    - LiveFindingsCollector, heartbeat generation
  synthesis    - synthesize_with_revision, relevance_gate, strip_moralizing
  pipeline     - LangGraph state, pipeline nodes, graph builder
"""

import sys

from .models import AtomicCondition, SubagentResult, ResearchNode
from .scoring import trust_score_url, serendipity_score
from .config import *  # noqa: F401,F403 - re-export all config constants
from .tool_defs import NATIVE_TOOLS, LANGCHAIN_TOOLS


def _get_http_client():
    """Look up http_client through the proxy module at call time.

    Tests mock ``shared`` and then ``patch.object(proxy, 'http_client', ...)``.
    A direct ``from shared import http_client`` binds at import time and
    misses the per-test patch.  This helper resolves the name through the
    proxy module (if already loaded) so the patched version is used.
    """
    proxy = sys.modules.get("persistent_deep_research_proxy")
    if proxy is not None and hasattr(proxy, "http_client"):
        return proxy.http_client()
    # Fallback: resolve from shared directly
    return sys.modules["shared"].http_client()
