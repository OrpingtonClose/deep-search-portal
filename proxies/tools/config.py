"""
Shared configuration: environment variables, constants, LLM factories,
logging, and per-request shared state used across all tools/ modules.

This module is imported by every other module in the tools/ package.
It reads from the environment at import time (module-level), so all
env vars must be set before the first import.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_openai import ChatOpenAI

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    env_int,
    require_env,
    setup_logging,
)
from research_metrics import MetricsCollector

# --- Logging ---
LOG_DIR = os.getenv("PERSISTENT_RESEARCH_LOG_DIR", "/opt/persistent_research_logs")
log = setup_logging("persistent-research", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "mistral-small-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("PERSISTENT_RESEARCH_PORT", 9300, minimum=1)
PORTAL_PUBLIC_URL = os.getenv("PORTAL_PUBLIC_URL", "").rstrip("/")
GATEWAY_INTERNAL_URL = os.getenv(
    "GATEWAY_INTERNAL_URL",
    os.getenv("OWUI_INTERNAL_URL", "http://localhost:3000"),
)


# --- LangChain Model Factories ---

def _get_llm(
    model: str = "",
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: float = 300.0,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance pointing at the Mistral API.

    Note: We pass max_tokens via extra_body instead of the native parameter
    because langchain-openai >=1.0 converts max_tokens to
    max_completion_tokens, which the Mistral API rejects with a 422.
    """
    return ChatOpenAI(
        model=model or UPSTREAM_MODEL,
        api_key=UPSTREAM_KEY,
        base_url=UPSTREAM_BASE,
        temperature=temperature,
        timeout=timeout,
        extra_body={"max_tokens": max_tokens},
    )


def _get_synthesis_llm(**kwargs: Any) -> ChatOpenAI:
    """LLM for synthesis / revision (upstream large model)."""
    return _get_llm(model=UPSTREAM_MODEL, max_tokens=8192, temperature=0.3, **kwargs)


def _get_subagent_llm(**kwargs: Any) -> ChatOpenAI:
    """LLM for subagents, heartbeat, relevance gate (small/fast model)."""
    return _get_llm(model=SUBAGENT_MODEL, max_tokens=4096, temperature=0.3, **kwargs)


# --- Numeric / string constants ---
MAX_SUBAGENT_TURNS = env_int("MAX_SUBAGENT_TURNS", 15, minimum=1)
MAX_CONCURRENT = env_int("MAX_CONCURRENT_PERSISTENT", 2, minimum=1)
RESEARCH_NAMESPACE = os.getenv("RESEARCH_NAMESPACE", "research")
JSONL_LOG_DIR = os.getenv("JSONL_LOG_DIR", "/opt/persistent_research_logs/jsonl")
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000
MAX_PRIOR_CONDITIONS = 20

# Saturation detection thresholds
NOVELTY_EXPAND_THRESHOLD = 0.3
NOVELTY_STOP_THRESHOLD = 0.05

# Legacy constants used by plan_research() and run_subagent() recursive spawning
MAX_SUBAGENTS = env_int("MAX_SUBAGENTS", 7, minimum=1)
MAX_RECURSIVE_DEPTH = env_int("MAX_RECURSIVE_DEPTH", 2, minimum=0)

# --- Tree Research Reactor Config ---
TREE_MAX_CONCURRENT = env_int("TREE_MAX_CONCURRENT", 10, minimum=1)
TREE_MAX_DEPTH = env_int("TREE_MAX_DEPTH", 5, minimum=1)
TREE_MAX_NODES = env_int("TREE_MAX_NODES", 50, minimum=5)
TREE_PRESSURE_THRESHOLD = float(os.getenv("TREE_PRESSURE_THRESHOLD", "0.15"))
TREE_WORKER_IDLE_TIMEOUT = float(os.getenv("TREE_WORKER_IDLE_TIMEOUT", "60.0"))
RESEARCH_TIME_LIMIT = float(os.getenv("RESEARCH_TIME_LIMIT", "300"))  # seconds; 0 = no limit
# Hard pipeline-level wall-clock timeout.  When exceeded, ALL remaining
# phases (entities, verify, reflect, persist) are skipped and synthesis
# runs immediately with whatever findings exist.  Separate from
# RESEARCH_TIME_LIMIT which only gates tree exploration.
PIPELINE_HARD_TIMEOUT = float(os.getenv("PIPELINE_HARD_TIMEOUT", "0"))  # 0 = derive from RESEARCH_TIME_LIMIT

# --- Enhanced Web Scraping Config ---
APIFY_API_KEY = os.getenv("APIFY_API_TOKEN", "")
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY", "")
BRIGHT_DATA_HOST = os.getenv("BRIGHT_DATA_HOST", "")
BRIGHT_DATA_CUSTOMER_ID = os.getenv("BRIGHT_DATA_CUSTOMER_ID", "hl_dc044bf4")
BRIGHT_DATA_ZONE = os.getenv("BRIGHT_DATA_ZONE", "mcp_unlocker")
OXYLABS_USERNAME = os.getenv("OXYLABS_USERNAME", "")
OXYLABS_PASSWORD = os.getenv("OXYLABS_PASSWORD", "")

# Playwright JS rendering: auto-detected at import time
try:
    from playwright.async_api import async_playwright  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

# Selenium fallback for JS rendering when Playwright is unavailable
try:
    from selenium import webdriver  # noqa: F401
    from selenium.webdriver.chrome.options import Options as ChromeOptions  # noqa: F401
    from selenium.webdriver.chrome.service import Service as ChromeService  # noqa: F401
    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False

# Veritas integration
VERITAS_VERIFY_ENABLED = os.getenv("VERITAS_VERIFY_ENABLED", "true").lower() in ("1", "true", "yes")
VERITAS_MIN_CONDITIONS = env_int("VERITAS_MIN_CONDITIONS", 3, minimum=1)
VERITAS_HALLUCINATION_THRESHOLD = float(os.getenv("VERITAS_HALLUCINATION_THRESHOLD", "0.3"))

# Commercial search APIs
COMMERCIAL_SEARCH_ENABLED = os.getenv("COMMERCIAL_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")
BRIGHT_DATA_SERP_ZONE = os.getenv("BRIGHT_DATA_SERP_ZONE", "mcp_unlocker")
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "mistral-small-latest")

log.info(
    f"Config: synthesis_model={UPSTREAM_MODEL}, subagent_model={SUBAGENT_MODEL}, "
    f"upstream={UPSTREAM_BASE}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, "
    f"tree_concurrent={TREE_MAX_CONCURRENT}, tree_depth={TREE_MAX_DEPTH}, "
    f"tree_nodes={TREE_MAX_NODES}, subagent_turns={MAX_SUBAGENT_TURNS}, "
    f"research_ns={RESEARCH_NAMESPACE}, "
    f"apify={'yes' if APIFY_API_KEY else 'no'}, "
    f"bright_data={'yes' if BRIGHT_DATA_API_KEY else 'no'}, "
    f"oxylabs={'yes' if OXYLABS_USERNAME else 'no'}, "
    f"playwright={'yes' if _PLAYWRIGHT_AVAILABLE else 'no'}, "
    f"selenium={'yes' if _SELENIUM_AVAILABLE else 'no'}"
)

# --- Shared state ---
tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)

# Sentinel used to signal the SSE output queue that the pipeline is done.
_STREAM_DONE = object()

# Per-request live findings collectors, keyed by req_id.
_live_collectors: dict = {}

# Per-request curated event queues for tree reactor -> heartbeat.
_curated_queues: dict[str, asyncio.Queue] = {}

# Per-request metrics collectors, keyed by req_id.
_metrics_collectors: dict[str, MetricsCollector] = {}
