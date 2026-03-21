"""Shared configuration constants for the research pipeline.

All environment-variable-driven configuration lives here so that
every sub-module can import a single source of truth.
"""
import os

from shared import env_int, require_env, setup_logging

# --- Logging ---
LOG_DIR = os.getenv("PERSISTENT_RESEARCH_LOG_DIR", "/opt/persistent_research_logs")
log = setup_logging("persistent-research", LOG_DIR)

# --- Upstream LLM ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "mistral-small-latest")
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "mistral-small-latest")

# --- Network ---
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("PERSISTENT_RESEARCH_PORT", 9300, minimum=1)
PORTAL_PUBLIC_URL = os.getenv("PORTAL_PUBLIC_URL", "").rstrip("/")
OWUI_INTERNAL_URL = os.getenv("OWUI_INTERNAL_URL", "http://localhost:3000")

# --- Concurrency / limits ---
MAX_SUBAGENT_TURNS = env_int("MAX_SUBAGENT_TURNS", 15, minimum=1)
MAX_CONCURRENT = env_int("MAX_CONCURRENT_PERSISTENT", 2, minimum=1)
RESEARCH_NAMESPACE = os.getenv("RESEARCH_NAMESPACE", "research")
JSONL_LOG_DIR = os.getenv("JSONL_LOG_DIR", "/opt/persistent_research_logs/jsonl")
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000
MAX_PRIOR_CONDITIONS = 20

# --- Saturation detection ---
NOVELTY_EXPAND_THRESHOLD = 0.3
NOVELTY_STOP_THRESHOLD = 0.05

# --- Legacy constants ---
MAX_SUBAGENTS = env_int("MAX_SUBAGENTS", 7, minimum=1)
MAX_RECURSIVE_DEPTH = env_int("MAX_RECURSIVE_DEPTH", 2, minimum=0)

# --- Tree Research Reactor ---
TREE_MAX_CONCURRENT = env_int("TREE_MAX_CONCURRENT", 10, minimum=1)
TREE_MAX_DEPTH = env_int("TREE_MAX_DEPTH", 5, minimum=1)
TREE_MAX_NODES = env_int("TREE_MAX_NODES", 50, minimum=5)
TREE_PRESSURE_THRESHOLD = float(os.getenv("TREE_PRESSURE_THRESHOLD", "0.15"))
TREE_WORKER_IDLE_TIMEOUT = float(os.getenv("TREE_WORKER_IDLE_TIMEOUT", "60.0"))

# --- Commercial scraping ---
APIFY_API_KEY = os.getenv("APIFY_API_TOKEN", "")
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY", "")
BRIGHT_DATA_HOST = os.getenv("BRIGHT_DATA_HOST", "")
BRIGHT_DATA_CUSTOMER_ID = os.getenv("BRIGHT_DATA_CUSTOMER_ID", "hl_dc044bf4")
BRIGHT_DATA_ZONE = os.getenv("BRIGHT_DATA_ZONE", "web_unlocker1")
BRIGHT_DATA_SERP_ZONE = os.getenv("BRIGHT_DATA_SERP_ZONE", "serp")
OXYLABS_USERNAME = os.getenv("OXYLABS_USERNAME", "")
OXYLABS_PASSWORD = os.getenv("OXYLABS_PASSWORD", "")

# --- Veritas ---
VERITAS_VERIFY_ENABLED = os.getenv("VERITAS_VERIFY_ENABLED", "true").lower() in ("1", "true", "yes")
VERITAS_MIN_CONDITIONS = env_int("VERITAS_MIN_CONDITIONS", 3, minimum=1)
VERITAS_HALLUCINATION_THRESHOLD = float(os.getenv("VERITAS_HALLUCINATION_THRESHOLD", "0.3"))

# --- Commercial search ---
COMMERCIAL_SEARCH_ENABLED = os.getenv("COMMERCIAL_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")

# --- Optional heavy dependencies ---
try:
    from playwright.async_api import async_playwright  # noqa: F401
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    async_playwright = None  # type: ignore[assignment, misc]
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver  # noqa: F401
    from selenium.webdriver.chrome.options import Options as ChromeOptions  # noqa: F401
    SELENIUM_AVAILABLE = True
except ImportError:
    webdriver = None  # type: ignore[assignment]
    ChromeOptions = None  # type: ignore[assignment, misc]
    SELENIUM_AVAILABLE = False
