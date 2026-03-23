"""
Search Result Cache — SQLite-backed cache for tool results with TTL and query normalization.

Prevents redundant network requests when multiple subagents issue similar queries.
Results are keyed by (tool_name, normalized_query) and expire after a configurable TTL.

Query normalization:
  - Lowercase
  - Strip extra whitespace
  - Sort words alphabetically (catches reordered queries)
  - Remove common stop words for fuzzy matching

Configuration (environment variables):
  SEARCH_CACHE_DB        — path to SQLite database (default: /tmp/search_cache.db)
  SEARCH_CACHE_TTL_SEC   — default TTL in seconds (default: 3600 = 1 hour)
  SEARCH_CACHE_STATIC_TTL — TTL for static content like Wikipedia (default: 86400 = 24h)
  SEARCH_CACHE_ENABLED   — set to "0" or "false" to disable (default: enabled)
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Optional

log = logging.getLogger("search_cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DB_PATH = os.getenv("SEARCH_CACHE_DB", "/tmp/search_cache.db")
DEFAULT_TTL = int(os.getenv("SEARCH_CACHE_TTL_SEC", "3600"))  # 1 hour
STATIC_TTL = int(os.getenv("SEARCH_CACHE_STATIC_TTL", "86400"))  # 24 hours
CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "1").lower() not in ("0", "false", "no")

# Tools whose results change slowly (long TTL)
_STATIC_TOOLS = {
    "wikipedia_search", "wikidata_query", "arxiv_search", "pubmed_search",
    "whois_lookup", "archiveorg_search",
}

# Tools that should NOT be cached (side-effectful or time-sensitive)
_UNCACHEABLE_TOOLS = {
    "python_exec", "fetch_webpage", "youtube_video_analyze",
    "knowledge_graph_search", "knowledge_discover",
}

# Stop words for query normalization
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "what", "which", "who", "how", "and", "but", "or", "if", "not",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she",
    "her", "it", "its", "they", "them", "their", "this", "that",
})

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_hits = 0
_misses = 0
_evictions = 0

# ---------------------------------------------------------------------------
# Thread-local SQLite connections
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the DB if needed."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(CACHE_DB_PATH) or "/tmp", exist_ok=True)
        conn = sqlite3.connect(CACHE_DB_PATH, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                cache_key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                raw_query TEXT NOT NULL,
                normalized_query TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON search_cache(expires_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_tool
            ON search_cache(tool_name)
        """)
        conn.commit()
        _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def normalize_query(query: str) -> str:
    """Normalize a search query for cache key generation.

    Steps:
      1. Lowercase
      2. Remove punctuation and extra whitespace
      3. Remove stop words
      4. Sort remaining words alphabetically
    """
    lower = query.lower().strip()
    # Remove punctuation except hyphens in compound words
    cleaned = re.sub(r"[^\w\s-]", " ", lower)
    words = cleaned.split()
    # Remove stop words and very short tokens
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    # Sort for order-independent matching
    meaningful.sort()
    return " ".join(meaningful)


def _is_url_like(text: str) -> bool:
    """Check if text looks like a URL or JSON (should not be normalized)."""
    stripped = text.strip()
    return (
        stripped.startswith("http://")
        or stripped.startswith("https://")
        or stripped.startswith("{")
    )


def _cache_key(tool_name: str, query: str) -> str:
    """Generate a cache key from tool name and query.

    For URL-like or JSON inputs (e.g. from wayback_fetch, youtube_transcript,
    or full argument dicts), uses the raw string to preserve structure.
    For natural-language queries, normalizes to catch near-duplicates.
    """
    if _is_url_like(query):
        key_input = query.strip()
    else:
        key_input = normalize_query(query)
    raw = f"{tool_name}:{key_input}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_get(tool_name: str, query: str) -> Optional[str]:
    """Look up a cached result. Returns None on miss or expired entry."""
    global _hits, _misses

    if not CACHE_ENABLED or tool_name in _UNCACHEABLE_TOOLS:
        return None

    key = _cache_key(tool_name, query)
    now = time.time()

    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT result, expires_at FROM search_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

        if row is None:
            _misses += 1
            return None

        result, expires_at = row
        if now > expires_at:
            # Expired — delete and return miss
            conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
            conn.commit()
            _misses += 1
            return None

        # Cache hit — update hit count
        conn.execute(
            "UPDATE search_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,),
        )
        conn.commit()
        _hits += 1
        log.debug(f"Cache HIT for {tool_name}: {query[:60]}")
        return result

    except Exception as e:
        log.warning(f"Cache get error: {e}")
        _misses += 1
        return None


def cache_put(tool_name: str, query: str, result: str) -> None:
    """Store a result in the cache with appropriate TTL."""
    if not CACHE_ENABLED or tool_name in _UNCACHEABLE_TOOLS:
        return

    # Don't cache error results or very short results
    if len(result) < 50:
        return
    lower_prefix = result.lower()[:80]
    if "error" in lower_prefix or "failed" in lower_prefix or "timed out" in lower_prefix:
        return

    ttl = STATIC_TTL if tool_name in _STATIC_TOOLS else DEFAULT_TTL
    key = _cache_key(tool_name, query)
    now = time.time()

    try:
        conn = _get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO search_cache
               (cache_key, tool_name, raw_query, normalized_query, result, created_at, expires_at, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (key, tool_name, query[:500], normalize_query(query), result, now, now + ttl),
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Cache put error: {e}")


def cache_evict_expired() -> int:
    """Remove all expired entries. Returns number evicted."""
    global _evictions
    try:
        conn = _get_conn()
        cursor = conn.execute(
            "DELETE FROM search_cache WHERE expires_at < ?", (time.time(),)
        )
        evicted = cursor.rowcount
        conn.commit()
        _evictions += evicted
        return evicted
    except Exception as e:
        log.warning(f"Cache eviction error: {e}")
        return 0


def cache_stats() -> dict:
    """Return cache statistics for health endpoints."""
    total_requests = _hits + _misses
    hit_rate = (_hits / total_requests * 100) if total_requests > 0 else 0.0

    size = 0
    try:
        conn = _get_conn()
        row = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()
        size = row[0] if row else 0
    except Exception:
        pass

    return {
        "enabled": CACHE_ENABLED,
        "hits": _hits,
        "misses": _misses,
        "hit_rate_pct": round(hit_rate, 1),
        "evictions": _evictions,
        "entries": size,
        "db_path": CACHE_DB_PATH,
        "default_ttl_sec": DEFAULT_TTL,
        "static_ttl_sec": STATIC_TTL,
    }


def cache_clear() -> int:
    """Clear all cache entries. Returns number deleted."""
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM search_cache")
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        log.warning(f"Cache clear error: {e}")
        return 0
