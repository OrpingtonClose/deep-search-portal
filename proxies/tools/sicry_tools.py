"""
Sicry Dark Web Search Integration — Tor/.onion access layer for the research pipeline.

Wraps Sicry's 6 tools as async functions matching the tool executor pattern.
Sicry provides:
  - sicry_search: Search 18 dark web engines simultaneously (Ahmia, Tor66, etc.)
  - sicry_fetch: Fetch any .onion or clearnet URL through Tor
  - sicry_check_tor: Verify Tor connectivity and get exit IP
  - sicry_renew_identity: Rotate Tor circuit for a new exit node

Sicry is expected to be installed at SICRY_PATH (default: /opt/sicry/sicry.py).
Tor daemon must be running on SOCKS port 9050 with ControlPort 9051.

If Sicry is not installed, all tools gracefully degrade to [TOOL_ERROR] messages
explaining how to set it up.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Optional

log = logging.getLogger("sicry_tools")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SICRY_PATH = os.getenv("SICRY_PATH", "/opt/sicry/sicry.py")
TOR_SOCKS_PROXY = os.getenv("TOR_SOCKS_PROXY", "socks5h://127.0.0.1:9050")

# Lazy-loaded Sicry module reference
_sicry_module = None
_sicry_available: Optional[bool] = None


def _load_sicry():
    """Attempt to load the Sicry module. Returns True if available."""
    global _sicry_module, _sicry_available

    if _sicry_available is not None:
        return _sicry_available

    if not os.path.isfile(SICRY_PATH):
        log.warning(
            f"Sicry not found at {SICRY_PATH}. "
            "Dark web search will be unavailable. "
            "Install: git clone https://github.com/JacobJandon/Sicry /opt/sicry && "
            "pip install requests[socks] beautifulsoup4 python-dotenv stem"
        )
        _sicry_available = False
        return False

    # Add Sicry's directory to sys.path so we can import it
    sicry_dir = os.path.dirname(SICRY_PATH)
    if sicry_dir not in sys.path:
        sys.path.insert(0, sicry_dir)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sicry", SICRY_PATH)
        if spec is None or spec.loader is None:
            log.error(f"Failed to create module spec from {SICRY_PATH}")
            _sicry_available = False
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _sicry_module = module
        _sicry_available = True
        log.info(f"Sicry loaded from {SICRY_PATH}")
        return True
    except Exception as e:
        log.error(f"Failed to load Sicry from {SICRY_PATH}: {e}")
        _sicry_available = False
        return False


def _not_available_msg(tool_name: str) -> str:
    """Return a standardized error message when Sicry is not available."""
    return (
        f"[TOOL_ERROR] {tool_name}: Sicry dark web module not available. "
        f"Expected at {SICRY_PATH}. "
        "Install: git clone https://github.com/JacobJandon/Sicry /opt/sicry && "
        "pip install requests[socks] beautifulsoup4 python-dotenv stem && "
        "apt install tor && tor &"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def tool_sicry_search(
    query: str,
    max_results: int = 20,
    engines: Optional[str] = None,
) -> str:
    """Search 18 dark web search engines via Sicry.

    Searches Ahmia, Tor66, OnionLand, Torgle, and 14 other dark web indexes
    simultaneously. Returns deduplicated results with titles, URLs, and source
    engine names.

    Args:
        query: Search query (keywords work best — Sicry refines internally).
        max_results: Maximum results to return (default 20).
        engines: Comma-separated engine names (optional, default: all 18).

    Returns:
        Formatted search results or [TOOL_ERROR] message.
    """
    if not _load_sicry():
        return _not_available_msg("sicry_search")

    try:
        loop = asyncio.get_running_loop()
        engine_list = [e.strip() for e in engines.split(",")] if engines else None

        results = await loop.run_in_executor(
            None,
            lambda: _sicry_module.search(
                query,
                engines=engine_list,
                max_results=max_results,
            ),
        )

        if not results:
            return f"Sicry dark web search returned 0 results for: {query}"

        # Format results
        lines = [f"**Dark Web Search Results ({len(results)} found)** — query: {query}\n"]
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            engine = r.get("engine", "unknown")
            lines.append(f"{i}. [{title}]({url}) — via {engine}")

        return "\n".join(lines)
    except Exception as e:
        return f"[TOOL_ERROR] sicry_search failed: {e}"


async def tool_sicry_fetch(url: str) -> str:
    """Fetch any URL (including .onion) through Tor via Sicry.

    Retrieves the full text content of a page, with HTML stripped, links
    extracted, and content truncated to 8000 chars. Works for both .onion
    hidden services and clearnet URLs routed through Tor for anonymity.

    Args:
        url: The URL to fetch (http:// or https://, .onion supported).

    Returns:
        Page content (title + text + links) or [TOOL_ERROR] message.
    """
    if not _load_sicry():
        return _not_available_msg("sicry_fetch")

    if not url or not url.startswith(("http://", "https://")):
        return "[TOOL_ERROR] sicry_fetch requires a valid http:// or https:// URL."

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _sicry_module.fetch(url),
        )

        if not result:
            return f"[TOOL_ERROR] sicry_fetch returned empty for: {url}"

        if result.get("error"):
            return f"[TOOL_ERROR] sicry_fetch error for {url}: {result['error']}"

        # Format the page content
        title = result.get("title", "")
        text = result.get("text", "")
        is_onion = result.get("is_onion", False)
        status = result.get("status", 0)
        links = result.get("links", [])

        parts = []
        if title:
            parts.append(f"**{title}**")
        parts.append(f"URL: {url} (status: {status}, onion: {is_onion})")
        if text:
            parts.append(f"\n{text}")
        if links:
            link_strs = [f"- [{l.get('text', 'link')}]({l.get('href', '')})" for l in links[:20]]
            parts.append(f"\n**Links ({len(links)} total, showing first 20):**\n" + "\n".join(link_strs))

        return "\n".join(parts)
    except Exception as e:
        return f"[TOOL_ERROR] sicry_fetch failed for {url}: {e}"


async def tool_sicry_check_tor() -> str:
    """Check Tor connectivity and return the current exit IP.

    Verifies that the Tor daemon is running and the machine is routing
    through a Tor exit node. Call this to diagnose connectivity issues.

    Returns:
        Tor status (active/inactive, exit IP) or [TOOL_ERROR] message.
    """
    if not _load_sicry():
        return _not_available_msg("sicry_check_tor")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _sicry_module.check_tor)

        if not result:
            return "[TOOL_ERROR] sicry_check_tor returned empty."

        active = result.get("tor_active", False)
        exit_ip = result.get("exit_ip", "unknown")
        error = result.get("error")

        if active:
            return f"Tor is ACTIVE. Exit IP: {exit_ip}."
        else:
            return f"[TOOL_ERROR] Tor is NOT active. Error: {error or 'connection failed'}."
    except Exception as e:
        return f"[TOOL_ERROR] sicry_check_tor failed: {e}"


async def tool_sicry_renew_identity() -> str:
    """Rotate the Tor circuit to get a new exit node and fresh identity.

    Useful when a hidden service blocks the current exit IP or when you
    want to appear as a different user.

    Returns:
        Success/failure status or [TOOL_ERROR] message.
    """
    if not _load_sicry():
        return _not_available_msg("sicry_renew_identity")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _sicry_module.renew_identity)

        if not result:
            return "[TOOL_ERROR] sicry_renew_identity returned empty."

        if result.get("success"):
            return "Tor identity renewed successfully. New circuit established."
        else:
            error = result.get("error", "unknown error")
            return f"[TOOL_ERROR] Failed to renew Tor identity: {error}"
    except Exception as e:
        return f"[TOOL_ERROR] sicry_renew_identity failed: {e}"


async def tool_sicry_scrape_all(urls_json: str) -> str:
    """Batch-fetch multiple dark web pages concurrently via Sicry.

    Takes a JSON array of search result objects (from sicry_search) and
    fetches all pages in parallel. Content is capped at 2000 chars per page.

    Args:
        urls_json: JSON array of {"title": ..., "url": ...} objects, or
                   a JSON array of URL strings.

    Returns:
        Combined page contents or [TOOL_ERROR] message.
    """
    if not _load_sicry():
        return _not_available_msg("sicry_scrape_all")

    try:
        url_list = json.loads(urls_json)
    except (json.JSONDecodeError, TypeError):
        return "[TOOL_ERROR] sicry_scrape_all: invalid JSON input. Expected array of {title, url} objects."

    # Normalize input — accept both [{"url": ...}] and ["url1", "url2"]
    if url_list and isinstance(url_list[0], str):
        url_list = [{"title": "", "url": u} for u in url_list]

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _sicry_module.scrape_all(url_list),
        )

        if not results:
            return "Sicry batch scrape returned 0 pages."

        parts = [f"**Batch Scrape Results ({len(results)} pages fetched)**\n"]
        for url, content in results.items():
            parts.append(f"--- {url} ---\n{content}\n")

        return "\n".join(parts)
    except Exception as e:
        return f"[TOOL_ERROR] sicry_scrape_all failed: {e}"
