"""Health-aware router — reads tool_health.py state, skips degraded services.

Integrates with the existing ToolHealthMonitor to make routing decisions.
Unhealthy MCP servers are skipped in the capability matrix ordering so the
dispatcher falls through to the next available provider.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional

# Add proxies to path so we can import the health monitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proxies"))

log = logging.getLogger("search-dispatcher.router")


def _get_monitor():
    """Lazily import and return the singleton ToolHealthMonitor."""
    from tools.tool_health import get_monitor

    return get_monitor()


def is_server_healthy(server_name: str) -> bool:
    """Check if an MCP server is considered healthy.

    Reads the ToolHealthMonitor's rolling statistics for the server.
    A server is unhealthy if it has hit the consecutive failure threshold
    or its rolling failure rate exceeds the alert threshold.
    """
    try:
        monitor = _get_monitor()
        status = monitor.get_tool_status(server_name)
        return status.get("status") in ("healthy", "unknown")
    except Exception:
        # If we can't read health status, assume healthy (fail open)
        return True


def filter_healthy_servers(
    server_tool_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Filter a capability matrix entry to only healthy servers.

    Given an ordered list of (server_name, tool_name) pairs from the
    capability matrix, returns only those whose server is currently healthy.
    If ALL servers are unhealthy, returns the original list unchanged
    (fail-open: better to try a degraded server than return nothing).
    """
    healthy = [
        (server, tool)
        for server, tool in server_tool_pairs
        if is_server_healthy(server)
    ]

    if not healthy:
        log.warning(
            "All servers unhealthy for tool chain %s — failing open",
            [s for s, _ in server_tool_pairs],
        )
        return server_tool_pairs

    return healthy


def get_server_health_summary() -> dict[str, dict]:
    """Return a summary of all MCP server health statuses.

    Used by the dispatcher's /health endpoint to expose server status.
    """
    try:
        monitor = _get_monitor()
        return monitor.get_all_status()
    except Exception as e:
        log.error("Failed to get health summary: %s", e)
        return {}


def record_server_outcome(
    server_name: str, success: bool, error: Optional[str] = None
) -> None:
    """Record a success or failure for a specific MCP server.

    Called by the dispatcher after each tool invocation so the health
    monitor can update its rolling statistics.
    """
    try:
        monitor = _get_monitor()
        if success:
            monitor.record_success(server_name)
        else:
            monitor.record_failure(server_name, error or "unknown error")
    except Exception as e:
        log.debug("Failed to record outcome for %s: %s", server_name, e)
