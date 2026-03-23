"""
Tool Health Monitor — real-time failure detection with LLM-driven root cause analysis.

Tracks success/failure rates for every tool in the research engine.  When a tool
fails repeatedly (3+ consecutive failures), an LLM is asked to diagnose the
probable root cause.  All issues are stored in a SQLite database and exposed
via a ``/tool-health`` REST endpoint so the user can inspect them.

Architecture:
  - ``record_outcome()`` — called after every tool execution (success or failure)
  - ``ToolHealthMonitor`` — singleton that manages failure streaks and triggers analysis
  - Issues stored in SQLite ``tool_issues`` table with LLM diagnosis
  - Background cleanup removes resolved issues older than 7 days

Configuration (environment variables):
  TOOL_HEALTH_DB          — path to SQLite database (default: /tmp/tool_health.db)
  TOOL_HEALTH_STREAK      — consecutive failures before triggering analysis (default: 3)
  TOOL_HEALTH_WINDOW      — rolling window size for failure rate calc (default: 100)
  TOOL_HEALTH_RATE_ALERT  — failure rate threshold for alerts (default: 0.5 = 50%)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("tool_health")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEALTH_DB_PATH = os.getenv("TOOL_HEALTH_DB", "/tmp/tool_health.db")
FAILURE_STREAK_THRESHOLD = int(os.getenv("TOOL_HEALTH_STREAK", "3"))
ROLLING_WINDOW = int(os.getenv("TOOL_HEALTH_WINDOW", "100"))
FAILURE_RATE_ALERT = float(os.getenv("TOOL_HEALTH_RATE_ALERT", "0.5"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolStats:
    """Rolling statistics for a single tool."""
    outcomes: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    last_analysis_time: float = 0.0  # Avoid re-analyzing too frequently

    @property
    def failure_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        failures = sum(1 for ok in self.outcomes if not ok)
        return failures / len(self.outcomes)

    @property
    def is_healthy(self) -> bool:
        return (
            self.consecutive_failures < FAILURE_STREAK_THRESHOLD
            and self.failure_rate < FAILURE_RATE_ALERT
        )


# ---------------------------------------------------------------------------
# SQLite persistence for issues
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating tables if needed."""
    conn = getattr(_local, "health_conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(HEALTH_DB_PATH) or "/tmp", exist_ok=True)
        conn = sqlite3.connect(HEALTH_DB_PATH, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                error_pattern TEXT NOT NULL,
                error_samples TEXT NOT NULL,
                llm_diagnosis TEXT DEFAULT '',
                failure_streak INTEGER DEFAULT 0,
                failure_rate REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                resolved_at REAL DEFAULT 0.0,
                status TEXT DEFAULT 'open'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_issues_tool
            ON tool_issues(tool_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_issues_status
            ON tool_issues(status)
        """)
        conn.commit()
        _local.health_conn = conn
    return conn


# ---------------------------------------------------------------------------
# Singleton monitor
# ---------------------------------------------------------------------------

class ToolHealthMonitor:
    """Tracks tool health and triggers LLM analysis on repeated failures."""

    def __init__(self) -> None:
        self._stats: dict[str, ToolStats] = defaultdict(ToolStats)
        self._lock = threading.Lock()
        self._analysis_queue: asyncio.Queue[tuple[str, ToolStats]] = asyncio.Queue()
        self._analysis_task: Optional[asyncio.Task] = None

    def record_success(self, tool_name: str) -> None:
        """Record a successful tool execution."""
        with self._lock:
            stats = self._stats[tool_name]
            stats.outcomes.append(True)
            stats.consecutive_failures = 0
            stats.total_calls += 1

    def record_failure(self, tool_name: str, error: str) -> None:
        """Record a tool failure and check if analysis is needed."""
        with self._lock:
            stats = self._stats[tool_name]
            stats.outcomes.append(False)
            stats.consecutive_failures += 1
            stats.total_calls += 1
            stats.total_failures += 1
            stats.last_error = error[:2000]
            stats.last_error_time = time.time()

            # Check if we should trigger analysis
            needs_analysis = (
                stats.consecutive_failures >= FAILURE_STREAK_THRESHOLD
                and (time.time() - stats.last_analysis_time) > 300  # 5 min cooldown
            )

        if needs_analysis:
            log.warning(
                f"Tool '{tool_name}' has {stats.consecutive_failures} consecutive failures. "
                f"Triggering root cause analysis."
            )
            self._store_issue(tool_name, stats)
            # Schedule async LLM analysis
            try:
                self._analysis_queue.put_nowait((tool_name, stats))
            except asyncio.QueueFull:
                log.warning("Analysis queue full, skipping")

    def _store_issue(self, tool_name: str, stats: ToolStats) -> None:
        """Store an issue in the database (pre-analysis)."""
        try:
            conn = _get_conn()
            conn.execute(
                """INSERT INTO tool_issues
                   (tool_name, error_pattern, error_samples, failure_streak, failure_rate, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'open')""",
                (
                    tool_name,
                    stats.last_error[:500],
                    stats.last_error[:2000],
                    stats.consecutive_failures,
                    round(stats.failure_rate, 3),
                    time.time(),
                ),
            )
            conn.commit()
        except Exception as e:
            log.warning(f"Failed to store tool issue: {e}")

    async def _run_llm_analysis(self, tool_name: str, stats: ToolStats) -> str:
        """Use LLM to diagnose the root cause of tool failures."""
        try:
            from .llm import call_llm

            prompt = (
                f"A research tool named '{tool_name}' has failed {stats.consecutive_failures} "
                f"times in a row. Its overall failure rate is {stats.failure_rate:.0%} "
                f"over the last {len(stats.outcomes)} calls.\n\n"
                f"Latest error message:\n{stats.last_error[:1500]}\n\n"
                f"Diagnose the most likely root cause. Consider:\n"
                f"1. Is this a rate-limiting issue (HTTP 429, Cloudflare challenge)?\n"
                f"2. Is this an API key or authentication problem?\n"
                f"3. Is this a network/connectivity issue?\n"
                f"4. Is this a bug in the tool implementation?\n"
                f"5. Is the external service down or unreachable?\n\n"
                f"Provide a concise diagnosis (2-3 sentences) and suggest a fix."
            )

            diagnosis = await call_llm(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                req_id="tool-health-analysis",
            )

            # Update the latest issue with the diagnosis
            try:
                conn = _get_conn()
                conn.execute(
                    """UPDATE tool_issues SET llm_diagnosis = ?
                       WHERE tool_name = ? AND status = 'open'
                       ORDER BY created_at DESC LIMIT 1""",
                    (diagnosis, tool_name),
                )
                conn.commit()
            except Exception as e:
                log.warning(f"Failed to update issue diagnosis: {e}")

            stats.last_analysis_time = time.time()
            log.info(f"LLM diagnosis for {tool_name}: {diagnosis[:200]}")
            return diagnosis

        except Exception as e:
            log.warning(f"LLM analysis failed for {tool_name}: {e}")
            return f"Analysis failed: {e}"

    async def start_analysis_worker(self) -> None:
        """Background task that processes the analysis queue."""
        while True:
            try:
                tool_name, stats = await asyncio.wait_for(
                    self._analysis_queue.get(), timeout=60.0
                )
                await self._run_llm_analysis(tool_name, stats)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.warning(f"Analysis worker error: {e}")
                await asyncio.sleep(5)

    def get_tool_status(self, tool_name: str) -> dict:
        """Get current status for a specific tool."""
        with self._lock:
            stats = self._stats.get(tool_name)
            if stats is None:
                return {"tool": tool_name, "status": "unknown", "calls": 0}
            return {
                "tool": tool_name,
                "status": "healthy" if stats.is_healthy else "degraded",
                "total_calls": stats.total_calls,
                "total_failures": stats.total_failures,
                "consecutive_failures": stats.consecutive_failures,
                "failure_rate_pct": round(stats.failure_rate * 100, 1),
                "last_error": stats.last_error[:300] if stats.last_error else None,
                "last_error_time": stats.last_error_time or None,
            }

    def get_all_status(self) -> dict:
        """Get status for all tracked tools."""
        with self._lock:
            tools = {}
            degraded = []
            for name, stats in self._stats.items():
                status = "healthy" if stats.is_healthy else "degraded"
                tools[name] = {
                    "status": status,
                    "calls": stats.total_calls,
                    "failures": stats.total_failures,
                    "failure_rate_pct": round(stats.failure_rate * 100, 1),
                    "consecutive_failures": stats.consecutive_failures,
                }
                if not stats.is_healthy:
                    degraded.append(name)

            return {
                "total_tools_tracked": len(self._stats),
                "degraded_tools": degraded,
                "tools": tools,
            }

    def get_open_issues(self) -> list[dict]:
        """Get all open issues from the database."""
        try:
            conn = _get_conn()
            rows = conn.execute(
                """SELECT id, tool_name, error_pattern, llm_diagnosis,
                          failure_streak, failure_rate, created_at, status
                   FROM tool_issues
                   WHERE status = 'open'
                   ORDER BY created_at DESC
                   LIMIT 50"""
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "tool_name": row[1],
                    "error_pattern": row[2],
                    "llm_diagnosis": row[3] or "Analysis pending...",
                    "failure_streak": row[4],
                    "failure_rate_pct": round(row[5] * 100, 1),
                    "created_at": row[6],
                    "status": row[7],
                }
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Failed to get open issues: {e}")
            return []

    def resolve_issue(self, issue_id: int) -> bool:
        """Mark an issue as resolved."""
        try:
            conn = _get_conn()
            conn.execute(
                "UPDATE tool_issues SET status = 'resolved', resolved_at = ? WHERE id = ?",
                (time.time(), issue_id),
            )
            conn.commit()
            return True
        except Exception as e:
            log.warning(f"Failed to resolve issue: {e}")
            return False

    def clear_old_issues(self, days: int = 7) -> int:
        """Remove resolved issues older than N days."""
        cutoff = time.time() - (days * 86400)
        try:
            conn = _get_conn()
            cursor = conn.execute(
                "DELETE FROM tool_issues WHERE status = 'resolved' AND resolved_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted
        except Exception as e:
            log.warning(f"Failed to clear old issues: {e}")
            return 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_monitor: Optional[ToolHealthMonitor] = None


def get_monitor() -> ToolHealthMonitor:
    """Return the singleton ToolHealthMonitor instance.

    Lazily starts the background LLM analysis worker the first time
    the monitor is accessed inside a running event loop.
    """
    global _monitor
    if _monitor is None:
        _monitor = ToolHealthMonitor()
    # Start the analysis worker if we're inside an event loop and it's not running yet
    if _monitor._analysis_task is None:
        try:
            loop = asyncio.get_running_loop()
            _monitor._analysis_task = loop.create_task(_monitor.start_analysis_worker())
            log.info("Started tool health analysis background worker")
        except RuntimeError:
            # No running event loop — worker will be started on next call
            pass
    return _monitor


def record_outcome(tool_name: str, success: bool, error: str = "") -> None:
    """Convenience function to record a tool outcome."""
    monitor = get_monitor()
    if success:
        monitor.record_success(tool_name)
    else:
        monitor.record_failure(tool_name, error)
