"""Slack alerting for model errors — fires on WARNING/ERROR logs from proxies.

Hooks into the standard Python logging system so every proxy that uses
``shared.setup_logging()`` gets Slack alerts for free — no per-proxy changes
required.

Activation
----------
Set the ``SLACK_WEBHOOK_URL`` environment variable to a Slack Incoming Webhook
URL.  When unset the module is a silent no-op.

The handler posts to Slack when a log message at WARNING or ERROR level
matches a known model-error pattern (HTTP failures, timeouts, connection
errors, unhandled exceptions during model calls).

Rate-limiting
-------------
To avoid flooding a Slack channel during a multi-model race (where several
models may fail simultaneously), messages are batched: the handler collects
errors for up to ``BATCH_WINDOW`` seconds, then sends a single Slack message
summarising all errors in that window.  A per-model cooldown
(``MODEL_COOLDOWN``) further suppresses repeated alerts for the same model.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger("slack_alerter")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")

# How long to batch errors before sending a single Slack message (seconds).
BATCH_WINDOW: float = float(os.getenv("SLACK_BATCH_WINDOW", "5"))

# Per-model cooldown — suppress duplicate alerts for the same model within
# this window (seconds).  Prevents noise when a provider is fully down.
MODEL_COOLDOWN: float = float(os.getenv("SLACK_MODEL_COOLDOWN", "60"))

# ---------------------------------------------------------------------------
# Patterns that identify a model-related error in a log message
# ---------------------------------------------------------------------------

_MODEL_ERROR_PATTERNS: list[re.Pattern[str]] = [
    # HTTP errors from model providers
    re.compile(r"returned (\d{3})", re.I),
    # Timeouts
    re.compile(r"timed? ?out", re.I),
    # Stream errors
    re.compile(r"stream .+ error \d{3}", re.I),
    # Generic model/provider exceptions
    re.compile(r"(?:model|provider|upstream|openrouter|xai|venice|anthropic|google|openai|deepseek|mistral).* (?:error|exception|failed)", re.I),
    # Connection errors
    re.compile(r"connect(?:ion)? error", re.I),
    # Passthrough errors
    re.compile(r"passthrough.*error", re.I),
    # Agent loop errors
    re.compile(r"agent loop error", re.I),
    # Unhandled exceptions during streaming
    re.compile(r"stream .+ exception", re.I),
    # stream-as-call errors
    re.compile(r"stream-as-call error", re.I),
]


def _is_model_error(message: str) -> bool:
    """Return True if the log message looks like a model/provider error."""
    for pat in _MODEL_ERROR_PATTERNS:
        if pat.search(message):
            return True
    return False


# Known LLM provider prefixes used in model IDs (e.g. "openai/gpt-4o").
_KNOWN_PROVIDERS = {
    "openai", "anthropic", "google", "x-ai", "deepseek", "mistralai",
    "perplexity", "qwen", "moonshotai", "cohere", "minimax", "groq",
    "stepfun", "z-ai", "nvidia", "meta-llama", "nousresearch", "xiaomi",
}


def _extract_model_name(message: str) -> str:
    """Try to extract a model identifier from a log message."""
    # Pattern: provider/model-name — only match known LLM provider prefixes
    # to avoid false positives on URL path segments like "openrouter.ai/api".
    for m in re.finditer(r"([\w-]+/[\w.-]+)", message):
        candidate = m.group(1)
        prefix = candidate.split("/", 1)[0]
        if prefix.lower() in _KNOWN_PROVIDERS:
            return candidate
    # Pattern: "xAI model-name" or "OpenRouter model-name"
    m = re.search(r"(?:xAI|OpenRouter|Venice|upstream)\s+([\w.-]+)", message)
    if m:
        return m.group(1)
    return "unknown"


def _extract_status_code(message: str) -> Optional[int]:
    """Try to extract an HTTP status code from a log message."""
    m = re.search(r"returned (\d{3})", message)
    if m:
        return int(m.group(1))
    m = re.search(r"error (\d{3})", message)
    if m:
        return int(m.group(1))
    m = re.search(r"HTTP (\d{3})", message)
    if m:
        return int(m.group(1))
    return None


_REQ_ID_RE = re.compile(
    r"\[(?!WARNING|ERROR|INFO|DEBUG|CRITICAL\])"  # skip log-level brackets
    r"([^\]]+)\]"
)


def _extract_req_id(message: str) -> str:
    """Extract the request ID (e.g. [tier-abc12345]) from a log message.

    The formatted log line looks like:
      ``2025-01-01 00:00:00 [WARNING] tier-chooser: [tier-abc12345] ...``
    This skips the ``[WARNING]`` bracket and grabs the actual request ID.
    """
    m = _REQ_ID_RE.search(message)
    return m.group(1) if m else ""


def _classify_error(message: str) -> str:
    """Classify the error into a human-readable category."""
    lower = message.lower()
    if "timed out" in lower or "timeout" in lower:
        return "Timeout"
    if "connect" in lower and "error" in lower:
        return "Connection Error"
    if re.search(r"returned [45]\d{2}", lower):
        code = _extract_status_code(message)
        if code and 400 <= code < 500:
            return f"Client Error (HTTP {code})"
        if code and 500 <= code < 600:
            return f"Server Error (HTTP {code})"
        return f"HTTP Error ({code})"
    if "exception" in lower or "error:" in lower:
        return "Exception"
    if "empty" in lower:
        return "Empty Response"
    return "Error"


# ---------------------------------------------------------------------------
# Pending-error buffer & background sender
# ---------------------------------------------------------------------------

_PendingError = dict  # keys: timestamp, proxy, model, category, message, level


class _ErrorBuffer:
    """Thread-safe buffer that collects errors and flushes them as a single
    Slack message after the batch window elapses."""

    # Evict stale cooldown entries after this many seconds (default 1 hour).
    _COOLDOWN_EVICT_INTERVAL: float = 3600.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[_PendingError] = []
        self._model_last_alert: dict[str, float] = {}  # model -> timestamp
        self._timer: Optional[threading.Timer] = None
        self._last_eviction: float = time.time()

    def add(self, entry: _PendingError) -> None:
        model = entry.get("model", "unknown")
        now = time.time()

        # Per-model cooldown check
        with self._lock:
            # Periodically evict stale cooldown entries to prevent unbounded growth
            if now - self._last_eviction > self._COOLDOWN_EVICT_INTERVAL:
                cutoff = now - MODEL_COOLDOWN
                self._model_last_alert = {
                    k: v for k, v in self._model_last_alert.items() if v > cutoff
                }
                self._last_eviction = now

            last = self._model_last_alert.get(model, 0.0)
            if now - last < MODEL_COOLDOWN:
                return  # suppress duplicate
            self._pending.append(entry)
            self._model_last_alert[model] = now

            # Schedule flush if not already pending
            if self._timer is None:
                self._timer = threading.Timer(BATCH_WINDOW, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
            self._timer = None

        if not batch:
            return

        try:
            _send_to_slack(batch)
        except Exception as exc:
            log.debug("Failed to send Slack alert: %s", exc)


_buffer = _ErrorBuffer()

# ---------------------------------------------------------------------------
# Slack message formatting & sending
# ---------------------------------------------------------------------------


def _format_slack_blocks(errors: list[_PendingError]) -> dict:
    """Build a Slack Block Kit message from a batch of errors."""
    count = len(errors)
    header = f":rotating_light: *{count} Model Error{'s' if count != 1 else ''}*"

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{count} Model Error{'s' if count != 1 else ''}", "emoji": True}},
    ]

    for err in errors[:20]:  # cap at 20 to stay within Slack limits
        ts = err.get("timestamp", "")
        proxy = err.get("proxy", "unknown")
        model = err.get("model", "unknown")
        category = err.get("category", "Error")
        message = err.get("message", "")
        req_id = err.get("req_id", "")
        level = err.get("level", "ERROR")

        # Truncate message for Slack (max ~2900 chars per block)
        if len(message) > 1500:
            message = message[:1500] + "…"

        detail_lines = [
            f"*Proxy:* `{proxy}`",
            f"*Model:* `{model}`",
            f"*Type:* {category}",
            f"*Level:* {level}",
        ]
        if req_id:
            detail_lines.append(f"*Request ID:* `{req_id}`")
        detail_lines.append(f"*Time:* {ts}")

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(detail_lines)},
        })

        # Error details in a code block
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"```\n{message}\n```"},
        })

        blocks.append({"type": "divider"})

    if count > 20:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"_… and {count - 20} more errors (truncated)_"},
        })

    return {
        "text": header,  # fallback for notifications
        "blocks": blocks,
    }


def _send_to_slack(errors: list[_PendingError]) -> None:
    """POST the formatted message to the Slack webhook (synchronous)."""
    if not SLACK_WEBHOOK_URL:
        return

    payload = _format_slack_blocks(errors)
    try:
        # Use a short-lived sync client for the webhook POST.
        # This runs in the background timer thread, not the async event loop.
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                SLACK_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                log.debug(
                    "Slack webhook returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
    except Exception as exc:
        log.debug("Slack webhook request failed: %s", exc)


# ---------------------------------------------------------------------------
# Custom logging handler
# ---------------------------------------------------------------------------


class SlackAlertHandler(logging.Handler):
    """Logging handler that sends model-error alerts to Slack.

    Filters for WARNING and ERROR level messages that match known
    model-error patterns, then queues them into the batching buffer.
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if not _is_model_error(message):
                return

            entry: _PendingError = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "proxy": record.name,
                "model": _extract_model_name(message),
                "category": _classify_error(message),
                "message": message,
                "req_id": _extract_req_id(message),
                "level": record.levelname,
            }
            _buffer.add(entry)
        except Exception:
            # Never let the alerter crash the application
            pass


def get_handler() -> Optional[SlackAlertHandler]:
    """Return a configured SlackAlertHandler, or None if Slack is not configured."""
    if not SLACK_WEBHOOK_URL:
        return None
    handler = SlackAlertHandler(level=logging.WARNING)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    return handler
