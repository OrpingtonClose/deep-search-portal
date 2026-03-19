"""Tests for langfuse_config.py — Langfuse observability integration."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure proxies/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))


# ---------------------------------------------------------------------------
# Helpers — we need to re-import langfuse_config with controlled env vars
# so we use importlib.reload in fixtures.
# ---------------------------------------------------------------------------

def _reload_config(**env_overrides):
    """Reload langfuse_config with the given env vars patched."""
    import importlib
    import langfuse_config

    with patch.dict(os.environ, env_overrides, clear=False):
        importlib.reload(langfuse_config)
    return langfuse_config


# ---------------------------------------------------------------------------
# Tests: module-level flags
# ---------------------------------------------------------------------------

class TestModuleFlags:
    def test_disabled_when_no_keys(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
            LANGFUSE_ENABLED="true",
        )
        assert not cfg._langfuse_available

    def test_disabled_when_enabled_false(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_ENABLED="false",
        )
        assert not cfg._langfuse_available

    def test_enabled_when_keys_present(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_ENABLED="true",
        )
        assert cfg._langfuse_available

    def test_default_base_url(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        assert cfg.LANGFUSE_BASE_URL  # should have a default

    def test_custom_base_url(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_BASE_URL="http://langfuse.local:3001",
        )
        assert cfg.LANGFUSE_BASE_URL == "http://langfuse.local:3001"


# ---------------------------------------------------------------------------
# Tests: is_enabled / get_trace_url / create_trace_id (no real Langfuse)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_is_enabled_false_without_keys(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        # Reset init state so _get_langfuse is re-evaluated
        cfg._init_attempted = False
        cfg._langfuse_client = None
        assert not cfg.is_enabled()

    def test_create_trace_id_deterministic(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        tid1 = cfg.create_trace_id("req-abc123")
        tid2 = cfg.create_trace_id("req-abc123")
        assert tid1 == tid2

    def test_create_trace_id_different_for_different_reqs(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        tid1 = cfg.create_trace_id("req-abc123")
        tid2 = cfg.create_trace_id("req-def456")
        assert tid1 != tid2

    def test_get_trace_url_empty_when_disabled(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None
        url = cfg.get_trace_url("some-trace-id")
        assert url == ""

    def test_get_trace_url_with_mock_client(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        mock_client = MagicMock()
        mock_client.get_trace_url.return_value = "https://langfuse.example.com/trace/abc"
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        url = cfg.get_trace_url("abc")
        assert url == "https://langfuse.example.com/trace/abc"
        mock_client.get_trace_url.assert_called_once_with("abc")


# ---------------------------------------------------------------------------
# Tests: create_callback_handler
# ---------------------------------------------------------------------------

class TestCallbackHandler:
    def test_returns_none_when_disabled(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None
        handler = cfg.create_callback_handler(trace_id="test")
        assert handler is None

    def test_creates_handler_when_enabled(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        # Mock the client so is_enabled() returns True
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        with patch("langfuse_config.CallbackHandler", create=True) as MockCB:
            # Patch the import inside create_callback_handler
            mock_handler = MagicMock()
            with patch.dict("sys.modules", {"langfuse.langchain": MagicMock(CallbackHandler=lambda **kw: mock_handler)}):
                handler = cfg.create_callback_handler(
                    trace_id="trace-123",
                    session_id="session-456",
                    tags=["persistent-research"],
                )
                assert handler is mock_handler

    def test_handler_exception_returns_none(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        mock_client = MagicMock()
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        with patch.dict("sys.modules", {"langfuse.langchain": MagicMock(CallbackHandler=MagicMock(side_effect=RuntimeError("boom")))}):
            handler = cfg.create_callback_handler(trace_id="test")
            assert handler is None


# ---------------------------------------------------------------------------
# Tests: flush / shutdown
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_flush_noop_when_disabled(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None
        cfg.flush()  # should not raise

    def test_flush_calls_client(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        mock_client = MagicMock()
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True
        cfg.flush()
        mock_client.flush.assert_called_once()

    def test_shutdown_clears_state(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        mock_client = MagicMock()
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        cfg.shutdown()
        assert cfg._langfuse_client is None
        assert not cfg._init_attempted
        mock_client.shutdown.assert_called_once()

    def test_shutdown_noop_when_no_client(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None
        cfg.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# Tests: _get_langfuse initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_init_only_attempted_once(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None

        result1 = cfg._get_langfuse()
        assert cfg._init_attempted is True
        assert result1 is None

        # Second call should not re-attempt
        result2 = cfg._get_langfuse()
        assert result2 is None

    def test_init_with_auth_failure(self):
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-bad",
            LANGFUSE_SECRET_KEY="sk-bad",
            LANGFUSE_ENABLED="true",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None

        # Mock Langfuse to raise on auth_check
        mock_langfuse_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.auth_check.side_effect = Exception("auth failed")
        mock_langfuse_cls.return_value = mock_instance

        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
            result = cfg._get_langfuse()
            assert result is None
            assert cfg._init_attempted is True


# ---------------------------------------------------------------------------
# Tests: proxy integration patterns (verify the wiring works)
# ---------------------------------------------------------------------------

class TestProxyIntegration:
    """Verify that the Langfuse integration pattern used in proxies works correctly."""

    def test_trace_url_emitted_before_think(self):
        """Simulate the proxy pattern: trace URL emitted, then <think>."""
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )

        # Mock the client
        mock_client = MagicMock()
        mock_client.get_trace_url.return_value = "https://langfuse.example.com/trace/test-trace"
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        req_id = "req-test123"
        trace_id = cfg.create_trace_id(req_id)
        trace_url = cfg.get_trace_url(trace_id)

        assert trace_url.startswith("https://langfuse.example.com/trace/")

        # Simulate the SSE emission order
        messages = []
        if trace_url:
            messages.append(f"[Langfuse trace]({trace_url})")
        messages.append("<think>")

        assert messages[0].startswith("[Langfuse trace](")
        assert messages[1] == "<think>"

    def test_callbacks_list_includes_langfuse_handler(self):
        """Verify the callbacks list pattern used in all three proxies."""
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        mock_client = MagicMock()
        cfg._langfuse_client = mock_client
        cfg._init_attempted = True

        mock_handler = MagicMock()
        with patch.dict("sys.modules", {"langfuse.langchain": MagicMock(CallbackHandler=lambda **kw: mock_handler)}):
            handler = cfg.create_callback_handler(
                trace_id="trace-123",
                session_id="req-456",
                tags=["persistent-research"],
            )

        # Simulate what the proxy does
        metrics_callback = MagicMock()  # ResearchMetricsCallback
        callbacks = [metrics_callback]
        if handler is not None:
            callbacks.append(handler)

        assert len(callbacks) == 2
        assert callbacks[0] is metrics_callback
        assert callbacks[1] is mock_handler

    def test_callbacks_list_works_without_langfuse(self):
        """When Langfuse is disabled, callbacks list only has metrics."""
        cfg = _reload_config(
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        )
        cfg._init_attempted = False
        cfg._langfuse_client = None

        handler = cfg.create_callback_handler(trace_id="test")

        metrics_callback = MagicMock()
        callbacks = [metrics_callback]
        if handler is not None:
            callbacks.append(handler)

        assert len(callbacks) == 1
        assert callbacks[0] is metrics_callback
