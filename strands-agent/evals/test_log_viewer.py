# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Eval tests for the log viewer HTML endpoint.

Verifies router configuration and the configure() injection mechanism
without requiring a running FastAPI server.
"""

from __future__ import annotations

import log_viewer


class TestConfigure:
    """Verify log viewer configuration."""

    def test_configure_sets_getter(self) -> None:
        def fake_getter(rid: str) -> dict | None:
            return {"model": "test", "timestamp": "now"}

        log_viewer.configure(fake_getter)
        assert log_viewer._get_log is fake_getter

    def test_router_exists(self) -> None:
        assert log_viewer.router is not None
        routes = [r.path for r in log_viewer.router.routes]
        assert "/logs/{request_id}" in routes

    def test_router_has_get_method(self) -> None:
        for route in log_viewer.router.routes:
            if hasattr(route, "path") and route.path == "/logs/{request_id}":
                assert "GET" in route.methods
                break
        else:
            raise AssertionError("Log viewer route not found")
