# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Pytest configuration and fixtures for the Strands agent eval suite.

Adapted from the deepagents evals framework conftest pattern.
Provides agent fixtures, plugin instances, and CLI options for
filtering by eval category.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

logger = logging.getLogger(__name__)

# Ensure the strands-agent directory is on the Python path so that
# imports like ``from agent import ...`` resolve correctly.
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add eval-specific CLI options."""
    parser.addoption(
        "--eval-category",
        action="store",
        default=None,
        help="Run only evals in this category (e.g. tool_use, budget, display)",
    )
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:8100",
        help="Base URL of the live Strands Agent API for integration evals",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Filter tests by --eval-category if specified."""
    category = config.getoption("--eval-category")
    if not category:
        return

    selected = []
    deselected = []
    for item in items:
        markers = [m.name for m in item.iter_markers()]
        if category in markers or f"eval_{category}" in markers:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def venice_api_key() -> str:
    """Require VENICE_API_KEY to be set for evals."""
    key = os.environ.get("VENICE_API_KEY", "")
    if not key:
        pytest.skip("VENICE_API_KEY not set — skipping evals")
    return key


@pytest.fixture(scope="session")
def plugins():
    """Create a fresh set of plugin instances for eval use."""
    from plugins.budget import BudgetPlugin
    from plugins.stream_capture import StreamCapturePlugin
    from plugins.thought_refiner import ThoughtRefinerPlugin
    from plugins.tool_display import ToolDisplayPlugin

    return {
        "budget": BudgetPlugin(),
        "stream_capture": StreamCapturePlugin(),
        "thought_refiner": ThoughtRefinerPlugin(),
        "tool_display": ToolDisplayPlugin(),
    }
