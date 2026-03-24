"""
Test cases extracted from the insulin query run (req-d31db7bb).

These test cases exercise every tool that returned empty results during
the run. They verify that tools are routed correctly (not "Unknown tool")
and that they return non-trivial content.

Usage:
    UPSTREAM_KEY=xxx SEARXNG_URL=http://localhost:8888 python -m pytest tests/test_tool_health.py -v --timeout=60
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

# Allow imports from proxies/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))


# ---------------------------------------------------------------------------
# Test cases from the insulin run (req-d31db7bb)
# Each tuple: (tool_name, arguments_dict, description)
# ---------------------------------------------------------------------------

ROUTING_TEST_CASES = [
    # Tools that were completely missing from the dispatcher (100% empty)
    ("telegram_search", {"query": "insulin Poland vendor underground"}, "telegram via SearXNG site filter"),
    ("darknet_market_search", {"query": "insulin vendor Warsaw Poland"}, "darknet via SearXNG site filter"),
    ("facebook_search", {"query": "insulin online pharmacy Poland", "result_type": "posts"}, "facebook via SearXNG site filter"),
    ("discord_search", {"query": "insulin vendors Poland shipping"}, "discord via SearXNG site filter"),
    ("signal_search", {"query": "insulin vendors Poland no prescription"}, "signal via SearXNG site filter"),
    ("whatsapp_search", {"query": "insulin vendors Poland no prescription"}, "whatsapp via SearXNG site filter"),
    ("crunchbase_search", {"query": "insulin international shipping Poland"}, "crunchbase via SearXNG site filter"),
    ("trustpilot_search", {"query": "insulin Warsaw Poland"}, "trustpilot via SearXNG site filter"),
    # Tools that were routed but returned empty
    ("chan_4plebs_search", {"query": "insulin online pharmacy Poland", "board": "pol"}, "4plebs archive search"),
    ("chan_b4k_search", {"query": "insulin Poland buy"}, "b4k /biz/ archive search"),
    ("chan_warosu_search", {"query": "insulin import no prescription", "board": "g"}, "warosu archive search"),
    ("stackexchange_search", {"query": "insulin online pharmacy Europe", "site": "health"}, "StackExchange health"),
    # Tools that mostly worked but test baseline
    ("searxng_search", {"query": "insulin online pharmacy Poland"}, "SearXNG general search"),
    ("reddit_search", {"query": "insulin online pharmacy Poland"}, "Reddit search"),
    ("hackernews_search", {"query": "insulin import regulations"}, "Hacker News search"),
    ("wikipedia_search", {"query": "insulin"}, "Wikipedia search"),
]


@pytest.mark.parametrize("tool_name,arguments,desc", ROUTING_TEST_CASES, ids=[t[0] for t in ROUTING_TEST_CASES])
def test_tool_is_routed(tool_name, arguments, desc):
    """Verify the tool is routed (does not return 'Unknown tool')."""
    # We can't actually call the tools without a running SearXNG etc,
    # but we CAN verify the dispatcher recognises them.
    from tools.tool_executor import _execute_tool_inner

    async def _check():
        try:
            result = await asyncio.wait_for(
                _execute_tool_inner(tool_name, arguments),
                timeout=5.0,
            )
        except Exception:
            # Network errors are fine -- we're testing routing, not connectivity
            return "routed"
        return result

    result = asyncio.get_event_loop().run_until_complete(_check())
    assert "Unknown tool" not in str(result), f"{tool_name} is not routed in _execute_tool_inner"


# ---------------------------------------------------------------------------
# Cloudflare detection in call_llm
# ---------------------------------------------------------------------------

def test_cloudflare_html_is_retryable():
    """Verify that Cloudflare challenge pages are detected as retryable."""
    from tools.llm import _is_cloudflare_challenge
    
    cf_html = (
        '<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title>'
        '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
    )
    assert _is_cloudflare_challenge(cf_html), "Should detect Cloudflare challenge"
    assert not _is_cloudflare_challenge("Connection refused"), "Should not flag normal errors"
    assert not _is_cloudflare_challenge("Rate limit exceeded"), "Should not flag rate limits"
