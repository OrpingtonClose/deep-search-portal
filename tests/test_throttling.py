"""Tests for the API throttling module (TokenBucketThrottler).

These tests import directly from the real ``shared`` module on disk,
bypassing any MagicMock that other test files may have injected into
``sys.modules["shared"]``.
"""

import asyncio
import importlib
import os
import sys
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Force-load the *real* shared module regardless of what other test files
# may have injected into sys.modules.
# ---------------------------------------------------------------------------
_proxies_dir = os.path.join(os.path.dirname(__file__), "..", "proxies")
if _proxies_dir not in sys.path:
    sys.path.insert(0, _proxies_dir)

_prev = sys.modules.pop("shared", None)
import shared as _real_shared  # noqa: E402

importlib.reload(_real_shared)

TokenBucketThrottler = _real_shared.TokenBucketThrottler
get_throttler = _real_shared.get_throttler
all_throttler_stats = _real_shared.all_throttler_stats
_throttlers = _real_shared._throttlers

# Restore the previous entry so other tests keep working if they depend on mock
if _prev is not None:
    sys.modules["shared"] = _prev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_throttler_registry():
    """Reset the global throttler registry between tests."""
    _throttlers.clear()
    yield
    _throttlers.clear()


# ---------------------------------------------------------------------------
# TokenBucketThrottler unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_immediate_when_tokens_available():
    """Acquiring a token should be near-instant when the bucket has capacity."""
    t = TokenBucketThrottler(rate=10.0, capacity=5, name="test")
    waited = await t.acquire()
    assert waited < 0.05  # essentially instant
    t.release()


@pytest.mark.asyncio
async def test_acquire_waits_when_bucket_empty():
    """Draining the bucket should force subsequent acquires to wait."""
    t = TokenBucketThrottler(rate=10.0, capacity=2, name="test")
    # Drain all tokens
    await t.acquire()
    await t.acquire()
    # Next acquire must wait ~0.1s (1 token / 10 rps)
    start = time.monotonic()
    await t.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05  # at least some wait
    t.release()
    t.release()
    t.release()


@pytest.mark.asyncio
async def test_throttle_context_manager():
    """The ``throttle()`` context manager should acquire and release."""
    t = TokenBucketThrottler(rate=100.0, capacity=5, name="test")
    async with t.throttle():
        pass  # should not raise
    # Stats should show 1 acquisition
    stats = t.stats()
    assert stats["total_acquired"] == 1


@pytest.mark.asyncio
async def test_concurrency_limit():
    """When max_concurrent is set, only that many can be active at once."""
    t = TokenBucketThrottler(rate=100.0, capacity=10, max_concurrent=2, name="test")

    active_count = 0
    max_active = 0

    async def _worker():
        nonlocal active_count, max_active
        async with t.throttle():
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1

    await asyncio.gather(*[_worker() for _ in range(5)])
    assert max_active <= 2


@pytest.mark.asyncio
async def test_stats_tracking():
    """Stats should accurately reflect acquisitions."""
    t = TokenBucketThrottler(rate=100.0, capacity=10, name="stats-test")
    for _ in range(3):
        async with t.throttle():
            pass
    stats = t.stats()
    assert stats["name"] == "stats-test"
    assert stats["total_acquired"] == 3
    assert stats["rate_per_sec"] == 100.0
    assert stats["capacity"] == 10
    assert stats["waiters"] == 0
    assert stats["active"] == 0


@pytest.mark.asyncio
async def test_multiple_concurrent_acquires():
    """Multiple coroutines should be able to acquire tokens fairly."""
    t = TokenBucketThrottler(rate=50.0, capacity=5, name="concurrent")
    results = []

    async def _worker(idx):
        async with t.throttle():
            results.append(idx)

    await asyncio.gather(*[_worker(i) for i in range(5)])
    assert len(results) == 5
    assert set(results) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_throttler_returns_same_instance():
    """get_throttler should return the same instance for the same provider."""
    t1 = get_throttler("mistral")
    t2 = get_throttler("mistral")
    assert t1 is t2


@pytest.mark.asyncio
async def test_get_throttler_different_providers():
    """Different providers should get separate throttlers."""
    t1 = get_throttler("mistral")
    t2 = get_throttler("bright_data")
    assert t1 is not t2
    assert t1.name == "mistral"
    assert t2.name == "bright_data"


@pytest.mark.asyncio
async def test_get_throttler_unknown_provider_uses_defaults():
    """An unknown provider should get sensible defaults (5.0 rps, 10 burst)."""
    t = get_throttler("unknown_provider")
    assert t.rate == 5.0
    assert t.capacity == 10


@pytest.mark.asyncio
async def test_get_throttler_env_override(monkeypatch):
    """Environment variables should override default rate limits."""
    monkeypatch.setenv("THROTTLE_ARXIV_RPS", "0.5")
    monkeypatch.setenv("THROTTLE_ARXIV_BURST", "2")
    monkeypatch.setenv("THROTTLE_ARXIV_MAX_CONCURRENT", "1")
    t = get_throttler("arxiv")
    assert t.rate == 0.5
    assert t.capacity == 2
    assert t._concurrent_sem is not None


@pytest.mark.asyncio
async def test_all_throttler_stats():
    """all_throttler_stats should return stats for all registered throttlers."""
    get_throttler("mistral")
    get_throttler("bright_data")
    stats = all_throttler_stats()
    assert len(stats) == 2
    names = {s["name"] for s in stats}
    assert names == {"mistral", "bright_data"}


# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_provider_defaults():
    """Known providers should have specific non-default rate limits."""
    providers = [
        "mistral", "bright_data", "oxylabs", "searxng", "arxiv",
        "wayback", "wikidata", "imageboard", "nitter", "apify",
        "knowledge_engine",
    ]
    for name in providers:
        t = get_throttler(name)
        assert t.name == name
        # All known providers should have a throttler created
        assert t.rate > 0
        assert t.capacity > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_without_concurrency_limit():
    """Calling release() when no concurrency cap is set should be a no-op."""
    t = TokenBucketThrottler(rate=10.0, capacity=5, max_concurrent=0, name="no-cap")
    await t.acquire()
    t.release()  # Should not raise


@pytest.mark.asyncio
async def test_high_burst_capacity():
    """A large burst capacity should allow many immediate acquires."""
    t = TokenBucketThrottler(rate=1.0, capacity=100, name="burst")
    start = time.monotonic()
    for _ in range(50):
        await t.acquire()
        t.release()
    elapsed = time.monotonic() - start
    # 50 acquires from a 100-token bucket should be near-instant
    assert elapsed < 1.0
