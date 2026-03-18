"""Tests for the SSE keepalive pattern in both research proxies.

Covers BUG_0003: the asyncio.wait_for wrapper must emit keepalive dots
when the LangGraph astream iterator stalls (e.g. during long LLM calls
or subagent runs) to prevent reverse proxies from timing out the SSE
connection.

These tests verify the pattern in isolation — no actual LangGraph graph
is run.  Instead we build a minimal async iterator that simulates stalls
and verify that the keepalive wrapper (a) emits dots on timeout and
(b) forwards real state updates when they arrive.
"""

import asyncio
import time

import pytest


# ---------------------------------------------------------------------------
# Reproduce the keepalive wrapper pattern from both proxies
# ---------------------------------------------------------------------------

async def keepalive_wrapper(astream_iter, keepalive_interval, chunk_fn):
    """Extracted keepalive wrapper — identical to the pattern in
    deep_research_proxy.run_deep_research and
    persistent_deep_research_proxy.run_persistent_research.

    Yields (event_type, payload) tuples:
      ("keepalive", ".")   — when no state update arrives within interval
      ("state", <dict>)    — when a state update arrives
    """
    done = False
    while not done:
        try:
            state_update = await asyncio.wait_for(
                astream_iter.__anext__(), timeout=keepalive_interval,
            )
        except asyncio.TimeoutError:
            yield ("keepalive", chunk_fn("."))
            continue
        except StopAsyncIteration:
            done = True
            break

        yield ("state", state_update)


def _chunk(text):
    """Trivial chunk formatter for testing."""
    return f"data: {text}\n\n"


# ---------------------------------------------------------------------------
# Helpers: async iterators that simulate graph behaviour
# ---------------------------------------------------------------------------

class FastIterator:
    """Yields items immediately without delay."""

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


class SlowIterator:
    """Yields items after a wall-clock deadline, surviving cancellation.

    Unlike a naive asyncio.sleep-based approach, this uses monotonic
    deadlines so that asyncio.wait_for cancellations don't cause items
    to be skipped -- the next __anext__ call simply re-checks the
    deadline and sleeps for the remaining time.
    """

    def __init__(self, items_with_delays):
        """items_with_delays: list of (delay_seconds, item) tuples."""
        self._items = list(items_with_delays)
        self._idx = 0
        self._deadline = None  # set on first __anext__ call for current item

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration

        delay, item = self._items[self._idx]

        # Set deadline on first attempt for this item
        if self._deadline is None:
            self._deadline = time.monotonic() + delay

        # Wait until deadline
        remaining = self._deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Deadline reached -- advance and return item
        self._idx += 1
        self._deadline = None
        return item


class NeverYieldsIterator:
    """Never yields — simulates an indefinitely stalled graph."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Sleep forever (will be interrupted by wait_for timeout)
        await asyncio.sleep(3600)
        raise StopAsyncIteration


# ============================================================================
# Tests
# ============================================================================

class TestKeepaliveEmitsDotsOnStall:
    """Verify that keepalive dots are emitted when the iterator stalls."""

    @pytest.mark.asyncio
    async def test_emits_dot_on_timeout(self):
        """A stalled iterator should produce keepalive events."""
        # Iterator that takes 0.5s per item, keepalive interval 0.05s
        # Generous ratio to ensure at least 1 keepalive dot
        items = SlowIterator([(0.5, {"progress_log": ["step1"]})])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 0.05, _chunk):
            events.append((event_type, payload))

        keepalives = [e for e in events if e[0] == "keepalive"]
        states = [e for e in events if e[0] == "state"]

        assert len(keepalives) >= 1, f"Expected >= 1 keepalive dots, got {len(keepalives)}"
        assert len(states) == 1
        assert states[0][1]["progress_log"] == ["step1"]

    @pytest.mark.asyncio
    async def test_no_keepalive_when_fast(self):
        """A fast iterator should produce no keepalive events."""
        items = FastIterator([
            {"progress_log": ["a"]},
            {"progress_log": ["a", "b"]},
        ])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 1.0, _chunk):
            events.append((event_type, payload))

        keepalives = [e for e in events if e[0] == "keepalive"]
        states = [e for e in events if e[0] == "state"]

        assert len(keepalives) == 0
        assert len(states) == 2

    @pytest.mark.asyncio
    async def test_keepalive_dot_format(self):
        """Keepalive events should use the chunk function."""
        items = SlowIterator([(0.25, {"done": True})])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 0.05, _chunk):
            events.append((event_type, payload))

        keepalives = [e for e in events if e[0] == "keepalive"]
        assert len(keepalives) >= 1
        for _, dot in keepalives:
            assert dot == "data: .\n\n"

    @pytest.mark.asyncio
    async def test_empty_iterator(self):
        """An immediately-exhausted iterator should produce no events."""
        items = FastIterator([])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 0.1, _chunk):
            events.append((event_type, payload))

        assert len(events) == 0


class TestKeepaliveWithMultipleUpdates:
    """Verify correct interleaving of keepalives and state updates."""

    @pytest.mark.asyncio
    async def test_mixed_fast_and_slow(self):
        """Mix of fast and slow updates should produce keepalives only
        during the slow segments."""
        items = SlowIterator([
            (0.0, {"step": 1}),   # immediate
            (0.5, {"step": 2}),   # slow — should trigger keepalive(s)
            (0.0, {"step": 3}),   # immediate
        ])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 0.05, _chunk):
            events.append((event_type, payload))

        states = [e for e in events if e[0] == "state"]
        keepalives = [e for e in events if e[0] == "keepalive"]

        assert len(states) == 3
        assert [s[1]["step"] for s in states] == [1, 2, 3]
        assert len(keepalives) >= 1  # at least one during the 0.5s stall

    @pytest.mark.asyncio
    async def test_progress_log_forwarded(self):
        """State updates with progress_log should be forwarded intact."""
        items = FastIterator([
            {"progress_log": ["Searching..."], "turn": 1},
            {"progress_log": ["Searching...", "Found 5 results"], "turn": 2},
        ])
        events = []
        async for event_type, payload in keepalive_wrapper(items, 1.0, _chunk):
            events.append((event_type, payload))

        assert events[0][1]["progress_log"] == ["Searching..."]
        assert events[1][1]["progress_log"] == ["Searching...", "Found 5 results"]


class TestKeepaliveNeverYields:
    """Verify behaviour when iterator never yields (simulates hung graph)."""

    @pytest.mark.asyncio
    async def test_continuous_keepalive_on_hang(self):
        """A hung iterator should continuously produce keepalive dots."""
        items = NeverYieldsIterator()
        events = []
        count = 0
        async for event_type, payload in keepalive_wrapper(items, 0.05, _chunk):
            events.append((event_type, payload))
            count += 1
            if count >= 5:
                break  # stop after 5 keepalive dots

        assert all(e[0] == "keepalive" for e in events)
        assert len(events) == 5


class TestKeepaliveIntervalRespected:
    """Verify that the keepalive interval is approximately respected."""

    @pytest.mark.asyncio
    async def test_interval_timing(self):
        """Keepalive dots should arrive approximately at the interval."""
        import time

        items = SlowIterator([(1.0, {"done": True})])
        interval = 0.1
        timestamps = []

        start = time.monotonic()
        async for event_type, payload in keepalive_wrapper(items, interval, _chunk):
            if event_type == "keepalive":
                timestamps.append(time.monotonic() - start)

        # Should have several keepalive dots (1.0 / 0.1)
        assert len(timestamps) >= 3, f"Expected >= 3 timestamps, got {len(timestamps)}: {timestamps}"

        # Each timestamp should be roughly at multiples of interval (generous tolerance)
        for i, ts in enumerate(timestamps):
            expected = interval * (i + 1)
            assert abs(ts - expected) < 0.15, (
                f"Keepalive {i} at {ts:.3f}s, expected ~{expected:.3f}s"
            )
