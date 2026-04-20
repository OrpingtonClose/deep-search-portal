# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Integration evals for thought refinement quality.

These evals hit the real Venice API to verify that the ThoughtRefinerPlugin
produces engaging, user-friendly prose from raw chain-of-thought input.

Requires: VENICE_API_KEY

Usage::

    pytest evals/test_integ_thought_refiner.py -v
"""

from __future__ import annotations

import re

import pytest

from plugins.thought_refiner import ThoughtRefinerPlugin

pytestmark = pytest.mark.integ


# ── Raw thinking samples ─────────────────────────────────────────────

RAW_THINKING_RESEARCH = """\
The user wants to know about mesh networking for censorship circumvention.
Let me search for this. I should check Tor, I2P, Nym, and Briar.
First I'll use brave_web_search to find recent papers on mesh networking
protocols. Then I'll look at Nym's mixnet architecture since it's newer
and uses Sphinx packet format. I should also check Briar which uses
Tor hidden services for peer-to-peer messaging. The key innovation in
Nym is that it adds timing obfuscation via Poisson mixing, which defeats
traffic analysis better than Tor's onion routing. Let me also look at
academic papers from USENIX and IEEE S&P on circumvention effectiveness.
I notice that most censorship circumvention tools have a bootstrapping
problem — how do you distribute the tool if the internet is already censored?
Briar solves this with Bluetooth/WiFi direct distribution.
"""

RAW_THINKING_SIMPLE = """\
The user is asking what 2+2 is. This is a straightforward arithmetic
question. I don't need any tools for this. The answer is 4. I should
respond directly without overthinking this. No need for a todo list
since this is fewer than three trivial steps.
"""

RAW_THINKING_MULTI_TOOL = """\
I need to research quantum computing threats to RSA encryption. Let me
start with a broad search. Found several papers from NIST on post-quantum
cryptography. The Shor's algorithm can factor large primes in polynomial
time on a quantum computer, which would break RSA-2048. Current estimates
suggest this needs about 4000 logical qubits. IBM's roadmap shows 100k
qubits by 2033. Google's Willow chip demonstrated 105 physical qubits
with below-threshold error correction. Let me now look at lattice-based
alternatives — CRYSTALS-Kyber was selected by NIST as the primary KEM.
I should also check the timeline estimates from various research groups.
The consensus seems to be "harvest now, decrypt later" is the real threat.
"""


# ── Quality assertion helpers ─────────────────────────────────────────


def assert_refined_prose(refined: str, min_sentences: int = 2, max_sentences: int = 6) -> None:
    """Assert the refined text meets prose quality standards."""
    # Not empty
    assert len(refined.strip()) > 0, "refined text is empty"

    # No markdown artifacts
    assert "```" not in refined, "contains code block"
    assert "**" not in refined, "contains bold markdown"
    assert "`" not in refined, "contains inline code"
    assert "- " not in refined.split("\n")[0][:3] if "\n" in refined else True, "starts with bullet"

    # No meta-commentary
    meta_phrases = ["the agent", "the model", "the system", "the AI", "I will"]
    lower = refined.lower()
    for phrase in meta_phrases:
        assert phrase not in lower, f"contains meta-phrase: {phrase!r}"

    # Sentence count (rough — split on . ! ?)
    sentences = [s.strip() for s in re.split(r"[.!?]+", refined) if s.strip()]
    assert len(sentences) >= min_sentences, (
        f"too few sentences ({len(sentences)}): {refined[:100]}"
    )
    assert len(sentences) <= max_sentences, (
        f"too many sentences ({len(sentences)}): {refined[:200]}"
    )

    # Not just the raw input truncated
    assert "[...internal reasoning truncated...]" not in refined, "got truncated fallback"


# ── Evals ─────────────────────────────────────────────────────────────


class TestRefinerProseQuality:
    """Verify refined thinking is engaging prose with interesting specifics."""

    def test_research_thinking_refined(self, venice_api_key: str) -> None:
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(RAW_THINKING_RESEARCH)

        assert_refined_prose(refined)
        # Should preserve interesting specifics from the raw thinking
        lower = refined.lower()
        has_specifics = any(
            term in lower
            for term in ["mesh", "tor", "nym", "briar", "censorship", "mixnet", "bluetooth"]
        )
        assert has_specifics, f"no specifics preserved: {refined}"

    def test_simple_thinking_refined(self, venice_api_key: str) -> None:
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(RAW_THINKING_SIMPLE)

        assert_refined_prose(refined, min_sentences=1, max_sentences=4)

    def test_multi_tool_thinking_refined(self, venice_api_key: str) -> None:
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(RAW_THINKING_MULTI_TOOL)

        assert_refined_prose(refined)
        lower = refined.lower()
        has_specifics = any(
            term in lower
            for term in ["quantum", "rsa", "qubit", "lattice", "nist", "shor", "encryption"]
        )
        assert has_specifics, f"no specifics preserved: {refined}"


class TestRefinerAsyncQuality:
    """Verify async refinement produces equivalent quality."""

    @pytest.mark.asyncio
    async def test_async_matches_sync_quality(self, venice_api_key: str) -> None:
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = await refiner.refine_async(RAW_THINKING_RESEARCH)

        assert_refined_prose(refined)
        lower = refined.lower()
        has_specifics = any(
            term in lower
            for term in ["mesh", "tor", "nym", "briar", "censorship"]
        )
        assert has_specifics, f"no specifics preserved: {refined}"


class TestRefinerEdgeCases:
    """Verify refiner handles edge cases gracefully with real API."""

    def test_very_long_input(self, venice_api_key: str) -> None:
        long_thinking = "Analysing data point. " * 500  # ~11k chars
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(long_thinking)

        # Should still produce reasonable output, not crash
        assert len(refined) > 0
        assert len(refined) < len(long_thinking)

    def test_non_english_input(self, venice_api_key: str) -> None:
        thinking = (
            "Ich muss nach Quantencomputing-Bedrohungen für RSA suchen. "
            "Die Shor-Algorithmus kann große Primzahlen in polynomieller Zeit faktorisieren. "
            "NIST hat CRYSTALS-Kyber als primären KEM ausgewählt."
        )
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(thinking)

        # Should produce some output (may be in English or German)
        assert len(refined) > 0

    def test_technical_jargon_preserved(self, venice_api_key: str) -> None:
        thinking = (
            "The QUIC protocol uses TLS 1.3 handshake with 0-RTT resumption. "
            "This means the client can send encrypted data in the first flight. "
            "The initial packet contains a CRYPTO frame with the ClientHello. "
            "Connection migration works by using connection IDs instead of "
            "5-tuples, so NAT rebinding doesn't break the connection. "
            "Compared to TCP+TLS, QUIC reduces connection setup from 2-3 RTTs to 1 RTT."
        )
        refiner = ThoughtRefinerPlugin(enabled=True)
        refined = refiner.refine_sync(thinking)

        assert_refined_prose(refined)
        lower = refined.lower()
        has_technical = any(
            term in lower
            for term in ["quic", "tls", "rtt", "handshake", "connection", "encrypt"]
        )
        assert has_technical, f"lost technical content: {refined}"
