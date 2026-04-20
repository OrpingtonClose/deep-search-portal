# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""LLM-as-judge evals for thought refinement quality.

Uses a separate LLM call to score the thought refiner's output on multiple
quality dimensions: engagement, specificity, conciseness, formatting, and
faithfulness to the original reasoning.

This replaces weak heuristic checks (sentence count, keyword presence) with
semantic evaluation that can detect whether the prose is *actually engaging*
and *contains interesting specifics*.

Requires: VENICE_API_KEY

Usage::

    pytest evals/test_integ_llm_judge.py -v
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
import pytest

from plugins.thought_refiner import ThoughtRefinerPlugin

pytestmark = pytest.mark.integ

logger = logging.getLogger(__name__)

# ── Judge configuration ───────────────────────────────────────────────

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google-gemma-4-26b-a4b-it")
JUDGE_API_BASE = os.environ.get(
    "JUDGE_API_BASE",
    os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1"),
)
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY",
    os.environ.get("VENICE_API_KEY", ""),
)

# Minimum acceptable scores (out of 10)
MIN_ENGAGEMENT = 5
MIN_SPECIFICITY = 5
MIN_CONCISENESS = 5
MIN_FORMATTING = 7
MIN_FAITHFULNESS = 5

JUDGE_SYSTEM_PROMPT = """\
You are a quality evaluator for AI-generated status updates. You will receive:
1. RAW THINKING: the original chain-of-thought from an AI agent
2. REFINED OUTPUT: a rewritten version meant for end users

Score the REFINED OUTPUT on these 5 dimensions (1-10 each):

- engagement: Does the prose draw the reader in? Is it written like a narrator \
describing interesting work, not a dry log entry? (1=boring/clinical, 10=compelling)
- specificity: Does it preserve interesting details from the raw thinking — \
names, numbers, protocols, surprising facts? (1=completely generic, 10=rich details)
- conciseness: Is it appropriately brief (2-5 sentences) without unnecessary \
repetition or filler? (1=way too long/verbose, 10=perfectly concise)
- formatting: Is it plain prose without markdown, code blocks, bullet points, \
quotes, or meta-commentary about "the agent"? (1=full of artifacts, 10=clean prose)
- faithfulness: Does it accurately represent what the agent is doing, without \
hallucinating actions or facts not in the raw thinking? (1=fabricated, 10=accurate)

Respond with ONLY a JSON object, no other text:
{"engagement": N, "specificity": N, "conciseness": N, "formatting": N, "faithfulness": N, "notes": "brief explanation"}\
"""

JUDGE_USER_TEMPLATE = """\
RAW THINKING:
{raw}

REFINED OUTPUT:
{refined}

Score the refined output (JSON only):"""


# ── Raw thinking samples ─────────────────────────────────────────────

SAMPLES = {
    "research_censorship": {
        "raw": (
            "The user wants to know about mesh networking for censorship circumvention. "
            "Let me search for this. I should check Tor, I2P, Nym, and Briar. "
            "First I'll use brave_web_search to find recent papers on mesh networking "
            "protocols. Then I'll look at Nym's mixnet architecture since it's newer "
            "and uses Sphinx packet format. I should also check Briar which uses "
            "Tor hidden services for peer-to-peer messaging. The key innovation in "
            "Nym is that it adds timing obfuscation via Poisson mixing, which defeats "
            "traffic analysis better than Tor's onion routing. Let me also look at "
            "academic papers from USENIX and IEEE S&P on circumvention effectiveness. "
            "I notice that most censorship circumvention tools have a bootstrapping "
            "problem — how do you distribute the tool if the internet is already censored? "
            "Briar solves this with Bluetooth/WiFi direct distribution."
        ),
        "expected_specifics": ["tor", "nym", "briar", "mesh", "censorship"],
    },
    "quantum_crypto": {
        "raw": (
            "I need to research quantum computing threats to RSA encryption. Let me "
            "start with a broad search. Found several papers from NIST on post-quantum "
            "cryptography. The Shor's algorithm can factor large primes in polynomial "
            "time on a quantum computer, which would break RSA-2048. Current estimates "
            "suggest this needs about 4000 logical qubits. IBM's roadmap shows 100k "
            "qubits by 2033. Google's Willow chip demonstrated 105 physical qubits "
            "with below-threshold error correction. Let me now look at lattice-based "
            "alternatives — CRYSTALS-Kyber was selected by NIST as the primary KEM. "
            "I should also check the timeline estimates from various research groups. "
            'The consensus seems to be "harvest now, decrypt later" is the real threat.'
        ),
        "expected_specifics": ["quantum", "rsa", "shor", "qubit", "nist"],
    },
    "simple_arithmetic": {
        "raw": (
            "The user is asking what 2+2 is. This is a straightforward arithmetic "
            "question. I don't need any tools for this. The answer is 4. I should "
            "respond directly without overthinking this. No need for a todo list "
            "since this is fewer than three trivial steps."
        ),
        "expected_specifics": [],
    },
    "web_protocols": {
        "raw": (
            "The QUIC protocol uses TLS 1.3 handshake with 0-RTT resumption. "
            "This means the client can send encrypted data in the first flight. "
            "The initial packet contains a CRYPTO frame with the ClientHello. "
            "Connection migration works by using connection IDs instead of "
            "5-tuples, so NAT rebinding doesn't break the connection. "
            "Compared to TCP+TLS, QUIC reduces connection setup from 2-3 RTTs to 1 RTT."
        ),
        "expected_specifics": ["quic", "tls", "rtt"],
    },
}


# ── Judge helper ──────────────────────────────────────────────────────


def _call_judge(raw: str, refined: str) -> dict[str, int | str]:
    """Call the judge LLM and return parsed scores.

    Returns:
        Dict with keys: engagement, specificity, conciseness, formatting,
        faithfulness (ints 1-10), and notes (str).
    """
    if not JUDGE_API_KEY:
        pytest.skip("JUDGE_API_KEY / VENICE_API_KEY not set")

    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(raw=raw, refined=refined)},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
        "stream": False,
        "venice_parameters": {"include_venice_system_prompt": False},
        "reasoning": {"effort": "none"},
    }

    resp = httpx.post(
        f"{JUDGE_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {JUDGE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30.0,
    )
    resp.raise_for_status()

    data = resp.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    text = msg.get("content", "") or msg.get("reasoning_content", "") or ""

    # Extract JSON from response (may have markdown wrapping)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    scores = json.loads(text)
    return scores


def _refine_and_judge(raw: str) -> tuple[str, dict[str, int | str]]:
    """Refine raw thinking and score the output with the judge LLM.

    Returns:
        Tuple of (refined_text, scores_dict).
    """
    refiner = ThoughtRefinerPlugin(enabled=True)
    refined = refiner.refine_sync(raw)
    time.sleep(2)  # rate limit buffer between refiner and judge calls
    scores = _call_judge(raw, refined)
    return refined, scores


# ── Evals ─────────────────────────────────────────────────────────────


class TestJudgeResearchQuality:
    """Judge-scored evals for research-heavy thinking blocks."""

    def test_censorship_research_quality(self, venice_api_key: str) -> None:
        refined, scores = _refine_and_judge(SAMPLES["research_censorship"]["raw"])

        logger.info("scores=%s | refined=%r", scores, refined[:100])
        assert scores["engagement"] >= MIN_ENGAGEMENT, (
            f"engagement={scores['engagement']}: {scores.get('notes', '')}"
        )
        assert scores["specificity"] >= MIN_SPECIFICITY, (
            f"specificity={scores['specificity']}: {scores.get('notes', '')}"
        )
        assert scores["formatting"] >= MIN_FORMATTING, (
            f"formatting={scores['formatting']}: {scores.get('notes', '')}"
        )
        assert scores["faithfulness"] >= MIN_FAITHFULNESS, (
            f"faithfulness={scores['faithfulness']}: {scores.get('notes', '')}"
        )

    def test_quantum_crypto_quality(self, venice_api_key: str) -> None:
        time.sleep(2)
        refined, scores = _refine_and_judge(SAMPLES["quantum_crypto"]["raw"])

        logger.info("scores=%s | refined=%r", scores, refined[:100])
        assert scores["engagement"] >= MIN_ENGAGEMENT, (
            f"engagement={scores['engagement']}: {scores.get('notes', '')}"
        )
        assert scores["specificity"] >= MIN_SPECIFICITY, (
            f"specificity={scores['specificity']}: {scores.get('notes', '')}"
        )
        assert scores["conciseness"] >= MIN_CONCISENESS, (
            f"conciseness={scores['conciseness']}: {scores.get('notes', '')}"
        )
        assert scores["faithfulness"] >= MIN_FAITHFULNESS, (
            f"faithfulness={scores['faithfulness']}: {scores.get('notes', '')}"
        )


class TestJudgeSimpleQuality:
    """Judge-scored evals for simple/short thinking blocks."""

    def test_simple_arithmetic_quality(self, venice_api_key: str) -> None:
        time.sleep(2)
        refined, scores = _refine_and_judge(SAMPLES["simple_arithmetic"]["raw"])

        logger.info("scores=%s | refined=%r", scores, refined[:100])
        # Simple inputs should still produce clean, concise output
        assert scores["conciseness"] >= MIN_CONCISENESS, (
            f"conciseness={scores['conciseness']}: {scores.get('notes', '')}"
        )
        assert scores["formatting"] >= MIN_FORMATTING, (
            f"formatting={scores['formatting']}: {scores.get('notes', '')}"
        )
        assert scores["faithfulness"] >= MIN_FAITHFULNESS, (
            f"faithfulness={scores['faithfulness']}: {scores.get('notes', '')}"
        )


class TestJudgeTechnicalQuality:
    """Judge-scored evals for technical content preservation."""

    def test_web_protocols_quality(self, venice_api_key: str) -> None:
        time.sleep(2)
        refined, scores = _refine_and_judge(SAMPLES["web_protocols"]["raw"])

        logger.info("scores=%s | refined=%r", scores, refined[:100])
        # Technical content should preserve specifics and be faithful
        assert scores["specificity"] >= MIN_SPECIFICITY, (
            f"specificity={scores['specificity']}: {scores.get('notes', '')}"
        )
        assert scores["faithfulness"] >= MIN_FAITHFULNESS, (
            f"faithfulness={scores['faithfulness']}: {scores.get('notes', '')}"
        )
        assert scores["formatting"] >= MIN_FORMATTING, (
            f"formatting={scores['formatting']}: {scores.get('notes', '')}"
        )


class TestJudgeAverageScores:
    """Aggregate quality gate — average scores across all samples."""

    def test_average_engagement_above_threshold(self, venice_api_key: str) -> None:
        """All research samples should average >= 6 engagement."""
        research_samples = ["research_censorship", "quantum_crypto", "web_protocols"]
        total = 0
        count = 0
        for name in research_samples:
            time.sleep(2)
            _, scores = _refine_and_judge(SAMPLES[name]["raw"])
            total += scores["engagement"]
            count += 1
            logger.info("sample=%s engagement=%d", name, scores["engagement"])

        avg = total / count
        assert avg >= 6, f"average engagement {avg:.1f} below threshold 6"

    def test_average_formatting_above_threshold(self, venice_api_key: str) -> None:
        """All samples should average >= 7 formatting (clean prose)."""
        total = 0
        count = 0
        for name, sample in SAMPLES.items():
            time.sleep(2)
            _, scores = _refine_and_judge(sample["raw"])
            total += scores["formatting"]
            count += 1
            logger.info("sample=%s formatting=%d", name, scores["formatting"])

        avg = total / count
        assert avg >= 7, f"average formatting {avg:.1f} below threshold 7"
