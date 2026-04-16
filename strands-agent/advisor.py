# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""
LLM-as-advisor tool for Strands agents.

Provides a ``consult_advisor`` Strands tool that calls an external LLM
(Claude, GPT-4, DeepSeek, Gemini, etc.) for cognitive-heavy tasks like:
- Evaluating conflicting evidence
- Choosing between research strategies
- Synthesizing complex findings
- Reasoning about ambiguous or nuanced questions

The advisor is a *different* model from the agent's own backbone (Venice
GLM), giving the agent access to diverse reasoning capabilities.

Configuration (env vars):
    ADVISOR_API_KEY       – API key (auto-detects provider if not set)
    ADVISOR_API_BASE      – Base URL (auto-detects from key)
    ADVISOR_MODEL         – Model name (default varies by provider)
    ADVISOR_MAX_CALLS     – Max advisory calls per request (default 5)
    ADVISOR_MAX_TOKENS    – Max tokens per advisory response (default 2048)
"""

from __future__ import annotations

import logging
import os
import threading
import time

from strands import tool

log = logging.getLogger("advisor")

# ── Configuration ─────────────────────────────────────────────────────

ADVISOR_MAX_CALLS = int(os.environ.get("ADVISOR_MAX_CALLS", "5"))
ADVISOR_MAX_TOKENS = int(os.environ.get("ADVISOR_MAX_TOKENS", "2048"))

# Provider detection order: explicit key → OpenRouter → OpenAI → DeepSeek →
# Mistral → xAI → Groq → Google Gemini
_PROVIDER_CHAIN = [
    {
        "key_env": "ADVISOR_API_KEY",
        "base_env": "ADVISOR_API_BASE",
        "model_env": "ADVISOR_MODEL",
        "base_url": None,  # must be set explicitly
        "model": None,
    },
    {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4-20250514",
    },
    {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    {
        "key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
    },
    {
        "key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-3-mini",
    },
    {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
]

# ── Advisor system prompt ─────────────────────────────────────────────

ADVISOR_SYSTEM_PROMPT = """\
You are an expert cognitive advisor. A research agent is consulting you \
for help with complex reasoning tasks. Your role is to provide clear, \
structured, actionable advice.

When asked to evaluate evidence or make strategic decisions:
- Be direct and decisive — give a clear recommendation, not wishy-washy hedging
- Consider multiple angles but converge on the strongest position
- Highlight non-obvious connections or implications
- Point out logical fallacies or gaps in reasoning
- Suggest specific next steps or alternative approaches

When asked to synthesize or analyse:
- Identify the key themes and contradictions
- Weigh evidence quality (primary sources > secondary > anecdotal)
- Flag what's missing or uncertain
- Provide a structured summary

Keep responses focused and concise. The agent is mid-research and needs \
actionable guidance, not lengthy essays."""


# ── Lazy client initialisation ────────────────────────────────────────

_client = None
_client_model = None
_client_lock = threading.Lock()


def _get_client():
    """Lazily initialise the OpenAI-compatible client for the advisor LLM."""
    global _client, _client_model
    if _client is not None:
        return _client, _client_model

    with _client_lock:
        if _client is not None:
            return _client, _client_model

        from openai import OpenAI

        # Walk the provider chain and pick the first one with a key set
        for provider in _PROVIDER_CHAIN:
            key = os.environ.get(provider["key_env"], "")
            if not key:
                continue

            base_url = os.environ.get(
                provider.get("base_env", ""),
                provider.get("base_url"),
            ) if provider.get("base_env") else provider.get("base_url")
            model = os.environ.get(
                provider.get("model_env", ""),
                provider.get("model"),
            ) if provider.get("model_env") else provider.get("model")

            if not base_url or not model:
                continue

            _client = OpenAI(api_key=key, base_url=base_url)
            _client_model = model
            log.info(
                "Advisor initialised: provider=%s model=%s base=%s",
                provider["key_env"], model, base_url,
            )
            return _client, _client_model

        raise RuntimeError(
            "No advisor LLM configured. Set one of: "
            + ", ".join(p["key_env"] for p in _PROVIDER_CHAIN)
        )


# ── Per-request call counter ──────────────────────────────────────────

_call_count = 0
_call_lock = threading.Lock()


def reset_advisor():
    """Reset per-request advisory call counter. Call before each HTTP request."""
    global _call_count
    with _call_lock:
        _call_count = 0


def _increment_calls() -> int:
    """Increment and return the call count. Thread-safe."""
    global _call_count
    with _call_lock:
        _call_count += 1
        return _call_count


# ── Strands tool ──────────────────────────────────────────────────────


@tool
def consult_advisor(
    question: str,
    context: str = "",
) -> str:
    """Consult an external AI advisor for complex cognitive tasks.

    Use this tool when you need a second opinion or deeper reasoning on:
    - Evaluating conflicting or ambiguous evidence
    - Choosing between research strategies or search angles
    - Synthesizing complex multi-source findings
    - Reasoning about nuanced questions where your initial approach feels weak
    - Deciding whether to search more or synthesize what you have

    The advisor is a different, powerful reasoning model (e.g. Claude, GPT-4)
    that can provide fresh perspective on your research challenges.

    Args:
        question: The specific question or decision you need help with.
                  Be precise about what you're struggling with.
        context: Relevant context — findings so far, conflicting data points,
                 the original user query, etc. More context = better advice.

    Returns:
        The advisor's response with structured reasoning and recommendations.
    """
    count = _increment_calls()
    if count > ADVISOR_MAX_CALLS:
        return (
            f"[Advisory budget exceeded — {ADVISOR_MAX_CALLS} calls used this request. "
            f"Proceed with your own judgment.]"
        )

    try:
        client, model = _get_client()
    except RuntimeError as e:
        log.warning("Advisor unavailable: %s", e)
        return f"[Advisor unavailable: {e}. Proceed with your own judgment.]"

    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
    ]
    if context:
        messages.append({
            "role": "user",
            "content": f"**Research context:**\n{context}\n\n**Question:**\n{question}",
        })
    else:
        messages.append({"role": "user", "content": question})

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=ADVISOR_MAX_TOKENS,
            temperature=0.7,
        )
        advice = response.choices[0].message.content or "[Empty response from advisor]"
        elapsed = round(time.time() - start, 1)
        log.info(
            "Advisory call %d/%d completed in %.1fs (model=%s, tokens=%s)",
            count, ADVISOR_MAX_CALLS, elapsed, model,
            getattr(response.usage, "total_tokens", "?"),
        )
        return advice
    except Exception as e:
        log.error("Advisor call failed: %s", e)
        return f"[Advisor call failed: {e}. Proceed with your own judgment.]"
