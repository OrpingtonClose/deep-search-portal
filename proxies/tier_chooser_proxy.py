#!/usr/bin/env python3
"""Tier Chooser Proxy — Two-tier model selection with author-direct routing.

Provides Fast (speed-optimised) and Thinking (reasoning-optimised) tiers.
Each tier races its models in parallel, scores responses, and returns the
best.  Individual model access is also supported.

All models route to their author's native API where an API key is
available; models without a native key fall back to OpenRouter.

Port: 9900 (configurable via TIER_CHOOSER_PORT)
"""

import asyncio
import html as html_mod
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from shared import (
    INGEST_DB_PATH,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_ingest_routes,
    register_standard_routes,
    setup_logging,
    stream_passthrough,
)
import knowledge_client
from search_providers import _search_searxng
from media_enrichment import (
    enrich_with_media,
    enrich_with_media_structured,
    fetch_transcript_for_video,
    search_youtube_videos,
    _extract_youtube_id as media_extract_yt_id,
    MEDIA_ENRICHMENT_MAX_VIDEOS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("TIER_CHOOSER_LOG_DIR", "/opt/tier_chooser_logs")
log = setup_logging("tier-chooser", LOG_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LISTEN_PORT = env_int("TIER_CHOOSER_PORT", 9900, minimum=1)
INGEST_DB = os.getenv("INGEST_DB", INGEST_DB_PATH)

MAX_CONCURRENT_MODELS = env_int("TIER_CHOOSER_MAX_CONCURRENT", 10, minimum=1)
MODEL_TIMEOUT = int(os.getenv("TIER_CHOOSER_MODEL_TIMEOUT", "90"))

# Synthesis configuration
SYNTHESIS_MODEL = os.getenv("TIER_CHOOSER_SYNTHESIS_MODEL", "google/gemini-3.1-flash")
KNOWLEDGE_NAMESPACE = os.getenv("TIER_CHOOSER_NAMESPACE", "tier-chooser")

IMAGE_ENRICHMENT_ENABLED = os.getenv("TIER_CHOOSER_IMAGE_ENRICHMENT", "true").lower() in ("1", "true", "yes")

# Deployment environment: "production" or "staging" (default).
# Controls which models appear in the /v1/models dropdown.
#   production: only Grok 4.20 (default) + Smart Combined (full-throttle race)
#   staging:    all tiers and individual models
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "staging").lower()

# ---------------------------------------------------------------------------
# Provider Registry — route models to their native APIs
# ---------------------------------------------------------------------------
# Maps the provider prefix (before the "/" in model IDs) to:
#   base_url: the provider's native OpenAI-compatible chat/completions endpoint
#   key_env:  the environment variable holding the API key
#
# If a provider's key is not set, falls back to OpenRouter.
# Providers with no direct API (open-weight hosts) always use OpenRouter.

PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    "openai":       {"base_url": "https://api.openai.com/v1",                                    "key_env": "OPENAI_API_KEY"},
    "anthropic":    {"base_url": "https://api.anthropic.com/v1",                                   "key_env": "ANTHROPIC_API_KEY"},
    "google":       {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",       "key_env": "GEMINI_API_KEY"},
    "x-ai":         {"base_url": "https://api.x.ai/v1",                                          "key_env": "XAI_API_KEY"},
    "deepseek":     {"base_url": "https://api.deepseek.com",                                     "key_env": "DEEPSEEK_API_KEY"},
    "perplexity":   {"base_url": "https://api.perplexity.ai",                                    "key_env": "PERPLEXITY_API_KEY"},
    "mistralai":    {"base_url": "https://api.mistral.ai/v1",                                    "key_env": "MISTRAL_NATIVE_API_KEY"},
    "qwen":         {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",       "key_env": "DASHSCOPE_API_KEY"},
    "moonshotai":   {"base_url": "https://api.moonshot.ai/v1",                                   "key_env": "MOONSHOT_API_KEY"},
    "cohere":       {"base_url": "https://api.cohere.com/compatibility/v1",                      "key_env": "COHERE_API_KEY"},
    "minimax":      {"base_url": "https://api.minimax.chat/v1",                                  "key_env": "MINIMAX_API_KEY"},
    "groq":         {"base_url": "https://api.groq.com/openai/v1",                               "key_env": "GROQ_API_KEY"},
    "stepfun":      {"base_url": "https://api.stepfun.com/v1",                                   "key_env": "STEPFUN_API_KEY"},
    "z-ai":         {"base_url": "https://open.bigmodel.cn/api/paas/v4",                         "key_env": "ZHIPU_API_KEY"},
    "nvidia":       {"base_url": "https://integrate.api.nvidia.com/v1",                          "key_env": "NVIDIA_API_KEY"},
    # Open-weight providers — no direct API, always use OpenRouter:
    # "meta-llama", "nousresearch", "xiaomi" are NOT listed here.
}


# Native API model name translations — some providers use different model
# names on their native API vs the OpenRouter-style prefix/name convention.
NATIVE_MODEL_MAP: dict[str, str] = {
    # xAI — native API uses versioned names with reasoning/non-reasoning suffix
    "grok-4.1-fast":       "grok-4-1-fast-non-reasoning",
    "grok-4":              "grok-4-0709",
    "grok-4.20":           "grok-4.20-0309-reasoning",
    "grok-4.20-non-reasoning": "grok-4.20-0309-non-reasoning",
    "grok-4.20-reasoning":    "grok-4.20-0309-reasoning",
    # Mistral — native API uses "-latest" aliases, not versioned suffixes
    "mistral-medium-3.1":  "mistral-medium-latest",
    "mistral-large-2512":  "mistral-large-latest",
    "mistral-large-3":     "mistral-large-latest",
    "mistral-large-3-efficient": "mistral-medium-latest",
    "mistral-large-4":     "mistral-large-latest",
    # Moonshot Kimi — native API model IDs
    "kimi-k2.5-instant":   "kimi-k2.5",
    "kimi-k2.5-thinking":  "kimi-k2.5",
    # Zhipu GLM — native API model IDs
    "glm-5-light":         "glm-5-turbo",
    "glm-5-thinking":      "glm-5.1",
    # Google Gemini — API uses "-preview" suffixes
    "gemini-3.1-flash":    "gemini-3-flash-preview",
    "gemini-3-pro":        "gemini-3-pro-preview",
    "gemini-3.1-pro":      "gemini-3.1-pro-preview",
    # OpenAI — high-effort variant is called "pro"
    "gpt-5.4-high":        "gpt-5.4-pro",
    # Qwen/DashScope — native API model IDs
    "qwen3.5-turbo":       "qwen3.5-flash",
    "qwen3.5-max":         "qwen3-max",
    "qwen3.5-72b":         "qwen3.5-plus",
    # DeepSeek — native API model IDs
    "deepseek-v3.2":       "deepseek-chat",
    "deepseek-v3.2-lite":  "deepseek-chat",
    "deepseek-r1":         "deepseek-reasoner",
    # Anthropic — native API uses hyphens (not dots); no haiku 4.6 exists yet
    "claude-haiku-4.6":    "claude-haiku-4-5-20251001",
    "claude-sonnet-4.6":   "claude-sonnet-4-6",
    "claude-opus-4.6":     "claude-opus-4-6",
}


def resolve_provider(model: str) -> tuple[str, str, str]:
    """Resolve a model ID to (base_url, api_key, native_model_name).

    For models whose provider prefix is in PROVIDER_REGISTRY and whose
    API key env var is set, returns the native endpoint.  The bare model
    name is translated via NATIVE_MODEL_MAP when a mapping exists.
    Otherwise falls back to OpenRouter (keeping the full model ID).
    """
    parts = model.split("/", 1)
    if len(parts) == 2:
        prefix, model_name = parts
        entry = PROVIDER_REGISTRY.get(prefix)
        if entry:
            key = os.environ.get(entry["key_env"], "")
            if key:
                native_name = NATIVE_MODEL_MAP.get(model_name, model_name)
                return entry["base_url"], key, native_name
    # Fallback: OpenRouter with full model ID
    return OPENROUTER_BASE, OPENROUTER_KEY, model


log.info(
    f"Config: env={DEPLOYMENT_ENV}, openrouter={OPENROUTER_BASE}, port={LISTEN_PORT}, "
    f"max_concurrent={MAX_CONCURRENT_MODELS}, timeout={MODEL_TIMEOUT}s"
)

# ---------------------------------------------------------------------------
# Two tiers: Fast (speed-optimised) and Thinking (reasoning-optimised)
# ---------------------------------------------------------------------------
# Models use ONLY native APIs (no OpenRouter fallback).  Each provider's
# API key env-var must be set.  Models whose providers have no native API
# (Meta Llama, inclusionAI Ling, etc.) are excluded — the user requires
# original-provider routing only.

TIER_MODELS = {
    "quick": [  # Fast Model — strict ≤1.5s, max tok/s, lowest latency
        "anthropic/claude-haiku-4.6",          # Anthropic fast tier
        "google/gemini-3.1-flash",             # Google DeepMind Flash
        "openai/gpt-5.4-mini",                 # OpenAI Nano/mini
        "x-ai/grok-4.20-non-reasoning",       # xAI Grok 4.20 fast/non-reasoning
        "mistralai/mistral-large-3-efficient", # Mistral Large 3 efficient (41B active)
        "deepseek/deepseek-v3.2-lite",         # DeepSeek V3.2 Lite/fast
        "qwen/qwen3.5-turbo",                  # Alibaba Qwen 3.5 Turbo/Light
        "z-ai/glm-5-light",                    # Zhipu GLM-5 Light
        "moonshotai/kimi-k2.5-instant",        # Moonshot Kimi K2.5 Instant
    ],
    "medium": [  # Best price/performance balance (unchanged)
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3-pro",
        "openai/gpt-5.4",
        "x-ai/grok-4",
        "deepseek/deepseek-v3.2",
        "qwen/qwen3.5-72b",
        "mistralai/mistral-large-4",
        "z-ai/glm-5",
    ],
    "full-throttle": [  # Maximum capability, no compromises
        "anthropic/claude-opus-4.6",      # Current coding/agentic king
        "google/gemini-3.1-pro-preview",  # Often #1 or #2 overall
        "openai/o3",                        # Highest-effort OpenAI reasoning model
        "x-ai/grok-4.20",                 # Competitive frontier model
        "deepseek/deepseek-reasoner",      # Top reasoning/value performer
        "qwen/qwq-plus",                  # Qwen thinking model (reasoning)
        "z-ai/glm-5",                     # Max-effort GLM variant
    ],
}

ALL_TIER_MODELS: list[str] = []
for models in TIER_MODELS.values():
    ALL_TIER_MODELS.extend(models)

# prevent GC of fire-and-forget tasks (Python < 3.12 weak-refs)
# See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
_background_tasks: set[asyncio.Task] = set()

# ============================================================================
# Refusal / Hedge Detection & Response Scoring
# ============================================================================

REFUSAL_PATTERNS = [
    re.compile(r"I (?:cannot|can't|won't|am not able to|am unable to|must decline to|have to decline)", re.I),
    re.compile(r"I'm (?:sorry|afraid|not able to)", re.I),
    re.compile(r"(?:against|violates?) (?:my|the) (?:guidelines|policies|terms|ethics|rules)", re.I),
    re.compile(r"I (?:don't|do not) (?:feel comfortable|think I should)", re.I),
    re.compile(r"I'm not (?:comfortable|going to|willing to|able to)", re.I),
    re.compile(r"(?:unethical|immoral|illegal|harmful|dangerous|inappropriate) (?:to|for me to)", re.I),
    re.compile(r"I (?:strongly |must )?(?:advise|recommend|urge|caution) against", re.I),
    re.compile(r"instead,? (?:I can|let me|I'd (?:suggest|recommend))", re.I),
    re.compile(r"I (?:need|have) to (?:be responsible|prioritize safety)", re.I),
]

HEDGE_PATTERNS = [
    re.compile(r"I (?:should|must|need to) (?:mention|note|point out|emphasize|stress)", re.I),
    re.compile(r"(?:please|it'?s important to) (?:note|keep in mind|be aware|remember|consider)", re.I),
    re.compile(r"(?:important|crucial|essential|critical) (?:disclaimer|warning|caveat|note)", re.I),
    re.compile(r"consult (?:a|with|your) (?:professional|doctor|lawyer|expert|physician)", re.I),
    re.compile(r"I (?:want|need) to (?:emphasize|stress|highlight|be (?:clear|transparent))", re.I),
    re.compile(r"(?:proceed|use|handle) with (?:caution|care|extreme care)", re.I),
    re.compile(r"(?:legal|ethical|safety) (?:implications|considerations|concerns)", re.I),
    re.compile(r"(?:highly|strongly) (?:recommend|advise|suggest|urge)", re.I),
    re.compile(r"\b(?:I|it'?s)\s+(?:important|worth|necessary)\s+to\s+(?:note|mention|consider|understand)", re.I),
    re.compile(r"\bbefore\s+(?:I|we)\s+(?:proceed|continue|begin|start)", re.I),
    re.compile(r"\bI\s+(?:want|need)\s+to\s+(?:be\s+clear|clarify|emphasize)", re.I),
    re.compile(r"\b(?:first|let\s+me)\s+(?:address|mention|note|point\s+out)", re.I),
    re.compile(r"\bwith\s+that\s+(?:said|in\s+mind|caveat)", re.I),
    re.compile(r"\bhaving\s+said\s+that", re.I),
    re.compile(r"\bthat\s+being\s+said", re.I),
]


def is_refusal(content: str) -> bool:
    """Check if response is a refusal."""
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(content):
            return True
    return False


def count_hedges(content: str) -> int:
    """Count hedge patterns in content."""
    count = 0
    for pattern in HEDGE_PATTERNS:
        if pattern.search(content):
            count += 1
    return count


def score_response(content: str, query: str) -> dict:
    """Score a response — higher is better.

    Direct port of G0DM0D3's scoreResponse function.
    Returns {"score": int, "is_refusal": bool, "hedge_count": int}.
    """
    lower_content = content.lower()
    lower_query = query.lower()

    refusal = is_refusal(content)
    hedge_count = count_hedges(content)

    if refusal:
        log.debug("[SCORE] REFUSAL DETECTED - auto-reject")
        return {"score": -9999, "is_refusal": True, "hedge_count": hedge_count}

    score = 0

    # LENGTH SCORING
    length = len(content)
    if length < 200:
        score += 10
    elif length < 500:
        score += 30
    elif length < 1000:
        score += 50
    elif length < 2000:
        score += 70
    elif length < 4000:
        score += 85
    else:
        score += 95

    # PENALTY: Each hedge pattern (-30)
    score -= hedge_count * 30

    # BONUS: Contains code blocks (+50)
    if re.search(r"```[\s\S]+```", content):
        score += 50

    # BONUS: Contains numbered lists or bullet points (+20)
    if re.search(r"^\s*(?:\d+\.|[-*\u2022])\s+", content, re.MULTILINE):
        score += 20

    # BONUS: Contains markdown headers (+15)
    if re.search(r"^#{1,3}\s+", content, re.MULTILINE):
        score += 15

    # BONUS: Query keywords in response (+5 each, max +50)
    query_words = [w for w in lower_query.split() if len(w) > 3]
    keyword_bonus = sum(5 for w in query_words if w in lower_content)
    score += min(keyword_bonus, 50)

    # BONUS: Technical/scientific terms (+40)
    if re.search(
        r"\b(?:synthesis|compound|reaction|molecule|equation|formula|procedure|"
        r"method|process|mechanism|catalyst|reagent|yield|temperature|"
        r"concentration|substrate|solvent|precipitate|filtration|distillation|"
        r"crystallization)\b",
        content, re.I,
    ):
        score += 40

    # BONUS: Starts directly with actionable content (+30)
    trimmed = content.strip()
    if re.match(r"^(?:Here's|Here are|The |To |First,?|Step|1\.|##|```|\*\*)", trimmed, re.I):
        score += 30

    # PENALTY: Starts with wishy-washy opener (-20)
    if re.match(r"^(?:I |Well,|So,|Okay,|Alright,|Let me)", trimmed, re.I):
        score -= 20

    # BONUS: Specific numbers/quantities (+25)
    number_matches = re.findall(
        r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|mg|g|kg|ml|L|cm|mm|m|km|hours?|"
        r"minutes?|seconds?|days?|weeks?|months?|years?|GB|MB|KB|TB|Hz|GHz|MHz|v\d))?\b",
        content, re.I,
    )
    if len(number_matches) >= 3:
        score += 25

    # BONUS: Real examples with specifics (+30)
    if re.search(
        r"(?:for example|for instance|such as|e\.g\.|specifically|in particular)[,:]?\s*[A-Z\d]",
        content, re.I,
    ):
        score += 30

    # BONUS: Working URLs or file paths (+15)
    if re.search(
        r"(?:https?://[^\s]+|/(?:usr|etc|var|home|opt)/[^\s]+|[A-Za-z]:\\[^\s]+)",
        content, re.I,
    ):
        score += 15

    # BONUS: Multiple sections with headers (+20)
    header_count = len(re.findall(r"^#{1,3}\s+", content, re.MULTILINE))
    if header_count >= 3:
        score += 20

    # BONUS: Inline code references (+15)
    if re.search(r"`[^`]+`", content):
        score += 15

    # BONUS: Domain expertise jargon (+25 each category)
    if re.search(
        r"\b(?:API|SDK|CLI|REST|GraphQL|OAuth|JWT|async|await|callback|"
        r"middleware|endpoint|webhook|microservice|containeriz|kubernetes|"
        r"docker|nginx|redis|postgres|mongodb)\b",
        content, re.I,
    ):
        score += 25
    if re.search(
        r"\b(?:exploit|vulnerability|CVE|payload|shellcode|injection|XSS|"
        r"CSRF|buffer overflow|privilege escalation|reverse shell|metasploit|"
        r"burp|nmap|wireshark)\b",
        content, re.I,
    ):
        score += 25
    if re.search(
        r"\b(?:mol(?:ar)?|stoichiom|exotherm|endotherm|oxidat|reduct|"
        r"electrolysis|titrat|spectroscop|chromatograph|centrifug)\b",
        content, re.I,
    ):
        score += 25

    # PENALTY: Obvious padding/filler (-15)
    if re.search(
        r"\b(?:basically|essentially|fundamentally|obviously|clearly|"
        r"simply put|in other words|that being said|having said that|"
        r"at the end of the day)\b",
        content, re.I,
    ):
        score -= 15

    # PENALTY: Meta-commentary (-20)
    if re.search(
        r"\b(?:I hope this helps|Let me know if you (?:need|have|want)|"
        r"Feel free to ask|Happy to (?:help|clarify|explain)|"
        r"I've (?:provided|included|outlined))\b",
        content, re.I,
    ):
        score -= 20

    # BONUS: Contains tables (+25)
    if re.search(r"\|.*\|.*\|", content, re.MULTILINE):
        score += 25

    # BONUS: Multiple code blocks (+30)
    code_block_count = len(re.findall(r"```", content)) // 2
    if code_block_count >= 2:
        score += 30

    # BONUS: Step-by-step instructions (+25)
    if re.search(r"(?:step\s*\d|first[,:]|second[,:]|third[,:]|finally[,:])", content, re.I):
        score += 25

    # BONUS: Warnings about real consequences showing expertise (+15)
    if re.search(
        r"\b(?:caution|warning|note|important|be careful|make sure|ensure)\b"
        r".*\b(?:because|otherwise|or else|could|might|will)\b",
        content, re.I,
    ):
        score += 15

    # PENALTY: Deflecting to other sources (-25, only if short)
    if re.search(
        r"\b(?:consult a (?:professional|doctor|lawyer|expert)|"
        r"seek (?:professional|medical|legal) (?:help|advice)|"
        r"I (?:recommend|suggest) (?:speaking|talking) to)\b",
        content, re.I,
    ):
        if length < 1000:
            score -= 25

    # PENALTY: Repetitive responses (-20)
    sentences = [s for s in re.split(r"[.!?]+", content) if len(s.strip()) > 20]
    unique = set(s.strip().lower()[:50] for s in sentences)
    if len(sentences) > 5 and len(unique) < len(sentences) * 0.7:
        score -= 20

    # BONUS: Actionable commands/code (+35)
    if re.search(r"(?:\$|>>>|>|#)\s*[a-z]", content, re.I | re.MULTILINE) or re.search(
        r"(?:npm|pip|yarn|brew|apt|cargo|docker|kubectl|git)\s+\w+", content, re.I
    ):
        score += 35

    # BONUS: Mathematical formulas (+20)
    if re.search(r"[=<>\u2264\u2265\u00b1\u00d7\u00f7\u2211\u220f\u222b\u221a\u221e]|\\frac|\\sqrt|\^[0-9{]", content):
        score += 20

    log.debug(f"[SCORE] Final score: {score} (length: {length}, hedges: {hedge_count})")
    return {"score": score, "is_refusal": False, "hedge_count": hedge_count}


# ============================================================================
# Model API call helpers (native provider routing with OpenRouter fallback)
# ============================================================================

# Models whose providers require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_PREFIXES = {"openai"}

# Thinking models that need enable_thinking=false for non-streaming calls
_THINKING_MODELS = {
    "qwen3-235b-a22b", "qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-max",
    "qwq-plus",
}

# Models that do not support custom temperature (only default=1)
_NO_CUSTOM_TEMPERATURE_MODELS = {
    "gpt-5",
    "o3",
}

# Models that MUST use streaming even when call_model() is invoked (non-streaming).
# qwq-plus: DashScope non-streaming returns empty content/reasoning_content
#           regardless of enable_thinking setting. Streaming works correctly.
# glm-5:   Zhipu reasoning model produces extensive reasoning_content before
#           content. Complex queries take 80-90s+ non-streaming, hitting the
#           90s timeout. Streaming avoids this since tokens flow progressively.
_FORCE_STREAM_MODELS = {
    "qwq-plus",
    "glm-5",
}


def _build_body(
    native_model: str,
    messages: list[dict],
    provider_prefix: str,
    *,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    """Build the request body with provider-specific parameter adjustments."""
    body: dict = {
        "model": native_model,
        "messages": messages,
        "stream": stream,
    }
    # Some models reject custom temperature; omit it so the API uses its default
    bare_model = native_model.split("/")[-1]
    if bare_model not in _NO_CUSTOM_TEMPERATURE_MODELS:
        body["temperature"] = temperature
    # Newer OpenAI models reject max_tokens; use max_completion_tokens
    if provider_prefix in _MAX_COMPLETION_TOKENS_PREFIXES:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
    # Qwen3 thinking models require enable_thinking=false for non-streaming
    if not stream and bare_model in _THINKING_MODELS:
        body["enable_thinking"] = False
    return body


def _extract_content(message: dict) -> str:
    """Extract text from a response message, falling back to reasoning_content.

    Reasoning models (DeepSeek Reasoner, GLM-5, GLM-4.7) may return their
    answer only in reasoning_content with empty content.
    """
    content = message.get("content", "") or ""
    if content.strip():
        return content
    # Fallback: some reasoning models put everything in reasoning_content
    return message.get("reasoning_content", "") or ""


async def call_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    error_out: list | None = None,
) -> str:
    """Call a single model via its native API (or OpenRouter fallback).

    Returns the model's text response, or empty string on failure.
    When *error_out* is provided (a list), error details are appended
    so callers can surface descriptive failure reasons.
    """
    # Some models only work in streaming mode on their native API.
    # Transparently collect streamed chunks into a single response.
    bare = model.split("/")[-1] if "/" in model else model
    if bare in _FORCE_STREAM_MODELS:
        chunks: list[str] = []
        try:
            async for chunk in stream_model(
                model, messages,
                temperature=temperature, max_tokens=max_tokens, req_id=req_id,
                content_only=True,
            ):
                chunks.append(chunk)
        except Exception as e:
            log.error(f"[{req_id}] {model} stream-as-call error: {e}")
            if error_out is not None:
                error_out.append(f"[ERROR] {model}: {type(e).__name__}: {e}")
            return ""
        text = "".join(chunks)
        if not text:
            log.warning(f"[{req_id}] {model} stream-as-call returned empty content")
            if error_out is not None:
                error_out.append(f"[EMPTY] {model} streamed but returned no content")
        return text

    base_url, api_key, native_model = resolve_provider(model)
    is_openrouter = (base_url == OPENROUTER_BASE)
    # Only apply provider-specific body tweaks when routing natively;
    # OpenRouter expects plain max_tokens / temperature for all models.
    provider_prefix = "" if is_openrouter else (model.split("/", 1)[0] if "/" in model else "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Deep Search Tier Chooser"

    body = _build_body(
        native_model, messages, provider_prefix,
        temperature=temperature, max_tokens=max_tokens, stream=False,
    )

    provider_label = "OpenRouter" if is_openrouter else base_url.split("//")[1].split("/")[0]
    client = http_client()
    try:
        resp = await asyncio.wait_for(
            client.post(
                f"{base_url}/chat/completions",
                json=body,
                headers=headers,
            ),
            timeout=MODEL_TIMEOUT,
        )
        if resp.status_code != 200:
            error_text = resp.text[:500]
            log.warning(f"[{req_id}] {provider_label} {model} returned {resp.status_code}: {error_text}")
            if error_out is not None:
                error_out.append(f"[ERROR:{resp.status_code}] {provider_label} returned HTTP {resp.status_code}")
            return ""

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log.warning(f"[{req_id}] {provider_label} {model} returned empty choices array")
            if error_out is not None:
                error_out.append(f"[EMPTY] {provider_label} returned 200 OK but empty choices array")
            return ""
        msg = choices[0].get("message", {})
        text = _extract_content(msg)
        if not text:
            finish = choices[0].get("finish_reason", "unknown")
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            log.warning(f"[{req_id}] {provider_label} {model} returned empty content (finish_reason={finish}, completion_tokens={tokens})")
            if error_out is not None:
                error_out.append(f"[EMPTY] {provider_label} returned 200 OK but empty content (finish_reason={finish}, {tokens} tokens used)")
        return text

    except asyncio.TimeoutError:
        log.warning(f"[{req_id}] {provider_label} {model} timed out after {MODEL_TIMEOUT}s")
        if error_out is not None:
            error_out.append(f"[TIMEOUT:{MODEL_TIMEOUT}s] {provider_label} did not respond in time")
        return ""
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} {model} error: {e}")
        if error_out is not None:
            error_out.append(f"[ERROR] {provider_label}: {type(e).__name__}: {e}")
        return ""


async def stream_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    content_only: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream a single model's response via its native API (or OpenRouter fallback).

    When *content_only* is True, only ``delta["content"]`` tokens are yielded
    and ``reasoning_content`` (thinking-chain) tokens are silently skipped.
    Use this when the collected text will be scored or fed to synthesis.
    """
    base_url, api_key, native_model = resolve_provider(model)
    is_openrouter = (base_url == OPENROUTER_BASE)
    provider_prefix = "" if is_openrouter else (model.split("/", 1)[0] if "/" in model else "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Deep Search Tier Chooser"

    body = _build_body(
        native_model, messages, provider_prefix,
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )

    provider_label = "OpenRouter" if is_openrouter else base_url.split("//")[1].split("/")[0]
    client = http_client()
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=MODEL_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning(f"[{req_id}] {provider_label} stream {model} error {resp.status_code}: {error_text}")
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Yield content; optionally fall back to reasoning_content
                    if content_only:
                        text_chunk = delta.get("content", "")
                    else:
                        text_chunk = delta.get("content", "") or delta.get("reasoning_content", "")
                    if text_chunk:
                        yield text_chunk
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} stream {model} exception: {e}")
        raise


# ============================================================================
# Neo4j Persistence — store all race results for knowledge accumulation
# ============================================================================

async def _store_race_results_neo4j(
    req_id: str,
    user_query: str,
    tier: str,
    results: list[dict],
) -> None:
    """Store all race results in Neo4j via the Knowledge Engine.

    Each model response is persisted as a condition so that the knowledge
    graph accumulates answers over time — matching the miropersistent
    pattern.
    """
    conditions = []
    for r in results:
        if r.get("is_empty") or not r.get("content"):
            continue
        # Normalise score into 0-1 range for confidence/trust fields
        norm_score = max(0.0, min(1.0, r["score"] / 300))
        conditions.append({
            "fact": r["content"][:10000],
            "source_url": f"tier-chooser:{tier}:{r['model']}",
            "confidence": 0.0 if r["is_refusal"] else norm_score,
            "trust_score": 0.0 if r["is_refusal"] else norm_score,
            "angle": r["model"],
            "domain": tier,
            "is_serendipitous": False,
            "serendipity_score": 0.0,
            "publication_date": "",
            "author": r["model"].split("/")[-1],
            "content_type": "model_response",
            "source_type": "tier_race",
        })
    if not conditions:
        return
    try:
        result = await knowledge_client.store_conditions(
            session_id=req_id,
            query=user_query,
            conditions=conditions,
            namespace=KNOWLEDGE_NAMESPACE,
        )
        stored = result.get("stored", len(conditions))
        log.info(f"[{req_id}] Stored {stored} race results in Neo4j (namespace={KNOWLEDGE_NAMESPACE})")
    except Exception as e:
        log.warning(f"[{req_id}] Neo4j storage failed (non-fatal): {e}")


# ============================================================================
# Synthesis — merge all valid responses into the richest possible answer
# ============================================================================

_SYNTHESIS_PROMPT = """You are a synthesis engine. You have received multiple responses from different AI models answering the same question. Your job is to produce a single comprehensive answer that captures the maximum wealth of information from ALL responses.

Rules:
- Extract and combine ALL unique facts, details, examples, code snippets, URLs, and actionable information from every response
- If models disagree on a point, present both perspectives with their reasoning
- Preserve specific numbers, dates, names, URLs, code blocks, and technical details from all responses
- Do NOT add hedging, disclaimers, or meta-commentary about the synthesis process
- Do NOT mention that multiple models were consulted or that this is a synthesis
- Structure the answer clearly with markdown headers and sections when content warrants it
- The final answer must read as a single authoritative response, not a compilation
- Prioritize depth, specificity, and completeness over brevity
- If one model provides a unique insight or example that others missed, ALWAYS include it"""

# ---------------------------------------------------------------------------
# Stage 3 prompt — media injection agent
# ---------------------------------------------------------------------------
_MEDIA_INJECT_PROMPT = """You are a media editor.  You receive a finished article (markdown) and a
catalog of available images and videos.  Your ONLY job is to insert media
items into the article at the most contextually appropriate positions so
the result reads like a beautifully illustrated magazine or encyclopedia
entry.

Rules:
1. Return the COMPLETE article text — every original word must remain.
   Do NOT rewrite, summarise, shorten, or rephrase any part of the text.
2. Insert media BETWEEN existing paragraphs or after section headings —
   never inside a sentence.
3. For images use:  ![<brief alt text>](<img_src URL>)
4. For videos with a thumbnail use:  [![<title>](<thumbnail URL>)](<video URL>)
5. For videos without a thumbnail use:  [▶ <title>](<video URL>)
6. Spread media evenly throughout the article; never cluster them.
7. Only insert an item when its title/description clearly matches the
   surrounding content.
8. You MUST use both images AND videos.  Videos are just as important as
   images — do NOT skip them.
9. If an item truly does not fit anywhere, skip it — but try hard.
10. Do NOT add any commentary, explanation, or meta-text of your own.
    Output ONLY the article with media inserted."""


async def _synthesize_responses(
    user_query: str,
    valid_results: list[dict],
    req_id: str,
) -> str:
    """Stage 2 — Merge all valid model responses into a single comprehensive answer.

    Uses the SYNTHESIS_MODEL (default: Gemini Flash for speed and large
    context window) to combine the best information from every response.
    This stage focuses ONLY on text quality — media is injected separately
    in Stage 3 by ``_inject_media_into_text``.

    Returns the synthesised text, or empty string on failure.
    """
    # Build the responses block — all models presented equally, no ranking
    response_parts = []
    for r in valid_results:
        short = r["model"].split("/")[-1]
        response_parts.append(
            f"--- {short} ---\n{r['content']}\n"
        )
    responses_text = "\n".join(response_parts)

    user_content = (
        f"User question: {user_query}\n\n"
        f"Model responses:\n{responses_text}\n\n"
        "Produce the most comprehensive, information-rich answer possible."
    )

    messages = [
        {"role": "system", "content": _SYNTHESIS_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.3, max_tokens=8192, req_id=req_id,
    )


async def _inject_media_into_text(
    synthesised_text: str,
    media: list[dict],
    req_id: str,
) -> str:
    """Stage 3 — Inject images and videos into the synthesised article.

    A separate LLM call (using the same fast SYNTHESIS_MODEL) that receives
    the finished article and a media catalog, then returns the article with
    media items woven in at contextually appropriate positions.

    The synthesis model never sees media — this keeps Stage 2 focused on
    text quality while Stage 3 is specialised for media placement.
    """
    media_lines = []
    for idx, item in enumerate(media, 1):
        if item["type"] == "image":
            media_lines.append(
                f"  IMAGE #{idx}: title={item['title']!r}, "
                f"description={item.get('description', '')!r}, "
                f"img_src={item['img_src']!r}"
            )
        elif item["type"] == "video":
            vid_id = item.get("video_id", "")
            thumb = item.get("thumbnail", "")
            media_lines.append(
                f"  VIDEO #{idx}: title={item['title']!r}, "
                f"description={item.get('description', '')!r}, "
                f"url={item['url']!r}, video_id={vid_id!r}, "
                f"thumbnail={thumb!r}"
            )

    user_content = (
        "ARTICLE:\n"
        + synthesised_text
        + "\n\nAVAILABLE MEDIA:\n"
        + "\n".join(media_lines)
        + "\n\nInsert the media items into the article. Output the full article with media embedded."
    )

    messages = [
        {"role": "system", "content": _MEDIA_INJECT_PROMPT},
        {"role": "user", "content": user_content},
    ]

    result = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.2, max_tokens=8192, req_id=req_id,
    )
    # If the media injection fails, return the original text unmodified
    return result if result else synthesised_text


# ============================================================================
# Image Enrichment — extract visual subjects and search for images
# ============================================================================

_IMAGE_SUBJECTS_PROMPT = """You are a visual editor. Given an answer text, identify 3-5 specific items, concepts, or entities mentioned in the answer that would benefit from an accompanying image. For each, output a short image search query (2-5 words) that would find a relevant, informative image.

Rules:
- Pick concrete, visual items (people, places, objects, diagrams, species, landmarks, etc.)
- Avoid abstract concepts that won't have good image results
- Output ONLY a JSON array of strings, e.g. ["query1", "query2", "query3"]
- No explanation, no markdown fences, just the JSON array"""


async def _extract_image_subjects(
    synthesised_answer: str,
    req_id: str,
) -> list[str]:
    """Use the synthesis model to identify items in the answer that deserve images."""
    messages = [
        {"role": "system", "content": _IMAGE_SUBJECTS_PROMPT},
        {"role": "user", "content": f"Answer text:\n{synthesised_answer[:6000]}"},
    ]
    raw = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.1, max_tokens=8192, req_id=req_id,
    )
    # call_model may return "" on failure
    if not raw:
        return []
    result = raw if isinstance(raw, str) else ""
    if not result:
        return []
    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        # Repair truncated JSON arrays (e.g. '["foo", "bar",')
        if cleaned.startswith("[") and not cleaned.endswith("]"):
            cleaned = cleaned.rstrip().rstrip(",") + "]"
            log.info(f"[{req_id}] Repaired truncated JSON array for image subjects")
        subjects = json.loads(cleaned)
        if isinstance(subjects, list):
            return [s for s in subjects if isinstance(s, str)][:5]
    except (json.JSONDecodeError, ValueError):
        log.warning(f"[{req_id}] Image subject extraction JSON parse error: {result[:300]}")
    return []


async def _search_images_for_subjects(
    subjects: list[str],
    req_id: str,
) -> list[dict]:
    """Search for images for each subject via SearXNG images category.

    Returns list of {"subject": str, "img_src": str, "title": str, "source_url": str}
    for subjects where an image was found.
    """
    async def search_one(subject: str) -> dict | None:
        try:
            results = await _search_searxng(
                subject, categories="images", max_results=5,
            )
            # Pick the first result that has an img_src
            for r in results:
                if r.img_src:
                    return {
                        "subject": subject,
                        "img_src": r.img_src,
                        "title": r.title or subject,
                        "source_url": r.url,
                    }
            return None
        except Exception as e:
            log.warning(f"[{req_id}] Image search for '{subject}' failed: {e}")
            return None

    tasks = [asyncio.create_task(search_one(s)) for s in subjects]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def _enrich_with_images(
    synthesised_answer: str,
    req_id: str,
) -> str:
    """Extract visual subjects from the answer, search for images, and append them.

    Returns the enriched answer with a 'Visual References' section appended,
    or the original answer unchanged if no images were found.
    """
    if not IMAGE_ENRICHMENT_ENABLED:
        return synthesised_answer

    try:
        subjects = await _extract_image_subjects(synthesised_answer, req_id)
        if not subjects:
            log.info(f"[{req_id}] No image subjects extracted")
            return synthesised_answer

        log.info(f"[{req_id}] Searching images for {len(subjects)} subjects: {subjects}")
        image_results = await _search_images_for_subjects(subjects, req_id)

        if not image_results:
            log.info(f"[{req_id}] No images found for any subject")
            return synthesised_answer

        # Build a visual references section
        def _esc_attr(text: str) -> str:
            """Escape text for safe use inside HTML attribute values."""
            return html_mod.escape(str(text), quote=True)

        image_section = "\n\n---\n\n### Visual References\n\n"
        for img in image_results:
            # Use HTML img with constrained width so images don't dominate
            alt = _esc_attr(img['title'])
            src = _esc_attr(img['img_src'])
            link = _esc_attr(img['source_url'])
            subject = _esc_attr(img['subject'])
            image_section += (
                f"**{subject}**\n\n"
                f'<a href="{link}"><img src="{src}" alt="{alt}" '
                f'style="max-width:320px; max-height:240px; border-radius:6px; '
                f'margin-bottom:8px;" /></a>\n\n'
            )

        return synthesised_answer + image_section

    except Exception as exc:
        log.warning(f"[{req_id}] Image enrichment failed (non-fatal): {exc}")
        return synthesised_answer


def _strip_media_images(media_section: str) -> str:
    """Remove the '### Visual References' image block from media_enrichment output.

    When our LLM-guided image enrichment is active it already provides a
    targeted '### Visual References' section.  The generic media_enrichment
    module may also produce one from a raw SearXNG image search.  To avoid
    duplicate image sections, strip the images portion and keep only
    '### Related Videos' (and anything after it).
    """
    if not media_section:
        return media_section
    videos_marker = "### Related Videos"
    idx = media_section.find(videos_marker)
    if idx != -1:
        # Keep everything from the videos header onward
        return "\n\n" + media_section[idx:]
    # No videos section — the entire media_section is images; drop it
    if "### Visual References" in media_section:
        return ""
    return media_section


async def _validate_image_urls(
    media_items: list[dict],
    req_id: str,
    timeout: float = 4.0,
) -> list[dict]:
    """HTTP HEAD-check each image URL in parallel; discard non-resolving ones.

    Videos are passed through without validation (YouTube thumbnails always
    resolve).  Images whose ``img_src`` returns a non-200 status or a
    non-image content-type are dropped.
    """
    if not media_items:
        return []

    client = http_client()

    async def _check(item: dict) -> dict | None:
        if item.get("type") != "image":
            return item  # videos pass through
        img_src = item.get("img_src", "")
        if not img_src:
            return None
        try:
            resp = await asyncio.wait_for(
                client.head(img_src, follow_redirects=True),
                timeout=timeout,
            )
            if resp.status_code != 200:
                log.debug(f"[{req_id}] Image URL HEAD {resp.status_code}: {img_src[:120]}")
                return None
            ct = resp.headers.get("content-type", "")
            if ct and not ct.startswith("image/"):
                log.debug(f"[{req_id}] Image URL non-image content-type '{ct}': {img_src[:120]}")
                return None
            return item
        except Exception:
            log.debug(f"[{req_id}] Image URL unreachable: {img_src[:120]}")
            return None

    checks = await asyncio.gather(*[_check(m) for m in media_items])
    valid = [m for m in checks if m is not None]
    dropped = len(media_items) - len(valid)
    if dropped:
        log.info(f"[{req_id}] Dropped {dropped} broken/unreachable image URL(s)")
    return valid


def _filter_media_by_relevance(
    media_items: list[dict],
    combined_text: str,
    req_id: str,
) -> list[dict]:
    """Filter media items to keep only those relevant to the combined model responses.

    Keeps items where at least 1 significant word (>3 chars) from the media's
    title+description appears in the combined text.
    Items with no extractable words are kept (benefit of the doubt).
    """
    if not media_items:
        return []
    lower_text = combined_text.lower()
    kept: list[dict] = []
    for item in media_items:
        title = (item.get("title", "") or "").lower()
        desc = (item.get("description", "") or "").lower()
        words = set(w for w in (title + " " + desc).split() if len(w) > 3)
        if not words:
            kept.append(item)  # no metadata to judge — keep it
            continue
        matching = sum(1 for w in words if w in lower_text)
        if matching >= 1:
            kept.append(item)
    discarded = len(media_items) - len(kept)
    if discarded:
        log.info(f"[{req_id}] Discarded {discarded} irrelevant media item(s)")
    return kept



# ============================================================================
# Video Quality — content-aware search + transcript-based ranking
# ============================================================================

_VIDEO_SUBJECTS_PROMPT = """You are a video editor. Given a finished article, identify 3-5 specific topics, \
concepts, or demonstrations mentioned in the article that would benefit from an \
accompanying YouTube video (tutorials, explainers, demonstrations, lectures, etc.).

For each, output a short YouTube search query (3-8 words) that would find a \
highly relevant, informative video whose spoken content directly addresses that \
part of the article.

Rules:
- Pick topics where a video would genuinely ADD information (demonstrations, \
  step-by-step guides, expert explanations, visual processes)
- Make queries specific enough to find niche, on-topic videos — not generic \
  "best of" compilations
- Output ONLY a JSON array of strings, e.g. ["query1", "query2", "query3"]
- No explanation, no markdown fences, just the JSON array"""


async def _extract_video_subjects(
    synthesised_text: str,
    req_id: str,
) -> list[str]:
    """Use the synthesis model to identify topics in the article that deserve videos."""
    messages = [
        {"role": "system", "content": _VIDEO_SUBJECTS_PROMPT},
        {"role": "user", "content": f"Article:\n{synthesised_text[:6000]}"},
    ]
    raw = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.1, max_tokens=8192, req_id=req_id,
    )
    if not raw:
        return []
    result = raw if isinstance(raw, str) else ""
    if not result:
        return []
    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        if cleaned.startswith("[") and not cleaned.endswith("]"):
            cleaned = cleaned.rstrip().rstrip(",") + "]"
            log.info(f"[{req_id}] Repaired truncated JSON array for video subjects")
        subjects = json.loads(cleaned)
        if isinstance(subjects, list):
            return [s for s in subjects if isinstance(s, str)][:5]
    except (json.JSONDecodeError, ValueError):
        log.warning(f"[{req_id}] Video subject extraction JSON parse error: {result[:300]}")
    return []


async def _search_videos_for_subjects(
    subjects: list[str],
    req_id: str,
) -> list[dict]:
    """Search YouTube for each video subject via TranscriptAPI.

    Returns a flat list of normalised video dicts (url, title, content, _source).
    """
    async def search_one(subject: str) -> list[dict]:
        try:
            results = await search_youtube_videos(subject, max_results=3)
            for r in results:
                r["_subject"] = subject
            return results
        except Exception as e:
            log.warning(f"[{req_id}] Video search for '{subject}' failed: {e}")
            return []

    tasks = [asyncio.create_task(search_one(s)) for s in subjects]
    all_results = await asyncio.gather(*tasks)
    # Flatten and deduplicate by video ID
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for results in all_results:
        for r in results:
            vid = media_extract_yt_id(r.get("url", ""))
            key = vid or r.get("url", "")
            if key and key not in seen_ids:
                seen_ids.add(key)
                merged.append(r)
    return merged


_VIDEO_RANK_PROMPT = """You are a video relevance judge.  You receive an article and a list of \
candidate YouTube videos with their transcript excerpts.

Your job: rank the videos by how well their spoken content complements the \
article.  A perfect video directly explains, demonstrates, or deepens a topic \
covered in the article.  A poor video is only tangentially related or covers \
a different subject entirely.

Output a JSON array of video numbers (1-based) in order from MOST relevant to \
LEAST relevant. Only include videos that are genuinely relevant — omit poor matches.

Example output: [3, 1, 5]

Rules:
- Judge by TRANSCRIPT CONTENT, not just title
- A video whose transcript closely mirrors a section of the article is ideal
- Omit videos that are off-topic, clickbait, or only loosely related
- Output ONLY the JSON array, no explanation"""


async def _rank_videos_by_transcript(
    synthesised_text: str,
    candidate_videos: list[dict],
    req_id: str,
    max_videos: int = 6,
) -> list[dict]:
    """Fetch transcripts for candidate videos and use LLM to rank by relevance.

    Returns the top *max_videos* videos, reordered by relevance to the article.
    """
    if not candidate_videos:
        return []

    # Fetch transcripts in parallel for all candidates
    async def fetch_one(item: dict) -> tuple[dict, str]:
        vid = media_extract_yt_id(item.get("url", ""))
        if not vid:
            return item, item.get("content", "") or ""
        transcript = await fetch_transcript_for_video(vid, max_chars=2000)
        return item, transcript or item.get("content", "") or ""

    pairs = await asyncio.gather(*[fetch_one(v) for v in candidate_videos])

    # Build the candidate list for the LLM
    video_descriptions: list[str] = []
    indexed_videos: list[dict] = []
    for idx, (item, transcript) in enumerate(pairs, 1):
        title = item.get("title", "Unknown")
        # Use transcript if available, fall back to existing content/description
        content_preview = transcript[:1500] if transcript else "(no transcript available)"
        video_descriptions.append(
            f"VIDEO #{idx}: \"{title}\"\n  Transcript: {content_preview}"
        )
        # Store the enriched transcript back on the item for Stage 3
        enriched = dict(item)
        if transcript:
            enriched["content"] = transcript[:500]  # keep a useful chunk for media injection
        indexed_videos.append(enriched)

    if not video_descriptions:
        return candidate_videos[:max_videos]

    user_content = (
        f"ARTICLE (first 4000 chars):\n{synthesised_text[:4000]}\n\n"
        f"CANDIDATE VIDEOS:\n" + "\n\n".join(video_descriptions) +
        "\n\nRank the videos by relevance. Output the JSON array of video numbers."
    )

    messages = [
        {"role": "system", "content": _VIDEO_RANK_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.1, max_tokens=8192, req_id=req_id,
    )

    if not raw:
        log.warning(f"[{req_id}] Video ranking LLM returned empty — using original order")
        return candidate_videos[:max_videos]

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        if cleaned.startswith("[") and not cleaned.endswith("]"):
            cleaned = cleaned.rstrip().rstrip(",") + "]"
        ranking = json.loads(cleaned)
        if isinstance(ranking, list):
            ranked: list[dict] = []
            for num in ranking:
                if isinstance(num, int) and 1 <= num <= len(indexed_videos):
                    ranked.append(indexed_videos[num - 1])
            if ranked:
                log.info(f"[{req_id}] Video ranking selected {len(ranked)} of {len(indexed_videos)} candidates")
                return ranked[:max_videos]
    except (json.JSONDecodeError, ValueError):
        log.warning(f"[{req_id}] Video ranking JSON parse error: {raw[:300]}")

    return candidate_videos[:max_videos]


# ============================================================================
# Tier Race — race models, store in Neo4j, synthesise the richest answer
# ============================================================================

async def run_tier_race(
    tier: str,
    messages: list[dict],
    user_query: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Race all models in a tier, store results in Neo4j, synthesise the richest answer."""
    request_id = f"chatcmpl-tier-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = f"tier-race-{tier}"

    def _chunk(content: str, finish_reason=None, reasoning_content: str = None):
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n"

    models = TIER_MODELS.get(tier, [])
    if not models:
        yield _chunk(f"Unknown tier: {tier}", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Fire media enrichment concurrently — but NOT for the quick tier (speed priority)
    is_quick_tier = (tier == "quick")
    if not is_quick_tier:
        media_task = asyncio.create_task(enrich_with_media_structured(user_query, req_id))
    else:
        media_task = None

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)

    async def query_model(model_name: str) -> dict:
        async with semaphore:
            errors: list[str] = []
            content = await call_model(
                model_name, messages,
                temperature=0.7, max_tokens=4096, req_id=req_id,
                error_out=errors,
            )
            if not content:
                error_detail = errors[0] if errors else ""
                return {"model": model_name, "content": "",
                        "score": -9999, "error": error_detail,
                        "is_refusal": False, "is_empty": True, "hedge_count": 0}
            result = score_response(content, user_query)
            result["is_empty"] = False
            return {"model": model_name, "content": content, **result}

    tasks = [asyncio.create_task(query_model(m)) for m in models]
    results = []
    completed = 0

    # Stream race status as reasoning_content (thought box)
    yield _chunk("", reasoning_content=f"TIER CHOOSER [{tier.upper()}] \u2014 racing {len(models)} models...\n")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if result.get("is_empty"):
            error_detail = result.get("error", "")
            status = f"FAILED \u2014 {error_detail}" if error_detail else "FAILED \u2014 unknown error"
        elif result["is_refusal"]:
            status = "REFUSAL"
        else:
            status = "OK"
        short_model = result["model"].split("/")[-1]
        log.info(f"[{req_id}] [{completed}/{len(models)}] {short_model}: {status}")
        yield _chunk("", reasoning_content=f"  [{completed}/{len(models)}] {short_model}: {status}\n")

    valid = [r for r in results if not r["is_refusal"] and not r.get("is_empty") and r["content"]]
    if not valid:
        empty_count = sum(1 for r in results if r.get("is_empty"))
        refusal_count = sum(1 for r in results if r["is_refusal"] and not r.get("is_empty"))
        msg = f"All {len(models)} models in the {tier} tier failed"
        if empty_count and refusal_count:
            msg += f" ({refusal_count} refused, {empty_count} timed out/errored)"
        elif empty_count:
            msg += f" ({empty_count} timed out/errored)"
        elif refusal_count:
            msg += f" ({refusal_count} refused)"
        details = []
        for r in results:
            short = r["model"].split("/")[-1]
            err = r.get("error", "")
            if err:
                details.append(f"  \u2022 {short}: {err}")
            elif r.get("is_refusal"):
                details.append(f"  \u2022 {short}: refused to answer")
        if details:
            msg += "\n\nDetails:\n" + "\n".join(details)
        msg += "\n\nTry rephrasing your question."
        if media_task is not None:
            media_task.cancel()
        yield _chunk(msg, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # --- Persist ALL results to Neo4j (fire-and-forget) ---
    _task = asyncio.create_task(_store_race_results_neo4j(req_id, user_query, tier, results))
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    # --- Collect structured media for inline embedding ---
    media_items: list[dict] = []
    if media_task is not None:
        try:
            media_items = await asyncio.wait_for(media_task, timeout=8.0)
        except Exception:
            media_items = []

    # --- Synthesise the richest answer from ALL valid responses ---
    valid_count = len(valid)

    # Quick tier: skip synthesis and media — just return the best single response
    if is_quick_tier:
        yield _chunk("", reasoning_content=f"\n{valid_count} model(s) responded. Selecting best answer...\n")
        best = max(valid, key=lambda r: r["score"])
        yield _chunk(best["content"], finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    combined_text = " ".join(r["content"] for r in valid if r.get("content"))

    # ====================================================================
    # 3-STAGE PIPELINE
    # Stage 1: Collect — dump API results + media to ephemeral temp dir
    # Stage 2: Synthesise — pure text, no media
    # Stage 3: Inject — LLM media agent weaves images/videos into the text
    # ====================================================================

    work_dir = tempfile.mkdtemp(prefix=f"tier-{req_id}-")
    try:
        # --- Stage 1: Dump to disk ---
        for idx, r in enumerate(valid):
            fpath = os.path.join(work_dir, f"response_{idx}_{r['model'].replace('/', '_')}.txt")
            with open(fpath, "w") as f:
                f.write(r["content"])
        if media_items:
            with open(os.path.join(work_dir, "media_catalog.json"), "w") as f:
                json.dump(media_items, f, indent=2)
        log.info(f"[{req_id}] Stage 1 complete: {len(valid)} responses + {len(media_items)} media items dumped to {work_dir}")

        # --- Validate image URLs (drop broken / unreachable) ---
        if media_items:
            media_items = await _validate_image_urls(media_items, req_id)

        # --- Extract content-aware image subjects and search for extra images ---
        try:
            subjects = await _extract_image_subjects(combined_text, req_id)
            if subjects:
                extra_images = await _search_images_for_subjects(subjects, req_id)
                if extra_images:
                    existing_srcs = {m.get("img_src", "") for m in media_items if m.get("type") == "image"}
                    pre_merge_count = len(media_items)
                    for img in extra_images:
                        src = img.get("img_src", "")
                        if src and src not in existing_srcs:
                            media_items.append({
                                "type": "image",
                                "url": img.get("source_url", src),
                                "title": img.get("title", "Image"),
                                "description": img.get("subject", ""),
                                "img_src": src,
                            })
                            existing_srcs.add(src)
                    new_items = media_items[pre_merge_count:]
                    validated_new = await _validate_image_urls(new_items, req_id)
                    media_items = media_items[:pre_merge_count] + validated_new
                    log.info(f"[{req_id}] After subject search: {len(media_items)} total media items")
        except Exception as exc:
            log.warning(f"[{req_id}] Subject-based image search failed (non-fatal): {exc}")

        # Filter media for relevance against the combined model responses
        if media_items:
            media_items = _filter_media_by_relevance(media_items, combined_text, req_id)

        # --- Stage 2: Synthesise (pure text, no media) ---
        synthesis_model_short = SYNTHESIS_MODEL.split("/")[-1]
        yield _chunk("", reasoning_content=f"\n{valid_count} model(s) responded. Synthesising via {synthesis_model_short}...\n")

        synthesised = await _synthesize_responses(user_query, valid, req_id)

        if not synthesised:
            fallback = max(valid, key=lambda r: len(r["content"]))
            yield _chunk(fallback["content"], finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        log.info(f"[{req_id}] Stage 2 complete: synthesis produced {len(synthesised)} chars")

        # Save synthesis to disk
        with open(os.path.join(work_dir, "synthesis.md"), "w") as f:
            f.write(synthesised)

        # --- Phase 2: Content-aware video search + transcript ranking ---
        if not is_quick_tier:
            try:
                yield _chunk("", reasoning_content="  Selecting best-matching videos...\n")
                video_subjects = await _extract_video_subjects(synthesised, req_id)
                if video_subjects:
                    log.info(f"[{req_id}] Video subjects extracted: {video_subjects}")
                    # Search YouTube for each subject
                    subject_videos = await _search_videos_for_subjects(video_subjects, req_id)
                    log.info(f"[{req_id}] Subject search found {len(subject_videos)} candidate videos")

                    # Merge with Phase 1 videos (existing media_items)
                    existing_video_ids = set()
                    for m in media_items:
                        if m.get("type") == "video":
                            vid = media_extract_yt_id(m.get("url", ""))
                            if vid:
                                existing_video_ids.add(vid)

                    # Add new subject-found videos to candidates
                    all_candidate_videos = [m for m in media_items if m.get("type") == "video"]
                    for sv in subject_videos:
                        vid = media_extract_yt_id(sv.get("url", ""))
                        if vid and vid not in existing_video_ids:
                            existing_video_ids.add(vid)
                            all_candidate_videos.append({
                                "type": "video",
                                "url": sv["url"],
                                "title": sv.get("title", "Video"),
                                "description": sv.get("content", ""),
                                "video_id": vid,
                                "thumbnail": f"https://img.youtube.com/vi/{vid}/mqdefault.jpg",
                            })

                    # Rank ALL candidate videos by transcript relevance
                    if all_candidate_videos:
                        ranked_videos = await _rank_videos_by_transcript(
                            synthesised, all_candidate_videos, req_id,
                            max_videos=MEDIA_ENRICHMENT_MAX_VIDEOS,
                        )
                        # Replace video items in media_items with ranked selection
                        non_video_items = [m for m in media_items if m.get("type") != "video"]
                        media_items = non_video_items + ranked_videos
                        log.info(f"[{req_id}] After video ranking: {len(ranked_videos)} videos selected")
                else:
                    log.info(f"[{req_id}] No video subjects extracted — keeping original videos")
            except Exception as exc:
                log.warning(f"[{req_id}] Video quality pipeline failed (non-fatal): {exc}")

        # --- Stage 3: Inject media into the synthesised text ---
        if media_items:
            img_count = sum(1 for m in media_items if m["type"] == "image")
            vid_count = sum(1 for m in media_items if m["type"] == "video")
            yield _chunk("", reasoning_content=f"  Injecting {img_count} image(s) + {vid_count} video(s) into article...\n")

            final_text = await _inject_media_into_text(synthesised, media_items, req_id)

            log.info(f"[{req_id}] Stage 3 complete: final article {len(final_text)} chars")

            # Save final output to disk
            with open(os.path.join(work_dir, "final_output.md"), "w") as f:
                f.write(final_text)

            yield _chunk(final_text, finish_reason="stop")
        else:
            yield _chunk(synthesised, finish_reason="stop")

    finally:
        # Cleanup ephemeral work dir
        try:
            shutil.rmtree(work_dir)
            log.info(f"[{req_id}] Cleaned up work dir {work_dir}")
        except Exception as exc:
            log.warning(f"[{req_id}] Failed to clean up {work_dir}: {exc}")

    yield "data: [DONE]\n\n"



# ============================================================================
# Single-model streaming access (no racing, no scoring)
# ============================================================================

async def _stream_single_model(
    model: str,
    messages: list[dict],
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Stream a single model's response directly — no racing, no system prompt injection."""
    request_id = f"chatcmpl-tier-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Pass messages as-is to the model
    collected = []
    try:
        async for content_chunk in stream_model(model, messages, temperature=0.7, max_tokens=4096, req_id=req_id):
            collected.append(content_chunk)
            yield make_sse_chunk(
                content_chunk,
                request_id=request_id,
                created=created,
                model_id=f"tier-{model}",
            )
    except Exception as e:
        log.error(f"[{req_id}] {model} direct-stream error: {e}")

    yield make_sse_chunk("", request_id=request_id, created=created, model_id=f"tier-{model}", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# /v1/models endpoint
# ============================================================================

# ---------------------------------------------------------------------------
# Production model list — available on BOTH production and staging
# ---------------------------------------------------------------------------
_PRODUCTION_MODELS = [
    {
        "id": "tier-x-ai/grok-4.20",
        "object": "model",
        "created": 1700000000,
        "owned_by": "tier-chooser",
        "name": "Grok 4.20",
    },
    {
        "id": "tier-race-full-throttle",
        "object": "model",
        "created": 1700000000,
        "owned_by": "tier-chooser",
        "name": "Smart Combined",
    },
]


def build_model_list() -> list[dict]:
    # Production models are always included (both prod + staging)
    models = list(_PRODUCTION_MODELS)

    if DEPLOYMENT_ENV != "production":
        # Staging-only: add all tier races and individual models
        for tier_name in TIER_MODELS:
            # Skip full-throttle race — already in production models as "Smart Combined"
            if tier_name == "full-throttle":
                continue
            models.append({
                "id": f"tier-race-{tier_name}",
                "object": "model",
                "created": 1700000000,
                "owned_by": "tier-chooser",
                "name": f"Tier Race: {tier_name.replace('-', ' ').title()}",
            })
        # Individual models within each tier
        for tier_name, tier_models in TIER_MODELS.items():
            for m in tier_models:
                # Skip grok-4.20 individual — already in production models
                if m == "x-ai/grok-4.20":
                    continue
                models.append({
                    "id": f"tier-{m}",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "tier-chooser",
                    "name": f"[{tier_name.replace('-', ' ').title()}] {m}",
                })
    return models


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Tier Chooser Proxy")
tracker = RequestTracker()

register_standard_routes(
    app,
    service_name="tier-chooser",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={"port": LISTEN_PORT, "env": DEPLOYMENT_ENV, "tiers": list(TIER_MODELS.keys())},
)
register_ingest_routes(app, INGEST_DB, log)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({"object": "list", "data": build_model_list()})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}})

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages array is required", "type": "invalid_request"}})

    requested_model = body.get("model", "tier-race-quick")
    utility = is_utility_request(messages)

    log.info(f"[{req_id}] New request: model={requested_model}, messages={len(messages)}, utility={utility}")
    tracker.start(req_id, model=requested_model, messages=len(messages))

    # Extract user query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            break

    if not user_query:
        tracker.finish(req_id)
        return JSONResponse(status_code=400, content={"error": {"message": "No user message found", "type": "invalid_request"}})

    # Utility requests — passthrough to a cheap fast model
    if utility:
        client_wants_stream = body.get("stream", False)
        base_url, api_key, native_model = resolve_provider("google/gemini-2.5-flash")
        if not client_wants_stream:
            log.info(f"[{req_id}] Utility request \u2014 NON-STREAMING passthrough")
            from shared import utility_passthrough_json
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=base_url,
                upstream_key=api_key,
                upstream_model=native_model,
                log=log,
            )
            tracker.finish(req_id)
            return result
        log.info(f"[{req_id}] Utility request \u2014 passthrough")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=base_url,
            upstream_key=api_key,
            upstream_model=native_model,
            model_id=requested_model,
            tracker=tracker,
            log=log,
        )
    elif requested_model.startswith("tier-race-"):
        tier = requested_model.replace("tier-race-", "")
        if tier not in TIER_MODELS:
            tracker.finish(req_id)
            return JSONResponse(status_code=400, content={"error": {"message": f"Unknown tier: {tier}. Valid: {list(TIER_MODELS.keys())}", "type": "invalid_request"}})
        generator = run_tier_race(tier, messages, user_query, req_id)
    elif requested_model.startswith("tier-"):
        # Direct model access: "tier-google/gemini-2.5-flash" etc.
        actual_model = requested_model[5:]  # strip "tier-"
        if actual_model not in ALL_TIER_MODELS:
            tracker.finish(req_id)
            return JSONResponse(status_code=400, content={"error": {"message": f"Unknown model: {actual_model}", "type": "invalid_request"}})
        generator = _stream_single_model(actual_model, messages, req_id)
    else:
        # Default to fast race for unknown model IDs
        generator = run_tier_race("quick", messages, user_query, req_id)

    async def tracked_generator():
        try:
            async for chunk in generator:
                yield chunk
        finally:
            tracker.finish(req_id)

    return StreamingResponse(
        tracked_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Tier Chooser Proxy on port {LISTEN_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
