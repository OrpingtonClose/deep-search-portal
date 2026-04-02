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
import json
import os
import re
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
from media_enrichment import enrich_with_media, enrich_with_media_structured

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
SYNTHESIS_MODEL = os.getenv("TIER_CHOOSER_SYNTHESIS_MODEL", "google/gemini-2.5-flash")
KNOWLEDGE_NAMESPACE = os.getenv("TIER_CHOOSER_NAMESPACE", "tier-chooser")

IMAGE_ENRICHMENT_ENABLED = os.getenv("TIER_CHOOSER_IMAGE_ENRICHMENT", "true").lower() in ("1", "true", "yes")

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
    "moonshotai":   {"base_url": "https://api.moonshot.cn/v1",                                   "key_env": "MOONSHOT_API_KEY"},
    "cohere":       {"base_url": "https://api.cohere.com/compatibility/v1",                      "key_env": "COHERE_API_KEY"},
    "minimax":      {"base_url": "https://api.minimax.chat/v1",                                  "key_env": "MINIMAX_API_KEY"},
    "groq":         {"base_url": "https://api.groq.com/openai/v1",                               "key_env": "GROQ_API_KEY"},
    "stepfun":      {"base_url": "https://api.stepfun.com/v1",                                   "key_env": "STEPFUN_API_KEY"},
    "z-ai":         {"base_url": "https://open.bigmodel.cn/api/paas/v4",                         "key_env": "ZHIPU_API_KEY"},
    "nvidia":       {"base_url": "https://integrate.api.nvidia.com/v1",                          "key_env": "NVIDIA_API_KEY"},
    # Open-weight providers — no direct API, always use OpenRouter:
    # "meta-llama", "nousresearch", "xiaomi" are NOT listed here.
}


def resolve_provider(model: str) -> tuple[str, str, str]:
    """Resolve a model ID to (base_url, api_key, native_model_name).

    For models whose provider prefix is in PROVIDER_REGISTRY and whose
    API key env var is set, returns the native endpoint.
    Otherwise falls back to OpenRouter (keeping the full model ID).
    """
    parts = model.split("/", 1)
    if len(parts) == 2:
        prefix, model_name = parts
        entry = PROVIDER_REGISTRY.get(prefix)
        if entry:
            key = os.environ.get(entry["key_env"], "")
            if key:
                return entry["base_url"], key, model_name
    # Fallback: OpenRouter with full model ID
    return OPENROUTER_BASE, OPENROUTER_KEY, model


log.info(
    f"Config: openrouter={OPENROUTER_BASE}, port={LISTEN_PORT}, "
    f"max_concurrent={MAX_CONCURRENT_MODELS}, timeout={MODEL_TIMEOUT}s"
)

# ---------------------------------------------------------------------------
# Two tiers: Fast (speed-optimised) and Thinking (reasoning-optimised)
# ---------------------------------------------------------------------------
# Models use ONLY native APIs (no OpenRouter fallback) where the
# corresponding API key env-var is set.  Models whose providers have no
# native API key (Phi-5, Llama 4, Command A, Yi-1.5, Hunyuan-Lite) are
# excluded — they require OpenRouter which the user explicitly declined.

TIER_MODELS = {
    "quick": [  # Fastest inference, lowest cost, still usable quality
        "google/gemini-3-flash",          # Top fast multimodal + reasoning
        "x-ai/grok-4.1-fast",             # Proven high-throughput & cheap
        "deepseek/deepseek-v3.2-chat",    # Excellent speed/quality MoE
        "openai/gpt-5.4-mini",            # Upgraded from gpt-4o (faster + smarter)
        "mistralai/mistral-medium-4",     # Latest fast Mistral variant
        "qwen/qwen3-72b-instruct-fast",   # Strong fast Chinese option
    ],
    "medium": [  # Best price/performance balance
        "anthropic/claude-sonnet-4.6",    # Outstanding coding/writing balance
        "google/gemini-3-pro",            # Broad reasoning + multimodal leader
        "openai/gpt-5.4",                 # Versatile all-rounder
        "x-ai/grok-4",                    # Strong real-time/unfiltered edge
        "deepseek/deepseek-v3.2",         # Insane value MoE
        "qwen/qwen3.5-72b",               # Upgraded Qwen MoE
        "mistralai/mistral-large-4",      # Latest Mistral large
        "z-ai/glm-5",                     # Strong Chinese contender
    ],
    "full-throttle": [  # Maximum capability, no compromises
        "anthropic/claude-opus-4.6",      # Current coding/agentic king
        "google/gemini-3.1-pro-preview",  # Often #1 or #2 overall
        "openai/gpt-5.4-high",            # Highest-effort GPT-5 variant
        "x-ai/grok-4.20",                 # Competitive frontier model
        "deepseek/deepseek-r1",           # Top reasoning/value performer
        "qwen/qwen3-235b-a22b",           # Massive MoE power
        "z-ai/glm-5-thinking",            # Max-effort GLM variant
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
}

# Models that do not support custom temperature (only default=1)
_NO_CUSTOM_TEMPERATURE_MODELS = {
    "gpt-5",
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


# Models known to produce reasoning_content alongside their answer.
# Thinking harvesting collects this chain-of-thought during the
# full-throttle race and feeds it to the synthesis agent.
_REASONING_MODELS = {
    "grok-4.20", "grok-4.20-0309-reasoning",
    "deepseek-r1", "deepseek-reasoner",
    "glm-4.7", "glm-5",
    "qwen3-235b-a22b", "qwen3-235b-a22b-thinking-2507",
}


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


def _extract_content_and_reasoning(message: dict) -> tuple[str, str]:
    """Extract both content and reasoning_content from a response message.

    Returns (content, reasoning_content).  Content uses the same fallback
    logic as ``_extract_content`` so that models which put everything in
    reasoning_content still produce a usable answer.
    """
    content = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", "") or ""
    if not content.strip() and reasoning.strip():
        # Model put everything in reasoning_content — use it as content too
        return reasoning, reasoning
    return content, reasoning


async def call_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    harvest_reasoning: bool = False,
) -> str | dict:
    """Call a single model via its native API (or OpenRouter fallback).

    When *harvest_reasoning* is True, returns a dict
    ``{"content": str, "reasoning": str}`` so callers can access the
    chain-of-thought separately.  Otherwise returns a plain string.
    """
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
    _empty: str | dict = {"content": "", "reasoning": ""} if harvest_reasoning else ""
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
            return _empty

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return _empty
        msg = choices[0].get("message", {})
        if harvest_reasoning:
            content, reasoning = _extract_content_and_reasoning(msg)
            return {"content": content, "reasoning": reasoning}
        return _extract_content(msg)

    except asyncio.TimeoutError:
        log.warning(f"[{req_id}] {provider_label} {model} timed out after {MODEL_TIMEOUT}s")
        return _empty
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} {model} error: {e}")
        return _empty


async def stream_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
) -> AsyncGenerator[str, None]:
    """Stream a single model's response via its native API (or OpenRouter fallback)."""
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
                    # Yield content; fall back to reasoning_content for reasoning models
                    content = delta.get("content", "") or delta.get("reasoning_content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} stream {model} exception: {e}")


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
# Tidbits — stream interesting observations from model responses as thoughts
# ============================================================================

_TIDBITS_PROMPT = """You are an insightful observer analysing multiple AI model responses to the same question. Your job is to provide a brief, engaging stream of interesting tidbits — surprising agreements, notable disagreements, unique angles, or fascinating details that individual models contributed.

Rules:
- Be concise: 3-6 short bullet points, each one line
- Use an engaging, conversational tone — like a knowledgeable friend pointing things out
- Highlight: surprising facts one model found that others missed, interesting disagreements, unexpected angles, particularly clever explanations
- Use model names (short form) when attributing observations
- Do NOT summarise the responses — only highlight what's *interesting* about them
- Do NOT add disclaimers or meta-commentary
- Use bullet points (•) with no headers"""


async def _stream_tidbits(
    user_query: str,
    valid_results: list[dict],
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Stream interesting tidbits from model responses as thinking content.

    Calls the SYNTHESIS_MODEL in streaming mode to highlight notable
    observations about how the models responded differently.
    """
    response_parts = []
    for r in valid_results:
        short = r["model"].split("/")[-1]
        # Truncate to keep the tidbits prompt fast
        snippet = r["content"][:3000]
        response_parts.append(f"--- {short} ---\n{snippet}\n")
    responses_text = "\n".join(response_parts)

    messages = [
        {"role": "system", "content": _TIDBITS_PROMPT},
        {"role": "user", "content": (
            f"User question: {user_query}\n\n"
            f"Model responses:\n{responses_text}\n\n"
            f"What's interesting about these responses?"
        )},
    ]

    async for chunk in stream_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.5, max_tokens=1024, req_id=req_id,
    ):
        yield chunk


# ============================================================================
# Thinking Harvest — extract a tidbit from a single model's chain-of-thought
# ============================================================================

_THINKING_TIDBIT_PROMPT = """You are reading the internal chain-of-thought (reasoning trace) of an AI model that just answered a question. Extract ONE concise, interesting insight from this reasoning — something the model considered, a surprising connection it made, an internal debate it had, or a key step in its logic that a reader would find illuminating.

Rules:
- ONE sentence only, 20-40 words
- Be specific — reference the actual thought, not a vague summary
- Start with the model name in bold: **ModelName** thought...
- Capture the *flavour* of the reasoning — was it cautious? creative? methodical? conflicted?
- Do NOT add disclaimers
- If the reasoning is empty or trivial, respond with exactly: SKIP"""


async def _harvest_thinking_tidbit(
    model_name: str,
    reasoning: str,
    user_query: str,
    req_id: str,
) -> str:
    """Produce a single interesting tidbit from a model's reasoning trace.

    Returns a short string suitable for streaming as reasoning_content,
    or empty string if the reasoning is empty/trivial.
    """
    if not reasoning or len(reasoning.strip()) < 50:
        return ""

    short = model_name.split("/")[-1]
    # Take first 2000 chars of reasoning to keep the call fast
    snippet = reasoning[:2000]

    messages = [
        {"role": "system", "content": _THINKING_TIDBIT_PROMPT},
        {"role": "user", "content": (
            f"Question: {user_query}\n\n"
            f"Model: {short}\n\n"
            f"Internal reasoning:\n{snippet}"
        )},
    ]

    result = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.5, max_tokens=100, req_id=req_id,
    )
    if isinstance(result, dict):
        result = result.get("content", "")
    if not result or result.strip() == "SKIP":
        return ""
    return result.strip()


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

_SYNTHESIS_THINKING_ADDENDUM = """

You also have access to the internal chain-of-thought (reasoning traces) from some of the models. These reveal how the models actually thought through the problem — their doubts, alternative approaches they considered, and internal debates. Use these to:
- Identify insights that only appear in the reasoning but not in the final answers
- Note where a model's internal reasoning was more nuanced than its public answer
- If reasoning reveals a model was uncertain about a claim, reflect that uncertainty
- Do NOT reproduce the raw reasoning — distil the *insights* from it into the answer"""

_SYNTHESIS_MEDIA_ADDENDUM = """

You have been given a set of IMAGES and VIDEOS found for the user's query.
Embed them INLINE at contextually appropriate positions in the answer using
Markdown and LibreChat Artifacts:

- For images: use `![<brief alt text>](<img_src URL>)` placed right after the
  paragraph the image illustrates.  Verify the image title/description actually
  matches the surrounding content before including it.
- For YouTube videos: output a clickable thumbnail followed by an artifact:
  `[![<title>](<thumbnail URL>)](<video URL>)`
  then on a new line:
  `:::artifact{title="<title>" type="text/html" identifier="vid-<N>"}`
  `<iframe width="560" height="315" src="https://www.youtube.com/embed/<video_id>" frameborder="0" allowfullscreen></iframe>`
  `:::`
- For non-YouTube videos: use `[▶ <title>](<url>)` as a clickable link.
- Aim for at least one image per major section when relevant media is available.
- Do NOT dump all media at the end; scatter them throughout the text where they
  add value.  Skip any item whose title/description does not match the content.
- If no media is relevant to a section, omit media from that section entirely."""


async def _synthesize_responses(
    user_query: str,
    valid_results: list[dict],
    req_id: str,
    harvested_thinking: Optional[dict[str, str]] = None,
    media: Optional[list[dict]] = None,
) -> str:
    """Merge all valid model responses into a single comprehensive answer.

    Uses the SYNTHESIS_MODEL (default: Gemini Flash for speed and large
    context window) to combine the best information from every response.

    When *harvested_thinking* is provided (model_name -> reasoning text),
    the reasoning traces are included so the synthesis can extract insights
    that only appeared in the chain-of-thought.

    When *media* is provided (list of image/video dicts from
    enrich_with_media_structured), the synthesis model is instructed to
    embed them inline at contextually appropriate positions.

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

    # Build the user content
    user_content = (
        f"User question: {user_query}\n\n"
        f"Model responses:\n{responses_text}\n\n"
    )

    # Append harvested reasoning traces if available
    system_prompt = _SYNTHESIS_PROMPT
    if harvested_thinking:
        thinking_parts = []
        for model_name, reasoning in harvested_thinking.items():
            if reasoning and len(reasoning.strip()) > 50:
                short = model_name.split("/")[-1]
                # Cap each reasoning trace to keep the prompt manageable
                thinking_parts.append(
                    f"--- {short} internal reasoning ---\n{reasoning[:3000]}\n"
                )
        if thinking_parts:
            system_prompt += _SYNTHESIS_THINKING_ADDENDUM
            user_content += (
                "Model reasoning traces (chain-of-thought):\n"
                + "\n".join(thinking_parts) + "\n\n"
            )

    # Append media for inline placement
    if media:
        system_prompt += _SYNTHESIS_MEDIA_ADDENDUM
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
        user_content += (
            "Available media (embed inline where contextually relevant):\n"
            + "\n".join(media_lines) + "\n\n"
        )

    user_content += "Produce the most comprehensive, information-rich answer possible."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.3, max_tokens=8192, req_id=req_id,
    )


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
    result = await call_model(
        SYNTHESIS_MODEL, messages,
        temperature=0.1, max_tokens=256, req_id=req_id,
    )
    if not result:
        return []
    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        subjects = json.loads(cleaned)
        if isinstance(subjects, list):
            return [s for s in subjects if isinstance(s, str)][:5]
    except (json.JSONDecodeError, ValueError):
        log.warning(f"[{req_id}] Image subject extraction JSON parse error")
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
        def _esc_url(url: str) -> str:
            return url.replace('(', '%28').replace(')', '%29')

        def _esc_text(text: str) -> str:
            return text.replace('[', '\\[').replace(']', '\\]')

        image_section = "\n\n---\n\n### Visual References\n\n"
        for img in image_results:
            # Markdown image with link to source
            image_section += (
                f"**{_esc_text(img['subject'])}**\n\n"
                f"[![{_esc_text(img['title'])}]({_esc_url(img['img_src'])})]({_esc_url(img['source_url'])})\n\n"
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

    def _chunk(content: str, finish_reason=None, reasoning=None):
        delta = {}
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
        if content:
            delta["content"] = content
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

    yield _chunk("", reasoning=f"TIER CHOOSER [{tier.upper()}] \u2014 racing {len(models)} models...\n")

    # Fire media enrichment concurrently with model race (zero added latency)
    media_task = asyncio.create_task(enrich_with_media_structured(user_query, req_id))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)

    # Determine whether to harvest reasoning (thinking tier only)
    is_thinking_tier = (tier == "full-throttle")
    # Collected reasoning traces keyed by model name
    harvested_thinking: dict[str, str] = {}
    # Background tidbit tasks fired as models complete (full-throttle only)
    tidbit_tasks: list[asyncio.Task] = []

    async def query_model(model_name: str) -> dict:
        async with semaphore:
            # Pass messages as-is — NO system prompt injection
            raw = await call_model(
                model_name, messages,
                temperature=0.7, max_tokens=4096, req_id=req_id,
                harvest_reasoning=is_thinking_tier,
            )
            if is_thinking_tier and isinstance(raw, dict):
                content = raw.get("content", "")
                reasoning = raw.get("reasoning", "")
            else:
                content = raw if isinstance(raw, str) else raw.get("content", "")
                reasoning = ""
            if not content:
                # Empty string means timeout, HTTP error, or exception — not a refusal
                return {"model": model_name, "content": "", "reasoning": "",
                        "score": -9999,
                        "is_refusal": False, "is_empty": True, "hedge_count": 0}
            result = score_response(content, user_query)
            result["is_empty"] = False
            return {"model": model_name, "content": content, "reasoning": reasoning,
                    **result}

    tasks = [asyncio.create_task(query_model(m)) for m in models]
    results = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if result.get("is_empty"):
            status = "ERROR/TIMEOUT"
        elif result["is_refusal"]:
            status = "REFUSAL"
        else:
            status = "OK"
        short_model = result["model"].split("/")[-1]
        yield _chunk("", reasoning=f"  [{completed}/{len(models)}] {short_model}: {status}\n")

        # --- Progressive thinking harvest (thinking tier only) ---
        if is_thinking_tier and result.get("reasoning") and status == "OK":
            harvested_thinking[result["model"]] = result["reasoning"]
            # Fire a background task to produce a tidbit from this model's
            # reasoning — streamed progressively as each completes.
            t = asyncio.create_task(
                _harvest_thinking_tidbit(
                    result["model"], result["reasoning"],
                    user_query, req_id,
                )
            )
            tidbit_tasks.append(t)

        # Stream any completed tidbits immediately (progressive, not batched)
        newly_done = [t for t in tidbit_tasks if t.done()]
        for t in newly_done:
            tidbit_tasks.remove(t)
            try:
                tidbit = t.result()
                if tidbit:
                    yield _chunk("", reasoning=f"• {tidbit}\n")
            except Exception:
                pass

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
        msg += ". Try rephrasing."
        media_task.cancel()
        for t in tidbit_tasks:
            t.cancel()
        yield _chunk(msg, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # --- Persist ALL results to Neo4j (fire-and-forget) ---
    _task = asyncio.create_task(_store_race_results_neo4j(req_id, user_query, tier, results))
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    refusal_count = sum(1 for r in results if r["is_refusal"] and not r.get("is_empty"))
    empty_count = sum(1 for r in results if r.get("is_empty"))
    model_names = ", ".join(r["model"].split("/")[-1] for r in valid)

    yield _chunk("", reasoning=(
        f"\nResults: {len(valid)} valid / {refusal_count} refused / "
        f"{empty_count} error\n"
        f"Valid: {model_names}\n"
    ))

    # --- Drain any remaining thinking tidbits (most already streamed progressively) ---
    if tidbit_tasks:
        done, pending = await asyncio.wait(tidbit_tasks, timeout=5.0)
        for t in pending:
            t.cancel()
        for t in done:
            try:
                tidbit = t.result()
                if tidbit:
                    yield _chunk("", reasoning=f"• {tidbit}\n")
            except Exception:
                pass
    if harvested_thinking:
        thinking_count = sum(1 for v in harvested_thinking.values() if v.strip())
        if thinking_count:
            yield _chunk("", reasoning=(
                f"\nCaptured reasoning from {thinking_count} model(s) — "
                f"feeding into synthesis.\n"
            ))

    # --- Collect structured media for inline embedding ---
    try:
        media_items: list[dict] = await asyncio.wait_for(media_task, timeout=8.0)
    except Exception:
        media_items = []

    media_count = len(media_items)
    if media_count:
        img_count = sum(1 for m in media_items if m["type"] == "image")
        vid_count = sum(1 for m in media_items if m["type"] == "video")
        yield _chunk("", reasoning=(
            f"\nMedia found: {img_count} images, {vid_count} videos "
            f"— will embed inline.\n"
        ))

    # --- Synthesise the richest answer from ALL valid responses ---
    if len(valid) == 1:
        yield _chunk("", reasoning="Single valid response. Searching for relevant images...\n")
        enriched = await _enrich_with_images(valid[0]["content"], req_id)
        yield _chunk(enriched, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Stream interesting tidbits from the responses as thinking content
    yield _chunk("", reasoning=(
        f"\nAnalysing {len(valid)} responses for interesting tidbits...\n\n"
    ))
    async for tidbit_chunk in _stream_tidbits(user_query, valid, req_id):
        yield _chunk("", reasoning=tidbit_chunk)

    yield _chunk("", reasoning=(
        f"\n\nSynthesising comprehensive answer from {len(valid)} responses "
        f"(via {SYNTHESIS_MODEL.split('/')[-1]})...\n"
    ))

    synthesised = await _synthesize_responses(
        user_query, valid, req_id,
        harvested_thinking=harvested_thinking if harvested_thinking else None,
        media=media_items if media_items else None,
    )

    if synthesised:
        yield _chunk("", reasoning="Synthesis complete. Searching for relevant images...\n")
        enriched = await _enrich_with_images(synthesised, req_id)
        if enriched != synthesised:
            image_count = enriched.count("![") - synthesised.count("![")
            yield _chunk("", reasoning=f"Found {image_count} relevant images.\n")
        else:
            yield _chunk("", reasoning="No relevant images found.\n")
        yield _chunk(enriched, finish_reason="stop")
    else:
        # Synthesis failed — return the longest response as a reasonable fallback
        fallback = max(valid, key=lambda r: len(r["content"]))
        yield _chunk("", reasoning="Synthesis failed — returning longest response as fallback.\n")
        yield _chunk(fallback["content"], finish_reason="stop")
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
    async for content_chunk in stream_model(model, messages, temperature=0.7, max_tokens=4096, req_id=req_id):
        collected.append(content_chunk)
        yield make_sse_chunk(
            content_chunk,
            request_id=request_id,
            created=created,
            model_id=f"tier-{model}",
        )

    yield make_sse_chunk("", request_id=request_id, created=created, model_id=f"tier-{model}", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# /v1/models endpoint
# ============================================================================

def build_model_list() -> list[dict]:
    models = []
    # Race modes
    for tier_name in TIER_MODELS:
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
    health_extras={"port": LISTEN_PORT, "tiers": list(TIER_MODELS.keys())},
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
        log.info(f"[{req_id}] Utility request \u2014 passthrough")
        # Use Gemini Flash for utility (native, fast, cheap)
        base_url, api_key, native_model = resolve_provider("google/gemini-2.5-flash")
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
