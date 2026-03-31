#!/usr/bin/env python3
"""
Tier Chooser Proxy — Three-tier model selection with author-direct routing.

Provides Quick / Medium / Full Throttle tiers. Each tier races its models
in parallel, scores responses, and returns the best. Individual model
access is also supported.

All models route to their author's native API. Only Anthropic models
fall back to OpenRouter (Anthropic's API is not OpenAI-compatible).

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
    # NOTE: Anthropic is NOT listed here — its API is not OpenAI-compatible
    # (uses /v1/messages, x-api-key header, different body format).
    # Anthropic models always route through OpenRouter.
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
# Three tiers
# ---------------------------------------------------------------------------

TIER_MODELS = {
    "quick": [
        "google/gemini-2.5-flash",
        "x-ai/grok-4-1-fast-non-reasoning",
        "deepseek/deepseek-chat",
        "openai/gpt-4o",
        "mistralai/mistral-medium-3.1",
        "qwen/qwen-2.5-72b-instruct",
    ],
    "medium": [
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3-pro-preview",
        "openai/gpt-5.4-chat",
        "x-ai/grok-4-1-fast-reasoning",
        "deepseek/deepseek-v3.2",
        "qwen/qwen3.5-plus-02-15",
        "mistralai/mistral-large-2512",
        "z-ai/glm-5",
    ],
    "full-throttle": [
        "anthropic/claude-opus-4.6",
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5",
        "x-ai/grok-4.20-0309-reasoning",
        "deepseek/deepseek-r1",
        "qwen/qwen3-235b-a22b",
        "z-ai/glm-4.7",
    ],
}

ALL_TIER_MODELS: list[str] = []
for models in TIER_MODELS.values():
    ALL_TIER_MODELS.extend(models)

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

async def call_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
) -> str:
    """Call a single model via its native API (or OpenRouter fallback)."""
    base_url, api_key, native_model = resolve_provider(model)
    is_openrouter = (base_url == OPENROUTER_BASE)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Deep Search Tier Chooser"

    body = {
        "model": native_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

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
            return ""

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    except asyncio.TimeoutError:
        log.warning(f"[{req_id}] {provider_label} {model} timed out after {MODEL_TIMEOUT}s")
        return ""
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} {model} error: {e}")
        return ""


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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Deep Search Tier Chooser"

    body = {
        "model": native_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

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
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error(f"[{req_id}] {provider_label} stream {model} exception: {e}")


# ============================================================================
# Tier Race — race all models in a tier, score responses, return the best
# ============================================================================

async def run_tier_race(
    tier: str,
    messages: list[dict],
    user_query: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Race all models in a tier, score responses, return the best."""
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

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)

    async def query_model(model_name: str) -> dict:
        async with semaphore:
            # Pass messages as-is — NO system prompt injection
            content = await call_model(
                model_name, messages,
                temperature=0.7, max_tokens=4096, req_id=req_id,
            )
            result = score_response(content, user_query) if content else {
                "score": -9999, "is_refusal": True, "hedge_count": 0,
            }
            return {"model": model_name, "content": content, **result}

    tasks = [asyncio.create_task(query_model(m)) for m in models]
    results = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        status = "REFUSAL" if result["is_refusal"] else f"score={result['score']}"
        short_model = result["model"].split("/")[-1]
        yield _chunk("", reasoning=f"  [{completed}/{len(models)}] {short_model}: {status}\n")

    valid = [r for r in results if not r["is_refusal"] and r["content"]]
    if not valid:
        yield _chunk(
            f"All {len(models)} models in the {tier} tier refused or returned empty. Try rephrasing.",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    best = max(valid, key=lambda r: r["score"])
    short_winner = best["model"].split("/")[-1]
    avg_score = sum(r["score"] for r in valid) / len(valid)
    refusal_count = sum(1 for r in results if r["is_refusal"])

    yield _chunk("", reasoning=(
        f"\nResults: {len(valid)} valid / {refusal_count} refused / "
        f"{len(results) - len(valid) - refusal_count} empty\n"
        f"Winner: {short_winner} (score={best['score']}, avg={avg_score:.0f})\n"
    ))

    yield _chunk(best["content"], finish_reason="stop")
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

    requested_model = body.get("model", "tier-race-medium")
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
        # Default to medium race for unknown model IDs
        generator = run_tier_race("medium", messages, user_query, req_id)

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
