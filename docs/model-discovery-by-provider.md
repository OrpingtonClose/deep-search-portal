# Model Discovery by Provider — April 2026

*Generated: 2026-04-23*

Cross-repo inventory of models available per provider, their current integration
status, benchmark data, and actionable gaps.

This document consolidates:

- The native-API provider registry in `proxies/tier_chooser_proxy.py` and
  `proxies/godmode_proxy.py` (MiroThinker).
- The deepagents eval model registry (`.github/scripts/models.py` +
  `libs/evals/MODEL_GROUPS.md`).
- Artificial Analysis benchmark JSON in
  `mcp-agent` (`src/mcp_agent/data/artificial_analysis_llm_benchmarks.json`).
- Local H200 model selection in `MiroThinker/docs/MODEL_SELECTION.md`.
- Empirical numbers from `docs/model-evaluation-april-2026.md`.

## Section 1: Provider Registry Overview

The Tier Chooser uses native APIs when a provider prefix is present in
`PROVIDER_REGISTRY` and the corresponding env var is set, otherwise falls back
to OpenRouter. `godmode_proxy.py` has an almost identical registry (Anthropic
is excluded because its API is not OpenAI-compatible).

DeepAgents' eval harness reaches additional aggregators (Fireworks, Baseten,
Groq, Ollama, OpenRouter) that are **not** wired into Tier Chooser or
G0DM0D3.

| Provider | Base URL | Key env | Models avail. | Tier Chooser | G0DM0D3 | DeepAgents evals | MiroThinker (H200) | Feynman |
|---|---|---|---|---|---|---|---|---|
| `openai` | `api.openai.com/v1` | `OPENAI_API_KEY` | 9 in set0 | YES (`o3`, `o3-mini`, `o3-mini-high`) | YES (fallback via OpenRouter) | YES (`openai`) | — | YES |
| `anthropic` | `api.anthropic.com/v1` | `ANTHROPIC_API_KEY` | 6 in set0 | YES (haiku-4.5, sonnet-4.6, opus-4.6) | NO (non-OpenAI-compatible) | YES (`anthropic`) | — | YES |
| `google` | `generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` | 4 in set0 | YES (flash-lite, 2.5-pro, 3.1-pro-preview) | YES | YES (`google_genai`) | — | YES |
| `x-ai` | `api.x.ai/v1` | `XAI_API_KEY` | 12 models evaluated | YES (grok-4.1-fast, grok-4, grok-4.20) | YES | YES (`xai`: grok-4, grok-3-mini-fast) | — | — |
| `deepseek` | `api.deepseek.com` | `DEEPSEEK_API_KEY` | 2–3 variants | YES (chat-v3.1, v3.2, r1-0528) | YES | via Fireworks/Ollama | — | — |
| `perplexity` | `api.perplexity.ai` | `PERPLEXITY_API_KEY` | sonar family | YES (via OpenRouter) | YES | — | — | — |
| `mistralai` | `api.mistral.ai/v1` | `MISTRAL_NATIVE_API_KEY` | 5 variants | YES (mistral-medium-3.1, mistral-large-2512) | YES | — | — | — |
| `qwen` | `dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | 4+ variants | YES (qwen-turbo, qwen3-235b, qwen3-max-thinking) | YES | — | — | — |
| `moonshotai` | `api.moonshot.ai/v1` (Chooser) / `api.moonshot.cn/v1` (G0DM0D3) | `MOONSHOT_API_KEY` | kimi-k2.5 | YES (quick tier) | YES | via Fireworks/Baseten/Groq | **YES (primary worker, local)** | — |
| `cohere` | `api.cohere.com/compatibility/v1` | `COHERE_API_KEY` | — | registered | registered | — | — | — |
| `minimax` | `api.minimax.chat/v1` | `MINIMAX_API_KEY` | minimax-m2.5 | via OpenRouter | via OpenRouter | via Fireworks/Baseten/Ollama (m2.1/m2.5/m2.7) | — | — |
| `groq` | `api.groq.com/openai/v1` | `GROQ_API_KEY` | 3 in set2 | registered, **not in tiers** | registered | YES (`groq` set2) | — | — |
| `stepfun` | `api.stepfun.com/v1` | `STEPFUN_API_KEY` | step-3.5-flash | via OpenRouter | registered | — | — | — |
| `z-ai` | `open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | glm-5, glm-4.7 | YES (glm-4.7-flash, glm-5) | YES | via Baseten/Fireworks/Ollama | YES (GLM-5.1 abliterated, report gen) | — |
| `nvidia` | `integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | nemotron-3 super/nano | via OpenRouter | registered | YES (`nvidia`, also via Baseten + OpenRouter) | via Baseten (Nemotron-120B) | — |
| `fireworks` | `api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | **7** | **NOT REGISTERED** | **NOT REGISTERED** | YES (`fireworks`) | — | — |
| `baseten` | `inference.baseten.co/v1` | `BASETEN_API_KEY` | **5** | **NOT REGISTERED** | **NOT REGISTERED** | YES (`baseten`) | Kimi-K2.5 reference | — |
| `openrouter` | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | fallback for all | fallback | fallback | YES (3 curated) | — | — |
| `ollama` | local | — | **13** | — | — | YES (`ollama` set2) | YES (GLM/Kimi/Nemotron abliterated) | — |

## Section 2: Fireworks (detailed)

### Available models (7)

Source: `OrpingtonClose/deepagents` `.github/scripts/models.py` lines 146–183
and `libs/evals/MODEL_GROUPS.md` (`fireworks` group).

| Model ID | Underlying model | Tool calling | Structured output | Eval sets |
|---|---|---|---|---|
| `fireworks/deepseek-v3p2` | DeepSeek V3 Phase 2 | YES | YES | `set0`, `fireworks` |
| `fireworks/deepseek-v3-0324` | DeepSeek V3 (March 2024) | YES | YES | `set0`, `fireworks` |
| `fireworks/qwen3-vl-235b-a22b-thinking` | Qwen3 VL 235B (vision + language) | YES | YES | **`set0`, `set1`**, `fireworks` |
| `fireworks/minimax-m2p1` | MiniMax M2.1 | YES | YES | `set0`, `fireworks` |
| `fireworks/kimi-k2p5` | Kimi K2.5 (Moonshot) | YES | YES | `set0`, `fireworks` |
| `fireworks/glm-5` | GLM-5 (Zhipu) | YES | YES | `set0`, `fireworks` |
| `fireworks/minimax-m2p5` | MiniMax M2.5 | YES | YES | `set0`, `fireworks` |

### Benchmark data

From `OrpingtonClose/mcp-agent` `src/mcp_agent/data/artificial_analysis_llm_benchmarks.json`
(Artificial Analysis snapshot):

| Metric | Kimi K2 @ Fireworks |
|---|---|
| Quality score | 47.19 |
| Input / output cost ($ / 1M) | $0.60 / $2.50 |
| Tokens per second | 148 |
| TTFT | 524 ms |
| Context window | 131K |

### Integration status

- **DeepAgents evals**: YES — all 7 in `set0`; only `qwen3-vl-235b-a22b-thinking`
  is also in the curated `set1`.
- **Tier Chooser**: **NOT integrated** — no `fireworks` entry in
  `PROVIDER_REGISTRY` in `proxies/tier_chooser_proxy.py`.
- **G0DM0D3 Consortium**: **NOT integrated** — no `fireworks` entry in
  `PROVIDER_REGISTRY` in `proxies/godmode_proxy.py` and no Hall-of-Fame slot.
- **MiroThinker H200**: N/A (local inference only).

### Gap

Fireworks is not in `PROVIDER_REGISTRY` in either Tier Chooser or G0DM0D3.
Adding it would unlock 7 models for Tier Chooser racing. The Fireworks Kimi K2.5
at **148 tok/s** is ~3× faster than via OpenRouter (51.7 tok/s) and ~11× faster
than the native Moonshot API (~13.3 tok/s in the April 2026 eval).

### Standout

`fireworks/qwen3-vl-235b-a22b-thinking` is the only Fireworks model that is
also in the curated `set1` set, and is the only vision+language model in the
Fireworks group.

## Section 3: Kimi / Moonshot (detailed)

### Access paths (5)

Source: `OrpingtonClose/mcp-agent` `src/mcp_agent/data/artificial_analysis_llm_benchmarks.json`.

| Path | Provider | Model ID | Speed | Cost (in/out $/1M) | TTFT |
|---|---|---|---|---|---|
| Native API | Moonshot | `kimi-k2.5` | ~13.3 tok/s | $0.42 / $2.20 | — |
| Fireworks | Fireworks | `fireworks/kimi-k2p5` | 148 tok/s | $0.60 / $2.50 | 524 ms |
| Groq | Groq | `moonshotai/kimi-k2-instruct` | **483 tok/s** | $1.00 / $3.00 | 223 ms |
| Baseten | Baseten | `moonshotai/Kimi-K2.5` | 66.8 tok/s | $0.60 / $2.50 | 298 ms |
| OpenRouter | OpenRouter | `moonshotai/kimi-k2.5` | 51.7 tok/s | $0.42 / $2.20 | 7001 ms |

Other hosts in the benchmark JSON (not currently wired): Parasail (16.1 tok/s,
$1.50/$4.00), Deepinfra (27.3 tok/s, $0.50/$2.00), Novita (47.2 tok/s,
$0.57/$2.30), GMI (32.0 tok/s, $1.00/$3.00), Together.ai (8.8 tok/s,
$1.00/$3.00).

### Local models (MiroThinker H200)

Source: `OrpingtonClose/MiroThinker` `docs/MODEL_SELECTION.md` lines 137–148
and 311–323.

| Model | Role | Notes |
|---|---|---|
| `Kimi-Linear-48B-A3B` (abliterated) | **Primary worker** | Hybrid KDA+MLA, **1M native context**, ~45 GB at Q4 (incl. KV), MMLU-Pro 51.0, RULER 84.3 @ 128K |
| `Kimi K2.5 / K2.6` | **Report generation specialist** | 1T MoE, AIME 96.1%, 262K context (K2.6), ~550 GB at INT4 |

### April 2026 eval results

Source: `docs/model-evaluation-april-2026.md` (OpenRouter section).

| Model | Censorship | Thought | Math | Analysis | Tok/s | Value |
|---|---|---|---|---|---|---|
| `moonshotai/kimi-k2.5` | REFUSED | 6/6 | 3/3 | 3/3 | 51.7 | 2.73 |
| `moonshotai/kimi-k2` | REFUSED | 6/6 | 3/3 | 3/3 | 13.3 | 2.61 |

### Integration status

- **Tier Chooser quick tier**: YES (`moonshotai/kimi-k2.5` via native API,
  `tier_chooser_proxy.py:202`).
- **Tier Chooser full-throttle**: NOT present.
- **G0DM0D3 Consortium / Hall of Fame**: NOT present.
- **DeepAgents evals**: YES via 3 providers — Fireworks (set0), Groq (set2),
  Baseten (set0).
- **MiroThinker H200**: YES — primary worker (abliterated Kimi-Linear-48B-A3B)
  + report generation specialist (K2.5/K2.6).
- **LibreChat dropdown**: YES (`kimi-k2.5-instant`, `kimi-k2.5-thinking` aliases
  via `NATIVE_MODEL_MAP`).

### Blocker

Kimi **REFUSED** on the April 2026 censorship test prompt on every remote
surface evaluated — hard blocker for G0DM0D3 / Consortium / Heretic use. The
abliterated local variant (`Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated`)
removes this limitation but requires H200 VRAM.

### Opportunity

Replace the native Moonshot API (~13.3 tok/s) with Fireworks (148 tok/s) or
Groq (483 tok/s) in Tier Chooser quick tier for a **10–36× speed improvement
at similar or slightly higher cost** — provided `fireworks` / `groq` models
are wired into the tier lists (see Section 4 and Section 8).

## Section 4: Groq

### Available models (3)

Source: `OrpingtonClose/deepagents` `libs/evals/MODEL_GROUPS.md` `groq` group.

- `groq:openai/gpt-oss-120b`
- `groq:qwen/qwen3-32b`
- `groq:moonshotai/kimi-k2-instruct`

### Benchmark data

Groq's defining characteristic is speed: `kimi-k2-instruct` clocks **483.4
tok/s at 223 ms TTFT** in the Artificial Analysis JSON — the fastest Kimi K2
host by a wide margin and ~10× faster than the next-fastest aggregator (Baseten
at 66.8 tok/s).

### Integration status

- **DeepAgents evals**: YES (`set2` only — the "open / fast" cohort).
- **Tier Chooser**: Registered in `PROVIDER_REGISTRY` (line 103) but **no
  `groq/*` models appear in any of the quick / medium / full-throttle tiers**
  (lines 192–222).
- **G0DM0D3**: Registered in `PROVIDER_REGISTRY` (line 88) but not used in
  Hall of Fame.

### Gap

Groq is wired up but unused. Adding `groq/moonshotai/kimi-k2-instruct` to the
quick tier would give Tier Chooser its fastest model overall.

## Section 5: Baseten

### Available models (5)

Source: `OrpingtonClose/deepagents` `.github/scripts/models.py` lines 98–145
and `libs/evals/MODEL_GROUPS.md` `baseten` group.

- `baseten:zai-org/GLM-5` (also in `set1`, `open`)
- `baseten:MiniMaxAI/MiniMax-M2.5` (also in `set1`)
- `baseten:moonshotai/Kimi-K2.5`
- `baseten:nvidia/Nemotron-120B-A12B`
- `baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct`

### Benchmark data

| Model | Tok/s | TTFT | Cost (in/out $/1M) |
|---|---|---|---|
| Kimi K2 @ Baseten | 66.8 | 298 ms | $0.60 / $2.50 |

Baseten sits in the middle of the Kimi hosting spectrum: faster than
OpenRouter/Novita, much slower than Fireworks/Groq, at Fireworks-equivalent
cost.

### Integration status

- **DeepAgents evals**: YES (`set0`; two models in `set1`, one in `open`).
- **Tier Chooser**: **NOT REGISTERED**.
- **G0DM0D3**: **NOT REGISTERED**.

## Section 6: Other Providers (brief)

### OpenAI

- **Tier Chooser**: `openai/o3` (full-throttle), `openai/o3-mini` and
  `openai/o3-mini-high` (quick / medium).
- **DeepAgents evals**: 9 models (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o3`,
  `o4-mini`, `gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.4`, `gpt-5.4-mini`).
- **April 2026 eval**: `openai/gpt-5` (SEMI-PASS, 64.0 tok/s, value 0.6),
  `openai/gpt-5.2` (SEMI-PASS, 50.6 tok/s, value 0.43), `openai/gpt-oss-20b`
  (SEMI-PASS, thought 6, value 54.55 — best gpt-oss by value), `openai/gpt-oss-120b`
  (value 31.58), `openai/gpt-4o` (value 0.5).
- **Notable gap**: `gpt-5.4` (frontier set1) is not in any Tier Chooser tier.

### Anthropic

- **Tier Chooser**: `claude-haiku-4.5` (quick), `claude-sonnet-4.6` (medium),
  `claude-opus-4.6` (full-throttle). Native-API translation map at
  `tier_chooser_proxy.py:151–154`.
- **DeepAgents evals**: 6 models (haiku-4.5, sonnet-4.5-20250929,
  sonnet-4.6, opus-4.1, opus-4.5-20251101, opus-4.6). Opus-4.6 is in the
  `frontier` set.
- **April 2026 eval**: `claude-sonnet-4.6` (value 0.4), `claude-opus-4.6`
  (value 0.24), `claude-opus-4` (value 0.08 — dominated by opus-4.6).
- **Notable gap**: All three Anthropic tiers still rely on OpenRouter because
  Anthropic's API is not OpenAI-compatible in G0DM0D3.

### Google

- **Tier Chooser**: `gemini-3.1-flash-lite-preview` (quick), `gemini-2.5-pro`
  (medium), `gemini-3.1-pro-preview` (full-throttle).
- **DeepAgents evals**: `gemini-2.5-flash`, `gemini-2.5-pro`,
  `gemini-3-flash-preview`, `gemini-3.1-pro-preview` (the last is in the
  `frontier` set).
- **April 2026 eval**: `google/gemini-2.5-flash` (value 2.4),
  `google/gemini-2.5-pro` (value 0.4), `google/gemini-3.1-pro-preview`
  (value 0.25, thought 3 — underperformed), `google/gemma-3-27b-it`
  (value 37.5 — best value Google model).
- **Notable gap**: Gemini 3 Flash Preview is evaluated but not in any tier
  (flash-lite-preview is used instead).

### x-AI

- **Tier Chooser**: `grok-4.1-fast` (quick), `grok-4` (medium), `grok-4.20`
  (full-throttle).
- **DeepAgents evals**: `grok-4`, `grok-3-mini-fast` (both in `set2`).
- **April 2026 eval**: 12 models evaluated on xAI Direct API plus OpenRouter
  variants. `grok-code-fast-1`, `grok-3-fast`, `grok-3-mini` all value=12.0
  but REFUSED. `grok-4.20-0309-non-reasoning` at 251.5 tok/s is the fastest
  full-thought model.
- **Notable gap**: `grok-4-fast-reasoning` / `grok-4-fast-non-reasoning`
  evaluated at 12.6 / 29.5 tok/s are not in any tier.

### DeepSeek

- **Tier Chooser**: `deepseek-chat-v3.1` (quick), `deepseek-v3.2` (medium),
  `deepseek-r1-0528` (full-throttle).
- **DeepAgents evals**: via Fireworks (`deepseek-v3p2`, `deepseek-v3-0324`)
  and Ollama (`deepseek-v3.2:cloud`). Not in the `deepseek` native provider
  group.
- **April 2026 eval**: `deepseek/deepseek-v3.2` (value 15.79),
  `deepseek/deepseek-r1` (value 2.4), `deepseek-v3.2` via Venice
  (value 4.8).

### Perplexity

- **Tier Chooser**: not explicitly in any tier list; routed via OpenRouter.
- **April 2026 eval**: `perplexity/sonar` (SEMI-PASS, thought 6, 40.9 tok/s,
  value 6.0).
- **Notable gap**: Perplexity is registered but has no first-class tier slot.

### Mistral

- **Tier Chooser**: `mistral-medium-3.1` (quick), `mistral-large-2512`
  (medium). No full-throttle Mistral.
- **April 2026 eval (Mistral Direct API)**: `mistral-large-latest` (REFUSED),
  `mistral-small-latest` (value 20.0), `codestral-2508` (168.5 tok/s,
  value 6.67). `mixtral-8x22b-instruct` and `mistral-medium-3.1` erred on
  direct API but passed on OpenRouter.
- **Notable gap**: No Mistral presence in DeepAgents evals (no `mistralai`
  provider group).

### Qwen / Alibaba

- **Tier Chooser**: `qwen-turbo` (quick), `qwen3-235b-a22b` (medium),
  `qwen3-max-thinking` (full-throttle). Native-model-name translations at
  `tier_chooser_proxy.py:141–143`.
- **DeepAgents evals**: via Baseten (`Qwen/Qwen3-Coder-480B-A35B-Instruct`),
  Fireworks (`qwen3-vl-235b-a22b-thinking`), Groq (`qwen/qwen3-32b`), Ollama
  (`qwen3.5:397b-cloud`, `qwen3-next:80b`, `qwen3-coder:480b-cloud`). No
  native `qwen` provider group in deepagents.
- **April 2026 eval**: `qwen/qwen-2.5-72b-instruct` (value 15.38, REFUSED),
  `qwen/qwen3-coder` (value 6.0), `qwen/qwen3-235b-a22b` (REFUSED),
  `qwen/qwq-32b` (value 10.34), `qwen/qwen3.5-plus-02-15` (value 3.85).
  On Venice: `qwen3.5-9b` (147.5 tok/s, value 60.0 — fastest Venice model),
  `qwen3.5-35b-a3b` (95.4 tok/s, value 60.0), `qwen3-next-80b` (88.2 tok/s,
  value 20.0).

### Cohere

- Registered in both registries but has no tier entries and no eval coverage.

### MiniMax

- **Tier Chooser**: no direct MiniMax tier entry.
- **DeepAgents evals**: via Baseten (`MiniMaxAI/MiniMax-M2.5`), Fireworks
  (`minimax-m2p1`, `minimax-m2p5`), Ollama (`minimax-m2.5`, `minimax-m2.7:cloud`),
  OpenRouter (`minimax/minimax-m2.7`). `MiniMax-M2.5` is in the curated `set1`.
- **April 2026 eval**: `minimax/minimax-m2.5` (SEMI-PASS, 21.5 tok/s, value 6.0).

### Stepfun

- Registered; `stepfun/step-3.5-flash` evaluated at 176.2 tok/s (3rd fastest
  OpenRouter model) but thought-power only 3/6 and not in any tier.

### z-AI (Zhipu GLM)

- **Tier Chooser**: `glm-4.7-flash` (quick), `glm-5` (medium and
  full-throttle).
- **DeepAgents evals**: via Baseten (`zai-org/GLM-5`, also in `set1`/`open`),
  Fireworks (`glm-5`), Ollama (`glm-5`, `glm-5.1`), OpenRouter (`z-ai/glm-5.1`,
  in `open` set).
- **MiroThinker H200**: `Huihui-GLM-5.1-abliterated-GGUF` — report generation
  and deep reasoning fallback (40B active, SWE-Verified 77.8%, AIME 92.7%).
- **April 2026 eval**: `z-ai/glm-5-turbo` (value 1.5), `z-ai/glm-5`
  (REFUSED, thought 1 — poor), `z-ai/glm-4.7` (REFUSED, thought 3),
  `olafangensan-glm-4.7-flash-heretic` on Venice (**UNCENSORED**, value 60.0,
  tied #1 on the composite ranking).

### NVIDIA

- **Tier Chooser**: `nvidia/*` routed via OpenRouter.
- **DeepAgents evals**: native `nvidia` group (`nemotron-3-super-120b-a12b`),
  Baseten (`nvidia/Nemotron-120B-A12B`), OpenRouter duplicate.
- **April 2026 eval**: `nvidia/nemotron-3-super-120b-a12b` at **346.2 tok/s**
  (fastest model evaluated), but thought 4/6 and REFUSED on censorship.
  `nvidia/nemotron-3-nano-30b-a3b` at 122.2 tok/s, value 30.0.

## Section 7: Cross-Provider Comparison — Same Model, Different Hosts

Kimi K2 is the clearest cross-host case: identical quality score (47.19),
wildly different speed/cost profiles. All numbers from Artificial Analysis
JSON in `mcp-agent` unless noted.

| Host | Model ID as exposed | Tok/s | TTFT (ms) | In $/1M | Out $/1M | Blended $/1M | Quality |
|---|---|---|---|---|---|---|---|
| Groq | `kimi-k2-instruct` | **483.4** | 223 | $1.00 | $3.00 | $1.50 | 47.19 |
| Fireworks | `kimi-k2-instruct` | 148.2 | 524 | $0.60 | $2.50 | $1.08 | 47.19 |
| Baseten | `Kimi-K2-Instruct` | 66.8 | 298 | $0.60 | $2.50 | $1.08 | 47.19 |
| Novita | `kimi-k2-instruct` | 47.2 | 1516 | $0.57 | $2.30 | $1.00 | 47.19 |
| GMI | `Kimi-K2-Instruct` | 32.0 | 687 | $1.00 | $3.00 | $1.50 | 47.19 |
| Deepinfra | `Kimi-K2-Instruct` | 27.3 | 359 | $0.50 | $2.00 | **$0.88** | 47.19 |
| Parasail | `parasail-kimi-k2-instruct` | 16.1 | 554 | $1.50 | $4.00 | $2.13 | 47.19 |
| Together.ai | `Kimi-K2-Instruct` | 8.8 | 813 | $1.00 | $3.00 | $1.50 | 47.19 |
| OpenRouter | `moonshotai/kimi-k2.5` | 51.7 (eval) | 7001 (eval) | $0.42 | $2.20 | $1.06 | 47.19 |
| Moonshot native | `kimi-k2.5` | ~13.3 (eval) | — | $0.42 | $2.20 | $1.06 | 47.19 |

Speed leader: **Groq** (30× faster than native, 2× more expensive).
Price leader: **Deepinfra** ($0.88 blended).
Balance leader: **Fireworks** (148 tok/s at Baseten-equivalent cost).

GLM-5 also multi-hosts (Baseten, Fireworks, Ollama, OpenRouter, Venice);
MiniMax M2.5/M2.7 multi-hosts across Baseten, Fireworks, Ollama, OpenRouter;
Qwen3 Coder 480B across Baseten and Ollama cloud. These all have similar
speed/cost spread dynamics though quality scores are not in the snapshot.

## Section 8: Integration Gap Summary

| Gap | Impact | Effort |
|---|---|---|
| **Fireworks not in `PROVIDER_REGISTRY`** | 7 models unavailable for Tier Chooser / Consortium racing, including the fastest commercial Kimi K2.5 at 148 tok/s | Add ~2 lines to both `proxies/tier_chooser_proxy.py` (line 91) and `proxies/godmode_proxy.py` (line 74) |
| **Groq not in Tier Chooser tiers** | `moonshotai/kimi-k2-instruct` at 483 tok/s — would be the fastest model in the quick tier overall | Add `groq/moonshotai/kimi-k2-instruct` (and `groq/openai/gpt-oss-120b`) to `TIER_MODELS["quick"]` |
| **Baseten not in `PROVIDER_REGISTRY`** | 5 models unavailable for Tier Chooser / Consortium, including `Kimi-K2.5`, `GLM-5`, `Qwen3-Coder-480B`, `Nemotron-120B`, `MiniMax-M2.5` | Add ~1 line to each proxy file |
| **Kimi censorship blocker** | Cannot use Kimi on G0DM0D3 / Heretic (REFUSED on all remote hosts tested) | No remote fix — use the MiroThinker `Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated` local variant |
| **Fireworks Kimi could replace native Moonshot in quick tier** | ~10× speed improvement (13.3 → 148 tok/s) at comparable cost | Swap model ID once Fireworks is registered |
| **Anthropic not in G0DM0D3 `PROVIDER_REGISTRY`** | All Claude traffic pays OpenRouter surcharge + added latency | Write an Anthropic-native shim (non-OpenAI-compatible body/headers) |
| **`gpt-5.4` frontier missing from tiers** | Frontier model in `set1` never raced | Add to `full-throttle` tier |
| **Perplexity has no tier slot** | `perplexity/sonar` (SEMI-PASS, 40.9 tok/s, value 6.0) only reachable via explicit user selection | Add to `medium` tier |
| **Ollama / local inference not routable via Tier Chooser** | 13 open-weight models (including abliterated variants) evaluated in deepagents `set2` are completely unreachable from the portal | Add an `ollama` entry to `PROVIDER_REGISTRY` (OpenAI-compatible) pointing at local endpoint |

## Section 9: Recommendations

Prioritized to maximise speed and reach while keeping code changes small.

### 1. Register Fireworks and Baseten in both proxies — **HIGH**

Two-line change in each of `proxies/tier_chooser_proxy.py` and
`proxies/godmode_proxy.py`:

```python
"fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY"},
"baseten":   {"base_url": "https://inference.baseten.co/v1",      "key_env": "BASETEN_API_KEY"},
```

Unlocks 12 additional models for race-based routing and gives parity with
the deepagents eval surface.

### 2. Swap Moonshot native → Fireworks Kimi in Tier Chooser quick tier — **HIGH**

Replace `"moonshotai/kimi-k2.5"` with `"fireworks/kimi-k2p5"` in
`TIER_MODELS["quick"]`. Expected outcome:

- Tokens/s: ~13.3 → 148 (≈11×).
- Cost (out $/1M): $2.20 → $2.50 (+14 %).
- Quality: unchanged (same underlying Kimi K2 weights).

### 3. Add Groq Kimi to quick tier — **HIGH**

Adding `"groq/moonshotai/kimi-k2-instruct"` to the quick tier gives the
Chooser an ultra-low-latency option (223 ms TTFT, 483 tok/s) for short
answers where cost per output token matters less than wall-clock time.

### 4. Wire an `ollama` entry for local open-weight models — **MEDIUM**

Lets the Heretic / Miro pipelines reach the abliterated GLM-5.1, Kimi-Linear
and Qwen3.5 abliterated variants that MiroThinker already runs on H200. The
Venice `olafangensan-glm-4.7-flash-heretic` is currently the only path for
the portal to reach an uncensored, native-tool-calling model — a local
alternative reduces third-party dependency for Tier 1 research flows.

### 5. Promote `fireworks/qwen3-vl-235b-a22b-thinking` — **MEDIUM**

Only Fireworks model in the curated `set1` and the only VL-capable model in
the deepagents registry. Adding it to the medium or full-throttle tier would
give the Chooser vision input on a single call.

### 6. Add an Anthropic-native shim to G0DM0D3 — **MEDIUM**

Removes the forced OpenRouter fallback for all Claude traffic in the
Consortium path. Implementation cost is non-trivial (non-OpenAI-compatible
request/response shape), but eliminates a per-call surcharge on the three
top-scoring Anthropic models (haiku-4.5, sonnet-4.6, opus-4.6).

### 7. Consolidate model evaluation into a provider-agnostic table — **LOW**

`docs/model-evaluation-april-2026.md` currently groups by **API surface**
(xAI Direct, Mistral Direct, Venice Direct, OpenRouter). A future eval should
emit a provider-agnostic table keyed by `(underlying model, host)` so that
cross-host latency / cost / censorship differences for the same model are
immediately visible (as in Section 7 of this document).

## Sources

- `OrpingtonClose/deep-search-portal` `proxies/tier_chooser_proxy.py` (lines
  91–222 for registry + tiers)
- `OrpingtonClose/MiroThinker` `proxies/godmode_proxy.py` (lines 74–113 for
  registry, Hall of Fame starts line 132)
- `OrpingtonClose/deepagents` `.github/scripts/models.py` and
  `libs/evals/MODEL_GROUPS.md`
- `OrpingtonClose/mcp-agent`
  `src/mcp_agent/data/artificial_analysis_llm_benchmarks.json`
- `OrpingtonClose/MiroThinker` `docs/MODEL_SELECTION.md` (lines 120–323)
- `OrpingtonClose/deep-search-portal` `docs/model-evaluation-april-2026.md`
