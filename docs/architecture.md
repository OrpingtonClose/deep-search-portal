# MiroFlow Deep Research Proxy — Complete Architecture Document

## Purpose

MiroFlow is a self-hosted deep research agent that sits between Open WebUI (a ChatGPT-like web interface) and the Mistral AI API. When a user asks a question, instead of forwarding it directly to the LLM, the proxy orchestrates a multi-turn agentic research loop: the LLM reasons about the question, calls tools (web search, page fetching, Python execution), receives results, reasons again, calls more tools, and repeats this for up to 15 rounds before synthesizing a comprehensive final answer.

The entire research process is streamed to the user in real-time inside `<think>` tags (which Open WebUI renders as a collapsible "thinking" section), so the user can watch the agent work. The final polished answer appears after `</think>`.

---

## System Architecture

```
User's Browser
    │
    ▼
Open WebUI (port 3000)  ◄── Cloudflare Tunnel ── https://deep-search.uk
    │
    ├── Direct models (Mistral, Venice, Together, Perplexity, RunPod)
    │
    ├── Thinking Proxy (port 9100) ── wraps Mistral for <think> tag support
    │
    └── Deep Research Proxy / MiroFlow (port 9200) ◄── THIS DOCUMENT
            │
            ├── Mistral AI API (api.mistral.ai/v1) — LLM with native function calling
            ├── SearXNG (localhost:8888) — self-hosted meta-search engine
            ├── httpx direct fetch — reads web pages
            └── subprocess Python exec — calculations/analysis
```

### Infrastructure

- **Host**: Vast.ai GPU instance (1x RTX 2060, $0.037/hr)
- **Domain**: `deep-search.uk` via Cloudflare Tunnel (HTTPS, Google OAuth login)
- **Process management**: GNU Screen sessions (`owui`, `searxng`, `thinking-proxy`, `deep-research`, `cftunnel`)
- **Startup**: `/opt/startup.sh` launches all services in order with health-check waits

---

## The Proxy Application (deep_research_proxy.py) — 878 lines

### Framework & Dependencies

- **FastAPI** + **uvicorn** (async web server)
- **httpx** (async HTTP client for LLM calls, search, web fetching)
- Python stdlib: `asyncio`, `subprocess`, `json`, `re`, `html`, `logging`, `uuid`, `tempfile`

### Entry Points (FastAPI Routes)

| Route | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | Main endpoint — receives OpenAI-compatible chat requests from Open WebUI |
| `/chat/completions` | POST | Alias of above (some clients omit `/v1`) |
| `/v1/models` | GET | Returns model list (single model: `miroflow`) |
| `/models` | GET | Alias |
| `/health` | GET | Health check — shows config, active requests, current turn per request |
| `/logs` | GET | Returns last N lines from the rotating log file |

### Request Routing Logic

When a request arrives at `/v1/chat/completions`, the proxy inspects the message content:

1. **Utility requests** (title generation, tag generation, autocomplete — detected by pattern matching against known Open WebUI system prompts) → **Passthrough mode**: forwarded directly to Mistral with streaming, no agent loop
2. **Real user queries** → **Deep Research mode**: full agentic loop with tools

The utility detection uses substring matching against these patterns:
```python
UTILITY_PATTERNS = [
    "generate a concise",
    "generate 1-3 broad tags",
    "generate a title",
    "### task:\ngenerate",
    "create a concise title",
    "generate a search query",
    "autocomplete",
]
```

---

## Deep Research Agent Loop (core logic)

### Configuration Constants

| Parameter | Value | Purpose |
|---|---|---|
| `UPSTREAM_MODEL` | `mistral-large-latest` | The LLM used for reasoning |
| `MAX_AGENT_TURNS` | 15 | Maximum research rounds |
| `WEBPAGE_MAX_CHARS` | 15,000 | Truncation limit for fetched web pages |
| `PYTHON_TIMEOUT` | 30s | Subprocess execution timeout |
| `PYTHON_OUTPUT_MAX` | 5,000 | Max chars from Python output |
| `max_tokens` | 4,096 | Per-LLM-call token limit |
| `temperature` | 0.3 | Low temp for focused research |

### System Prompt

The system prompt (injected as the first message, with today's date) instructs the LLM to:

1. **Exhaust all 15 research rounds** — never stop early unless absolutely certain no new info exists
2. **Search from multiple angles** — rephrase queries, try synonyms, explore adjacent topics
3. **Read many sources** — at least 5-10+ pages, cross-reference claims
4. **Dig deeper** — follow up on interesting findings with more specific searches
5. **Verify conflicts** — if sources disagree, search specifically to resolve contradictions
6. **Use python_exec** for calculations and data analysis

The prompt also mandates:
- **No jargon** without explanation
- **No talking down** to the user — no unsolicited warnings, ethical disclaimers, safety preambles, or "consult a professional" hedging
- **Answer the spirit** of the question directly and confidently
- **Cite sources** with URLs
- **MUST use tools** — never answer from training data alone

### Tools (Native Mistral Function Calling)

Three tools are defined in OpenAI function-calling format and passed via the `tools` parameter:

#### 1. `searxng_search(query: str) → str`
- Calls the local SearXNG instance (`http://localhost:8888/search?q=...&format=json&categories=general`)
- Returns top 10 results formatted as numbered list with title, URL, and snippet (300 char max)
- SearXNG currently has **Brave** and **Bing** as active engines (Google/Startpage blocked from datacenter IP)
- Timeout: 20s

#### 2. `fetch_webpage(url: str, extract_info?: str) → str`
- Fetches a URL with httpx (follows redirects, 20s timeout)
- Strips `<script>` and `<style>` tags, removes all HTML tags, unescapes entities
- Truncates to 15,000 chars
- Optional `extract_info` parameter prepends a "looking for" hint (but no actual extraction — the LLM reads the full text)
- User-Agent: `Mozilla/5.0 (compatible; DeepResearchBot/1.0)`

#### 3. `python_exec(code: str) → str`
- Runs Python code in a subprocess (`subprocess.run`) with 30s timeout
- Working directory: system temp dir
- Captures stdout + stderr, truncates to 5,000 chars
- No sandbox beyond subprocess isolation — has access to standard library and any installed packages

### The Turn Loop (lines 452-707)

```
for turn in range(1, MAX_AGENT_TURNS + 1):
    1. Stream turn header "[Turn X/15]" to user via SSE
    2. Call Mistral with full message history + tools
    3. If error → increment consecutive_errors, retry up to circuit breaker
    4. If model returns tool_calls → execute each tool, feed results back, increment turns_with_tools
    5. If model returns NO tool_calls (wants to stop) → push-back logic decides whether to force continuation
    6. If model is allowed to stop → stream final answer after </think>
```

### Push-Back Logic (Anti-Early-Stopping)

When the model tries to stop researching (no tool calls), the proxy evaluates three conditions:

```python
can_stop = (
    turn >= MAX_AGENT_TURNS - 1     # last 2 turns (14 or 15)
    or consecutive_no_tool_turns >= 3  # model insisted 3 times in a row
    or turns_with_tools >= 10          # 10+ rounds of actual research
)
```

If `can_stop` is False, the proxy injects a "push-back" user message forcing the model to continue. Three rotating push-back prompts are used to suggest different research angles:

1. "You have barely scratched the surface. Search for different angles, alternative viewpoints..."
2. "Think about what perspectives you HAVEN'T covered yet. Contrarian views? Historical context?..."
3. "Look for original research papers, official reports, expert interviews, forum discussions..."

### Forced Final Answer (Turn 15)

If the model reaches `MAX_AGENT_TURNS` without voluntarily stopping, the proxy:
1. Injects a user message: "You have reached the maximum. Based on ALL information gathered, provide your final comprehensive answer."
2. Calls the LLM with `include_tools=False` (removes tool definitions so it can't call them)
3. Streams the response as the final answer

### Duplicate Detection

A `used_queries` set tracks all `(tool_name, arguments)` combinations. If the model tries to make an identical tool call, it's skipped and the model receives: "Duplicate call skipped. Please use previously gathered information or try a different query."

### Error Handling & Retry Logic

**Per-LLM-call retries:**
- Status codes 429, 500, 502, 503, 504 trigger up to 3 retries
- Backoff: 5s → 15s → 30s
- Timeout errors (ReadTimeout, ConnectTimeout) also retry
- LLM call timeout: 300s (5 minutes) with 30s connect timeout

**Circuit breaker (turn-level):**
- If 3 consecutive turns fail (even after per-call retries), the proxy aborts
- Streams a detailed error message to the user explaining exactly what happened

**All errors surface to the user** — nothing is silently swallowed. The user sees:
- The specific HTTP status code and error body
- How many retries were attempted
- Whether it was a timeout, auth error, overload, etc.

### SSE Streaming Protocol

All output is streamed as Server-Sent Events in OpenAI chat completion chunk format:

```json
data: {"id": "chatcmpl-dr-xxxx", "object": "chat.completion.chunk", "created": 1234567890, "model": "miroflow", "choices": [{"index": 0, "delta": {"content": "..."}, "finish_reason": null}]}
```

**Stream structure:**
1. `<think>\n` — opens the thinking block
2. Turn headers: `**[Turn X/15]**`
3. Tool call announcements: `🔍 Searching: \`query\`` / `📄 Reading: \`url\`` / `🐍 Running code: \`...\``
4. Tool result summaries: `→ 1.2s — 9 results: Title1, Title2, Title3 (+6 more)`
5. Model reasoning (trimmed to 500 chars max — first 400 + last 100)
6. Push-back notifications: `↻ X research rounds done — pushing deeper...`
7. Completion: `✅ Research complete (X rounds, Y tool calls). Generating answer...`
8. `</think>\n\n` — closes thinking block
9. Final answer in 200-char chunks
10. `finish_reason: "stop"` + `[DONE]`

### Keepalive Mechanism

During LLM calls (which can take 30-60+ seconds), a background coroutine sends `.` dots every 8 seconds via an `asyncio.Queue` to prevent SSE connection timeouts.

### Thinking Trace Summarization

Tool results are NOT shown raw in the thinking trace — a `_summarize_tool_result()` function creates concise one-liners:

- **Search**: `"1.2s — 9 results: Title1, Title2, Title3 (+6 more)"`
- **Fetch**: `"2.3s — fetched 15,193 chars from https://example.com/article..."`
- **Python**: `"0.1s — output text here..."` (or truncated at 150 chars)

The FULL tool results are still fed to the LLM — only the user-facing trace is summarized.

Model reasoning is also trimmed: content over 500 chars is shown as first 400 + `[...X chars trimmed...]` + last 100.

---

## Passthrough Mode

Utility requests (title/tag/autocomplete generation from Open WebUI) bypass the agent loop entirely. They are forwarded to Mistral with streaming enabled, and the response chunks are relayed back with the model ID rewritten to `miroflow`.

The passthrough strips Open WebUI-specific fields (`user`, `chat_id`, `tools`, `tool_choice`, `functions`, `function_call`) before forwarding.

---

## Logging

- **Console**: INFO level to stdout
- **File**: DEBUG level to `/opt/deep_research_logs/proxy.log` (rotating, 10MB max, 5 backups)
- Every request, turn, tool call, error, and timing is logged
- The `/logs` endpoint exposes the log file via HTTP

---

## SearXNG Configuration

SearXNG is a self-hosted meta-search engine that aggregates results from multiple backends.

- **Port**: 8888 (localhost only)
- **Settings file**: `/tmp/searxng/searx/settings.yml`
- **Key settings**:
  - `secret_key`: randomized (was default "ultrasecretkey" which caused startup refusal)
  - `formats: [html, csv, json, rss]` — JSON must be enabled for API access (was empty, causing 403)
  - `safe_search: 0` — no filtering
- **Active engines**: Brave, Bing, Wikipedia (Google suspended/access denied, Startpage CAPTCHA, DuckDuckGo timeout — expected from datacenter IP)
- **Typical results**: 18-30 per query from Brave + Bing

---

## Known Issues & Limitations

1. **Context window growth**: Each turn adds tool results to `agent_messages`. With 15 turns of full web pages (15,000 chars each), this can exceed context limits. No summarization or pruning of older messages is implemented.

2. **No conversation memory across requests**: Each request starts fresh. Previous research sessions don't inform new ones.

3. **HTML-to-text extraction is basic**: Regex-based strip of tags. No readability extraction (like Readability.js or trafilatura). Navigation, footers, ads, cookie banners all included in the text.

4. **Python sandbox is minimal**: Uses subprocess with timeout but no seccomp, no network isolation, no filesystem restrictions. The code runs as root.

5. **No parallel tool execution**: Tool calls within a turn are executed sequentially, even when Mistral returns multiple tool_calls in one response.

6. **No streaming from the LLM during agent turns**: Each LLM call uses `stream: False` (needed to get the complete `tool_calls` response). Only the final answer could potentially be streamed from the LLM, but it's also non-streaming (chunked manually at 200 chars).

7. **The forced final answer at turn 15 can be truncated**: If the answer is very long and hits `max_tokens: 4096`, it cuts off mid-sentence. There's no continuation or retry for truncated final answers.

8. **fetch_webpage doesn't handle JavaScript-rendered pages**: Uses raw httpx GET. SPAs and JS-heavy sites return empty or minimal content.

9. **No rate limiting on the proxy itself**: Any authenticated Open WebUI user can trigger unlimited deep research sessions, each consuming 15+ Mistral API calls.

10. **SearXNG engine availability is fragile**: Engines get suspended/blocked from datacenter IPs. Currently only Brave and Bing are reliable. No monitoring or auto-recovery.

11. **The `extract_info` parameter on `fetch_webpage` is cosmetic**: It prepends text but doesn't actually extract anything — the LLM receives the full page and must find the info itself.

12. **No document upload / file attachment support**: Users cannot attach PDFs, documents, or other files for the agent to research. This was requested as a feature.

13. **No knowledge graph / RAG integration**: Research results are not persisted or indexed. Each session is ephemeral. GraphRAG integration was discussed but not implemented.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `UPSTREAM_BASE` | `https://api.mistral.ai/v1` | LLM API base URL |
| `UPSTREAM_KEY` | (Mistral API key) | LLM API key |
| `UPSTREAM_MODEL` | `mistral-large-latest` | Model identifier |
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG instance URL |
| `DEEP_RESEARCH_PORT` | `9200` | Port to listen on |
| `MAX_AGENT_TURNS` | `15` | Maximum research rounds |

---

## File Locations on the VM

| File | Purpose |
|---|---|
| `/opt/deep_research_proxy.py` | The proxy application (this document describes) |
| `/opt/thinking_proxy.py` | Separate proxy for `<think>` tag support on regular Mistral calls |
| `/opt/startup.sh` | Starts all services on boot |
| `/opt/deep_research_logs/proxy.log` | Rotating debug log |
| `/tmp/searxng/` | SearXNG installation |
| `/tmp/searxng/searx/settings.yml` | SearXNG config |

---

## GitHub Repository

`OrpingtonClose/thinking-proxy` (private) — contains both `thinking_proxy.py` and `deep_research_proxy.py`.

---

## Design Intent (from the user)

This system was built with these explicit goals:
- **Anti-censorship**: No content filtering, no moralizing, no safety theater. Direct answers.
- **Exhaustive research**: All 15 rounds should be used. The system prompt and push-back logic both enforce this.
- **Self-hosted search**: SearXNG as the search backbone — no dependence on commercial search APIs.
- **Transparency**: All reasoning visible in thinking traces. All errors surfaced to the user.
- **API-provider based**: Uses Mistral's cloud API rather than local model hosting (switched away from OpenRouter due to key issues).
- **Future plans**: GraphRAG for persistent knowledge extraction, document upload support, subagent architecture for context conservation.
