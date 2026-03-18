# Deep Search Portal

A self-hosted, anti-censorship research portal combining [Open WebUI](https://github.com/open-webui/open-webui) with a custom multi-turn deep research agent (MiroFlow) and a thinking proxy, backed by SearXNG for private web search and Mistral AI for reasoning.

## Architecture

```
Browser → Cloudflare Tunnel (HTTPS + Google OAuth) → Open WebUI (port 3000)
                                                          │
                ┌─────────────────────────────────────────┤
                │                                         │
        Thinking Proxy (9100)                    MiroFlow Deep Research (9200)
                │                                    │         │         │
          Mistral AI API                      Mistral AI   SearXNG   Python exec
                                                          (8888)
```

## Components

| Component | Purpose | Port |
|---|---|---|
| **Open WebUI** | ChatGPT-like web interface with multi-provider support | 3000 |
| **MiroFlow** (`deep_research_proxy.py`) | Agentic deep research — up to 15 rounds of search/read/analyze | 9200 |
| **Thinking Proxy** (`thinking_proxy.py`) | Wraps Mistral for `<think>` tag streaming support | 9100 |
| **SearXNG** | Self-hosted meta-search engine (Brave, Bing, Wikipedia) | 8888 |
| **Cloudflare Tunnel** | HTTPS + domain routing to the VM | — |

## How MiroFlow Works

1. User asks a question via Open WebUI
2. Proxy injects a research-focused system prompt and sends to Mistral with tool definitions
3. Mistral reasons about the question and calls tools (search, fetch web pages, run Python)
4. Proxy executes tools, feeds results back to Mistral
5. Repeat for up to 15 rounds — push-back logic prevents early stopping
6. All reasoning streams to the user inside `<think>` tags (collapsible in Open WebUI)
7. Final comprehensive answer streams after `</think>`

### Tools

- **`searxng_search`** — Web search via local SearXNG (returns top 10 results with snippets)
- **`fetch_webpage`** — Fetches and extracts text from web pages (15K char limit)
- **`python_exec`** — Sandboxed Python execution for calculations/analysis (30s timeout)
- **`knowledge_search`** — Search through ingested text documents (books, papers, reports)

## Repo Structure

```
├── proxies/
│   ├── deep_research_proxy.py    # MiroFlow deep research agent
│   └── thinking_proxy.py         # Thinking tag proxy
├── scripts/
│   ├── startup.sh                # Master startup (all services)
│   └── start_openwebui.sh        # Open WebUI with provider config
├── config/
│   ├── searxng_settings_patch.yml  # Key SearXNG overrides
│   └── searxng_settings_full.yml   # Full SearXNG settings reference
├── docs/
│   └── architecture.md           # Detailed architecture document
├── .env.example                  # Environment variables template
└── README.md
```

## Deployment

### Prerequisites

- Ubuntu 22.04 VM (tested on Vast.ai, 1x RTX 2060, $0.037/hr)
- Python 3.10+
- Cloudflare account with a domain
- Mistral AI API key
- Google OAuth client for login

### Quick Setup

1. Clone this repo to `/opt/deep-search-portal` on the VM
2. Copy `.env.example` to `/opt/.env` and fill in credentials
3. Install Open WebUI: `pip install open-webui==0.8.8` (into a venv at `/opt/openwebui-env`)
4. Clone SearXNG: `git clone https://github.com/searxng/searxng.git /tmp/searxng`
5. Apply `config/searxng_settings_patch.yml` to SearXNG's `settings.yml`
6. Install proxy dependencies: `pip install fastapi uvicorn httpx`
7. Copy proxy files to `/opt/` and scripts to `/opt/`
8. Set up Cloudflare Tunnel pointing to `localhost:3000`
9. Run `startup.sh`

### Endpoints

| Endpoint | Purpose |
|---|---|
| `/v1/chat/completions` | Main OpenAI-compatible endpoint (on each proxy) |
| `/health` | Health check with active request details |
| `/logs?lines=100` | Last N lines from the debug log |
| `POST /v1/ingest` | Ingest a large text document into the knowledge base |
| `GET /v1/documents` | List all ingested documents |
| `DELETE /v1/documents/{doc_id}` | Remove an ingested document |

## Provider Configuration

Open WebUI connects to 8 providers (configured in `start_openwebui.sh`):

| Index | Provider | Models |
|---|---|---|
| 0 | OpenRouter | Various |
| 1 | Venice.ai | Uncensored models |
| 2 | Together.ai | Open-source models |
| 3 | Perplexity | Perplexity models |
| 4 | RunPod | Custom deployments |
| 5 | Thinking Proxy | `mistral-large-thinking` |
| 6 | Deep Research | `miroflow` |
| 7 | Mistral Direct | `mistral-large-latest`, `mistral-medium-latest` |

## Text Ingestion

All three proxies support ingesting large text documents (books, papers, reports) that agents can search during research.

### Ingest a document

```bash
curl -X POST http://localhost:9200/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "My Research Paper",
    "text": "Full text content here...",
    "source": "https://example.com/paper.pdf"
  }'
```

The text is automatically chunked (~2000 chars with 200 char overlap) and indexed with SQLite FTS5 for fast full-text search.

### List / delete documents

```bash
# List all ingested documents
curl http://localhost:9200/v1/documents

# Delete a document
curl -X DELETE http://localhost:9200/v1/documents/doc-abc123
```

### Agent usage

The `knowledge_search` tool is available to the MiroFlow and Persistent Research agents. When ingested documents exist, agents will search them first before querying the web.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `INGEST_DB` | `/opt/ingested_texts/ingest.db` | Path to the ingestion SQLite database |
| `INGEST_CHUNK_SIZE` | `2000` | Characters per chunk |
| `INGEST_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

## Design Philosophy

- **No censorship** — system prompts demand direct answers without moralizing, disclaimers, or safety theater
- **Exhaustive research** — all 15 rounds used by default; push-back logic forces the model to keep digging
- **Full transparency** — all reasoning visible in thinking traces; all errors surfaced to the user
- **Self-hosted search** — SearXNG means no dependence on commercial search APIs
- **API-provider architecture** — uses Mistral cloud API, not local model hosting

## License

Private.
