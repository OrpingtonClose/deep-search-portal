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
                                    Persistent Research (9300)
                                         │         │
                                    Knowledge Engine (9400) ←── Neo4j

                                    Swarm Proxy (9500) ← fully self-contained
                                         │
                                  Background Swarm Workers
                                 (in-memory knowledge store,
                                  LLM-powered extraction)
```

## Components

| Component | Purpose | Port |
|---|---|---|
| **Open WebUI** | ChatGPT-like web interface with multi-provider support | 3000 |
| **MiroFlow** (`deep_research_proxy.py`) | Agentic deep research — up to 15 rounds of search/read/analyze | 9200 |
| **Persistent Research** (`persistent_deep_research_proxy.py`) | Multi-session research with knowledge accumulation | 9300 |
| **Thinking Proxy** (`thinking_proxy.py`) | Wraps Mistral for `<think>` tag streaming support | 9100 |
| **Knowledge Engine** (`services/knowledge-engine/`) | Neo4j-centric knowledge corpus ETL microservice | 9400 |
| **Swarm Proxy** (`swarm_proxy.py`) | Swarm-based corpus decomposition with non-disruptive querying | 9500 |
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
- **`knowledge_graph_search`** — Search the Neo4j knowledge graph (hybrid, keyword, or graph traversal)
- **`knowledge_discover`** — Run graph discovery algorithms (spreading activation, Swanson ABC, information gaps)

## Repo Structure

```
├── proxies/
│   ├── deep_research_proxy.py          # MiroFlow deep research agent
│   ├── persistent_deep_research_proxy.py # Multi-session persistent research
│   ├── swarm_proxy.py                  # Swarm-based corpus decomposition proxy
│   ├── thinking_proxy.py               # Thinking tag proxy
│   └── knowledge_client.py             # Lightweight async client for knowledge engine
├── services/
│   └── knowledge-engine/
│       ├── knowledge_engine/
│       │   ├── main.py                 # FastAPI application
│       │   ├── config.py               # Configuration & logging
│       │   ├── models.py               # Pydantic request/response models
│       │   ├── neo4j_client.py         # Neo4j connection & schema management
│       │   ├── ontology.py             # Epistemic ontology (node/relationship creation)
│       │   ├── chunker.py              # Text chunking with sentence boundaries
│       │   ├── extractor.py            # Multi-pass LLM entity/relationship extraction
│       │   ├── entity_resolver.py      # Entity resolution & deduplication
│       │   ├── pipeline.py             # ETL pipeline orchestrator
│       │   ├── algorithms.py           # Graph algorithms (spreading activation, Swanson ABC, etc.)
│       │   └── search.py               # Unified search (keyword, graph, hybrid)
│       └── requirements.txt
├── scripts/
│   ├── startup.sh                      # Master startup (all services)
│   └── start_openwebui.sh              # Open WebUI with provider config
├── config/
│   ├── searxng_settings_patch.yml      # Key SearXNG overrides
│   └── searxng_settings_full.yml       # Full SearXNG settings reference
├── docs/
│   └── architecture.md                 # Detailed architecture document
├── .env.example                        # Environment variables template
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

## Knowledge Engine

A standalone Neo4j-centric microservice for ingesting large text corpora, extracting knowledge with multi-pass LLM extraction, and performing graph-based discovery.

### Quick Start

```bash
# Install dependencies
cd services/knowledge-engine
pip install -r requirements.txt

# Set environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your-password
export UPSTREAM_KEY=your-mistral-api-key

# Run the service
python -m uvicorn knowledge_engine.main:app --host 0.0.0.0 --port 9400
```

### ETL Pipeline

When a corpus is ingested via `POST /v1/ingest`, the engine runs a full ETL pipeline:

1. **Chunk** — Split text into ~2000 char chunks with 200 char overlap at sentence boundaries
2. **Extract (Pass 1)** — Direct extraction: concepts, claims, evidence, methods
3. **Extract (Pass 2)** — Implicit extraction: hypotheses, anomalies, analogies, implicit relationships
4. **Extract (Pass 3)** — Cross-chunk relationship inference
5. **Entity Resolution** — Exact-match MERGE + fuzzy deduplication of near-duplicate concepts
6. **Load** — Write epistemic ontology nodes into Neo4j (Concept, Claim, Hypothesis, Anomaly, Evidence, Method)
7. **Graph Metrics** — Compute Louvain community detection, betweenness centrality, RNS serendipity scores

### Epistemic Ontology

| Node Type | Description |
|---|---|
| `Document` | Source document metadata |
| `Chunk` | Text chunk (~2K chars) with reading order |
| `Concept` | Named concept/entity with domains and abstraction level |
| `Claim` | Factual assertion with confidence and polarity |
| `Hypothesis` | Speculative statement with status and abductive origin |
| `Anomaly` | Surprising finding with surprise score |
| `Evidence` | Supporting evidence with strength |
| `Method` | Research method with domain and transferability |

Relationship types: `ANALOGOUS_TO`, `CONTRADICTS`, `EXPLAINS`, `SUPPORTED_BY`, `TRANSFERABLE_TO`, `INSTANCE_OF`, `REQUIRES`, `UPDATES`, `DERIVED_FROM`, `PART_OF`, `EXTRACTED_BY`, `OBSERVED_IN`

### Graph Algorithms

- **Spreading Activation** — Multi-hop activation propagation from seed concepts with configurable decay
- **Swanson ABC** — Literature-based bisociation discovery (A→B→C where A and C are not directly connected)
- **Information Gaps** — Find under-connected but frequently mentioned concepts
- **Serendipity Beam Search** — RNS-guided beam search (Relevance × Novelty × Surprise)
- **Community Detection** — Louvain algorithm via networkx (no GDS dependency)

### REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/ingest` | POST | Ingest text corpus (triggers full ETL pipeline) |
| `/v1/ingest/{job_id}` | GET | Check ingestion job status |
| `/v1/search` | POST | Unified search (hybrid, keyword, graph) |
| `/v1/algorithms/spreading-activation` | POST | Run spreading activation |
| `/v1/algorithms/swanson-abc` | POST | Run Swanson ABC discovery |
| `/v1/algorithms/information-gaps/{ns}` | GET | Find information gaps |
| `/v1/algorithms/serendipity-beam` | POST | Serendipity beam search |
| `/v1/graph/neighborhood/{ns}/{concept}` | GET | Get concept neighborhood |
| `/v1/namespaces` | GET | List all namespaces with stats |
| `/v1/namespaces/{name}` | DELETE | Delete namespace and all data |
| `/v1/graph/stats/{namespace}` | GET | Get graph statistics |

### Configuration

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `neo4j` | Neo4j password |
| `UPSTREAM_BASE` | `https://api.mistral.ai/v1` | LLM API base URL |
| `UPSTREAM_KEY` | — | Mistral API key |
| `EXTRACTION_MODEL` | `mistral-small-latest` | Model for entity/relationship extraction |
| `EMBEDDING_MODEL` | `mistral-embed` | Model for embeddings |
| `RAW_FILES_DIR` | `/opt/knowledge_corpus/files` | Raw file storage directory |
| `CHUNK_SIZE` | `2000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_EXTRACTION_CONCURRENCY` | `5` | Max parallel LLM extraction calls |
| `KNOWLEDGE_ENGINE_URL` | `http://localhost:9400` | URL for proxy client |

### Namespace Isolation

All data is namespaced by conversation/context. Every node gets a `namespace` property. Queries filter by namespace. Use `DELETE /v1/namespaces/{name}` to remove all data for a context. Set `rebuild=true` on ingest to clear the namespace before loading new data.

## Swarm Proxy

A swarm-based proxy (inspired by [swarms.world](https://swarms.world)) that decomposes large corpora of text using background worker agents. Unlike the other proxies, the swarm operates continuously — submitting text starts background processing that runs independently of any queries.

### Key Principles

- **Non-disruptive querying** — Sending a prompt does NOT interrupt the swarm's current work. Queries are answered from whatever knowledge the swarm has built so far.
- **Sincerity** — The proxy honestly reports what the swarm is doing: processing progress, active workers, knowledge coverage, and any gaps. No pretending work is done when it isn't.
- **Additive ingestion** — Further large corpora are treated identically to the initial send. They queue additively without resetting existing work.
- **Background processing** — Corpus decomposition (chunking, entity extraction, relationship extraction, knowledge graph construction) happens asynchronously via a worker pool.

### How It Works

1. User sends a large body of text via the chat interface
2. The swarm proxy detects it as a corpus (>5K chars, low question density) and queues it
3. Background workers chunk the text and use the LLM to extract entities, relationships, and claims
4. All extracted knowledge is stored in an in-memory knowledge store (no external DB)
5. While processing continues, the user can ask questions at any time
6. Queries search the in-memory knowledge store using TF-IDF-like scoring
7. The proxy prefixes every response with an honest status of what the swarm is doing
8. Sending more text adds to the queue — it never resets or interrupts existing work

### Swarm API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (auto-routes corpus vs query) |
| `/v1/swarm/status` | GET | Full swarm status (workers, corpora, knowledge stats) |
| `/v1/swarm/corpora` | GET | List all submitted corpora with processing status |
| `/v1/swarm/sincerity` | GET | Current sincerity preamble (what the swarm would tell a user) |
| `/v1/swarm/submit` | POST | Direct corpus submission API |
| `/v1/swarm/knowledge` | GET | Knowledge store statistics (entities, relationships, claims, chunks) |

### Configuration

| Variable | Default | Description |
|---|---|---|
| `SWARM_PROXY_PORT` | `9500` | Listen port |
| `SWARM_SYNTHESIS_MODEL` | `mistral-large-latest` | Model for query synthesis |
| `SWARM_WORKER_MODEL` | `mistral-small-latest` | Model for worker tasks |
| `SWARM_MAX_WORKERS` | `6` | Max concurrent background workers |
| `SWARM_MAX_CONCURRENT_QUERIES` | `4` | Max concurrent query handlers |
| `SWARM_CHUNK_SIZE` | `2000` | Characters per chunk |
| `SWARM_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SWARM_LARGE_DOC_THRESHOLD` | `5000` | Char threshold for corpus detection |

## Design Philosophy

- **No censorship** — system prompts demand direct answers without moralizing, disclaimers, or safety theater
- **Exhaustive research** — all 15 rounds used by default; push-back logic forces the model to keep digging
- **Full transparency** — all reasoning visible in thinking traces; all errors surfaced to the user
- **Self-hosted search** — SearXNG means no dependence on commercial search APIs
- **API-provider architecture** — uses Mistral cloud API, not local model hosting
- **Self-contained swarm** — the swarm proxy builds its own in-memory knowledge store with no external infrastructure dependencies

## License

Private.
