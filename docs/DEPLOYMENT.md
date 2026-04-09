# Deep Search Portal — Deployment Guide

## Architecture

```
Internet → Cloudflare Tunnel → LibreChat Docker (:3000 → :3080 inside container)
                              → Proxy services (:9100–9951)
```

- **LibreChat**: Docker Compose stack (`config/docker-compose.librechat.yml`) — API + MongoDB + Meilisearch
- **Proxies**: Python FastAPI processes in GNU `screen` sessions (managed by `scripts/startup.sh`)
- **Tunnel**: `cloudflared` routes `deep-search.uk` / `staging.deep-search.uk` to the VM
- **Host**: Vast.ai instances (Ubuntu). Instance IDs change on recreation — always check `vastai show instances`.

---

## Fresh VM Setup (One Command)

```bash
# 1. Create a Vast.ai instance (Ubuntu, >=32GB RAM, >=80GB disk)
vastai create instance <offer_id> --image ubuntu:22.04 --disk 100 --ssh --direct

# 2. Wait for SSH, then deploy
scp scripts/deploy-production.sh root@<host>:/tmp/deploy.sh
ssh root@<host> "
  export VENICE_API_KEY=...
  export XAI_API_KEY=...
  export OPENROUTER_API_KEY=...
  export GOOGLE_CLIENT_ID=...
  export GOOGLE_CLIENT_SECRET=...
  export CLOUDFLARE_TUNNEL_TOKEN=...
  bash /tmp/deploy.sh
"
```

The script installs Docker, clones the repo, writes `/opt/.env`, starts LibreChat via Docker Compose, launches all proxies, connects the Cloudflare tunnel, and runs health checks. ~5 minutes total.

Optional env vars: `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `MISTRAL_API_KEY`, `BRIGHT_DATA_API_KEY`.

---

## Deploying Code Changes (Existing VM)

```bash
# Pull latest
cd /opt/deep-search-portal && git pull origin main

# Copy the RIGHT config (back up first!)
cp /opt/LibreChat/librechat.yaml /opt/LibreChat/librechat.yaml.bak

# Production:
cp config/librechat.yaml /opt/LibreChat/librechat.yaml
# Staging:
cp config/librechat-staging.yaml /opt/LibreChat/librechat.yaml

# Restart LibreChat
bash scripts/start_librechat.sh restart

# Restart proxies (if proxy code changed)
bash scripts/startup.sh
```

> **CRITICAL:** `/opt/LibreChat/librechat.yaml` is a standalone copy. `git pull` does NOT update it. You must copy explicitly.

---

## Config Files

| File | Purpose |
|------|---------|
| `config/librechat.yaml` | **Production** — 3 models (Simple group), `enforce: true` |
| `config/librechat-staging.yaml` | **Staging** — 41+ models, "Simple PROD" marks production models, `enforce: false` |
| `config/docker-compose.librechat.yml` | Docker Compose: LibreChat API + MongoDB + Meilisearch |
| `scripts/deploy-production.sh` | Full fresh-VM setup (single command) |
| `scripts/startup.sh` | Master startup: all proxies + LibreChat + Cloudflare tunnel |
| `scripts/start_librechat.sh` | Docker Compose wrapper (`up`, `stop`, `restart`, `status`, `logs`) |
| `.env.example` | Template for all env vars |

---

## Production Models (Simple Group)

1. **Grok 4.20 Multi-Agent** — xAI flagship, parallel reasoning, web search
2. **Heretic Uncensored** — GLM-4.7 Flash with Firecrawl/Exa/Brave tools
3. **Tier Race (Full Throttle)** — Races 9 flagship models

All other models exist only in staging.

---

## Services & Ports

| Service | Port | Screen Session |
|---------|------|----------------|
| LibreChat (Docker) | 3000 (host) → 3080 (container) | N/A (Docker) |
| Thinking Proxy | 9100 | `thinking-proxy` |
| Deep Research Proxy | 9200 | `deep-research` |
| Persistent MiroFlow | 9300 | `persistent-research` |
| MiroFlow Sprint | 9400 | `miroflow-sprint` |
| Swarm Deep Search | 9500 | `swarm-proxy` |
| G0DM0D3 Proxy | 9600 | `godmode-proxy` |
| xAI Native Proxy | 9700 | `xai-native-proxy` |
| Knowledge Engine | 9850 | `knowledge-engine` |
| Tier Chooser Proxy | 9900 | `litellm` |
| Heretic Proxy | 9950 | `heretic-proxy` |
| Miro Proxy | 9951 | `miro-proxy` |
| SearXNG | 8888 | `searxng` |

---

## Environment Variables

All stored in `/opt/.env` on each instance. See `.env.example` for the full list.

**Required for deployment:**

| Variable | Used by |
|----------|---------|
| `VENICE_API_KEY` | Heretic proxy, Miro proxy, Swarm proxy |
| `XAI_API_KEY` | xAI native proxy, Deep Research proxy |
| `OPENROUTER_API_KEY` | G0DM0D3 proxy (55 models via OpenRouter) |
| `GOOGLE_CLIENT_ID` | LibreChat Google OAuth login |
| `GOOGLE_CLIENT_SECRET` | LibreChat Google OAuth login |
| `CLOUDFLARE_TUNNEL_TOKEN` | Cloudflare tunnel (public routing) |
| `CREDS_KEY` | LibreChat encryption (auto-generated if missing) |
| `JWT_SECRET` | LibreChat JWT signing (auto-generated if missing) |

**Optional (tools degrade gracefully without these):**

| Variable | Used by |
|----------|---------|
| `FIRECRAWL_API_KEY` | Heretic/Miro web scraping tools |
| `EXA_API_KEY` | Heretic/Miro semantic search tools |
| `BRAVE_SEARCH_API_KEY` | Heretic/Miro web search + media enrichment |
| `MISTRAL_API_KEY` | Thinking proxy |
| `BRIGHT_DATA_API_KEY` | SearXNG proxy routing |

---

## Vast.ai Access

```bash
# List instances
vastai show instances

# SSH into an instance
ssh -o StrictHostKeyChecking=no -p <port> root@<ssh_host>

# Find SSH details for an instance
vastai ssh-url <instance_id>
```

Requires `VAST_AI_API_KEY` (or `VASTAI_API_KEY`) env var.

---

## Adding a New Model to Production

1. Add proxy code under `proxies/` (if new endpoint)
2. Add modelSpec to `config/librechat.yaml` in Simple group
3. Add endpoint wiring to `config/librechat.yaml` under `endpoints.custom`
4. Add the same model to `config/librechat-staging.yaml` in the "Simple PROD" group
5. Add startup entry to `scripts/startup.sh`
6. Create PR, wait for CI
7. Deploy to both instances (see "Deploying Code Changes" above)

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Models not visible after deploy | Forgot to copy config to `/opt/LibreChat/librechat.yaml` | Copy the right config file and restart LibreChat |
| Wrong models on wrong server | Used production config on staging or vice versa | Check instance role with `vastai show instances`, use correct config file |
| LibreChat won't start | Missing `GOOGLE_CLIENT_ID`, `CREDS_KEY`, or `JWT_SECRET` | Check `/opt/.env` has all required vars. `CREDS_KEY`/`JWT_SECRET` auto-generate if missing. |
| Proxy won't start | Missing API key in `/opt/.env` | `source /opt/.env && env \| grep KEY` to verify |
| Config drift between repo and server | `/opt/LibreChat/librechat.yaml` is a standalone copy | Always copy from repo after `git pull` |
