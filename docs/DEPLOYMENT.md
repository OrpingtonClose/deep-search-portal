# Deep Search Portal — Deployment Guide

## Architecture

```
Internet → Cloudflare Tunnel → LibreChat (Node.js, :3000)
                              → Proxy services (:9100–9951)
```

- **LibreChat**: Standalone Node.js process (extracted from Docker image via `crane`). MongoDB + Meilisearch run natively.
- **Proxies**: Python FastAPI processes in GNU `screen` sessions (managed by `scripts/startup.sh`)
- **Tunnel**: `cloudflared` routes `deep-search.uk` / `staging.deep-search.uk` to the VM
- **Host**: Vast.ai instances (Ubuntu). Instance IDs change on recreation — always check `vastai show instances`.

### Current Instances

| Role | Instance ID | SSH | URL |
|------|-------------|-----|-----|
| **Production** | 34878427 | `ssh -p 38426 root@ssh7.vast.ai` | https://deep-search.uk |
| **Staging** | 34878429 | `ssh -p 38428 root@ssh3.vast.ai` | https://staging.deep-search.uk |

---

## Fresh VM Setup (Native — No Docker)

Vast.ai containers lack `CAP_NET_ADMIN`, so Docker-in-Docker is impossible. We run everything natively.

```bash
# 1. Create a Vast.ai instance (Ubuntu 22.04, >=32GB RAM, >=80GB disk)
vastai create instance <offer_id> --image ubuntu:22.04 --disk 100 --ssh --direct

# 2. Wait for SSH, then run Phase 1 (system packages)
ssh root@<host> "apt-get update -qq && apt-get install -y git curl wget screen python3 python3-pip gnupg lsb-release ca-certificates"

# 3. Install MongoDB 7
ssh root@<host> 'curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg && echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" > /etc/apt/sources.list.d/mongodb-org-7.0.list && apt-get update -qq && apt-get install -y mongodb-org && mkdir -p /data/db && nohup mongod --dbpath /data/db --bind_ip 127.0.0.1 --noauth > /var/log/mongod.log 2>&1 &'

# 4. Install Meilisearch
ssh root@<host> 'curl -fsSL https://install.meilisearch.com | sh && mkdir -p /data/meili && nohup meilisearch --http-addr 127.0.0.1:7700 --db-path /data/meili --no-analytics > /var/log/meilisearch.log 2>&1 &'

# 5. Install cloudflared
ssh root@<host> 'curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb'

# 6. Install Node.js 20
ssh root@<host> 'curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs'

# 7. Clone repo
ssh root@<host> 'git clone https://github.com/OrpingtonClose/deep-search-portal.git /opt/deep-search-portal'

# 8. Extract pre-built LibreChat from Docker image (avoids fragile source builds)
ssh root@<host> 'curl -fsSL https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz | tar xz -C /usr/local/bin crane && mkdir -p /opt/LibreChat/app && cd /opt/LibreChat/app && crane export ghcr.io/danny-avila/librechat:latest - | tar x'

# 9. Copy config (PRODUCTION or STAGING)
ssh root@<host> 'cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/app/librechat.yaml'
# For staging:
# ssh root@<host> 'cp /opt/deep-search-portal/config/librechat-staging.yaml /opt/LibreChat/app/librechat.yaml'

# 10. SCP the .env file (contains all API keys — see .env.example)
scp /tmp/prod.env root@<host>:/opt/.env

# 11. Install Python dependencies
ssh root@<host> 'pip3 install -q fastapi uvicorn httpx pydantic aiohttp requests beautifulsoup4 langgraph langgraph-checkpoint-sqlite langchain-openai langchain-core langchain'

# 12. Start LibreChat natively
ssh root@<host> 'set -a; source /opt/.env; set +a; cd /opt/LibreChat/app && nohup env HOST=0.0.0.0 PORT=3000 NODE_ENV=production MONGO_URI=mongodb://127.0.0.1:27017/LibreChat MEILI_HOST=http://127.0.0.1:7700 ENDPOINTS=custom ALLOW_SOCIAL_LOGIN=true ALLOW_SOCIAL_REGISTRATION=true ALLOW_EMAIL_LOGIN=true ALLOW_REGISTRATION=false GOOGLE_CALLBACK_URL=/oauth/google/callback node api/server/index.js > /var/log/librechat.log 2>&1 &'

# 13. Start all proxies and tunnel
ssh root@<host> 'cd /opt/deep-search-portal && bash scripts/startup.sh'
```

> **NOTE:** `scripts/startup.sh` attempts Docker Compose for LibreChat first. On native deployments (no Docker), this fails gracefully (patched with `|| true`) and the script continues to start all proxies and the Cloudflare tunnel. LibreChat must already be running from step 12.

Optional env vars: `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `MISTRAL_API_KEY`, `BRIGHT_DATA_API_KEY`, `APIFY_API_TOKEN`.

---

## Deploying Code Changes (Existing VM)

```bash
# Pull latest
cd /opt/deep-search-portal && git pull origin main

# Copy the RIGHT config (back up first!)
cp /opt/LibreChat/app/librechat.yaml /opt/LibreChat/app/librechat.yaml.bak

# Production:
cp config/librechat.yaml /opt/LibreChat/app/librechat.yaml
# Staging:
cp config/librechat-staging.yaml /opt/LibreChat/app/librechat.yaml

# Restart LibreChat (kill old process and relaunch)
pkill -f 'node api/server/index.js' || true
cd /opt/LibreChat/app && set -a && source /opt/.env && set +a && nohup env HOST=0.0.0.0 PORT=3000 NODE_ENV=production MONGO_URI=mongodb://127.0.0.1:27017/LibreChat MEILI_HOST=http://127.0.0.1:7700 ENDPOINTS=custom ALLOW_SOCIAL_LOGIN=true ALLOW_SOCIAL_REGISTRATION=true ALLOW_EMAIL_LOGIN=true ALLOW_REGISTRATION=false GOOGLE_CALLBACK_URL=/oauth/google/callback node api/server/index.js > /var/log/librechat.log 2>&1 &

# Restart proxies (if proxy code changed)
bash scripts/startup.sh
```

> **CRITICAL:** `/opt/LibreChat/app/librechat.yaml` is a standalone copy. `git pull` does NOT update it. You must copy explicitly.

---

## Config Files

| File | Purpose |
|------|---------|
| `config/librechat.yaml` | **Production** — 3 models (Simple group), `enforce: true` |
| `config/librechat-staging.yaml` | **Staging** — 41+ models, "Simple PROD" marks production models, `enforce: false` |
| `config/docker-compose.librechat.yml` | Docker Compose (legacy, not used on Vast.ai — kept for local dev) |
| `scripts/deploy-production.sh` | Full fresh-VM setup (single command) |
| `scripts/startup.sh` | Master startup: all proxies + LibreChat + Cloudflare tunnel |
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
| LibreChat (Node.js) | 3000 | N/A (nohup) |
| Thinking Proxy | 9100 | `thinking-proxy` |
| Deep Research Proxy | 9200 | `deep-research` |
| Persistent MiroFlow | 9300 | `persistent-research` |
| MiroFlow Sprint | 9400 | `miroflow-sprint` |
| Swarm Deep Search | 9500 | `swarm-proxy` |
| G0DM0D3 Proxy | 9600 | `godmode-proxy` |
| xAI Native Proxy | 9700 | `xai-native-proxy` |
| Persistent MiroFlow Wiki | 9800 | `persistent-research-wiki` |
| MiroFlow Sprint Wiki | 9850 | `miroflow-sprint-wiki` |
| Tier Chooser Proxy | 9900 | `tier-chooser` |
| Heretic Proxy | 9950 | `heretic-proxy` |
| Miro Proxy | 9951 | `miro-proxy` |

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

## Cloudflare Tunnels

Production and staging use **separate** Cloudflare tunnels:

| Role | Tunnel ID | Hostname |
|------|-----------|----------|
| Production | `6ef95cd2-6405-4257-ba6c-6790c7d7b97e` | `deep-search.uk` |
| Staging | `ba0b804c-5969-4865-b4dd-7fb1f8b6a1b6` | `staging.deep-search.uk` |

Each tunnel has its own token. The production `CLOUDFLARE_TUNNEL_TOKEN` must NOT be used on the staging server (and vice versa). To get the staging tunnel token, use the Cloudflare API:

```bash
curl -s "https://api.cloudflare.com/client/v4/accounts/<account_id>/cfd_tunnel/<tunnel_id>/token" \
  -H "Authorization: Bearer <CF_API_TOKEN>" | jq -r '.result'
```

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
| Models not visible after deploy | Forgot to copy config to `/opt/LibreChat/app/librechat.yaml` | Copy the right config file and restart LibreChat |
| Wrong models on wrong server | Used production config on staging or vice versa | Check instance role with `vastai show instances`, use correct config file |
| LibreChat won't start | Missing `GOOGLE_CLIENT_ID`, `CREDS_KEY`, or `JWT_SECRET` | Check `/opt/.env` has all required vars. `CREDS_KEY`/`JWT_SECRET` auto-generate if missing. |
| Proxy won't start | Missing API key in `/opt/.env` | `source /opt/.env && env \| grep KEY` to verify |
| Config drift between repo and server | `/opt/LibreChat/app/librechat.yaml` is a standalone copy | Always copy from repo after `git pull` |
| Staging tunnel uses wrong token | Each tunnel has its own token | Get the staging-specific token from the Cloudflare API (see Cloudflare Tunnels section) |
