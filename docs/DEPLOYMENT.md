# Deep Search Portal — Deployment Guide

## Architecture

```
Internet → Cloudflare Tunnel → LibreChat (Node.js, :3000)
                              → Proxy services (:9100–9951)
```

- **LibreChat**: Standalone Node.js process (cloned from source at `/opt/LibreChat/`). MongoDB + Meilisearch run natively.
- **Proxies**: Python FastAPI processes in GNU `screen` sessions (managed by `scripts/startup.sh`)
- **Tunnel**: `cloudflared` routes `deep-search.uk` / `staging.deep-search.uk` to the VM
- **Host**: Vast.ai instances (Ubuntu). Instance IDs change on recreation — always check `vastai show instances`.

### Current Instances

| Role | Instance ID | SSH | URL |
|------|-------------|-----|-----|
| **Production** | 35040116 | `ssh -p 10116 root@ssh1.vast.ai` | https://deep-search.uk |
| **Staging** | 35040117 | `ssh -p 10116 root@ssh4.vast.ai` | https://staging.deep-search.uk |

---

## Fresh VM Setup (Native — No Docker)

Vast.ai containers lack `CAP_NET_ADMIN`, so Docker-in-Docker is impossible. We run everything natively.

```bash
# 1. Create a Vast.ai instance (Ubuntu 22.04, >=32GB RAM, >=80GB disk)
vastai create instance <offer_id> --image ubuntu:22.04 --disk 100 --ssh --direct

# 2. Wait for SSH, then install system packages
ssh root@<host> "apt-get update -qq && apt-get install -y git curl wget screen python3 python3-pip gnupg lsb-release ca-certificates"

# 3. Install MongoDB (check AVX support first: grep avx /proc/cpuinfo)
ssh root@<host> 'curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg && echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" > /etc/apt/sources.list.d/mongodb-org-7.0.list && apt-get update && apt-get install -y mongodb-org && mkdir -p /data/db && mongod --dbpath /data/db --bind_ip 127.0.0.1 --fork --logpath /var/log/mongod.log'

# 4. Install Meilisearch
ssh root@<host> 'curl -fsSL https://install.meilisearch.com | sh && mkdir -p /data/meili && nohup meilisearch --http-addr 127.0.0.1:7700 --db-path /data/meili --no-analytics > /var/log/meilisearch.log 2>&1 &'

# 5. Install cloudflared
ssh root@<host> 'curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb'

# 6. Install Node.js 20
ssh root@<host> 'curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs'

# 7. Clone repos
ssh root@<host> 'git clone https://github.com/OrpingtonClose/deep-search-portal.git /opt/deep-search-portal'
ssh root@<host> 'git clone https://github.com/danny-avila/LibreChat.git /opt/LibreChat && cd /opt/LibreChat && npm ci'

# 8. Copy config (PRODUCTION or STAGING)
ssh root@<host> 'cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/librechat.yaml'
# For staging:
# ssh root@<host> 'cp /opt/deep-search-portal/config/librechat-staging.yaml /opt/LibreChat/librechat.yaml'

# 9. Fix host.docker.internal references (native deployment uses localhost)
ssh root@<host> "sed -i 's/host\.docker\.internal/localhost/g' /opt/LibreChat/librechat.yaml"

# 10. Create /opt/.env with all API keys (see .env.example)
# IMPORTANT: Must include PORT=3000, HOST=0.0.0.0, NODE_ENV=production
# IMPORTANT: DOMAIN_CLIENT and DOMAIN_SERVER must match the server role

# 11. Install Python dependencies
ssh root@<host> 'pip3 install -q fastapi uvicorn httpx pydantic aiohttp requests beautifulsoup4 langgraph langgraph-checkpoint-sqlite langchain-openai langchain-core langchain'

# 12. Create helper script for screen sessions
ssh root@<host> 'cat > /opt/start_proxy.sh << "SCRIPT"
#!/bin/bash
set -a
source /opt/.env
set +a
cd /opt/deep-search-portal/proxies
exec python3 "$@"
SCRIPT
chmod +x /opt/start_proxy.sh'

# 13. Start LibreChat
ssh root@<host> 'screen -dmS librechat bash -c "set -a; source /opt/.env; set +a; cd /opt/LibreChat && node api/server/index.js 2>&1 | tee /var/log/librechat.log"'

# 14. Start Cloudflare tunnel
ssh root@<host> 'screen -dmS cftunnel bash -c "set -a; source /opt/.env; set +a; cloudflared tunnel run --token \$CLOUDFLARE_TUNNEL_TOKEN 2>&1 | tee /var/log/cftunnel.log"'

# 15. Start all proxies
ssh root@<host> 'bash /opt/deep-search-portal/scripts/startup.sh'
```

> **NOTE:** Use `set -a; source /opt/.env; set +a` inside screen sessions to properly export env vars.

> **NOTE:** LibreChat is cloned to `/opt/LibreChat/` — config file goes at `/opt/LibreChat/librechat.yaml`.

Optional env vars: `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `MISTRAL_API_KEY`, `BRIGHT_DATA_API_KEY`, `APIFY_API_TOKEN`.

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

# Fix host.docker.internal references
sed -i 's/host\.docker\.internal/localhost/g' /opt/LibreChat/librechat.yaml

# Restart LibreChat (kill old process and relaunch)
pkill -f 'node api/server/index.js' || true
screen -S librechat -X quit 2>/dev/null || true
screen -dmS librechat bash -c "set -a; source /opt/.env; set +a; cd /opt/LibreChat && node api/server/index.js 2>&1 | tee /var/log/librechat.log"

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
| `scripts/deploy-production.sh` | Full fresh-VM setup (single command) |
| `scripts/startup.sh` | Master startup: all proxies + Cloudflare tunnel |
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
| LibreChat (Node.js) | 3000 | `librechat` |
| Thinking Proxy | 9100 | `thinking-proxy` |
| Deep Research Proxy | 9200 | `deep-research` |
| Persistent MiroFlow | 9300 | `persistent-research` |
| MiroFlow Sprint | 9400 | `miroflow-sprint` |
| Swarm Deep Search | 9500 | `swarm-proxy` |
| G0DM0D3 Proxy | 9600 | `godmode-proxy` |
| xAI Native Proxy | 9700 | `xai-native-proxy` |
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
| `CREDS_KEY` | LibreChat encryption (generate via `openssl rand -hex 32`) |
| `CREDS_IV` | LibreChat encryption IV (generate via `openssl rand -hex 16`) |
| `JWT_SECRET` | LibreChat JWT signing (generate via `openssl rand -hex 32`) |
| `JWT_REFRESH_SECRET` | LibreChat JWT refresh (generate via `openssl rand -hex 32`) |

**Required in .env but often forgotten:**

| Variable | Value | Notes |
|----------|-------|-------|
| `PORT` | `3000` | LibreChat defaults to 3080 without this |
| `HOST` | `0.0.0.0` | Bind to all interfaces |
| `NODE_ENV` | `production` | |
| `MONGO_URI` | `mongodb://127.0.0.1:27017/LibreChat` | |
| `MEILI_HOST` | `http://127.0.0.1:7700` | |
| `DOMAIN_CLIENT` | `https://deep-search.uk` (prod) or `https://staging.deep-search.uk` (staging) | Must match server role |
| `DOMAIN_SERVER` | Same as `DOMAIN_CLIENT` | Must match server role |

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

Each tunnel has its own token. The production `CLOUDFLARE_TUNNEL_TOKEN` must NOT be used on the staging server (and vice versa).

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
| LibreChat won't start | Missing `GOOGLE_CLIENT_ID`, `CREDS_KEY`, or `JWT_SECRET` | Check `/opt/.env` has all required vars |
| LibreChat on wrong port | Missing `PORT=3000` in `.env` | LibreChat defaults to 3080; add `PORT=3000` to `/opt/.env` |
| Login fails ("unknown error") | Missing `JWT_REFRESH_SECRET` or `CREDS_IV` in `.env` | Generate and add: `openssl rand -hex 32` / `openssl rand -hex 16` |
| Login redirects to wrong domain | `DOMAIN_CLIENT`/`DOMAIN_SERVER` set to wrong URL in `.env` | Ensure production uses `https://deep-search.uk`, staging uses `https://staging.deep-search.uk` |
| Proxy won't start in screen | `source /opt/.env` doesn't export vars | Use `set -a; source /opt/.env; set +a` or `/opt/start_proxy.sh` helper |
| `host.docker.internal` in config | Config written for Docker but running natively | Run `sed -i 's/host\.docker\.internal/localhost/g'` on the config |
| Config drift between repo and server | `/opt/LibreChat/librechat.yaml` is a standalone copy | Always copy from repo after `git pull` |
| Staging tunnel uses wrong token | Each tunnel has its own token | Use the staging-specific `CLOUDFLARE_TUNNEL_TOKEN` |
