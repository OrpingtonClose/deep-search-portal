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
| **Production** | 34951881 | `ssh -p 31880 root@ssh6.vast.ai` | https://deep-search.uk |
| **Staging** | 34954619 | `ssh -p 34618 root@ssh7.vast.ai` | https://staging.deep-search.uk |

---

## Fresh VM Setup (Native — No Docker)

Vast.ai containers lack `CAP_NET_ADMIN`, so Docker-in-Docker is impossible. We run everything natively.

```bash
# 1. Create a Vast.ai instance (Ubuntu 22.04, >=32GB RAM, >=80GB disk)
vastai create instance <offer_id> --image ubuntu:22.04 --disk 100 --ssh --direct

# 2. Wait for SSH, then run Phase 1 (system packages)
ssh root@<host> "apt-get update -qq && apt-get install -y git curl wget screen python3 python3-pip gnupg lsb-release ca-certificates"

# 3. Install Percona MongoDB 4.4 (MongoDB 7.0 requires AVX — many Vast.ai CPUs lack it)
ssh root@<host> 'wget https://repo.percona.com/apt/percona-release_latest.$(lsb_release -sc)_all.deb && dpkg -i percona-release_latest.*.deb && percona-release setup psmdb44 && apt-get install -y percona-server-mongodb && mkdir -p /var/lib/mongodb && mongod --dbpath /var/lib/mongodb --bind_ip 127.0.0.1 --fork --logpath /var/log/mongodb/mongod.log'

# 4. Install Meilisearch
ssh root@<host> 'curl -fsSL https://install.meilisearch.com | sh && mkdir -p /data/meili && nohup meilisearch --http-addr 127.0.0.1:7700 --db-path /data/meili --no-analytics > /var/log/meilisearch.log 2>&1 &'

# 5. Install cloudflared
ssh root@<host> 'curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb'

# 6. Install Node.js 22
ssh root@<host> 'curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs'

# 7. Clone repo
ssh root@<host> 'git clone https://github.com/OrpingtonClose/deep-search-portal.git /opt/deep-search-portal'

# 8. Extract pre-built LibreChat from Docker image (avoids fragile source builds)
ssh root@<host> 'curl -fsSL https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz | tar xz -C /usr/local/bin crane && mkdir -p /opt/LibreChat/app && cd /opt/LibreChat/app && crane export ghcr.io/danny-avila/librechat:latest - | tar x'

# 9. Copy config (PRODUCTION or STAGING)
ssh root@<host> 'cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/app/app/librechat.yaml'
# For staging:
# ssh root@<host> 'cp /opt/deep-search-portal/config/librechat-staging.yaml /opt/LibreChat/app/app/librechat.yaml'

# 10. Fix host.docker.internal references (native deployment uses localhost)
ssh root@<host> "sed -i 's/host\.docker\.internal/localhost/g' /opt/LibreChat/app/app/librechat.yaml"

# 11. SCP the .env file (contains all API keys — see .env.example)
scp /tmp/prod.env root@<host>:/opt/.env
# IMPORTANT: Also copy to LibreChat's working directory (dotenv loads from cwd)
ssh root@<host> 'cp /opt/.env /opt/LibreChat/app/app/.env'

# 12. Install Python dependencies
ssh root@<host> 'pip3 install -q fastapi uvicorn httpx pydantic aiohttp requests beautifulsoup4 langgraph langgraph-checkpoint-sqlite langchain-openai langchain-core langchain'

# 13. Start LibreChat natively
ssh root@<host> 'cd /opt/LibreChat/app/app && export $(grep -v "^#" /opt/.env | xargs) && nohup node api/server/index.js > /var/log/librechat.log 2>&1 &'

# 14. Start all proxies via screen sessions
ssh root@<host> 'cd /opt/deep-search-portal && bash scripts/startup.sh'

# 15. Start Cloudflare tunnel
ssh root@<host> 'screen -dmS cftunnel bash -c "cloudflared tunnel run --token $CLOUDFLARE_TUNNEL_TOKEN 2>&1 | tee /var/log/cftunnel.log"'
```

> **NOTE:** `source /opt/.env` does NOT work inside screen sessions. Use `export $(grep -v "^#" /opt/.env | xargs)` instead.

> **NOTE:** LibreChat extracted via `crane` ends up at `/opt/LibreChat/app/app/` (nested), not `/opt/LibreChat/app/`.

Optional env vars: `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `MISTRAL_API_KEY`, `BRIGHT_DATA_API_KEY`, `APIFY_API_TOKEN`.

---

## Deploying Code Changes (Existing VM)

```bash
# Pull latest
cd /opt/deep-search-portal && git pull origin main

# Copy the RIGHT config (back up first!)
cp /opt/LibreChat/app/app/librechat.yaml /opt/LibreChat/app/app/librechat.yaml.bak

# Production:
cp config/librechat.yaml /opt/LibreChat/app/app/librechat.yaml
# Staging:
# cp config/librechat-staging.yaml /opt/LibreChat/app/app/librechat.yaml

# Fix host.docker.internal references
sed -i 's/host\.docker\.internal/localhost/g' /opt/LibreChat/app/app/librechat.yaml

# Restart LibreChat (kill old process and relaunch)
pkill -f 'node api/server/index.js' || true
cd /opt/LibreChat/app/app && export $(grep -v "^#" /opt/.env | xargs) && nohup node api/server/index.js > /var/log/librechat.log 2>&1 &

# Restart proxies (if proxy code changed)
bash scripts/startup.sh
```

> **CRITICAL:** `/opt/LibreChat/app/app/librechat.yaml` is a standalone copy. `git pull` does NOT update it. You must copy explicitly.

---

## Config Files

| File | Purpose |
|------|---------|
| `config/librechat.yaml` | **Production** — 3 models (Simple group), `enforce: true` |
| `config/librechat-staging.yaml` | **Staging** — 41+ models, "Simple PROD" marks production models, `enforce: false` |
| `config/docker-compose.librechat.yml` | Docker Compose (legacy, not used on Vast.ai — kept for local dev) |
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
| LibreChat (Node.js) | 3000 | N/A (nohup) |
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
| Models not visible after deploy | Forgot to copy config to `/opt/LibreChat/app/app/librechat.yaml` | Copy the right config file and restart LibreChat |
| Wrong models on wrong server | Used production config on staging or vice versa | Check instance role with `vastai show instances`, use correct config file |
| LibreChat won't start | Missing `GOOGLE_CLIENT_ID`, `CREDS_KEY`, or `JWT_SECRET` | Check `/opt/.env` has all required vars |
| Login fails ("unknown error") | Missing `JWT_REFRESH_SECRET` or `CREDS_IV` in `.env` | Generate and add: `openssl rand -hex 32` / `openssl rand -hex 16` |
| Login redirects to wrong domain | `DOMAIN_CLIENT`/`DOMAIN_SERVER` set to wrong URL in `.env` | Ensure production uses `https://deep-search.uk`, staging uses `https://staging.deep-search.uk` |
| Proxy won't start in screen | `source /opt/.env` doesn't work in screen sessions | Use `export $(grep -v "^#" /opt/.env | xargs)` instead |
| MongoDB won't start | CPU lacks AVX (required by MongoDB 5.0+) | Use Percona MongoDB 4.4 instead (check AVX: `grep avx /proc/cpuinfo`) |
| Config drift between repo and server | `/opt/LibreChat/app/app/librechat.yaml` is a standalone copy | Always copy from repo after `git pull` |
| Staging tunnel uses wrong token | Each tunnel has its own token | Get the staging-specific token from the Cloudflare API (see Cloudflare Tunnels section) |
| `host.docker.internal` in config | Config written for Docker but running natively | Run `sed -i 's/host\.docker\.internal/localhost/g'` on the config |
