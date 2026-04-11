# Deep Search Portal — Deployment Guide

## Architecture

```
Internet → Cloudflare Tunnel → nginx (:3000) → LibreChat Node.js (:3001)
                              → Proxy services (:9800–9951)
```

- **LibreChat**: Standalone Node.js (`node api/server/index.js`) — NOT Docker Compose (Vast.ai containers lack Docker-in-Docker permissions)
- **nginx**: Reverse proxy on port 3000 forwarding to LibreChat on port 3001
- **MongoDB**: Standalone `mongod` process (started manually, no systemd)
- **Proxies**: Python FastAPI processes in GNU `screen` sessions (managed by `scripts/startup.sh`)
- **Tunnel**: `cloudflared` routes `deep-search.uk` to the VM
- **Host**: Vast.ai instances (Ubuntu). Instance IDs change on recreation — always check `vastai show instances`.

### Current Production Instance

| Field | Value |
|-------|-------|
| Instance ID | 34657856 |
| SSH | `ssh -p 17856 root@ssh2.vast.ai` |
| GPU | RTX 2080 Ti |
| RAM | 20 GB |
| Disk | 40 GB |
| Location | Ohio, US |
| Cost | ~$0.07/hr |
| URL | https://deep-search.uk |

### Current Staging Instance

| Field | Value |
|-------|-------|
| Instance ID | 34657920 |
| SSH | `ssh -p 17920 root@ssh2.vast.ai` |
| GPU | GTX 1060 |
| RAM | 31 GB |
| Disk | 40 GB |
| Location | Norway, NO |
| Cost | ~$0.09/hr |
| URL | N/A (no Cloudflare tunnel — access via SSH port forwarding) |

---

## Fresh VM Setup

```bash
# 1. Create a Vast.ai instance (GPU optional, >=16GB RAM, >=40GB disk)
vastai create instance <offer_id> --image ubuntu:22.04 --disk 40 --label production-deep-search

# 2. Wait for instance to boot (status: running)
vastai show instances --raw | python3 -c "import json,sys; [print(i['id'],i['actual_status']) for i in json.load(sys.stdin)]"

# 3. SSH in and install system deps
ssh -p <port> root@<ssh_host> "apt-get update && apt-get install -y git nginx screen curl gnupg ca-certificates python3-pip"

# 4. Install Node.js 20
ssh ... "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs"

# 5. Install MongoDB
ssh ... "curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg && echo 'deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' > /etc/apt/sources.list.d/mongodb-org-7.0.list && apt-get update && apt-get install -y mongodb-org"

# 6. Install cloudflared
ssh ... "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"

# 7. Clone repos
ssh ... "cd /opt && git clone https://github.com/OrpingtonClose/deep-search-portal.git"
ssh ... "git clone https://github.com/danny-avila/LibreChat.git /opt/LibreChat"

# 8. Build LibreChat
ssh ... "cd /opt/LibreChat && npm ci && npm run build:packages && npm run frontend"

# 9. Install Python deps for proxies
ssh ... "pip3 install fastapi uvicorn sse-starlette httpx openai aiohttp youtube-transcript-api yt-dlp google-api-python-client beautifulsoup4 langchain-openai langchain-core"

# 10. Create /opt/.env (see .env.example for all vars)
# 11. Copy production config
ssh ... "cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/librechat.yaml"
ssh ... "cp /opt/.env /opt/LibreChat/.env"

# 12. Add host.docker.internal to /etc/hosts (required for proxy routing)
ssh ... "echo '127.0.0.1 host.docker.internal' >> /etc/hosts"

# 13. Configure nginx (port 3000 -> 3001) — see nginx config below

# 14. Start MongoDB, LibreChat, proxies, and tunnel (see Services section)
```

> **NOTE:** Docker Compose does NOT work inside Vast.ai containers (overlayfs/iptables permission restrictions). Use standalone Node.js.

Optional env vars: `FIRECRAWL_API_KEY`, `EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `MISTRAL_API_KEY`, `BRIGHT_DATA_API_KEY`.

### nginx Config

```nginx
server {
    listen 3000;
    server_name _;
    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

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

# Restart LibreChat (kill old screen, start new)
screen -X -S librechat quit
screen -dmS librechat bash -c 'set -a; source /opt/.env; set +a; cd /opt/LibreChat && NODE_ENV=production PORT=3001 HOST=0.0.0.0 node api/server/index.js 2>&1 | tee /var/log/librechat.log'

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
| `scripts/startup.sh` | Master startup: all proxies + LibreChat + Cloudflare tunnel |
| `.env.example` | Template for all env vars |

---

## Production Models (Simple Group)

1. **Grok 4.20 Multi-Agent** — xAI flagship, parallel reasoning, web search
2. **Heretic Uncensored** — GLM-4.7 Flash with Firecrawl/Exa/Brave tools
3. **Tier Race (Full Throttle)** — Races 9 flagship models, enriches answers with YouTube videos

All other models exist only in staging.

---

## Services & Ports

| Service | Port | Screen Session |
|---------|------|----------------|
| nginx | 3000 | N/A (foreground) |
| LibreChat (Node.js) | 3001 | `librechat` |
| MongoDB | 27017 | N/A (forked) |
| xAI Native Proxy | 9800 | `xai-native-proxy` |
| Tier Chooser Proxy | 9900 | `tier-chooser` |
| Heretic Proxy | 9950 | `heretic-proxy` |
| Miro Proxy | 9951 | `miro-proxy` |
| Cloudflare Tunnel | — | `cftunnel` |

---

## Environment Variables

All stored in `/opt/.env` on each instance. See `.env.example` for the full list.

**Required for deployment:**

| Variable | Used by |
|----------|---------|
| `VENICE_API_KEY` | Heretic proxy, Miro proxy |
| `XAI_API_KEY` | xAI native proxy |
| `GOOGLE_CLIENT_ID` | LibreChat Google OAuth login (production only) |
| `GOOGLE_CLIENT_SECRET` | LibreChat Google OAuth login (production only) |
| `CLOUDFLARE_TUNNEL_TOKEN` | Cloudflare tunnel (production only) |
| `UPSTREAM_KEY` | Heretic tool-calling (langchain agent) |
| `CREDS_KEY` | LibreChat encryption (auto-generated if missing) |
| `JWT_SECRET` | LibreChat JWT signing (auto-generated if missing) |
| `JWT_REFRESH_SECRET` | LibreChat JWT refresh (auto-generated if missing) |

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

## Starting Services on a Running VM

```bash
# Start MongoDB (if not already running)
mkdir -p /var/lib/mongodb /var/log/mongodb
chown mongodb:mongodb /var/lib/mongodb /var/log/mongodb
mongod --dbpath /var/lib/mongodb --logpath /var/log/mongodb/mongod.log --fork --bind_ip 127.0.0.1

# Start nginx
nginx

# Start LibreChat
screen -dmS librechat bash -c 'set -a; source /opt/.env; set +a; cd /opt/LibreChat && NODE_ENV=production PORT=3001 HOST=0.0.0.0 node api/server/index.js 2>&1 | tee /var/log/librechat.log'

# Start proxies
screen -dmS heretic-proxy bash -c 'set -a; source /opt/.env; set +a; cd /opt/deep-search-portal/proxies && HERETIC_PROXY_PORT=9950 python3 heretic_proxy.py 2>&1 | tee /var/log/heretic_proxy.log'
screen -dmS xai-native-proxy bash -c 'set -a; source /opt/.env; set +a; cd /opt/deep-search-portal/proxies && XAI_PROXY_PORT=9800 python3 xai_native_proxy.py 2>&1 | tee /var/log/xai_native_proxy.log'
screen -dmS tier-chooser bash -c 'set -a; source /opt/.env; set +a; cd /opt/deep-search-portal/proxies && TIER_CHOOSER_PORT=9900 python3 tier_chooser_proxy.py 2>&1 | tee /var/log/tier_chooser_proxy.log'

# Start Cloudflare tunnel (production only)
screen -dmS cftunnel bash -c 'cloudflared tunnel run --token $CLOUDFLARE_TUNNEL_TOKEN 2>&1 | tee /var/log/cftunnel.log'

# Health checks
curl -s http://localhost:3001/ > /dev/null && echo "LibreChat: OK"
curl -s http://localhost:9950/health > /dev/null && echo "Heretic: OK"
curl -s http://localhost:9800/health > /dev/null && echo "xAI: OK"
curl -s http://localhost:9900/health > /dev/null && echo "Tier Chooser: OK"
```

---

## Adding a New Model to Production

1. Add proxy code under `proxies/` (if new endpoint)
2. Add modelSpec to `config/librechat.yaml` in Simple group
3. Add endpoint wiring to `config/librechat.yaml` under `endpoints.custom`
4. Add the same model to `config/librechat-staging.yaml` in the "Simple PROD" group
5. Add startup entry to `scripts/startup.sh`
6. Create PR, wait for CI
7. Deploy to production (see "Deploying Code Changes" above)

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Models not visible after deploy | Forgot to copy config to `/opt/LibreChat/librechat.yaml` | Copy the right config file and restart LibreChat |
| Wrong models on wrong server | Used production config on staging or vice versa | Check instance role with `vastai show instances`, use correct config file |
| LibreChat won't start | Missing `GOOGLE_CLIENT_ID`, `CREDS_KEY`, or `JWT_SECRET` | Check `/opt/.env` has all required vars. `CREDS_KEY`/`JWT_SECRET` auto-generate if missing. |
| Proxy won't start | Missing API key in `/opt/.env` | `source /opt/.env && env \| grep KEY` to verify |
| Config drift between repo and server | `/opt/LibreChat/librechat.yaml` is a standalone copy | Always copy from repo after `git pull` |
