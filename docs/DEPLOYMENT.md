# Deep Search Portal — Deployment Guide

## Architecture Overview

The Deep Search Portal runs on **two Vast.ai GPU instances**, both accessible via SSH using the Vast.ai CLI or API.

| Instance | Role | SSH | Description |
|----------|------|-----|-------------|
| 34445148 | **Production** | `ssh5.vast.ai:15148` | End-user facing at `https://deep-search.uk`. **3 models only** (Simple group). RTX 3070, 64GB RAM, 14 CPUs, Norway. |
| *(TBD)* | **Staging** | *(TBD)* | Testing/dev. Full model list + "Simple PROD" group marking the curated production models. Accessible via `https://staging.deep-search.uk`. |

> **Note:** Instance IDs, SSH hosts, and ports may change if instances are recreated. Always verify via `vastai show instances` or the Vast.ai API.

---

## Config Files

### Production (`config/librechat.yaml`)

The **production config** contains only the **Simple** group with exactly **3 models**:

1. **Grok 4.20 Multi-Agent** — xAI flagship, parallel reasoning, web search
2. **Heretic Uncensored** — GLM-4.7 Flash with Firecrawl/Exa/Brave tools
3. **Tier Race (Full Throttle)** — Races 9 flagship models

All other models (Miro, Experimental, Raw-*) exist **only in staging**. `enforce: true` ensures users see nothing beyond these 3.

### Staging (`config/librechat-staging.yaml`)

The **staging config** combines:
1. All modelSpecs (41 total) — the **Simple** group is renamed to **"Simple PROD"** to mark the 3 production models. All other groups (Miro, Experimental, Raw-*) keep their normal names
2. All backend endpoints with their **full model lists** (e.g. G0DM0D3 with 62 models, Singular endpoint)
3. `enforce: false` so all models from all endpoints are visible (not just modelSpec entries)

This lets testers see the curated production view ("Simple PROD") alongside all other models.

---

## LibreChat Config Location on Servers

**CRITICAL:** The LibreChat config at `/opt/LibreChat/librechat.yaml` on both instances is a **standalone copy**, NOT a symlink to the repo. Running `git pull` on the server does NOT update the LibreChat config. You must explicitly copy the config file after pulling code:

```bash
# Production
cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/librechat.yaml

# Staging
cp /opt/deep-search-portal/config/librechat-staging.yaml /opt/LibreChat/librechat.yaml
```

The deploy backup at `/opt/deep-search-portal-deploy/config/librechat.yaml` is a snapshot of the production config. It may be outdated.

---

## Deploying Changes

### 1. Pull latest code on the server

```bash
cd /opt/deep-search-portal && git pull origin main
```

### 2. Copy the correct config

```bash
# PRODUCTION instance:
cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/librechat.yaml

# STAGING instance:
cp /opt/deep-search-portal/config/librechat-staging.yaml /opt/LibreChat/librechat.yaml
```

### 3. Restart LibreChat

LibreChat runs as a standalone Node process in a GNU `screen` session (NOT Docker):

```bash
screen -S librechat -X quit
sleep 1
cd /opt/LibreChat && screen -dmS librechat bash -c 'set -a; source /opt/.env 2>/dev/null; source .env 2>/dev/null; set +a; node api/server/index.js 2>&1 | tee /var/log/librechat.log'
```

nginx reverse-proxies port 3000 → LibreChat on port 3001 (Cloudflare tunnel connects to port 3000).

### 4. Restart proxies (if proxy code changed)

All proxies run via `scripts/startup.sh`:

```bash
cd /opt/deep-search-portal && bash scripts/startup.sh
```

Or restart individual proxies:

```bash
# Example: Heretic proxy
screen -S heretic-proxy -X quit
screen -dmS heretic-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && HERETIC_PROXY_PORT=9950 python3 heretic_proxy.py 2>&1 | tee /var/log/heretic_proxy.log"
```

### 5. Verify

```bash
# Check LibreChat is up
curl -s -o /dev/null -w '%{http_code}' http://localhost:3000
# Should return 200

# Check model groups via API
curl -s http://localhost:3000/api/config | python3 -c "
import sys, json; data = json.loads(sys.stdin.read())
for s in data.get('modelSpecs',{}).get('list',[]):
    print(f\"{s.get('group','?'):30s} {s.get('label','')}\")"
```

---

## Services & Ports

| Service | Port | Screen Session |
|---------|------|----------------|
| nginx (reverse proxy) | 3000 | — |
| LibreChat | 3001 | `librechat` |
| Mistral Thinking Proxy | 9100 | `thinking-proxy` |
| Deep Research Proxy | 9200 | `deep-research` |
| Persistent MiroFlow | 9300 | `persistent-research` |
| MiroFlow Sprint | 9400 | `miroflow-sprint` |
| Swarm Deep Search | 9500 | `swarm-proxy` |
| G0DM0D3 Proxy | 9600 | `godmode-proxy` |
| xAI Native Proxy | 9700 | `xai-native-proxy` |
| Persistent MiroFlow Wiki | 9800 | (shared) |
| MiroFlow Sprint Wiki | 9850 | (shared) |
| Tier Chooser Proxy | 9900 | `tier-chooser` |
| Heretic Proxy | 9950 | `heretic-proxy` |
| Miro Proxy | 9951 | `miro-proxy` |
| SearXNG | 8888 | `searxng` |

---

## Environment Variables

All env vars are stored in `/opt/.env` on each instance. Key variables:

- `VENICE_API_KEY` — Required for Heretic proxy (port 9950) and Miro proxy (port 9951)
- `XAI_API_KEY` — Required for xAI Native proxy (port 9700)
- `UPSTREAM_KEY` — Derived from VENICE_API_KEY, used by proxy shared module
- `UPSTREAM_BASE` — Venice API base URL (`https://api.venice.ai/api/v1`)
- `UPSTREAM_MODEL` — Default model for Heretic (`olafangensan-glm-4.7-flash-heretic`)
- `CLOUDFLARE_TUNNEL_TOKEN` — Cloudflare tunnel token for deep-search.uk
- `GOOGLE_CLIENT_ID` — Google OAuth client ID for social login
- `GOOGLE_CLIENT_SECRET` — Google OAuth client secret
- `GOOGLE_CALLBACK_URL` — Set to `/oauth/google/callback`
- `FIRECRAWL_API_KEY` — Used by Heretic proxy for web scraping tools
- `EXA_API_KEY` — Used by Heretic proxy for semantic search tools
- `BRAVE_SEARCH_API_KEY` — Used by Heretic proxy and search providers

---

## Vast.ai Access

Use the `vastai` CLI (requires `VAST_AI_API_KEY`):

```bash
# List instances
vastai show instances

# SSH into an instance
vastai ssh-url <instance_id>
# Or directly:
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ssh_host>

# Attach SSH key (if needed)
vastai attach ssh-key <instance_id> "$(cat ~/.ssh/id_ed25519.pub)"
```

---

## Adding a New Model to Production

1. Add the proxy code (if new endpoint) under `proxies/`
2. Add modelSpec entry to `config/librechat.yaml` in the Simple group (production only has Simple)
3. Add endpoint wiring to `config/librechat.yaml` under `endpoints.custom`
4. Regenerate staging config: add the new model to the staging spec list, ensure Simple models get "Simple PROD" group
5. Add startup entry to `scripts/startup.sh`
6. Create PR, wait for CI
7. Deploy to both instances using the steps above

---

## Common Pitfalls

- **Config drift**: The server's `/opt/LibreChat/librechat.yaml` is a standalone copy. Always copy the repo config after `git pull`.
- **Wrong config on wrong instance**: Production uses `config/librechat.yaml`, staging uses `config/librechat-staging.yaml`. Deploying the wrong one breaks the model list.
- **LibreChat is NOT Docker**: It runs as `node api/server/index.js` in a screen session. Do not try `docker compose restart`.
- **nginx on port 3000**: Cloudflare tunnel connects to port 3000 (nginx), which reverse-proxies to LibreChat on port 3001. Do not run LibreChat on port 3000 directly.
- **Miro trace side-channel**: nginx proxies `/miro-trace/` → Miro proxy (port 9951) for execution trace REST endpoints.
- **Docker does NOT work**: Vast.ai instances lack overlayfs/iptables permissions for Docker-in-Docker. LibreChat runs as standalone Node.js with a separate MongoDB process — not via Docker Compose.
- **Proxy won't start**: Check that the required API keys are set in `/opt/.env`. Use `source /opt/.env && env | grep KEY` to verify.
- **Models not visible after deploy**: You forgot to copy the config to `/opt/LibreChat/librechat.yaml` and restart LibreChat.
