# Staging VM Deployment & Testing

## Overview
The Deep Search Portal staging environment runs on a Vast.ai instance accessible via SSH. All services run in GNU screen sessions.

## Architecture
- **Open WebUI**: port 3000
- **nginx reverse proxy**: port 3001 (Cloudflare tunnel routes here)
- **Persistent research proxy**: port 9300
- **SearXNG**: port 8888
- **Knowledge Engine**: port 9400
- **Thinking Proxy**: port 9100
- **Veritas Inquisitor**: port 9500

### Routing
- `https://staging.deep-search.uk/research/*` -> nginx:3001 -> proxy:9300
- `https://staging.deep-search.uk/*` (everything else) -> nginx:3001 -> OWUI:3000

## Devin Secrets Needed
- `VAST_AI_API_KEY` — for managing Vast.ai instances
- SSH key access is pre-configured for `root@ssh5.vast.ai -p 18770`

## Deploying Code Changes
1. SCP the updated Python files to `/opt/` on the staging VM
2. Kill and restart the relevant screen session with correct env vars
3. Verify the service is responding

### Restarting the Persistent Proxy
The proxy requires these env vars (found in `/opt/start_production.sh`):
```bash
export UPSTREAM_KEY='...'  # Mistral API key
export UPSTREAM_BASE='https://api.mistral.ai/v1'
export UPSTREAM_MODEL='mistral-large-latest'
export SUBAGENT_MODEL='mistral-small-latest'
export SEARXNG_URL='http://127.0.0.1:8888'
export KNOWLEDGE_ENGINE_URL='http://127.0.0.1:9400'
export B2_KEY_ID='...'
export B2_APP_KEY='...'
export B2_BUCKET_NAME='miroflow-publications'
export VERITAS_VERIFY_ENABLED=true
export PERSISTENT_RESEARCH_PORT=9300
export JSONL_LOG_DIR=/opt/persistent_research_logs/jsonl
```

Restart command:
```bash
screen -S persistent-proxy -X quit
screen -dmS persistent-proxy bash -c '
  export UPSTREAM_KEY="..." 
  # ... all env vars ...
  cd /opt
  python3 persistent_deep_research_proxy.py 2>&1 | tee /var/log/persistent-proxy.log
'
```

**Important**: The `.env` file at `/opt/.env` only contains Google OAuth vars, NOT the proxy's required env vars. The proxy env vars come from the screen session startup command. Check `/opt/start_production.sh` for the canonical list of env vars.

## Verifying Services
```bash
# Check proxy is responding
curl -s -o /dev/null -w '%{http_code}' http://localhost:9300/v1/models
# Should return 200

# Check dashboard endpoint
curl -s -o /dev/null -w '%{http_code}' http://localhost:9300/research/dashboard
# Should return 200

# Check via public URL
curl -s -o /dev/null -w '%{http_code}' https://staging.deep-search.uk/research/dashboard
# Should return 200
```

## Common Issues
- **Proxy won't start**: Usually missing env vars. Check the error in `/tmp/proxy.log` or `/var/log/persistent-proxy.log`.
- **`UPSTREAM_KEY is not set`**: The screen session didn't export the env vars. Use the full startup command from `/opt/start_production.sh`.
- **Cloudflare tunnel down**: Restart with `screen -dmS cftunnel cloudflared tunnel run --token <TOKEN>`
- **Google OAuth broken**: Check that `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` are set in the OWUI screen session. These must match what's configured in Google Cloud Console.
- **Models not visible**: The OWUI database at `/opt/staging-openwebui-data/webui.db` may have stale API endpoint URLs. Check Admin > Settings > Connections in OWUI.

## Testing Dashboard Endpoints
- `/research/dashboard` — HTML dashboard with KPI cards, tool usage, sessions table
- `/research/dashboard/data` — JSON endpoint for programmatic consumption
- `/research/dashboard?days=N` — changes lookback window (default 7, max 90)
- Local metrics files are at `/opt/persistent_research_logs/metrics/*.json`
- Langfuse sections only appear when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` env vars are set
