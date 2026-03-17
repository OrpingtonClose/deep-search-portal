#!/bin/bash
# =============================================================================
# Deep Search Portal — Master Startup Script
# Starts all services in order with health-check waits.
# Deploy to /opt/startup.sh on the VM.
# =============================================================================

set -euo pipefail

# Load environment variables if .env exists
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

MISTRAL_API_KEY="${MISTRAL_API_KEY:?MISTRAL_API_KEY not set}"
CLOUDFLARE_TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:?CLOUDFLARE_TUNNEL_TOKEN not set}"

# --- SearXNG ---
if ! pgrep -f "searx.webapp" > /dev/null; then
    cd /tmp/searxng
    screen -dmS searxng /opt/searxng-env/bin/python -m searx.webapp
    echo "SearXNG started"
fi

for i in $(seq 1 30); do
    curl -s http://localhost:8888 > /dev/null 2>&1 && break
    sleep 1
done

# --- Open WebUI ---
if ! pgrep -f "open-webui serve" > /dev/null; then
    screen -dmS owui bash /opt/start_openwebui.sh
    echo "Open WebUI started"
fi

for i in $(seq 1 60); do
    curl -s http://localhost:3000 > /dev/null 2>&1 && break
    sleep 1
done

# --- Cloudflare Tunnel ---
if ! pgrep -f "cloudflared tunnel" > /dev/null; then
    screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
    echo "Cloudflare tunnel started"
fi

echo "All services started. Portal: ${WEBUI_URL:-https://deep-search.uk}"

# --- Thinking Proxy (Mistral Direct API) ---
pip3 install fastapi uvicorn httpx -q
screen -dmS thinking-proxy bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export UPSTREAM_MODEL='mistral-large-latest' && export THINKING_PROXY_PORT='9100' && python3 /opt/thinking_proxy.py 2>&1 | tee /var/log/thinking_proxy.log"
sleep 2
echo "Thinking Proxy started on port 9100"

# --- Deep Research Proxy (MiroFlow) ---
screen -dmS deep-research bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export UPSTREAM_MODEL='mistral-large-latest' && export SEARXNG_URL='http://localhost:8888' && export DEEP_RESEARCH_PORT='9200' && python3 /opt/deep_research_proxy.py 2>&1 | tee /var/log/deep_research_proxy.log"
sleep 2
echo "Deep Research Proxy (MiroFlow) started on port 9200"
