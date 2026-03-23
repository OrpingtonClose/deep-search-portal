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

# --- Helper: wait for an HTTP endpoint to become healthy ---
wait_for_health() {
    local url="$1"
    local label="$2"
    local timeout="${3:-30}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "$label is healthy ($url)"
            return 0
        fi
        sleep 1
    done
    echo "WARNING: $label did not become healthy within ${timeout}s ($url)"
    return 1
}

# --- Signal trapping for clean shutdown ---
cleanup() {
    echo "Shutting down services..."
    for session in swarm-proxy persistent-research deep-research thinking-proxy cftunnel owui searxng; do
        screen -S "$session" -X quit 2>/dev/null || true
    done
    echo "All services stopped."
}
trap cleanup SIGTERM SIGINT

# --- SearXNG ---
if ! pgrep -f "searx.webapp" > /dev/null; then
    cd /tmp/searxng
    screen -dmS searxng /opt/searxng-env/bin/python -m searx.webapp
    echo "SearXNG started"
fi
wait_for_health "http://localhost:8888" "SearXNG" 30

# --- Open WebUI ---
if ! pgrep -f "open-webui serve" > /dev/null; then
    screen -dmS owui bash /opt/start_openwebui.sh
    echo "Open WebUI started"
fi
wait_for_health "http://localhost:3000" "Open WebUI" 60

# --- Sync Models (YAML → DB) ---
if [ -f /opt/sync_models.py ] && [ -f /opt/models.yaml ]; then
    echo "Syncing models from models.yaml..."
    python3 /opt/sync_models.py /opt/models.yaml --db-path /opt/openwebui-data/webui.db || echo "WARNING: sync_models.py failed (exit $?), continuing startup"
fi

# --- Cloudflare Tunnel ---
if ! pgrep -f "cloudflared tunnel" > /dev/null; then
    screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
    echo "Cloudflare tunnel started"
fi

# --- Thinking Proxy (Mistral Direct API) ---
pip3 install fastapi uvicorn httpx -q
if ! pgrep -f "thinking_proxy.py" > /dev/null; then
    screen -dmS thinking-proxy bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export UPSTREAM_MODEL='mistral-large-latest' && export THINKING_PROXY_PORT='9100' && python3 /opt/thinking_proxy.py 2>&1 | tee /var/log/thinking_proxy.log"
    echo "Thinking Proxy starting..."
fi
wait_for_health "http://localhost:9100/health" "Thinking Proxy" 15

# --- Deep Research Proxy (MiroFlow) ---
if ! pgrep -f "deep_research_proxy.py" > /dev/null; then
    screen -dmS deep-research bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export UPSTREAM_MODEL='mistral-large-latest' && export SEARXNG_URL='http://localhost:8888' && export DEEP_RESEARCH_PORT='9200' && python3 /opt/deep_research_proxy.py 2>&1 | tee /var/log/deep_research_proxy.log"
    echo "Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9200/health" "Deep Research Proxy" 15

# --- Persistent Deep Research Proxy (Subagent Map-Reduce + AoT) ---
if ! pgrep -f "persistent_deep_research_proxy.py" > /dev/null; then
    screen -dmS persistent-research bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export UPSTREAM_MODEL='mistral-large-latest' && export SUBAGENT_MODEL='mistral-small-latest' && export SEARXNG_URL='http://localhost:8888' && export PERSISTENT_RESEARCH_PORT='9300' && python3 /opt/persistent_deep_research_proxy.py 2>&1 | tee /var/log/persistent_research_proxy.log"
    echo "Persistent Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9300/health" "Persistent Deep Research Proxy" 15

# --- Swarm Deep Search Proxy ---
if ! pgrep -f "swarm_proxy.py" > /dev/null; then
    screen -dmS swarm-proxy bash -c "export UPSTREAM_BASE='https://api.mistral.ai/v1' && export UPSTREAM_KEY='${MISTRAL_API_KEY}' && export SWARM_SYNTHESIS_MODEL='mistral-large-latest' && export SWARM_WORKER_MODEL='mistral-small-latest' && export SWARM_PROXY_PORT='9500' && python3 /opt/swarm_proxy.py 2>&1 | tee /var/log/swarm_proxy.log"
    echo "Swarm Deep Search Proxy starting..."
fi
wait_for_health "http://localhost:9500/health" "Swarm Deep Search Proxy" 15

echo "All services started. Portal: ${WEBUI_URL:-https://deep-search.uk}"
