#!/bin/bash
# =============================================================================
# Deep Search Portal — Master Startup Script (Native Deployment)
# Starts all services in order with health-check waits.
# Supports native (Vast.ai) deployments without Docker.
# Deploy to /opt/startup.sh on the VM.
# =============================================================================

set -euo pipefail

# Load environment variables if .env exists
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

MISTRAL_API_KEY="${MISTRAL_API_KEY:?MISTRAL_API_KEY not set}"
CLOUDFLARE_TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:-}"

REPO_DIR="${REPO_DIR:-/opt/deep-search-portal}"
PROXY_DIR="$REPO_DIR/proxies"

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
    for session in swarm-proxy persistent-research deep-research thinking-proxy cftunnel librechat searxng; do
        screen -S "$session" -X quit 2>/dev/null || true
    done
    # Stop MongoDB if running
    pkill mongod 2>/dev/null || true
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

# --- MongoDB (required for LibreChat) ---
if ! pgrep -f "mongod" > /dev/null; then
    mkdir -p /data/db
    nohup mongod --dbpath /data/db --bind_ip 127.0.0.1 > /var/log/mongod.log 2>&1 &
    echo "MongoDB started"
    sleep 3
fi

# --- LibreChat (native Node.js) ---
LIBRECHAT_DIR="${LIBRECHAT_DIR:-/opt/LibreChat}"
if [ -d "$LIBRECHAT_DIR" ]; then
    if ! pgrep -f "node api/server/index.js" > /dev/null; then
        screen -dmS librechat bash -c "cd $LIBRECHAT_DIR && npm run backend 2>&1 | tee /var/log/librechat.log"
        echo "LibreChat starting..."
    fi
    wait_for_health "http://localhost:3000" "LibreChat" 60
else
    echo "WARNING: LIBRECHAT_DIR ($LIBRECHAT_DIR) not found, skipping LibreChat"
fi

# --- Cloudflare Tunnel ---
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ] && ! pgrep -f "cloudflared tunnel" > /dev/null; then
    screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
    echo "Cloudflare tunnel started"
fi

# --- Thinking Proxy (Mistral Direct API) ---
if ! pgrep -f "thinking_proxy:app" > /dev/null; then
    screen -dmS thinking-proxy bash -c "if [ -f /opt/.env ]; then set -a && source /opt/.env && set +a; fi && cd $PROXY_DIR && python3 -m uvicorn thinking_proxy:app --host 0.0.0.0 --port 9100 2>&1 | tee /var/log/thinking_proxy.log"
    echo "Thinking Proxy starting..."
fi
wait_for_health "http://localhost:9100/health" "Thinking Proxy" 15

# --- Deep Research Proxy (MiroFlow) ---
if ! pgrep -f "deep_research_proxy:app" > /dev/null; then
    screen -dmS deep-research bash -c "if [ -f /opt/.env ]; then set -a && source /opt/.env && set +a; fi && cd $PROXY_DIR && python3 -m uvicorn deep_research_proxy:app --host 0.0.0.0 --port 9200 2>&1 | tee /var/log/deep_research_proxy.log"
    echo "Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9200/health" "Deep Research Proxy" 15

# --- Persistent Deep Research Proxy (Tree Reactor + Subagent Map-Reduce) ---
if ! pgrep -f "persistent_deep_research_proxy:app" > /dev/null; then
    screen -dmS persistent-research bash -c "if [ -f /opt/.env ]; then set -a && source /opt/.env && set +a; fi && cd $PROXY_DIR && python3 -m uvicorn persistent_deep_research_proxy:app --host 0.0.0.0 --port 9300 2>&1 | tee /var/log/persistent_research_proxy.log"
    echo "Persistent Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9300/health" "Persistent Deep Research Proxy" 15

# --- Swarm Deep Search Proxy ---
if ! pgrep -f "swarm_proxy:app" > /dev/null; then
    screen -dmS swarm-proxy bash -c "if [ -f /opt/.env ]; then set -a && source /opt/.env && set +a; fi && cd $PROXY_DIR && python3 -m uvicorn swarm_proxy:app --host 0.0.0.0 --port 9500 2>&1 | tee /var/log/swarm_proxy.log"
    echo "Swarm Deep Search Proxy starting..."
fi
wait_for_health "http://localhost:9500/health" "Swarm Deep Search Proxy" 15

echo "All services started. Portal: ${DOMAIN_CLIENT:-https://deep-search.uk}"
