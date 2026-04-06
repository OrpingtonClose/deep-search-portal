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

CLOUDFLARE_TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:?CLOUDFLARE_TUNNEL_TOKEN not set}"

# Model API keys — read from .env, with sensible defaults
# Priority: VENICE_API_KEY (uncensored) > XAI_API_KEY (Grok) > MISTRAL_API_KEY (fallback)
# See docs/model-evaluation-april-2026.md for the full evaluation.
# The base URL and model defaults are set to match whichever key was resolved.
if [ -n "${UPSTREAM_KEY:-}" ]; then
    # Explicit UPSTREAM_KEY — user controls UPSTREAM_BASE themselves
    UPSTREAM_BASE="${UPSTREAM_BASE:-https://api.venice.ai/api/v1}"
    UPSTREAM_MODEL="${UPSTREAM_MODEL:-olafangensan-glm-4.7-flash-heretic}"
    SUBAGENT_MODEL="${SUBAGENT_MODEL:-qwen3.5-9b}"
elif [ -n "${VENICE_API_KEY:-}" ]; then
    UPSTREAM_KEY="$VENICE_API_KEY"
    UPSTREAM_BASE="${UPSTREAM_BASE:-https://api.venice.ai/api/v1}"
    # miro-long:  synthesis, final answers — uncensored + tool calling + strong reasoning
    UPSTREAM_MODEL="${UPSTREAM_MODEL:-olafangensan-glm-4.7-flash-heretic}"
    # miro-short: sub-tasks, planning, verification — fast + tool calling + cheap
    SUBAGENT_MODEL="${SUBAGENT_MODEL:-qwen3.5-9b}"
elif [ -n "${XAI_API_KEY:-}" ]; then
    UPSTREAM_KEY="$XAI_API_KEY"
    UPSTREAM_BASE="${UPSTREAM_BASE:-https://api.x.ai/v1}"
    UPSTREAM_MODEL="${UPSTREAM_MODEL:-grok-3-fast}"
    SUBAGENT_MODEL="${SUBAGENT_MODEL:-grok-3-fast}"
elif [ -n "${MISTRAL_API_KEY:-}" ]; then
    UPSTREAM_KEY="$MISTRAL_API_KEY"
    UPSTREAM_BASE="${UPSTREAM_BASE:-https://api.mistral.ai/v1}"
    UPSTREAM_MODEL="${UPSTREAM_MODEL:-mistral-large-latest}"
    SUBAGENT_MODEL="${SUBAGENT_MODEL:-mistral-small-latest}"
else
    echo "FATAL: No LLM API key set (need UPSTREAM_KEY, VENICE_API_KEY, XAI_API_KEY, or MISTRAL_API_KEY)" >&2
    exit 1
fi
# Export derived variables so screen-launched proxies (which re-source .env
# independently) inherit them even when .env only has e.g. VENICE_API_KEY.
export UPSTREAM_KEY UPSTREAM_BASE UPSTREAM_MODEL SUBAGENT_MODEL
VENICE_API_KEY="${VENICE_API_KEY:-}"
XAI_API_KEY="${XAI_API_KEY:-}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
SEARCH_BACKEND="${SEARCH_BACKEND:-legacy}"

# Warn if API keys are missing (services will fail to authenticate)
if [ -z "$VENICE_API_KEY" ]; then
    echo "WARNING: VENICE_API_KEY not set — persistent-research, miroflow-sprint, and swarm proxy will fail"
fi
if [ -z "$XAI_API_KEY" ]; then
    echo "WARNING: XAI_API_KEY not set — deep-research and xAI native proxy will fall back to MISTRAL_API_KEY"
fi

# Data-source credential warnings
BRIGHT_DATA_API_KEY="${BRIGHT_DATA_API_KEY:-}"
APIFY_API_TOKEN="${APIFY_API_TOKEN:-}"
if [ -z "$BRIGHT_DATA_API_KEY" ]; then
    echo "WARNING: BRIGHT_DATA_API_KEY not set — Reddit, Twitter, and commercial SERP search will be unavailable"
fi
if [ -z "$APIFY_API_TOKEN" ]; then
    echo "WARNING: APIFY_API_TOKEN not set — Reddit/Instagram/TikTok/LinkedIn Apify fallback will be unavailable"
fi
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
    for session in miro-proxy heretic-proxy mcp-exa mcp-firecrawl xai-native-proxy godmode-proxy swarm-proxy miroflow-sprint persistent-research deep-research thinking-proxy knowledge-engine search-dispatcher mcp-searxng litellm cftunnel searxng; do
        screen -S "$session" -X quit 2>/dev/null || true
    done
    # Stop LibreChat Docker stack
    if [ -f "${LIBRECHAT_COMPOSE:-}" ]; then
        docker compose -f "$LIBRECHAT_COMPOSE" down 2>/dev/null || true
    fi
    echo "All services stopped."
}
trap cleanup SIGTERM SIGINT

# --- Neo4j (knowledge graph database — MUST start before Knowledge Engine and proxies) ---
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
if ! pgrep -f "org.neo4j.server" > /dev/null; then
    if command -v neo4j > /dev/null 2>&1; then
        neo4j start 2>&1 || echo "WARNING: neo4j start failed"
        echo "Neo4j starting..."
    else
        echo "Neo4j is NOT installed — attempting auto-install..."
        if command -v apt-get > /dev/null 2>&1; then
            # Install Java runtime (required by Neo4j)
            apt-get install -y openjdk-21-jre-headless 2>/dev/null || apt-get install -y default-jre-headless 2>/dev/null || true
            # Add Neo4j repo and install
            if [ ! -f /usr/share/keyrings/neo4j-archive-keyring.gpg ]; then
                curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor -o /usr/share/keyrings/neo4j-archive-keyring.gpg 2>/dev/null || true
                echo 'deb [signed-by=/usr/share/keyrings/neo4j-archive-keyring.gpg] https://debian.neo4j.com stable latest' > /etc/apt/sources.list.d/neo4j.list 2>/dev/null || true
                apt-get update -qq 2>/dev/null || true
            fi
            apt-get install -y neo4j 2>/dev/null
            if command -v neo4j > /dev/null 2>&1; then
                # Set default password for unattended operation
                neo4j-admin dbms set-initial-password "${NEO4J_PASSWORD:-neo4j_deep_search}" 2>/dev/null || true
                neo4j start 2>&1 || echo "WARNING: neo4j start failed after install"
                echo "Neo4j installed and starting..."
            else
                echo "ERROR: Neo4j auto-install failed."
                echo "  → Prior knowledge retrieval will return empty"
                echo "  → Condition persistence will fail"
                echo "  → Cross-session knowledge accumulation is disabled"
            fi
        else
            echo "ERROR: Neo4j is NOT installed and apt-get is unavailable."
            echo "  Install manually: apt-get install -y openjdk-21-jre-headless neo4j"
        fi
    fi
fi
wait_for_health "http://localhost:${NEO4J_HTTP_PORT}" "Neo4j" 30 || true

# --- Knowledge Engine (Neo4j API layer — MUST start before proxies) ---
KE_PORT="${KE_PORT:-9850}"  # Note: 9400 is taken by MiroFlow Sprint
if ! pgrep -f "knowledge_engine.main" > /dev/null; then
    REPO_DIR="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)}"
    if [ -d "$REPO_DIR/services/knowledge-engine" ]; then
        screen -dmS knowledge-engine bash -c "set -a; source /opt/.env 2>/dev/null; set +a; export KE_PORT=${KE_PORT}; cd $REPO_DIR/services/knowledge-engine && python3 -c \"
import uvicorn
from knowledge_engine.main import app
uvicorn.run(app, host='0.0.0.0', port=${KE_PORT})
\" 2>&1 | tee /var/log/knowledge-engine.log"
        echo "Knowledge Engine starting on port ${KE_PORT}..."
    else
        echo "WARNING: Knowledge Engine not found at $REPO_DIR/services/knowledge-engine"
    fi
fi
wait_for_health "http://localhost:${KE_PORT}/health" "Knowledge Engine" 30 || true

# --- Tor Daemon (required for Sicry dark web search + onion_fetch) ---
if ! pgrep -f "tor " > /dev/null 2>&1 && ! pgrep -x "tor" > /dev/null 2>&1; then
    if command -v tor > /dev/null 2>&1; then
        tor --RunAsDaemon 1 --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 1 2>&1 || echo "WARNING: Tor daemon failed to start"
        echo "Tor daemon starting on SOCKS5 :9050, ControlPort :9051..."
        sleep 3
    else
        echo "WARNING: Tor is NOT installed. Dark web search (Sicry) will be unavailable."
        echo "  → Install with: apt-get install -y tor"
    fi
fi

# --- Sicry (dark web search engine — clone if not present) ---
SICRY_PATH="${SICRY_PATH:-/opt/sicry/sicry.py}"
if [ ! -f "$SICRY_PATH" ]; then
    echo "Cloning Sicry dark web search engine..."
    git clone --depth 1 https://github.com/JacobJandon/Sicry.git "$(dirname "$SICRY_PATH")" 2>/dev/null || echo "WARNING: Failed to clone Sicry"
    if [ -f "$(dirname "$SICRY_PATH")/requirements.txt" ]; then
        pip3 install -q -r "$(dirname "$SICRY_PATH")/requirements.txt" 2>/dev/null || true
    fi
fi

# --- SearXNG ---
if ! pgrep -f "searx.webapp" > /dev/null; then
    cd /tmp/searxng
    screen -dmS searxng /opt/searxng-env/bin/python -m searx.webapp
    echo "SearXNG started"
fi
wait_for_health "http://localhost:8888" "SearXNG" 30

# --- LibreChat (Docker Compose: API + MongoDB + Meilisearch) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
LIBRECHAT_COMPOSE="$REPO_ROOT/config/docker-compose.librechat.yml"
if [ -f "$LIBRECHAT_COMPOSE" ]; then
    if ! docker ps --format '{{.Names}}' | grep -q librechat-api; then
        echo "Starting LibreChat stack..."
        bash "$REPO_ROOT/scripts/start_librechat.sh" up
    fi
    echo "Waiting for LibreChat..."
    wait_for_health "http://localhost:3000" "LibreChat" 90
else
    echo "WARNING: $LIBRECHAT_COMPOSE not found, skipping LibreChat"
fi

# --- Cloudflare Tunnel ---
if ! pgrep -f "cloudflared tunnel" > /dev/null; then
    screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
    echo "Cloudflare tunnel started"
fi

# --- LiteLLM Proxy (LLM routing + cost tracking) ---
if [ "${SEARCH_BACKEND:-legacy}" = "mcp" ]; then
    if ! pgrep -f "litellm" > /dev/null; then
        screen -dmS litellm bash -c "litellm --config /opt/deep-search-portal/config/litellm_config.yaml --port ${LITELLM_PORT:-4000} 2>&1 | tee /var/log/litellm.log"
        echo "LiteLLM Proxy starting..."
    fi
    wait_for_health "http://localhost:${LITELLM_PORT:-4000}/health" "LiteLLM Proxy" 30
fi

# --- MCP Search Servers + Dispatcher (only when SEARCH_BACKEND=mcp) ---
if [ "${SEARCH_BACKEND:-legacy}" = "mcp" ]; then
    REPO_DIR="${REPO_ROOT:-/opt/deep-search-portal}"

    # Start MCP SearXNG server
    if ! pgrep -f "mcp_searxng" > /dev/null; then
        screen -dmS mcp-searxng bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd ${REPO_DIR}/mcp_servers/searxng && python3 server.py 2>&1 | tee /var/log/mcp_searxng.log"
        echo "MCP SearXNG server starting on port 9814..."
    fi
    sleep 2

    # --- Exa MCP Server ---
    if ! pgrep -f "mcp_servers/exa/server.py" > /dev/null; then
        screen -dmS mcp-exa bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal && python3 -m mcp_servers.exa.server 2>&1 | tee /var/log/mcp_exa.log"
        echo "Exa MCP Server starting..."
    fi

    # --- Firecrawl MCP Server ---
    if ! pgrep -f "mcp_servers/firecrawl/server.py" > /dev/null; then
        screen -dmS mcp-firecrawl bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal && python3 -m mcp_servers.firecrawl.server 2>&1 | tee /var/log/mcp_firecrawl.log"
        echo "Firecrawl MCP Server starting..."
    fi

    # Start Search Dispatcher (LangGraph router)
    if ! pgrep -f "search_dispatcher" > /dev/null; then
        screen -dmS search-dispatcher bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd ${REPO_DIR}/services/search_dispatcher && python3 main.py 2>&1 | tee /var/log/search_dispatcher.log"
        echo "Search Dispatcher starting on port ${MCP_DISPATCHER_PORT:-9801}..."
    fi
    wait_for_health "http://localhost:${MCP_DISPATCHER_PORT:-9801}/health" "Search Dispatcher" 15
fi

# --- Thinking Proxy (stays on Mistral — override global xAI defaults) ---
pip3 install fastapi uvicorn httpx -q
if ! pgrep -f "thinking_proxy.py" > /dev/null; then
    screen -dmS thinking-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && UPSTREAM_BASE='https://api.mistral.ai/v1' UPSTREAM_KEY=\"${MISTRAL_API_KEY:-}\" UPSTREAM_MODEL='mistral-large-latest' THINKING_PROXY_PORT=9100 python3 thinking_proxy.py 2>&1 | tee /var/log/thinking_proxy.log"
    echo "Thinking Proxy starting..."
fi
wait_for_health "http://localhost:9100/health" "Thinking Proxy" 15

# --- Deep Research Proxy (MiroFlow) — Grok via xAI direct API ---
if ! pgrep -f "deep_research_proxy.py" > /dev/null; then
    screen -dmS deep-research bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && DEEP_RESEARCH_PORT=9200 python3 deep_research_proxy.py 2>&1 | tee /var/log/deep_research_proxy.log"
    echo "Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9200/health" "Deep Research Proxy" 15

# --- Persistent Deep Research Proxy (Subagent Map-Reduce + AoT) — Venice AI (uncensored) ---
# miro-long:  olafangensan-glm-4.7-flash-heretic (UNCENSORED, thought=6/6, 82.4 tok/s, native tool calling)
# miro-short: qwen3.5-9b (thought=6/6, 147.5 tok/s, native tool calling, $0.1/M)
if ! pgrep -f "persistent_deep_research_proxy.py" > /dev/null; then
    screen -dmS persistent-research bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && PERSISTENT_RESEARCH_PORT=9300 python3 persistent_deep_research_proxy.py 2>&1 | tee /var/log/persistent_research_proxy.log"
    echo "Persistent Deep Research Proxy starting..."
fi
wait_for_health "http://localhost:9300/health" "Persistent Deep Research Proxy" 15

# --- MiroFlow Sprint Proxy (quick 2-round variant) — Venice AI (uncensored) ---
if ! pgrep -f "miroflow_sprint_proxy.py" > /dev/null; then
    screen -dmS miroflow-sprint bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && MIROFLOW_SPRINT_PORT=9400 python3 miroflow_sprint_proxy.py 2>&1 | tee /var/log/miroflow_sprint_proxy.log"
    echo "MiroFlow Sprint Proxy starting..."
fi
wait_for_health "http://localhost:9400/health" "MiroFlow Sprint Proxy" 15

# --- Swarm Deep Search Proxy — Venice AI (uncensored, override global xAI defaults) ---
if ! pgrep -f "swarm_proxy.py" > /dev/null; then
    screen -dmS swarm-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && UPSTREAM_BASE='https://api.venice.ai/api/v1' UPSTREAM_KEY=\"${VENICE_API_KEY:-${MISTRAL_API_KEY:-}}\" UPSTREAM_MODEL='venice-uncensored' SWARM_SYNTHESIS_MODEL='venice-uncensored' SWARM_WORKER_MODEL='venice-uncensored' SWARM_PROXY_PORT=9500 python3 swarm_proxy.py 2>&1 | tee /var/log/swarm_proxy.log"
    echo "Swarm Deep Search Proxy starting..."
fi
wait_for_health "http://localhost:9500/health" "Swarm Deep Search Proxy" 15

# --- G0DM0D3 Proxy (multi-provider native routing) ---
if ! pgrep -f "godmode_proxy.py" > /dev/null; then
    screen -dmS godmode-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && GODMODE_PROXY_PORT=9600 python3 godmode_proxy.py 2>&1 | tee /var/log/godmode_proxy.log"
    echo "G0DM0D3 Proxy starting..."
fi
wait_for_health "http://localhost:9600/health" "G0DM0D3 Proxy" 15

# --- xAI Native Proxy (direct xAI API access + race modes) ---
if ! pgrep -f "xai_native_proxy.py" > /dev/null; then
    screen -dmS xai-native-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && XAI_PROXY_PORT=9700 python3 xai_native_proxy.py 2>&1 | tee /var/log/xai_native_proxy.log"
    echo "xAI Native Proxy starting..."
fi
wait_for_health "http://localhost:9700/health" "xAI Native Proxy" 15

# --- Heretic Proxy (GLM-4.7 Flash Heretic + Firecrawl/Exa/Brave tools) ---
if [ -z "$VENICE_API_KEY" ]; then
    echo "WARNING: Skipping Heretic Proxy — VENICE_API_KEY not set"
elif ! pgrep -f "heretic_proxy.py" > /dev/null; then
    screen -dmS heretic-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && HERETIC_PROXY_PORT=9950 python3 heretic_proxy.py 2>&1 | tee /var/log/heretic_proxy.log"
    echo "Heretic Proxy starting..."
    wait_for_health "http://localhost:9950/health" "Heretic Proxy" 15
fi

# --- Miro Proxy (MiroThinker-style deep research on GLM Heretic + tools) ---
if [ -z "$VENICE_API_KEY" ]; then
    echo "WARNING: Skipping Miro Proxy — VENICE_API_KEY not set"
elif ! pgrep -f "miro_proxy.py" > /dev/null; then
    screen -dmS miro-proxy bash -c "set -a; source /opt/.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && MIRO_PROXY_PORT=9951 python3 miro_proxy.py 2>&1 | tee /var/log/miro_proxy.log"
    echo "Miro Proxy starting..."
    wait_for_health "http://localhost:9951/health" "Miro Proxy" 15
fi

# --- Self-hosted GPU VM health check (optional) ---
if [ -n "${GPU_VM_URL:-}" ]; then
    echo "Checking self-hosted GPU VM at $GPU_VM_URL..."
    if curl -sf "${GPU_VM_URL}/health" > /dev/null 2>&1; then
        echo "GPU VM is online: $GPU_VM_URL"
    else
        echo "WARNING: GPU VM at $GPU_VM_URL is not responding (may be stopped)"
    fi
fi
echo "All services started. Portal: ${DOMAIN_CLIENT:-https://deep-search.uk}"
