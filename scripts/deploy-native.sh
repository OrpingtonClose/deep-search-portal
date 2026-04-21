#!/bin/bash
# =============================================================================
# Deep Search Portal — Native Deployment for Vast.ai VMs
#
# This script sets up a fresh Vast.ai Ubuntu instance WITHOUT Docker.
# Docker-in-Docker is not available on Vast.ai, so everything runs natively:
#   - MongoDB via apt
#   - Meilisearch via binary
#   - LibreChat via Node.js (npm)
#   - All proxies via Python venv + screen sessions
#   - Cloudflare tunnel via cloudflared binary
#
# Usage:
#   export VENICE_API_KEY=... XAI_API_KEY=... CLOUDFLARE_TUNNEL_TOKEN=... ...
#   bash scripts/deploy-native.sh [--env prod|staging]
#
# The script is IDEMPOTENT — safe to re-run. It skips already-installed
# components and only restarts services that aren't healthy.
#
# Required env vars (from /opt/.env or exported before running):
#   VENICE_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY,
#   GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, CLOUDFLARE_TUNNEL_TOKEN
#
# Optional env vars:
#   FIRECRAWL_API_KEY, EXA_API_KEY, BRAVE_SEARCH_API_KEY, BRAVE_API_KEY,
#   MISTRAL_API_KEY, BRIGHT_DATA_API_KEY, APIFY_API_TOKEN,
#   KAGI_API_KEY, DEPLOY_ENV (prod|staging)
# =============================================================================

set -euo pipefail

log() { echo "[deploy-native] $(date '+%H:%M:%S') $*"; }
warn() { echo "[deploy-native] $(date '+%H:%M:%S') WARNING: $*" >&2; }

# ---------------------------------------------------------------------------
# 0. Load existing env FIRST (before --env parsing)
# ---------------------------------------------------------------------------
# This must happen BEFORE the env-specific DOMAIN_CLIENT/DOMAIN_SERVER
# assignments below. If sourced after, stale domain values from a previous
# prod deployment would clobber the staging domain on a cross-env re-run.
if [ -f /opt/.env ]; then
    set -a; source /opt/.env 2>/dev/null; set +a
    log "Loaded existing /opt/.env"
fi

# ---------------------------------------------------------------------------
# 0b. Determine environment (prod vs staging)
# ---------------------------------------------------------------------------
DEPLOY_ENV="${DEPLOY_ENV:-prod}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) if [[ $# -lt 2 ]]; then echo "ERROR: --env requires an argument" >&2; exit 1; fi; DEPLOY_ENV="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Domain assignments are UNCONDITIONAL (not :- defaults) because /opt/.env
# was sourced above and may contain stale domain values from a previous run.
if [[ "$DEPLOY_ENV" == "prod" ]]; then
    DOMAIN_CLIENT="https://deep-search.uk"
    DOMAIN_SERVER="https://deep-search.uk"
    LIBRECHAT_PORT=3000
    LIBRECHAT_YAML="librechat.yaml"
    MONGO_DB="LibreChat"
elif [[ "$DEPLOY_ENV" == "staging" ]]; then
    DOMAIN_CLIENT="https://staging.deep-search.uk"
    DOMAIN_SERVER="https://staging.deep-search.uk"
    LIBRECHAT_PORT=3002
    LIBRECHAT_YAML="librechat-staging.yaml"
    MONGO_DB="LibreChat-staging"
else
    echo "ERROR: DEPLOY_ENV must be 'prod' or 'staging', got '$DEPLOY_ENV'" >&2
    exit 1
fi

log "Deploying as: $DEPLOY_ENV (domain: $DOMAIN_CLIENT, port: $LIBRECHAT_PORT)"

# ---------------------------------------------------------------------------
# 0c. Pre-flight: check required vars
# ---------------------------------------------------------------------------
REQUIRED_VARS=(VENICE_API_KEY XAI_API_KEY OPENROUTER_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET CLOUDFLARE_TUNNEL_TOKEN)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then MISSING+=("$var"); fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:" >&2
    printf '  - %s\n' "${MISSING[@]}" >&2
    exit 1
fi

# Derived model config (Venice AI primary)
# UPSTREAM_KEY is always re-derived from VENICE_API_KEY (not :- default)
# so that rotating VENICE_API_KEY in /opt/.env propagates to all proxies.
UPSTREAM_KEY="$VENICE_API_KEY"
UPSTREAM_BASE="${UPSTREAM_BASE:-https://api.venice.ai/api/v1}"
UPSTREAM_MODEL="${UPSTREAM_MODEL:-olafangensan-glm-4.7-flash-heretic}"
SUBAGENT_MODEL="${SUBAGENT_MODEL:-qwen3.5-9b}"

# Map Brave key variants
if [ -n "${BRAVE_SEARCH_API_KEY:-}" ] && [ -z "${BRAVE_API_KEY:-}" ]; then
    export BRAVE_API_KEY="$BRAVE_SEARCH_API_KEY"
fi

# ---------------------------------------------------------------------------
# Helper: wait for an HTTP endpoint to become healthy
# ---------------------------------------------------------------------------
wait_for_health() {
    local url="$1" label="$2" timeout="${3:-30}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log "  OK: $label ($url)"
            return 0
        fi
        sleep 1
    done
    warn "$label did not become healthy within ${timeout}s ($url)"
    return 1
}

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>/dev/null
apt-get install -y -qq \
    git curl wget screen gnupg lsb-release ca-certificates \
    python3 python3-pip python3-venv python3.12-venv \
    > /dev/null 2>&1 || true

# Ensure python3.12-venv is available (Ubuntu 24.04)
apt-get install -y -qq python3.12-venv 2>/dev/null || apt-get install -y -qq python3-venv 2>/dev/null || true
log "System packages installed."

# ---------------------------------------------------------------------------
# 2. Node.js 20 (required by LibreChat — v18 is too old)
# ---------------------------------------------------------------------------
NODE_VERSION=$(node --version 2>/dev/null || echo "none")
if [[ "$NODE_VERSION" != v20* ]] && [[ "$NODE_VERSION" != v22* ]]; then
    log "Installing Node.js 20 (current: $NODE_VERSION)..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null 2>&1
    log "Node.js $(node --version) installed."
else
    log "Node.js $NODE_VERSION already installed."
fi

# ---------------------------------------------------------------------------
# 3. MongoDB (native, no Docker)
# ---------------------------------------------------------------------------
if ! command -v mongod &>/dev/null; then
    log "Installing MongoDB 7.0..."
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg 2>/dev/null
    CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu ${CODENAME}/mongodb-org/7.0 multiverse" \
        > /etc/apt/sources.list.d/mongodb-org-7.0.list
    apt-get update -qq 2>/dev/null
    apt-get install -y -qq mongodb-org > /dev/null 2>&1
    log "MongoDB installed."
else
    log "MongoDB already installed."
fi
# Start MongoDB if not running
if ! pgrep -x mongod > /dev/null 2>&1; then
    mkdir -p /var/lib/mongodb /var/log/mongodb /tmp/mongodb
    chown -R mongodb:mongodb /var/lib/mongodb /var/log/mongodb 2>/dev/null || true
    mongod --dbpath /var/lib/mongodb --logpath /var/log/mongodb/mongod.log --fork --bind_ip 127.0.0.1 2>/dev/null || \
        mongod --dbpath /tmp/mongodb --logpath /var/log/mongodb/mongod.log --fork --bind_ip 127.0.0.1 2>/dev/null || true
    log "MongoDB started."
fi

# ---------------------------------------------------------------------------
# 4. Meilisearch (binary install)
# ---------------------------------------------------------------------------
if ! command -v meilisearch &>/dev/null && [ ! -f /usr/local/bin/meilisearch ]; then
    log "Installing Meilisearch..."
    curl -fsSL https://install.meilisearch.com | sh > /dev/null 2>&1
    mv meilisearch /usr/local/bin/ 2>/dev/null || true
    log "Meilisearch installed."
else
    log "Meilisearch already installed."
fi
if ! pgrep -f meilisearch > /dev/null 2>&1; then
    screen -dmS meilisearch bash -c "meilisearch --master-key meilisearch_master_key --db-path /tmp/meilisearch_data --http-addr 127.0.0.1:7700 2>&1 | tee /var/log/meilisearch.log"
    log "Meilisearch started on port 7700."
fi

# ---------------------------------------------------------------------------
# 5. cloudflared
# ---------------------------------------------------------------------------
if ! command -v cloudflared &>/dev/null; then
    log "Installing cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    dpkg -i /tmp/cloudflared.deb > /dev/null 2>&1
    rm -f /tmp/cloudflared.deb
    log "cloudflared installed."
else
    log "cloudflared already installed."
fi

# ---------------------------------------------------------------------------
# 6. Clone / update the repo
# ---------------------------------------------------------------------------
if [ ! -d /opt/deep-search-portal/.git ]; then
    log "Cloning deep-search-portal..."
    git clone https://github.com/OrpingtonClose/deep-search-portal.git /opt/deep-search-portal
else
    log "Updating deep-search-portal..."
    cd /opt/deep-search-portal && git pull origin main 2>/dev/null || true
fi
log "Repo ready at /opt/deep-search-portal."

# ---------------------------------------------------------------------------
# 7. Python venv with ALL dependencies
# ---------------------------------------------------------------------------
VENV=/opt/venv
PYTHON="$VENV/bin/python3"
PIP="$VENV/bin/pip"

if [ ! -f "$PYTHON" ]; then
    log "Creating Python venv..."
    python3 -m venv "$VENV"
fi

log "Installing Python dependencies (comprehensive)..."
"$PIP" install -q --upgrade pip 2>/dev/null
"$PIP" install -q \
    fastapi uvicorn httpx python-dotenv openai pydantic \
    langchain-core langchain-openai langchain-community langgraph \
    langgraph-checkpoint-sqlite \
    litellm duckduckgo-search beautifulsoup4 requests PySocks stem \
    strands-agents strands-agents-tools strands-agents-builder \
    mcp fastmcp langfuse \
    2>&1 | tail -3
log "Python dependencies installed."

# ---------------------------------------------------------------------------
# 8. Write /opt/.env (preserves existing secrets on re-run)
# ---------------------------------------------------------------------------
CREDS_KEY="${CREDS_KEY:-$(openssl rand -hex 32)}"
JWT_SECRET="${JWT_SECRET:-$(openssl rand -hex 32)}"
JWT_REFRESH_SECRET="${JWT_REFRESH_SECRET:-$(openssl rand -hex 32)}"

# Generate staging secrets early so they can be persisted in /opt/.env.
# This ensures staging user sessions survive re-deploys.
STAGING_CREDS_KEY="${STAGING_CREDS_KEY:-$(openssl rand -hex 32)}"
STAGING_JWT_SECRET="${STAGING_JWT_SECRET:-$(openssl rand -hex 32)}"
STAGING_JWT_REFRESH="${STAGING_JWT_REFRESH:-$(openssl rand -hex 32)}"

# Select the correct auth secrets for this environment's LibreChat .env.
# This prevents --env staging from using prod JWT signing keys.
if [[ "$DEPLOY_ENV" == "staging" ]]; then
    LC_CREDS_KEY="$STAGING_CREDS_KEY"
    LC_JWT_SECRET="$STAGING_JWT_SECRET"
    LC_JWT_REFRESH_SECRET="$STAGING_JWT_REFRESH"
else
    LC_CREDS_KEY="$CREDS_KEY"
    LC_JWT_SECRET="$JWT_SECRET"
    LC_JWT_REFRESH_SECRET="$JWT_REFRESH_SECRET"
fi

# write_env_single_quoted writes a KEY='VALUE' line to a file, escaping
# embedded single quotes so $, backticks, and \ are preserved verbatim.
write_env_sq() {
    local file="$1" key="$2" val="$3"
    val="${val//\'/\'\"\'\"\'}"
    printf "%s='%s'\n" "$key" "$val" >> "$file"
}

log "Writing /opt/.env..."
install -m 600 /dev/null /opt/.env
{
    echo "# === Auto-generated by deploy-native.sh at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    echo "# === Environment: ${DEPLOY_ENV} ==="
    echo ""
    echo "# --- LLM API Keys ---"
} > /opt/.env
write_env_sq /opt/.env VENICE_API_KEY "$VENICE_API_KEY"
write_env_sq /opt/.env XAI_API_KEY "$XAI_API_KEY"
write_env_sq /opt/.env OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
write_env_sq /opt/.env MISTRAL_API_KEY "${MISTRAL_API_KEY:-}"
echo "" >> /opt/.env
echo "# --- Search/Tool API Keys ---" >> /opt/.env
write_env_sq /opt/.env FIRECRAWL_API_KEY "${FIRECRAWL_API_KEY:-}"
write_env_sq /opt/.env EXA_API_KEY "${EXA_API_KEY:-}"
write_env_sq /opt/.env BRAVE_SEARCH_API_KEY "${BRAVE_SEARCH_API_KEY:-}"
write_env_sq /opt/.env BRAVE_API_KEY "${BRAVE_API_KEY:-${BRAVE_SEARCH_API_KEY:-}}"
write_env_sq /opt/.env BRIGHT_DATA_API_KEY "${BRIGHT_DATA_API_KEY:-}"
write_env_sq /opt/.env APIFY_API_TOKEN "${APIFY_API_TOKEN:-}"
write_env_sq /opt/.env KAGI_API_KEY "${KAGI_API_KEY:-}"
echo "" >> /opt/.env
echo "# --- Google OAuth ---" >> /opt/.env
write_env_sq /opt/.env GOOGLE_CLIENT_ID "$GOOGLE_CLIENT_ID"
write_env_sq /opt/.env GOOGLE_CLIENT_SECRET "$GOOGLE_CLIENT_SECRET"
echo "" >> /opt/.env
echo "# --- LibreChat Secrets (stable across re-deploys) ---" >> /opt/.env
write_env_sq /opt/.env CREDS_KEY "$CREDS_KEY"
write_env_sq /opt/.env JWT_SECRET "$JWT_SECRET"
write_env_sq /opt/.env JWT_REFRESH_SECRET "$JWT_REFRESH_SECRET"
echo "" >> /opt/.env
echo "# --- Staging LibreChat Secrets (stable across re-deploys) ---" >> /opt/.env
write_env_sq /opt/.env STAGING_CREDS_KEY "$STAGING_CREDS_KEY"
write_env_sq /opt/.env STAGING_JWT_SECRET "$STAGING_JWT_SECRET"
write_env_sq /opt/.env STAGING_JWT_REFRESH "$STAGING_JWT_REFRESH"
echo "" >> /opt/.env
echo "# --- Domain ---" >> /opt/.env
write_env_sq /opt/.env DOMAIN_CLIENT "$DOMAIN_CLIENT"
write_env_sq /opt/.env DOMAIN_SERVER "$DOMAIN_SERVER"
echo "" >> /opt/.env
echo "# --- Cloudflare ---" >> /opt/.env
write_env_sq /opt/.env CLOUDFLARE_TUNNEL_TOKEN "$CLOUDFLARE_TUNNEL_TOKEN"
echo "" >> /opt/.env
echo "# --- Derived (model routing) ---" >> /opt/.env
write_env_sq /opt/.env UPSTREAM_KEY "$UPSTREAM_KEY"
write_env_sq /opt/.env UPSTREAM_BASE "$UPSTREAM_BASE"
write_env_sq /opt/.env UPSTREAM_MODEL "$UPSTREAM_MODEL"
write_env_sq /opt/.env SUBAGENT_MODEL "$SUBAGENT_MODEL"
log "/opt/.env written."

# ---------------------------------------------------------------------------
# 9. Install & build LibreChat (native, no Docker)
# ---------------------------------------------------------------------------
LIBRECHAT_DIR="/opt/LibreChat"
if [ ! -f "$LIBRECHAT_DIR/package.json" ]; then
    log "Cloning LibreChat..."
    rm -rf "$LIBRECHAT_DIR"
    git clone --depth 1 https://github.com/danny-avila/LibreChat.git "$LIBRECHAT_DIR"
fi

# Write LibreChat .env — CRITICAL: callback URL must be RELATIVE to avoid
# double-domain bug (DOMAIN_SERVER + absolute URL = malformed redirect_uri)
log "Writing LibreChat .env..."
install -m 600 /dev/null "$LIBRECHAT_DIR/.env"
{
    echo "# --- Database ---"
    echo "MONGO_URI=mongodb://127.0.0.1:27017/${MONGO_DB}"
    echo "MEILI_HOST=http://127.0.0.1:7700"
    echo "MEILI_MASTER_KEY=meilisearch_master_key"
    echo ""
    echo "# --- Auth secrets (stable across re-deploys) ---"
    echo "# When DEPLOY_ENV=staging, use staging-specific secrets to avoid sharing"
    echo "# JWT signing keys with prod (a JWT minted by prod must not be valid on staging)."
} > "$LIBRECHAT_DIR/.env"
write_env_sq "$LIBRECHAT_DIR/.env" CREDS_KEY "$LC_CREDS_KEY"
write_env_sq "$LIBRECHAT_DIR/.env" JWT_SECRET "$LC_JWT_SECRET"
write_env_sq "$LIBRECHAT_DIR/.env" JWT_REFRESH_SECRET "$LC_JWT_REFRESH_SECRET"
echo "" >> "$LIBRECHAT_DIR/.env"
echo "# --- Domain ---" >> "$LIBRECHAT_DIR/.env"
write_env_sq "$LIBRECHAT_DIR/.env" DOMAIN_CLIENT "$DOMAIN_CLIENT"
write_env_sq "$LIBRECHAT_DIR/.env" DOMAIN_SERVER "$DOMAIN_SERVER"
{
    echo ""
    echo "# --- Google OAuth ---"
    echo "# IMPORTANT: GOOGLE_CALLBACK_URL must be a RELATIVE path."
    echo "# LibreChat prepends DOMAIN_SERVER to this value internally."
    echo "# Using an absolute URL here causes a double-domain bug:"
    echo "#   https://deep-search.ukhttps://deep-search.uk/oauth/google/callback"
    echo "ALLOW_SOCIAL_LOGIN=true"
    echo "ALLOW_SOCIAL_REGISTRATION=true"
} >> "$LIBRECHAT_DIR/.env"
write_env_sq "$LIBRECHAT_DIR/.env" GOOGLE_CLIENT_ID "$GOOGLE_CLIENT_ID"
write_env_sq "$LIBRECHAT_DIR/.env" GOOGLE_CLIENT_SECRET "$GOOGLE_CLIENT_SECRET"
{
    echo "GOOGLE_CALLBACK_URL=/oauth/google/callback"
    echo ""
    echo "# --- Server ---"
    echo "HOST=0.0.0.0"
    echo "PORT=${LIBRECHAT_PORT}"
} >> "$LIBRECHAT_DIR/.env"

# Copy the correct LibreChat config
cp "/opt/deep-search-portal/config/${LIBRECHAT_YAML}" "$LIBRECHAT_DIR/librechat.yaml"
log "LibreChat config: ${LIBRECHAT_YAML}"

# Build LibreChat frontend (idempotent — skip if already built)
if [ ! -f "$LIBRECHAT_DIR/client/dist/index.html" ]; then
    log "Building LibreChat (npm install + frontend)..."
    cd "$LIBRECHAT_DIR"
    npm install 2>&1 | tail -3
    npm run frontend 2>&1 | tail -5
    log "LibreChat built."
else
    log "LibreChat already built (client/dist/index.html exists)."
fi

# Add host.docker.internal alias (some LibreChat endpoints reference it)
if ! grep -q 'host.docker.internal' /etc/hosts 2>/dev/null; then
    echo '127.0.0.1 host.docker.internal' >> /etc/hosts
fi

# ---------------------------------------------------------------------------
# 10. Install MCP tool servers (npm global packages)
# ---------------------------------------------------------------------------
log "Installing MCP tool servers..."
npm install -g exa-mcp-server 2>/dev/null || true
# Install uv for uvx-based MCP servers (kagi, etc.)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi
log "MCP tool servers ready."

# ---------------------------------------------------------------------------
# 11. Strands Agent venv (separate from proxy venv)
# ---------------------------------------------------------------------------
STRANDS_DIR="/opt/deep-search-portal/strands-agent"
STRANDS_VENV="$STRANDS_DIR/.venv"
if [ ! -f "$STRANDS_VENV/bin/python" ]; then
    log "Creating Strands Agent venv..."
    python3 -m venv "$STRANDS_VENV"
    "$STRANDS_VENV/bin/pip" install -q --upgrade pip
    "$STRANDS_VENV/bin/pip" install -q \
        strands-agents strands-agents-tools strands-agents-builder \
        openai python-dotenv "fastapi[standard]" "uvicorn[standard]" \
        duckduckgo-search httpx \
        2>&1 | tail -3
    log "Strands Agent venv ready."
else
    log "Strands Agent venv already exists."
fi

# ---------------------------------------------------------------------------
# 12. Start all services
# ---------------------------------------------------------------------------
log "Starting services..."

# --- Cloudflare Tunnel ---
if ! pgrep -f "cloudflared tunnel" > /dev/null 2>&1; then
    screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
    log "  Cloudflare tunnel started."
fi

# --- LibreChat ---
# CRITICAL: We must explicitly export DOMAIN_CLIENT/DOMAIN_SERVER in the screen
# command. Node's dotenv does NOT override env vars already set in the process.
# If /opt/.env was sourced in this shell (it was), those prod domain values
# propagate to child processes and override the .env file values.
# This caused staging OAuth to redirect to deep-search.uk instead of
# staging.deep-search.uk — a recurring "OAuth doesn't work" bug.
LIBRECHAT_SCREEN_NAME="librechat"
if [[ "$DEPLOY_ENV" == "staging" ]]; then
    LIBRECHAT_SCREEN_NAME="librechat-staging"
fi
if ! curl -sf "http://localhost:${LIBRECHAT_PORT}" > /dev/null 2>&1; then
    screen -S "$LIBRECHAT_SCREEN_NAME" -X quit 2>/dev/null || true
    screen -dmS "$LIBRECHAT_SCREEN_NAME" bash -c "
        export DOMAIN_CLIENT='${DOMAIN_CLIENT}'
        export DOMAIN_SERVER='${DOMAIN_SERVER}'
        export PORT=${LIBRECHAT_PORT}
        export ALLOW_SOCIAL_LOGIN=true
        export ALLOW_SOCIAL_REGISTRATION=true
        cd $LIBRECHAT_DIR && node api/server/index.js 2>&1 | tee /var/log/${LIBRECHAT_SCREEN_NAME}.log
    "
    log "  LibreChat ($DEPLOY_ENV) starting on port ${LIBRECHAT_PORT}..."
fi

# --- Staging LibreChat (when deploying prod, also set up staging on port 3002) ---
# Both prod and staging LibreChat run on the SAME VM. The Cloudflare tunnel
# routes deep-search.uk → :3000 and staging.deep-search.uk → :3002.
if [[ "$DEPLOY_ENV" == "prod" ]]; then
    STAGING_DIR="/opt/LibreChat-staging"
    if [ ! -d "$STAGING_DIR" ]; then
        log "  Setting up staging LibreChat (copy from prod)..."
        cp -r "$LIBRECHAT_DIR" "$STAGING_DIR"
    fi
    # Write staging .env with DIFFERENT domain
    # Staging secrets were generated earlier and persisted in /opt/.env
    install -m 600 /dev/null "$STAGING_DIR/.env"
    {
        echo "MONGO_URI=mongodb://127.0.0.1:27017/LibreChat-staging"
        echo "MEILI_HOST=http://127.0.0.1:7700"
        echo "MEILI_MASTER_KEY=meilisearch_master_key"
    } > "$STAGING_DIR/.env"
    write_env_sq "$STAGING_DIR/.env" CREDS_KEY "$STAGING_CREDS_KEY"
    write_env_sq "$STAGING_DIR/.env" JWT_SECRET "$STAGING_JWT_SECRET"
    write_env_sq "$STAGING_DIR/.env" JWT_REFRESH_SECRET "$STAGING_JWT_REFRESH"
    {
        echo "DOMAIN_CLIENT=https://staging.deep-search.uk"
        echo "DOMAIN_SERVER=https://staging.deep-search.uk"
        echo "ALLOW_SOCIAL_LOGIN=true"
        echo "ALLOW_SOCIAL_REGISTRATION=true"
    } >> "$STAGING_DIR/.env"
    write_env_sq "$STAGING_DIR/.env" GOOGLE_CLIENT_ID "$GOOGLE_CLIENT_ID"
    write_env_sq "$STAGING_DIR/.env" GOOGLE_CLIENT_SECRET "$GOOGLE_CLIENT_SECRET"
    {
        echo "GOOGLE_CALLBACK_URL=/oauth/google/callback"
        echo "HOST=0.0.0.0"
        echo "PORT=3002"
    } >> "$STAGING_DIR/.env"
    cp "/opt/deep-search-portal/config/librechat-staging.yaml" "$STAGING_DIR/librechat.yaml"

    if ! curl -sf "http://localhost:3002" > /dev/null 2>&1; then
        screen -S librechat-staging -X quit 2>/dev/null || true
        screen -dmS librechat-staging bash -c "
            export DOMAIN_CLIENT='https://staging.deep-search.uk'
            export DOMAIN_SERVER='https://staging.deep-search.uk'
            export PORT=3002
            export ALLOW_SOCIAL_LOGIN=true
            export ALLOW_SOCIAL_REGISTRATION=true
            cd $STAGING_DIR && node api/server/index.js 2>&1 | tee /var/log/librechat-staging.log
        "
        log "  Staging LibreChat starting on port 3002..."
    fi
fi

# --- Proxy services (all use /opt/venv/bin/python3) ---
# write_proxy_env writes a per-service env file to avoid double shell expansion.
# Values are single-quoted so $, backticks, and \ in API keys are preserved
# verbatim when the inner bash -c shell sources the file.
# Callers pass KEY=VALUE pairs; this function splits on the first '=' and
# wraps the value in single quotes (escaping any embedded single quotes).
write_proxy_env() {
    local name="$1"; shift
    local envfile="/tmp/${name}.env"
    install -m 600 /dev/null "$envfile"
    for pair in "$@"; do
        local key="${pair%%=*}"
        local val="${pair#*=}"
        # Escape any single quotes in the value: replace ' with '"'"'
        val="${val//\'/\'\"\'\"\'}"
        printf "%s='%s'\n" "$key" "$val" >> "$envfile"
    done
}

start_proxy() {
    local name="$1" port="$2" script="$3"
    shift 3
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        log "  OK: $name already healthy on port $port"
        return 0
    fi
    # Write per-service env file with any extra vars (already quoted by caller)
    write_proxy_env "$name" "$@"
    screen -S "$name" -X quit 2>/dev/null || true
    screen -dmS "$name" bash -c "set -a; source /opt/.env 2>/dev/null; source /tmp/${name}.env 2>/dev/null; set +a; cd /opt/deep-search-portal/proxies && $PYTHON $script 2>&1 | tee /var/log/${name}.log"
    log "  Starting $name on port $port..."
}

start_proxy "thinking-proxy" 9100 "thinking_proxy.py" \
    "UPSTREAM_BASE=https://api.mistral.ai/v1" \
    "UPSTREAM_KEY=${MISTRAL_API_KEY:-}" \
    "UPSTREAM_MODEL=mistral-large-latest" \
    "THINKING_PROXY_PORT=9100"

start_proxy "deep-research" 9200 "deep_research_proxy.py" \
    "DEEP_RESEARCH_PORT=9200"

start_proxy "persistent-research" 9300 "persistent_deep_research_proxy.py" \
    "PERSISTENT_RESEARCH_PORT=9300"

start_proxy "miroflow-sprint" 9400 "miroflow_sprint_proxy.py" \
    "MIROFLOW_SPRINT_PORT=9400"

start_proxy "swarm-proxy" 9500 "swarm_proxy.py" \
    "UPSTREAM_BASE=https://api.venice.ai/api/v1" \
    "UPSTREAM_KEY=${VENICE_API_KEY:-}" \
    "UPSTREAM_MODEL=venice-uncensored" \
    "SWARM_SYNTHESIS_MODEL=venice-uncensored" \
    "SWARM_WORKER_MODEL=venice-uncensored" \
    "SWARM_PROXY_PORT=9500"

start_proxy "godmode-proxy" 9600 "godmode_proxy.py" \
    "GODMODE_PROXY_PORT=9600"

start_proxy "xai-native-proxy" 9700 "xai_native_proxy.py" \
    "XAI_PROXY_PORT=9700"

start_proxy "tier-chooser" 9900 "tier_chooser_proxy.py" \
    "TIER_CHOOSER_PORT=9900"

start_proxy "heretic-proxy" 9950 "heretic_proxy.py" \
    "HERETIC_PROXY_PORT=9950"

start_proxy "miro-proxy" 9951 "miro_proxy.py" \
    "MIRO_PROXY_PORT=9951"

# --- Deep Agents proxy (deepagents SDK, port 8200) ---
# The deepagents proxy runs in its own Python 3.11+ venv because the SDK
# requires >=3.11 while the VM's system Python may be 3.10.
# It uses the main /opt/venv for langchain deps but needs the deepagents
# package from /opt/deepagents/.venv.
DEEPAGENTS_PORT="${DEEPAGENTS_PORT:-8200}"
if ! curl -sf "http://localhost:${DEEPAGENTS_PORT}/health" > /dev/null 2>&1; then
    screen -S deepagents-proxy -X quit 2>/dev/null || true
    # Use /opt/deepagents/.venv if it exists (has deepagents SDK), else /opt/venv
    DAGENT_PYTHON="/opt/deepagents/.venv/bin/python"
    if [ ! -f "$DAGENT_PYTHON" ]; then
        DAGENT_PYTHON="$PYTHON"
    fi
    write_proxy_env "deepagents-proxy" \
        "DEEPAGENTS_PORT=${DEEPAGENTS_PORT}" \
        "VENICE_API_KEY=${VENICE_API_KEY:-}" \
        "VENICE_API_BASE=${UPSTREAM_BASE}" \
        "DEEPAGENTS_MODEL=${UPSTREAM_MODEL}"
    screen -dmS deepagents-proxy bash -c "
        set -a; source /opt/.env 2>/dev/null; source /tmp/deepagents-proxy.env 2>/dev/null; set +a
        cd /opt/deep-search-portal/proxies && $DAGENT_PYTHON deepagents_proxy.py 2>&1 | tee /var/log/deepagents_proxy.log
    "
    log "  Deep Agents proxy starting on port ${DEEPAGENTS_PORT}..."
fi

# --- Strands Agent ---
STRANDS_AGENT_PORT="${STRANDS_AGENT_PORT:-8100}"
if ! curl -sf "http://localhost:${STRANDS_AGENT_PORT}/health" > /dev/null 2>&1; then
    screen -S strands-agent -X quit 2>/dev/null || true
    REPO_ROOT="/opt/deep-search-portal"
    # Write strands-agent env file to avoid double expansion of API keys
    write_proxy_env "strands-agent" \
        "BRAVE_API_KEY=${BRAVE_SEARCH_API_KEY:-}"
    # PYTHONPATH needs variable expansion at runtime, so set it in bash -c
    # (write_proxy_env single-quotes values which would prevent expansion)
    screen -dmS strands-agent bash -c "
        set -a; source /opt/.env 2>/dev/null; source /tmp/strands-agent.env 2>/dev/null; set +a
        export PYTHONPATH='${REPO_ROOT}/proxies':\${PYTHONPATH:-}
        cd '${STRANDS_DIR}'
        ${STRANDS_VENV}/bin/python -m uvicorn main:app --host 0.0.0.0 --port ${STRANDS_AGENT_PORT} 2>&1 | tee /var/log/strands-agent.log
    "
    log "  Strands Agent starting on port ${STRANDS_AGENT_PORT}..."
fi

# ---------------------------------------------------------------------------
# 13. Wait for services and report
# ---------------------------------------------------------------------------
log ""
log "Waiting for services to become healthy..."
sleep 10

SERVICES=(
    "LibreChat-${DEPLOY_ENV}:${LIBRECHAT_PORT}"
)
if [[ "$DEPLOY_ENV" == "prod" ]]; then
    SERVICES+=("LibreChat-staging:3002")
fi
SERVICES+=(
    "Thinking:9100"
    "DeepResearch:9200"
    "PersistentResearch:9300"
    "MiroFlowSprint:9400"
    "Swarm:9500"
    "G0DM0D3:9600"
    "xAI:9700"
    "TierChooser:9900"
    "Heretic:9950"
    "Miro:9951"
    "DeepAgents:${DEEPAGENTS_PORT}"
    "StrandsAgent:${STRANDS_AGENT_PORT}"
)

HEALTHY=0
TOTAL=${#SERVICES[@]}
for svc in "${SERVICES[@]}"; do
    NAME="${svc%%:*}"; PORT="${svc##*:}"
    if curl -sf -m 3 "http://localhost:${PORT}/health" > /dev/null 2>&1 || \
       curl -sf -m 3 "http://localhost:${PORT}" > /dev/null 2>&1; then
        log "  OK: ${NAME} (${PORT})"
        ((HEALTHY++)) || true
    else
        warn "FAIL: ${NAME} (${PORT})"
    fi
done

log ""
log "=========================================="
log "  Deployment Complete: ${DEPLOY_ENV}"
log "=========================================="
log "  Domain:   ${DOMAIN_CLIENT}"
log "  Services: ${HEALTHY}/${TOTAL} healthy"
log "  Python:   ${PYTHON}"
log "  LibreChat: ${LIBRECHAT_DIR} (port ${LIBRECHAT_PORT})"
log ""

if [ "$HEALTHY" -lt "$TOTAL" ]; then
    warn "Some services failed. Check: screen -ls && screen -r <session>"
    warn "Logs in: /var/log/*.log"
    exit 1
else
    log "All services healthy. ${DEPLOY_ENV} is live at ${DOMAIN_CLIENT}"
fi
