#!/bin/bash
# =============================================================================
# Deep Search Portal — Full Production VM Setup (Single Command)
#
# This script sets up a fresh Vast.ai Ubuntu instance as a production server.
# It installs Docker, clones the repo, writes the .env, starts LibreChat via
# Docker Compose, starts all proxies, and connects the Cloudflare tunnel.
#
# Usage (from Devin or any machine with SSH access):
#   scp scripts/deploy-production.sh root@<host>:/tmp/deploy.sh
#   ssh root@<host> 'bash /tmp/deploy.sh'
#
# Or run directly on the VM:
#   curl -fsSL https://raw.githubusercontent.com/OrpingtonClose/deep-search-portal/main/scripts/deploy-production.sh | bash
#
# Required env vars (must be set before running, or passed inline):
#   VENICE_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY,
#   GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, CLOUDFLARE_TUNNEL_TOKEN,
#   DOMAIN_CLIENT (default: https://deep-search.uk),
#   DOMAIN_SERVER (default: https://deep-search.uk)
#
# Optional env vars:
#   FIRECRAWL_API_KEY, EXA_API_KEY, BRAVE_SEARCH_API_KEY,
#   MISTRAL_API_KEY, BRIGHT_DATA_API_KEY
# =============================================================================

set -euo pipefail

log() { echo "[deploy] $(date '+%H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
log "Starting production deployment..."

REQUIRED_VARS=(VENICE_API_KEY XAI_API_KEY OPENROUTER_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET CLOUDFLARE_TUNNEL_TOKEN)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:" >&2
    printf '  - %s\n' "${MISSING[@]}" >&2
    echo "" >&2
    echo "Set them before running this script, e.g.:" >&2
    echo "  export VENICE_API_KEY=... XAI_API_KEY=... && bash $0" >&2
    exit 1
fi

DOMAIN_CLIENT="${DOMAIN_CLIENT:-https://deep-search.uk}"
DOMAIN_SERVER="${DOMAIN_SERVER:-https://deep-search.uk}"

# ---------------------------------------------------------------------------
# 1. Install system packages (single apt-get call)
# ---------------------------------------------------------------------------
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl wget screen python3 python3-pip \
    apt-transport-https ca-certificates gnupg lsb-release \
    > /dev/null 2>&1
log "System packages installed."

# ---------------------------------------------------------------------------
# 2. Install Docker (if not already installed)
# ---------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    log "Docker installed."
else
    log "Docker already installed."
fi

# ---------------------------------------------------------------------------
# 3. Install cloudflared (if not already installed)
# ---------------------------------------------------------------------------
if ! command -v cloudflared &> /dev/null; then
    log "Installing cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    dpkg -i /tmp/cloudflared.deb
    rm -f /tmp/cloudflared.deb
    log "cloudflared installed."
else
    log "cloudflared already installed."
fi

# ---------------------------------------------------------------------------
# 4. Clone the repo
# ---------------------------------------------------------------------------
if [ ! -d /opt/deep-search-portal/.git ]; then
    log "Cloning repo..."
    git clone https://github.com/OrpingtonClose/deep-search-portal.git /opt/deep-search-portal
else
    log "Repo already exists, pulling latest..."
    cd /opt/deep-search-portal && git pull origin main
fi
log "Repo ready at /opt/deep-search-portal."

# ---------------------------------------------------------------------------
# 5. Install Python dependencies
# ---------------------------------------------------------------------------
log "Installing Python dependencies..."
pip3 install -q -r /opt/deep-search-portal/requirements.txt 2>/dev/null || true
pip3 install -q fastapi uvicorn httpx 2>/dev/null || true
log "Python dependencies installed."

# ---------------------------------------------------------------------------
# 6. Write /opt/.env
# ---------------------------------------------------------------------------
log "Writing /opt/.env..."

# Preserve existing secrets on rerun (CREDS_KEY/JWT_SECRET from prior deploy)
if [ -f /opt/.env ]; then
    set -a; source /opt/.env 2>/dev/null; set +a
    log "Loaded existing /opt/.env (preserving CREDS_KEY/JWT_SECRET)."
fi
CREDS_KEY="${CREDS_KEY:-$(openssl rand -hex 32)}"
JWT_SECRET="${JWT_SECRET:-$(openssl rand -hex 32)}"

install -m 600 /dev/null /opt/.env
cat > /opt/.env << ENVEOF
# === Auto-generated by deploy-production.sh at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===

# --- LLM API Keys ---
VENICE_API_KEY=${VENICE_API_KEY}
XAI_API_KEY=${XAI_API_KEY}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
MISTRAL_API_KEY=${MISTRAL_API_KEY:-}

# --- Search/Tool API Keys ---
FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY:-}
EXA_API_KEY=${EXA_API_KEY:-}
BRAVE_SEARCH_API_KEY=${BRAVE_SEARCH_API_KEY:-}
BRIGHT_DATA_API_KEY=${BRIGHT_DATA_API_KEY:-}

# --- Google OAuth ---
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}

# --- LibreChat Secrets ---
CREDS_KEY=${CREDS_KEY}
JWT_SECRET=${JWT_SECRET}
JWT_REFRESH_SECRET=${JWT_SECRET}

# --- Domain ---
DOMAIN_CLIENT=${DOMAIN_CLIENT}
DOMAIN_SERVER=${DOMAIN_SERVER}

# --- Cloudflare ---
CLOUDFLARE_TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}

# --- Derived (set by startup.sh) ---
UPSTREAM_KEY=${VENICE_API_KEY}
UPSTREAM_BASE=https://api.venice.ai/api/v1
UPSTREAM_MODEL=olafangensan-glm-4.7-flash-heretic
SUBAGENT_MODEL=qwen3.5-9b
ENVEOF

log "/opt/.env written."

# ---------------------------------------------------------------------------
# 7. Copy production LibreChat config
# ---------------------------------------------------------------------------
log "Setting up LibreChat config..."
mkdir -p /opt/LibreChat
cp /opt/deep-search-portal/config/librechat.yaml /opt/LibreChat/librechat.yaml
# Docker Compose mounts from config/ dir, so also ensure it's there
cp /opt/deep-search-portal/config/librechat.yaml /opt/deep-search-portal/config/librechat.yaml.bak 2>/dev/null || true
log "LibreChat config ready (production: 3 models, enforce: true)."

# ---------------------------------------------------------------------------
# 8. Add host.docker.internal to /etc/hosts (required by LibreChat endpoints)
# ---------------------------------------------------------------------------
if ! grep -q 'host.docker.internal' /etc/hosts 2>/dev/null; then
    echo '127.0.0.1 host.docker.internal' >> /etc/hosts
    log "Added host.docker.internal to /etc/hosts."
fi

# ---------------------------------------------------------------------------
# 9. Start all services via startup.sh
#    (LibreChat Docker Compose + all proxies + Cloudflare tunnel)
# ---------------------------------------------------------------------------
log "Starting all services via startup.sh..."
cd /opt/deep-search-portal
bash scripts/startup.sh || log "WARNING: Some services may have failed to start."

# ---------------------------------------------------------------------------
# 10. Health check summary
# ---------------------------------------------------------------------------
log ""
log "=========================================="
log "  Production Deployment Complete"
log "=========================================="
log ""
log "Public URL: ${DOMAIN_CLIENT}"
log ""

SERVICES=(
    "LibreChat:3000"
    "Heretic Proxy:9950"
    "Miro Proxy:9951"
    "xAI Native Proxy:9700"
    "Tier Chooser:9900"
    "G0DM0D3 Proxy:9600"
    "Deep Research:9200"
    "Thinking Proxy:9100"
)

ALL_OK=true
for svc in "${SERVICES[@]}"; do
    NAME="${svc%%:*}"
    PORT="${svc##*:}"
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || curl -sf "http://localhost:${PORT}" > /dev/null 2>&1; then
        log "  ✓ ${NAME} (port ${PORT})"
    else
        log "  ✗ ${NAME} (port ${PORT}) — NOT RESPONDING"
        ALL_OK=false
    fi
done

log ""
if [ "$ALL_OK" = true ]; then
    log "All services healthy. Production is live at ${DOMAIN_CLIENT}"
else
    log "Some services failed. Check logs: screen -ls && screen -r <session>"
fi
