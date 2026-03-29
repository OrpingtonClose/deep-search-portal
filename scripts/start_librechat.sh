#!/bin/bash
# =============================================================================
# LibreChat Gateway Startup — Docker Compose stack for the chat UI.
# Deploy to /opt/start_librechat.sh on the VM.
# Expects environment variables from .env or parent process.
# =============================================================================

set -euo pipefail

# Load environment variables if .env exists
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/config/docker-compose.librechat.yml"

# Auto-generate secrets if not set
if [ -z "${CREDS_KEY:-}" ]; then
    echo "CREDS_KEY not set — generating one..."
    export CREDS_KEY="$(openssl rand -hex 32)"
    echo "Generated CREDS_KEY=$CREDS_KEY"
    echo "Add this to your .env file to persist it."
fi

if [ -z "${JWT_SECRET:-}" ]; then
    echo "JWT_SECRET not set — generating one..."
    export JWT_SECRET="$(openssl rand -hex 32)"
    echo "Generated JWT_SECRET=$JWT_SECRET"
    echo "Add this to your .env file to persist it."
fi

# Map legacy OAUTH env vars to LibreChat's expected names
export GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID:-${OAUTH_CLIENT_ID:-}}"
export GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET:-${OAUTH_CLIENT_SECRET:-}}"
export DOMAIN_CLIENT="${DOMAIN_CLIENT:-${WEBUI_URL:-http://localhost:3000}}"
export DOMAIN_SERVER="${DOMAIN_SERVER:-${WEBUI_URL:-http://localhost:3000}}"

ACTION="${1:-up}"

case "$ACTION" in
    up|start)
        echo "Starting LibreChat stack..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo ""
        echo "LibreChat is starting at: ${DOMAIN_CLIENT}"
        echo "Google OAuth is configured — users can sign in with their Google accounts."
        ;;
    stop|down)
        echo "Stopping LibreChat stack..."
        docker compose -f "$COMPOSE_FILE" down
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f --tail=100
        ;;
    restart)
        docker compose -f "$COMPOSE_FILE" restart
        ;;
    status)
        docker compose -f "$COMPOSE_FILE" ps
        ;;
    *)
        echo "Usage: $0 {up|stop|logs|restart|status}"
        exit 1
        ;;
esac
