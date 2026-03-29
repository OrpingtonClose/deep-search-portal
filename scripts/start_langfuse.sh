#!/bin/bash
# =============================================================================
# Langfuse Self-Hosted Startup — LLM Observability for Deep Search Portal
#
# Starts the Langfuse stack (Postgres, ClickHouse, Redis, Langfuse Web)
# using Docker Compose.  Reuses the same Google OAuth credentials as
# LibreChat so users get single sign-on across both dashboards.
#
# Prerequisites:
#   - Docker and Docker Compose installed
#   - .env file with GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
#
# Usage:
#   ./scripts/start_langfuse.sh          # start in background
#   ./scripts/start_langfuse.sh stop     # stop all containers
#   ./scripts/start_langfuse.sh logs     # tail logs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/config/docker-compose.langfuse.yml"

# Load environment variables if .env exists
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
elif [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# Auto-generate encryption key if not set
if [ -z "${LANGFUSE_ENCRYPTION_KEY:-}" ]; then
    echo "LANGFUSE_ENCRYPTION_KEY not set — generating one..."
    export LANGFUSE_ENCRYPTION_KEY="$(openssl rand -hex 32)"
    echo "Generated LANGFUSE_ENCRYPTION_KEY=$LANGFUSE_ENCRYPTION_KEY"
    echo "Add this to your .env file to persist it."
fi

# Auto-generate NextAuth secret if not set
if [ -z "${LANGFUSE_NEXTAUTH_SECRET:-}" ]; then
    export LANGFUSE_NEXTAUTH_SECRET="$(openssl rand -hex 32)"
fi

ACTION="${1:-up}"

case "$ACTION" in
    up|start)
        echo "Starting Langfuse stack..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo ""
        echo "Langfuse is starting at: ${LANGFUSE_URL:-http://localhost:3001}"
        echo "Google OAuth is configured — users can sign in with their Google accounts."
        echo ""
        echo "Set these in your proxy .env to enable tracing:"
        echo "  LANGFUSE_PUBLIC_KEY=<from Langfuse UI -> Settings -> API Keys>"
        echo "  LANGFUSE_SECRET_KEY=<from Langfuse UI -> Settings -> API Keys>"
        echo "  LANGFUSE_BASE_URL=${LANGFUSE_URL:-http://localhost:3001}"
        ;;
    stop|down)
        echo "Stopping Langfuse stack..."
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
