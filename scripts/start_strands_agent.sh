#!/bin/bash
# =============================================================================
# Strands Agent — Startup Script
# Starts the Strands Venice research agent with observability configured.
#
# Usage:
#   bash scripts/start_strands_agent.sh
#
# Prerequisites:
#   - deep-search-portal repo at $REPO_ROOT (auto-detected from script location)
#   - Python venv at $REPO_ROOT/strands-agent/.venv
#   - /opt/.env with API keys (VENICE_API_KEY, BRAVE_API_KEY, etc.)
#
# The script:
#   1. Sources API keys from /opt/.env
#   2. Adds deep-search-portal/proxies to PYTHONPATH (observability + adaptive plugin)
#   3. Installs logrotate config for JSONL log files
#   4. Starts the agent via screen on port $STRANDS_AGENT_PORT (default: 8100)
# =============================================================================

set -euo pipefail

# Load environment variables
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

# Resolve paths — agent code now lives inside deep-search-portal
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)}"
STRANDS_AGENT_APP="${REPO_ROOT}/strands-agent"
STRANDS_AGENT_PORT="${STRANDS_AGENT_PORT:-8100}"

# Validate prerequisites
if [ ! -f "$STRANDS_AGENT_APP/main.py" ]; then
    echo "ERROR: Strands agent not found at $STRANDS_AGENT_APP"
    echo "  Expected deep-search-portal/strands-agent/main.py"
    exit 1
fi

if [ -z "${VENICE_API_KEY:-}" ]; then
    echo "WARNING: VENICE_API_KEY not set — Strands agent will fail to initialise"
fi

# Map Brave key name (deep-search uses BRAVE_SEARCH_API_KEY, MCP expects BRAVE_API_KEY)
if [ -n "${BRAVE_SEARCH_API_KEY:-}" ] && [ -z "${BRAVE_API_KEY:-}" ]; then
    export BRAVE_API_KEY="$BRAVE_SEARCH_API_KEY"
fi

# ── Install logrotate config (idempotent) ──
LOGROTATE_SRC="${REPO_ROOT}/config/strands-logrotate.conf"
LOGROTATE_DST="/etc/logrotate.d/strands-agent"
if [ -f "$LOGROTATE_SRC" ]; then
    if ! diff -q "$LOGROTATE_SRC" "$LOGROTATE_DST" > /dev/null 2>&1; then
        cp "$LOGROTATE_SRC" "$LOGROTATE_DST" 2>/dev/null || \
            sudo cp "$LOGROTATE_SRC" "$LOGROTATE_DST" 2>/dev/null || \
            echo "WARNING: Could not install logrotate config (no sudo?)"
    fi
fi

# ── Ensure log directory is writable ──
for logfile in /var/log/strands-metrics.jsonl /var/log/strands-agent-debug.jsonl /var/log/strands-agent.log; do
    touch "$logfile" 2>/dev/null || sudo touch "$logfile" 2>/dev/null || true
    chmod 666 "$logfile" 2>/dev/null || sudo chmod 666 "$logfile" 2>/dev/null || true
done

# ── Ensure qualitative research data directory exists ──
QUAL_RESEARCH_DATA_DIR="${QUAL_RESEARCH_DATA_DIR:-/opt/qualitative-research}"
mkdir -p "$QUAL_RESEARCH_DATA_DIR" 2>/dev/null || sudo mkdir -p "$QUAL_RESEARCH_DATA_DIR" 2>/dev/null || true
chmod 755 "$QUAL_RESEARCH_DATA_DIR" 2>/dev/null || sudo chmod 755 "$QUAL_RESEARCH_DATA_DIR" 2>/dev/null || true
export QUAL_RESEARCH_DATA_DIR

# ── Install qualitative research MCP server if missing ──
QUAL_RESEARCH_DIR="${QUAL_RESEARCH_DIR:-/opt/qualitativeresearch}"
if [ ! -f "$QUAL_RESEARCH_DIR/index.js" ]; then
    echo "Installing qualitative research MCP server..."
    git clone https://github.com/tejpalvirk/qualitativeresearch.git "$QUAL_RESEARCH_DIR" 2>/dev/null || true
    if [ -d "$QUAL_RESEARCH_DIR" ]; then
        cd "$QUAL_RESEARCH_DIR" && npm install 2>/dev/null && npm run build 2>/dev/null || true
        cd -
    fi
fi
export QUAL_RESEARCH_DIR

# ── Start the agent ──
if pgrep -f "strands-agent.*main:app" > /dev/null 2>&1 || pgrep -f "uvicorn.*main:app.*${STRANDS_AGENT_PORT}" > /dev/null 2>&1; then
    echo "Strands agent is already running on port ${STRANDS_AGENT_PORT}"
    exit 0
fi

# Add deep-search-portal/proxies to PYTHONPATH so the agent can import
# strands_observability and strands_adaptive without copying modules.
export PYTHONPATH="${REPO_ROOT}/proxies:${PYTHONPATH:-}"

# Determine the Python interpreter
if [ -d "${STRANDS_AGENT_APP}/.venv" ]; then
    PYTHON="${STRANDS_AGENT_APP}/.venv/bin/python"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON="python3"
else
    echo "ERROR: No Python interpreter found"
    exit 1
fi

screen -dmS strands-agent bash -c "
    set -a; source /opt/.env 2>/dev/null; set +a
    export PYTHONPATH='${REPO_ROOT}/proxies:${PYTHONPATH:-}'
    if [ -n '${BRAVE_SEARCH_API_KEY:-}' ] && [ -z '${BRAVE_API_KEY:-}' ]; then
        export BRAVE_API_KEY='${BRAVE_SEARCH_API_KEY:-}'
    fi
    cd '${STRANDS_AGENT_APP}'
    ${PYTHON} -m uvicorn main:app --host 0.0.0.0 --port ${STRANDS_AGENT_PORT} 2>&1 | tee /var/log/strands-agent.log
"

echo "Strands agent starting on port ${STRANDS_AGENT_PORT}..."
echo "  App: ${STRANDS_AGENT_APP}"
echo "  Logs: /var/log/strands-agent.log"
echo "  Metrics JSONL: /var/log/strands-metrics.jsonl"
echo "  SDK debug: /var/log/strands-agent-debug.jsonl"
echo "  Observability: ${REPO_ROOT}/proxies/strands_observability.py"
echo "  Adaptive plugin: ${REPO_ROOT}/proxies/strands_adaptive.py"
