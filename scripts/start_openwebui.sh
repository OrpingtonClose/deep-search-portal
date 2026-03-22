#!/bin/bash
# =============================================================================
# Open WebUI Startup — configures providers, OAuth, and launches the server.
# Deploy to /opt/start_openwebui.sh on the VM.
# Expects environment variables from .env or parent process.
# =============================================================================

# Load environment variables if .env exists
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

# --- Provider Configuration ---
# Semicolon-separated lists. Order matters — index used in OPENAI_API_CONFIGS.
# 0: OpenRouter, 1: Venice.ai, 2: Together.ai, 3: Perplexity, 4: Thinking Proxy, 5: Persistent Deep Research Proxy
export ENABLE_OPENAI_API=true
export OPENAI_API_BASE_URLS="https://openrouter.ai/api/v1;https://api.venice.ai/api/v1;https://api.together.xyz/v1;https://api.perplexity.ai;http://localhost:9100/v1;http://localhost:9300/v1"
export OPENAI_API_KEYS="${OPENROUTER_API_KEY:-not-set};${VENICE_API_KEY:-not-set};${TOGETHER_API_KEY:-not-set};${PERPLEXITY_API_KEY:-not-set};not-needed;not-needed"

# Provider-specific model filtering
export OPENAI_API_CONFIGS='{"4": {"enable": true, "model_ids": ["mistral-large-thinking"]}, "5": {"enable": true, "model_ids": ["persistent-miroflow"]}}'

# --- SearXNG for RAG Web Search ---
export ENABLE_RAG_WEB_SEARCH=true
export RAG_WEB_SEARCH_ENGINE=searxng
export SEARXNG_QUERY_URL="http://localhost:8888/search?q=<query>&format=json"

# --- Data Directory ---
export DATA_DIR="/opt/openwebui-data"

# --- Google OAuth via OIDC ---
# Uses OIDC provider (not built-in GOOGLE_CLIENT_ID) so the callback
# path is /oauth/oidc/callback — must match Google Cloud Console redirect URIs.
export ENABLE_OAUTH_SIGNUP=true
export OAUTH_PROVIDER_NAME=Google
export OPENID_PROVIDER_URL=https://accounts.google.com/.well-known/openid-configuration
export OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID:?OAUTH_CLIENT_ID not set}"
export OAUTH_CLIENT_SECRET="${OAUTH_CLIENT_SECRET:?OAUTH_CLIENT_SECRET not set}"
export OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true
export ENABLE_LOGIN_FORM=true

# --- Access Control ---
export ENABLE_SIGNUP=false
export BYPASS_ADMIN_ACCESS_CONTROL=false

export WEBUI_URL="${WEBUI_URL:-https://deep-search.uk}"

cd /opt/openwebui-env
exec /opt/openwebui-env/bin/python3.11 /opt/openwebui-env/bin/open-webui serve --host 0.0.0.0 --port 3000
