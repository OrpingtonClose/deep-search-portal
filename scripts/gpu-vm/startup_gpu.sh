#!/bin/bash
# =============================================================================
# GPU Inference VM — Startup Script
# Starts model server + auto-stop daemon + optional Cloudflare tunnel
# Deploy to /opt/startup.sh on the Vast.ai GPU VM.
# =============================================================================
set -euo pipefail

# Load env
if [ -f /opt/.env ]; then
    set -a; source /opt/.env; set +a
fi

MODEL_DIR="${MODEL_DIR:-/models/gpt-oss-120b}"
MODEL_ID="${MODEL_ID:-huizimao/gpt-oss-120b-uncensored-bf16}"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1200}"  # 20 minutes default
VAST_API_KEY="${VAST_API_KEY:?VAST_API_KEY not set}"

# --- Download model if not cached ---
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading model $MODEL_ID (first time only)..."
    pip install -q huggingface_hub[hf_transfer]
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

# --- Health check helper (same pattern as main startup.sh) ---
wait_for_health() {
    local url="$1" label="$2" timeout="${3:-120}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "$label is healthy ($url)"; return 0
        fi
        sleep 1
    done
    echo "WARNING: $label did not become healthy within ${timeout}s ($url)"
    return 1
}

# --- Signal trapping for clean shutdown ---
cleanup() {
    echo "Shutting down GPU VM services..."
    for session in model-server auto-stop cftunnel; do
        screen -S "$session" -X quit 2>/dev/null || true
    done
    echo "All GPU VM services stopped."
}
trap cleanup SIGTERM SIGINT

# --- Start inference server (vLLM preferred, SGLang as fallback) ---
# Use vLLM because it handles HuggingFace weights directly without engine build steps.
# SGLang is also fine. Do NOT use TensorRT-LLM (requires engine build step).
if command -v vllm > /dev/null 2>&1 || python3 -c "import vllm" 2>/dev/null; then
    screen -dmS model-server bash -c "python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --host 0.0.0.0 \
        --port $SERVE_PORT \
        --tensor-parallel-size $TP_SIZE \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        2>&1 | tee /var/log/model_server.log"
    echo "vLLM server starting on port $SERVE_PORT..."
elif python3 -c "import sglang" 2>/dev/null; then
    screen -dmS model-server bash -c "python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --host 0.0.0.0 \
        --port $SERVE_PORT \
        --tp $TP_SIZE \
        --trust-remote-code \
        2>&1 | tee /var/log/model_server.log"
    echo "SGLang server starting on port $SERVE_PORT..."
else
    echo "FATAL: Neither vLLM nor SGLang installed. Install with: pip install vllm"
    exit 1
fi

# Wait for model to load (can take several minutes for large models)
wait_for_health "http://localhost:$SERVE_PORT/health" "Model Server" 600

# --- Start auto-stop daemon ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
screen -dmS auto-stop bash -c "python3 ${SCRIPT_DIR}/auto_stop.py \
    --port $SERVE_PORT \
    --timeout $IDLE_TIMEOUT \
    --vast-api-key $VAST_API_KEY \
    2>&1 | tee /var/log/auto_stop.log"
echo "Auto-stop daemon started (timeout: ${IDLE_TIMEOUT}s)"

# --- Optional: Cloudflare Tunnel ---
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    if ! pgrep -f "cloudflared tunnel" > /dev/null; then
        screen -dmS cftunnel cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
        echo "Cloudflare tunnel started"
    fi
fi

echo "GPU VM ready. Model server: http://0.0.0.0:$SERVE_PORT/v1"
