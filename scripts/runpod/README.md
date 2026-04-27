# Kimi K2.6 Heretic — On-Demand RunPod Deployment

Self-hosted, uncensored Kimi K2.6 (abliterated "Heretic" version) running as
a GGUF model on RunPod GPU pods.  Designed for on-demand use: spin up when you
need it, stop when you're done to save costs.

## Architecture

```
LibreChat → Kimi Proxy (:9960) → RunPod Pod (llama-server :8000)
                                      │
                                      ├── GGUF model on Network Volume
                                      └── 4× H100/A100 80GB GPUs
```

- **`manage_kimi.py`** — CLI to create, start, stop, and destroy the RunPod pod
- **`proxies/kimi_proxy.py`** — FastAPI proxy on the Vast.ai staging VM that
  forwards requests to the RunPod pod
- **Network Volume** — Persistent storage for the ~584 GB model files (survives
  pod stop/restart, so you only download once)

## Quick Start

### 1. Set your RunPod API key

```bash
export RUNPOD_API_KEY=your-key-here
# Get from: https://www.runpod.io/console/user/settings
```

### 2. Create the pod (first time only)

```bash
python scripts/runpod/manage_kimi.py create
```

This:
- Creates a network volume (~800 GB) for model storage
- Launches a pod with 4× H100 80GB GPUs
- Downloads the Q4_K_M GGUF from HuggingFace (~584 GB, ~15-30 min)
- Starts `llama-server` with OpenAI-compatible API

### 3. Check status

```bash
python scripts/runpod/manage_kimi.py status
```

Shows pod state, GPU info, cost, and the endpoint URL.

### 4. Use it

Once the pod is running, you can:

**Direct API access:**
```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k26-heretic",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Via LibreChat** (staging): Select "Kimi K2.6 Heretic" from the model dropdown.
Requires the Kimi proxy to be running (see Proxy Setup below).

### 5. Stop when done (saves money)

```bash
python scripts/runpod/manage_kimi.py stop
```

Volume is preserved — next `start` won't re-download the model.

### 6. Destroy (full cleanup)

```bash
# Keep volume (model cached for next time):
python scripts/runpod/manage_kimi.py destroy

# Delete everything including model cache:
python scripts/runpod/manage_kimi.py destroy --include-volume
```

## Proxy Setup (Staging VM)

To make Kimi available in LibreChat, add the endpoint URL to the staging VM:

```bash
# SSH into staging
ssh -p 23934 root@ssh5.vast.ai

# Add to /opt/.env:
echo 'KIMI_RUNPOD_URL=https://<pod-id>-8000.proxy.runpod.net/v1' >> /opt/.env

# Start the proxy
screen -dmS kimi-proxy bash -c 'set -a; source /opt/.env; set +a; cd /opt/deep-search-portal/proxies && KIMI_PROXY_PORT=9960 python3 kimi_proxy.py'
```

Or restart all services: `bash /opt/startup.sh`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_API_KEY` | (required) | RunPod API key |
| `KIMI_QUANT` | `Q4_K_M` | GGUF quantization level |
| `KIMI_GPU_COUNT` | `4` | Number of GPUs per pod |
| `KIMI_CONTEXT_SIZE` | `32768` | Context window (tokens) |
| `KIMI_VOLUME_DISK_GB` | `800` | Network volume size |
| `KIMI_RUNPOD_URL` | (none) | Pod endpoint URL (for proxy) |

### Available Quantizations

| Quant | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `Q2_K` | ~340 GB | Lower | Fastest | Cheapest, noticeable quality loss |
| `Q4_K_M` | ~584 GB | Good | Good | **Recommended default** |
| `Q6_K` | ~595 GB | Very good | Slower | Near-lossless |
| `Q8_0` | ~large | Best | Slowest | Highest fidelity quant |

### GPU Requirements

| Quant | Min GPUs | Recommended | Approx Cost/hr |
|-------|----------|-------------|-----------------|
| `Q2_K` | 2× H100 | 4× H100 | ~$8-15 |
| `Q4_K_M` | 4× H100 | 4× H100 | ~$10-15 |
| `Q6_K` | 4× H100 | 8× H100 | ~$15-25 |

## Model Details

- **Model**: Youssofal/Kimi-K2.6-Abliterated-Heretic-GGUF
- **Architecture**: 1T total params, 32B active (MoE, DeepSeek V3-style)
- **Context**: Up to 256K tokens (limited by VRAM)
- **Capabilities**: Uncensored text generation, tool calling, multimodal (vision)
- **Server**: llama.cpp `llama-server` with OpenAI-compatible API

## Ports

| Service | Port | Description |
|---------|------|-------------|
| Kimi Proxy | 9960 | On staging VM, forwards to RunPod |
| llama-server | 8000 | On RunPod pod, serves the model |

## State File

Pod and volume IDs are stored in `~/.kimi_runpod_state.json` so you don't
need to track them manually.
