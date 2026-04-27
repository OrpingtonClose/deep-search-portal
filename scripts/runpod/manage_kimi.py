#!/usr/bin/env python3
"""
Manage an on-demand Kimi K2.6 Heretic (GGUF) pod on RunPod.

Provisions a GPU pod running llama-server with the abliterated/uncensored
Kimi K2.6 model.  Designed for on-demand use: spin up when needed, stop
when done to save costs.

Usage:
    export RUNPOD_API_KEY=...

    # First time: create network volume + pod
    python manage_kimi.py create

    # Day-to-day: start/stop as needed
    python manage_kimi.py start
    python manage_kimi.py stop

    # Check status / get endpoint URL
    python manage_kimi.py status

    # Tear down everything
    python manage_kimi.py destroy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUNPOD_API = "https://rest.runpod.io/v1"
POD_NAME = "kimi-k26-heretic"

# Model configuration
# Q4_K_M is a good balance of quality and size (~584 GB).
# Change to Q2_K (~340 GB) for cheaper/smaller, Q6_K (~595 GB) for higher quality.
MODEL_REPO = "Youssofal/Kimi-K2.6-Abliterated-Heretic-GGUF"
MODEL_QUANT = os.getenv("KIMI_QUANT", "Q4_K_M")
MODEL_DIR = "/workspace/models/kimi-k26-heretic"

# GPU configuration — 4× H100 80GB is the recommended minimum for Q4_K_M.
# Falls back to A100 80GB if H100 not available.
GPU_TYPE_IDS = ["NVIDIA H100 80GB HBM3", "NVIDIA A100 80GB"]
GPU_COUNT = int(os.getenv("KIMI_GPU_COUNT", "4"))

# Disk: 800 GB container disk for model download headroom
CONTAINER_DISK_GB = int(os.getenv("KIMI_CONTAINER_DISK_GB", "100"))
VOLUME_DISK_GB = int(os.getenv("KIMI_VOLUME_DISK_GB", "800"))

# Network volume for persistent model storage (survives pod stop/restart)
VOLUME_NAME = "kimi-k26-heretic-models"

# Server config
SERVE_PORT = 8000
CONTEXT_SIZE = int(os.getenv("KIMI_CONTEXT_SIZE", "32768"))

# State file to remember pod/volume IDs across invocations
STATE_FILE = os.path.expanduser("~/.kimi_runpod_state.json")


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        print("ERROR: RUNPOD_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return key


def _request(method: str, path: str, data: dict | None = None) -> dict | list:
    url = f"{RUNPOD_API}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        print(f"API error {e.code}: {err_body[:500]}", file=sys.stderr)
        sys.exit(1)


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  State saved to {STATE_FILE}")


# ---------------------------------------------------------------------------
# Network Volume management
# ---------------------------------------------------------------------------

def _find_or_create_volume() -> str:
    """Find existing network volume or create a new one."""
    state = _load_state()
    if state.get("volume_id"):
        # Verify it still exists
        try:
            vol = _request("GET", f"/network-volumes/{state['volume_id']}")
            if vol.get("id"):
                print(f"  Using existing volume: {vol['id']} ({vol.get('name', '')})")
                return vol["id"]
        except SystemExit:
            print("  Previous volume not found, creating new one...")

    # List existing volumes to check if one already exists
    volumes = _request("GET", "/network-volumes")
    if isinstance(volumes, list):
        for vol in volumes:
            if vol.get("name") == VOLUME_NAME:
                print(f"  Found existing volume: {vol['id']}")
                state["volume_id"] = vol["id"]
                _save_state(state)
                return vol["id"]

    # Create new volume
    print(f"  Creating network volume '{VOLUME_NAME}' ({VOLUME_DISK_GB} GB)...")
    vol = _request("POST", "/network-volumes", {
        "name": VOLUME_NAME,
        "size": VOLUME_DISK_GB,
        "dataCenterId": "US-TX-3",  # Dallas — good H100 availability
    })
    vol_id = vol.get("id", "")
    if not vol_id:
        print(f"ERROR: Failed to create volume: {vol}", file=sys.stderr)
        sys.exit(1)
    print(f"  Volume created: {vol_id}")
    state["volume_id"] = vol_id
    _save_state(state)
    return vol_id


# ---------------------------------------------------------------------------
# Pod management
# ---------------------------------------------------------------------------

def _build_startup_cmd() -> str:
    """Build the Docker start command that downloads the model and starts llama-server."""
    # The startup script is embedded here so the pod is fully self-contained.
    # It downloads the GGUF model files if not already cached on the volume,
    # then starts llama-server with the OpenAI-compatible API.
    return " && ".join([
        # Install dependencies
        "pip install -q huggingface_hub[hf_transfer]",
        # Download model if not cached
        f"mkdir -p {MODEL_DIR}",
        f'if [ ! -f "{MODEL_DIR}/done.marker" ]; then '
        f'echo "Downloading {MODEL_REPO} ({MODEL_QUANT})..." && '
        f'HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download '
        f'"{MODEL_REPO}" '
        f'"Kimi-K2.6-Abliterated-Heretic-{MODEL_QUANT}/*" '
        f'"mmproj-Kimi-K2.6-Abliterated-Heretic.gguf" '
        f'--local-dir "{MODEL_DIR}" && '
        f'touch "{MODEL_DIR}/done.marker" && '
        f'echo "Download complete"; '
        f'else echo "Model already cached"; fi',
        # Find the GGUF file
        f'GGUF_FILE=$(find {MODEL_DIR} -name "*.gguf" -not -name "mmproj*" | sort | head -1)',
        f'MMPROJ_FILE=$(find {MODEL_DIR} -name "mmproj*.gguf" | sort | head -1)',
        # Install llama.cpp if not present
        'if ! command -v llama-server > /dev/null 2>&1; then '
        'echo "Installing llama.cpp..." && '
        'apt-get update -qq && apt-get install -y -qq cmake build-essential > /dev/null && '
        'git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp && '
        'cd /tmp/llama.cpp && '
        'cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;89;90" > /dev/null 2>&1 && '
        'cmake --build build --config Release -j$(nproc) > /dev/null 2>&1 && '
        'cp build/bin/llama-server /usr/local/bin/ && '
        'echo "llama.cpp installed"; fi',
        # Start llama-server
        'echo "Starting llama-server on port 8000..."',
        f'llama-server '
        f'-m "$GGUF_FILE" '
        f'--mmproj "$MMPROJ_FILE" '
        f'-ngl 999 '
        f'-c {CONTEXT_SIZE} '
        f'--host 0.0.0.0 '
        f'--port {SERVE_PORT} '
        f'--jinja '
        f'-fa '
        f'--threads $(nproc) '
        f'2>&1 | tee /workspace/llama_server.log',
    ])


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new Kimi pod (with network volume for model persistence)."""
    state = _load_state()

    if state.get("pod_id"):
        print(f"Pod already exists: {state['pod_id']}")
        print("Use 'start' to resume it, or 'destroy' first to recreate.")
        return

    print("Setting up Kimi K2.6 Heretic on RunPod...")

    # Create or find network volume
    volume_id = _find_or_create_volume()

    # Create pod
    print(f"\nCreating pod ({GPU_COUNT}× GPU, {MODEL_QUANT} quant)...")
    pod_data = {
        "name": POD_NAME,
        "imageName": "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu22.04",
        "gpuTypeIds": GPU_TYPE_IDS,
        "gpuTypePriority": "availability",
        "gpuCount": GPU_COUNT,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/workspace",
        "ports": [f"{SERVE_PORT}/http", "22/tcp"],
        "supportPublicIp": True,
        "cloudType": "COMMUNITY",  # cheaper than SECURE
        "dockerStartCmd": ["/bin/bash", "-c", _build_startup_cmd()],
        "env": {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(GPU_COUNT)),
        },
    }

    pod = _request("POST", "/pods", pod_data)
    pod_id = pod.get("id", "")
    if not pod_id:
        print(f"ERROR: Failed to create pod: {pod}", file=sys.stderr)
        sys.exit(1)

    state["pod_id"] = pod_id
    _save_state(state)

    print(f"\nPod created: {pod_id}")
    print(f"  Cost: ~${pod.get('costPerHr', '?')}/hr")
    print(f"  GPU: {pod.get('gpu', {}).get('displayName', '?')} × {GPU_COUNT}")
    print(f"\nThe pod is now downloading the model and starting llama-server.")
    print(f"This may take 15-30 minutes for the first run (model download).")
    print(f"\nRun 'python manage_kimi.py status' to check progress.")
    print(f"Once ready, the endpoint will be at:")
    print(f"  https://{pod_id}-{SERVE_PORT}.proxy.runpod.net/v1")


def cmd_start(args: argparse.Namespace) -> None:
    """Start (resume) a stopped Kimi pod."""
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod found. Run 'create' first.")
        return

    print(f"Starting pod {pod_id}...")
    _request("POST", f"/pods/{pod_id}/start")
    print("Pod start requested.")
    print(f"  Endpoint (once ready): https://{pod_id}-{SERVE_PORT}.proxy.runpod.net/v1")
    print("  Run 'status' to check when it's ready.")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the Kimi pod (preserves volume, no GPU charges)."""
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod found.")
        return

    print(f"Stopping pod {pod_id}...")
    _request("POST", f"/pods/{pod_id}/stop")
    print("Pod stopped. Volume preserved — model won't need re-downloading.")
    print("  Volume storage costs continue (~$0.07/GB/month).")
    print("  Run 'start' to resume.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current pod status and endpoint URL."""
    state = _load_state()
    pod_id = state.get("pod_id")
    volume_id = state.get("volume_id")

    if not pod_id and not volume_id:
        print("No Kimi infrastructure found. Run 'create' first.")
        return

    if volume_id:
        try:
            vol = _request("GET", f"/network-volumes/{volume_id}")
            print(f"Volume: {vol.get('name', '?')} ({vol.get('size', '?')} GB)")
            print(f"  ID: {volume_id}")
            print(f"  DC: {vol.get('dataCenterId', '?')}")
        except SystemExit:
            print(f"Volume {volume_id} — not found (may have been deleted)")

    if pod_id:
        try:
            pod = _request("GET", f"/pods/{pod_id}")
            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime", {}) or {}
            uptime = runtime.get("uptimeInSeconds", 0)

            print(f"\nPod: {pod.get('name', '?')}")
            print(f"  ID: {pod_id}")
            print(f"  Status: {status}")
            print(f"  GPU: {pod.get('gpu', {}).get('displayName', '?')} × {pod.get('gpu', {}).get('count', '?')}")
            print(f"  Cost: ${pod.get('costPerHr', '?')}/hr")
            if uptime:
                hours = uptime / 3600
                print(f"  Uptime: {hours:.1f}h")

            if status == "RUNNING":
                endpoint = f"https://{pod_id}-{SERVE_PORT}.proxy.runpod.net"
                print(f"\n  Endpoint: {endpoint}/v1")
                print(f"  Health:   {endpoint}/health")
                print(f"\n  Test with:")
                print(f'    curl {endpoint}/v1/models')

                # Try to check if server is actually responding
                try:
                    health_req = urllib.request.Request(
                        f"{endpoint}/health",
                        method="GET",
                    )
                    with urllib.request.urlopen(health_req, timeout=10) as resp:
                        print(f"\n  Server health: OK ({resp.status})")
                except Exception:
                    print(f"\n  Server health: Not responding yet (model may still be loading)")
            elif status == "EXITED":
                print("\n  Pod is stopped. Run 'start' to resume.")
        except SystemExit:
            print(f"Pod {pod_id} — not found")


def cmd_destroy(args: argparse.Namespace) -> None:
    """Destroy the pod and optionally the network volume."""
    state = _load_state()
    pod_id = state.get("pod_id")
    volume_id = state.get("volume_id")

    if pod_id:
        print(f"Deleting pod {pod_id}...")
        try:
            _request("DELETE", f"/pods/{pod_id}")
            print("  Pod deleted.")
        except SystemExit:
            print("  Pod not found (may already be deleted).")
        state.pop("pod_id", None)

    if volume_id and args.include_volume:
        print(f"Deleting volume {volume_id}...")
        try:
            _request("DELETE", f"/network-volumes/{volume_id}")
            print("  Volume deleted (model cache removed).")
        except SystemExit:
            print("  Volume not found (may already be deleted).")
        state.pop("volume_id", None)
    elif volume_id:
        print(f"Volume {volume_id} preserved (model still cached).")
        print("  Add --include-volume to also delete the volume.")

    _save_state(state)
    print("Done.")


def cmd_endpoint(args: argparse.Namespace) -> None:
    """Print just the endpoint URL (for scripting / proxy config)."""
    state = _load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod found.", file=sys.stderr)
        sys.exit(1)
    print(f"https://{pod_id}-{SERVE_PORT}.proxy.runpod.net/v1")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Kimi K2.6 Heretic on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_kimi.py create          # First-time setup
  python manage_kimi.py start           # Resume stopped pod
  python manage_kimi.py status          # Check status + endpoint
  python manage_kimi.py stop            # Stop to save costs
  python manage_kimi.py destroy         # Remove pod (keep volume)
  python manage_kimi.py destroy --include-volume  # Remove everything

Environment variables:
  RUNPOD_API_KEY          Required. Your RunPod API key.
  KIMI_QUANT              GGUF quantization (default: Q4_K_M)
  KIMI_GPU_COUNT          Number of GPUs (default: 4)
  KIMI_CONTEXT_SIZE       Context window size (default: 32768)
  KIMI_CONTAINER_DISK_GB  Container disk size (default: 100)
  KIMI_VOLUME_DISK_GB     Network volume size (default: 800)
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create", help="Create pod + volume (first time)")
    sub.add_parser("start", help="Start/resume a stopped pod")
    sub.add_parser("stop", help="Stop pod (preserves volume)")
    sub.add_parser("status", help="Show pod status + endpoint URL")
    sub.add_parser("endpoint", help="Print endpoint URL (for scripting)")

    destroy_p = sub.add_parser("destroy", help="Delete pod (and optionally volume)")
    destroy_p.add_argument(
        "--include-volume",
        action="store_true",
        help="Also delete the network volume (removes cached model)",
    )

    args = parser.parse_args()

    cmds = {
        "create": cmd_create,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "destroy": cmd_destroy,
        "endpoint": cmd_endpoint,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
