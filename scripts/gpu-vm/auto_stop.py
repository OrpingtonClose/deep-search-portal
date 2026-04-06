#!/usr/bin/env python3
"""
Auto-stop daemon for GPU inference VM on Vast.ai.

Monitors the inference server for activity. When no requests have been
processed for IDLE_TIMEOUT seconds, stops the Vast.ai instance via API.

Much more reliable than counting TCP connections — uses the server's
own metrics endpoint to detect actual inference activity.
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [auto-stop] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto-stop")


def get_server_status(port: int) -> dict:
    """Query the inference server for current status.

    Tries multiple endpoints in priority order:
    1. /metrics — Prometheus endpoint (vLLM or SGLang); gives exact active request count.
    2. /health  — generic health check; confirms server is alive.
    3. /v1/models — OpenAI-compatible list; confirms server is alive.

    Returns a dict with keys:
        alive  (bool): whether the server responded at all
        active (int):  number of active requests (-1 = unknown)
        source (str):  which endpoint answered
    """
    for path in ["/metrics", "/health", "/v1/models"]:
        try:
            url = f"http://localhost:{port}{path}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                if path == "/metrics":
                    # Parse Prometheus metrics for active requests.
                    # Supports both vLLM and SGLang metric names.
                    running = 0
                    waiting = 0
                    found_known_metric = False
                    for line in body.split("\n"):
                        if line.startswith("#"):
                            continue
                        # vLLM metrics (may use "vllm:" or "vllm_" prefix)
                        if "vllm:num_requests_running" in line or "vllm_num_requests_running" in line:
                            running = int(float(line.split()[-1]))
                            found_known_metric = True
                        if "vllm:num_requests_waiting" in line or "vllm_num_requests_waiting" in line:
                            waiting = int(float(line.split()[-1]))
                            found_known_metric = True
                        # SGLang metrics
                        if "sglang:num_running_reqs" in line or "sglang_num_running_reqs" in line:
                            running = int(float(line.split()[-1]))
                            found_known_metric = True
                        if "sglang:num_waiting_reqs" in line or "sglang_num_waiting_reqs" in line:
                            waiting = int(float(line.split()[-1]))
                            found_known_metric = True
                    if found_known_metric:
                        return {"alive": True, "active": running + waiting, "source": "metrics"}
                    # /metrics responded but no recognized metric names —
                    # conservatively treat as unknown to avoid false idle.
                    return {"alive": True, "active": -1, "source": "metrics-unknown"}
                elif path == "/health":
                    return {"alive": True, "active": -1, "source": "health"}
                else:
                    return {"alive": True, "active": -1, "source": "models"}
        except Exception:
            continue
    return {"alive": False, "active": 0, "source": "none"}


def stop_instance(api_key: str) -> bool:
    """Stop this Vast.ai instance via the API."""
    instance_id = os.getenv("VAST_CONTAINERLABEL", "")
    if not instance_id:
        # Try to get from /etc/vast_containerlabel
        try:
            with open("/etc/vast_containerlabel", "r") as f:
                instance_id = f.read().strip()
        except FileNotFoundError:
            pass

    if not instance_id:
        log.error("Cannot determine Vast.ai instance ID. Set VAST_CONTAINERLABEL env var.")
        return False

    log.info("Stopping Vast.ai instance %s...", instance_id)
    try:
        url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
        data = json.dumps({"state": "stopped"}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method="PUT",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            log.info("Stop response: %s", resp.status)
            return resp.status in (200, 201, 204)
    except Exception as e:
        log.error("Failed to stop instance: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto-stop GPU VM on idle")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=1200, help="Idle timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between checks")
    parser.add_argument("--vast-api-key", type=str, default=os.getenv("VAST_API_KEY", ""))
    parser.add_argument("--startup-grace", type=int, default=3600,
                        help="Seconds to wait for server to come alive before allowing shutdown (default: 3600)")
    parser.add_argument("--dry-run", action="store_true", help="Log but don't actually stop")
    args = parser.parse_args()

    if not args.vast_api_key:
        log.error("VAST_API_KEY required (--vast-api-key or env var)")
        sys.exit(1)

    last_activity = time.time()
    daemon_start = time.time()
    server_seen_alive = False
    log.info(
        "Auto-stop daemon started. Port=%d, timeout=%ds, poll=%ds, startup_grace=%ds",
        args.port,
        args.timeout,
        args.poll_interval,
        args.startup_grace,
    )

    while True:
        time.sleep(args.poll_interval)

        status = get_server_status(args.port)

        if not status["alive"]:
            if not server_seen_alive:
                elapsed = time.time() - daemon_start
                log.info("Server not yet alive (%.0fs / %ds grace)", elapsed, args.startup_grace)
            else:
                log.warning("Server not responding — treating as idle")
        elif status["active"] > 0:
            if not server_seen_alive:
                log.info("Server is alive (first seen with active requests)")
            server_seen_alive = True
            last_activity = time.time()
            log.debug("Active requests: %d", status["active"])
        elif status["active"] == 0:
            if not server_seen_alive:
                # First time seeing server alive — reset last_activity so idle
                # timeout starts from NOW, not from daemon start time.
                log.info("Server is alive (first seen idle) — resetting idle timer")
                last_activity = time.time()
            server_seen_alive = True
            idle_secs = time.time() - last_activity
            log.info("Idle for %.0fs / %ds", idle_secs, args.timeout)
        else:
            # active == -1 means we couldn't determine (health-only endpoint)
            # Conservatively treat as active
            if not server_seen_alive:
                log.info("Server is alive (first seen, unknown activity) — resetting idle timer")
                last_activity = time.time()
            server_seen_alive = True
            last_activity = time.time()

        # During startup grace period, don't shut down if server has never been alive.
        # This prevents killing the VM during model download + loading.
        idle_duration = time.time() - last_activity
        if not server_seen_alive and (time.time() - daemon_start) < args.startup_grace:
            continue

        if idle_duration >= args.timeout:
            log.info("Idle timeout reached (%.0fs >= %ds)", idle_duration, args.timeout)
            if args.dry_run:
                log.info("[DRY RUN] Would stop instance now")
                last_activity = time.time()  # Reset to avoid repeated logs
            else:
                if stop_instance(args.vast_api_key):
                    log.info("Instance stop requested. Exiting.")
                    sys.exit(0)
                else:
                    log.error("Failed to stop instance. Will retry in 60s.")
                    time.sleep(60)


if __name__ == "__main__":
    main()
