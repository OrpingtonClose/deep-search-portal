#!/usr/bin/env python3
"""Live integration test -- sends a real prompt through the running VM proxy,
captures the full SSE stream, fetches server-side metrics and JSONL logs,
then generates:
  1. An HTML report with Mermaid graphs for human consumption
  2. A detailed JSON report with every observable data point for LLM analysis

NO MOCKS. This tests the live system end-to-end.

Environment variables:
  PROXY_URL       - Full URL to proxy (default: uses SSH tunnel to VM)
  VAST_SSH_HOST   - SSH host (default: ssh5.vast.ai)
  VAST_SSH_PORT   - SSH port (default: 18770)
  PROXY_PORT      - Proxy port inside VM (default: 9300)
  TEST_QUERY      - Prompt to send (default: microplastics question)
  REPORT_DIR      - Output directory (default: tests/)
  STREAM_TIMEOUT  - Max seconds to wait for stream (default: 1800)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VAST_SSH_HOST = os.getenv("VAST_SSH_HOST", "ssh5.vast.ai")
VAST_SSH_PORT = int(os.getenv("VAST_SSH_PORT", "18770"))
PROXY_PORT = int(os.getenv("PROXY_PORT", "9300"))
PROXY_URL = os.getenv("PROXY_URL", "")
TEST_QUERY = os.getenv(
    "TEST_QUERY",
    "What are the health risks of microplastics in drinking water?",
)
REPORT_DIR = os.getenv("REPORT_DIR", os.path.dirname(__file__) or ".")
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "1800"))
LOCAL_TUNNEL_PORT = 19300

# Add tests dir to path for the report generator
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# SSH Tunnel
# ---------------------------------------------------------------------------

class SSHTunnel:
    """Manage an SSH port-forward to the VM proxy."""

    def __init__(self, ssh_host, ssh_port, remote_port, local_port):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.remote_port = remote_port
        self.local_port = local_port
        self._proc = None

    def open(self):
        """Open tunnel; returns the local URL."""
        cmd = [
            "ssh", "-N", "-L",
            f"{self.local_port}:127.0.0.1:{self.remote_port}",
            f"root@{self.ssh_host}",
            "-p", str(self.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15",
        ]
        print(f"[tunnel] Opening SSH tunnel: localhost:{self.local_port} -> "
              f"{self.ssh_host}:{self.ssh_port} -> 127.0.0.1:{self.remote_port}")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Give the tunnel a moment to establish
        time.sleep(3)
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"SSH tunnel failed to start: {stderr}")
        url = f"http://127.0.0.1:{self.local_port}"
        print(f"[tunnel] Tunnel established: {url}")
        return url

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print("[tunnel] Tunnel closed")


# ---------------------------------------------------------------------------
# SSE Stream Capture
# ---------------------------------------------------------------------------

def parse_sse_line(line):
    """Parse a single SSE 'data: ...' line into a structured dict."""
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    payload = line[6:]
    if payload == "[DONE]":
        return {"type": "done", "raw": payload}
    try:
        obj = json.loads(payload)
        return {"type": "chunk", "parsed": obj, "raw": payload}
    except json.JSONDecodeError:
        return {"type": "unparsed", "raw": payload}


def capture_sse_stream(base_url, query, timeout=STREAM_TIMEOUT):
    """Send a real prompt and capture the full SSE stream with timestamps.

    Returns a dict with:
      - events: list of {wall_ts, elapsed, type, parsed, raw, content_delta}
      - full_content: accumulated text
      - think_content: accumulated <think> text
      - response_id: the chatcmpl-pdr-... id
      - model: model name from response
      - start_time / end_time: ISO timestamps
      - duration_secs: total wall-clock time
      - error: any error string (empty if ok)
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "deep-research",
        "messages": [{"role": "user", "content": query}],
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-integration",
    }

    print(f"[stream] POST {url}")
    print(f"[stream] Query: {query[:80]}...")
    print(f"[stream] Timeout: {timeout}s")

    result = {
        "events": [],
        "full_content": "",
        "think_content": "",
        "response_id": "",
        "model": "",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": "",
        "duration_secs": 0,
        "error": "",
    }

    t0 = time.monotonic()
    in_think = False
    event_count = 0
    last_print = t0

    try:
        resp = requests.post(url, json=payload, headers=headers,
                             stream=True, timeout=(30, timeout))
        resp.raise_for_status()

        for raw_line in resp.iter_lines(decode_unicode=True):
            now = time.monotonic()
            elapsed = now - t0

            parsed = parse_sse_line(raw_line)
            if parsed is None:
                continue

            event_count += 1
            event_record = {
                "wall_ts": datetime.now(timezone.utc).isoformat(),
                "elapsed": round(elapsed, 4),
                "type": parsed["type"],
                "raw": parsed.get("raw", ""),
                "content_delta": "",
            }

            if parsed["type"] == "done":
                result["events"].append(event_record)
                break

            if parsed["type"] == "chunk":
                obj = parsed["parsed"]
                # Extract response ID and model
                if not result["response_id"] and obj.get("id"):
                    result["response_id"] = obj["id"]
                if not result["model"] and obj.get("model"):
                    result["model"] = obj["model"]

                # Extract content delta
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        event_record["content_delta"] = content
                        # Track think vs normal content
                        if "<think>" in content:
                            in_think = True
                        if "</think>" in content:
                            in_think = False
                            # Split content at </think>
                            parts = content.split("</think>", 1)
                            result["think_content"] += parts[0]
                            if len(parts) > 1:
                                result["full_content"] += parts[1]
                        elif in_think:
                            result["think_content"] += content
                        else:
                            result["full_content"] += content

            result["events"].append(event_record)

            # Progress reporting every 30 seconds
            if now - last_print > 30:
                print(f"[stream] {elapsed:.0f}s elapsed, {event_count} events, "
                      f"{len(result['full_content'])} chars content, "
                      f"{len(result['think_content'])} chars think")
                last_print = now

    except requests.exceptions.Timeout:
        result["error"] = f"Stream timed out after {timeout}s"
        print(f"[stream] TIMEOUT after {timeout}s")
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection error: {e}"
        print(f"[stream] CONNECTION ERROR: {e}")
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        print(f"[stream] ERROR: {e}")

    t1 = time.monotonic()
    result["end_time"] = datetime.now(timezone.utc).isoformat()
    result["duration_secs"] = round(t1 - t0, 4)

    print(f"[stream] Completed in {result['duration_secs']:.1f}s, "
          f"{len(result['events'])} events, "
          f"{len(result['full_content'])} chars content")

    return result


# ---------------------------------------------------------------------------
# Phase & Curated Event Detection
# ---------------------------------------------------------------------------

PHASE_PATTERNS = [
    (r"\[Phase\s*(\d+)[:\s]+(.*?)\]", "phase"),
    (r"\[depth\s+(\d+)\]", "depth"),
    (r"Spawning\s+(\d+)\s+sub-investigation", "spawn"),
    (r"Branch\s+complete", "branch_complete"),
    (r"All\s+branches?\s+complete", "all_branches_complete"),
    (r"Synthesizing\s+final", "synthesis_start"),
]


def detect_phases(capture):
    """Detect phase transitions from the think block and content."""
    phases = []
    combined = capture.get("think_content", "") + capture.get("full_content", "")

    for pattern, ptype in PHASE_PATTERNS:
        for match in re.finditer(pattern, combined, re.IGNORECASE):
            phases.append({
                "type": ptype,
                "match": match.group(0),
                "groups": list(match.groups()),
                "position": match.start(),
            })

    # Also detect phases from SSE events with timestamps
    for evt in capture.get("events", []):
        content = evt.get("content_delta", "")
        if not content:
            continue
        for pattern, ptype in PHASE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                phases.append({
                    "type": ptype,
                    "match": match.group(0),
                    "groups": list(match.groups()),
                    "elapsed": evt.get("elapsed", 0),
                })

    return phases


def detect_curated_events(capture):
    """Detect curated thought messages from the SSE stream.

    Curated events appear as content deltas that contain heartbeat-style
    messages (findings, status updates, branch info).
    """
    curated = []

    for evt in capture.get("events", []):
        content = evt.get("content_delta", "")
        if not content:
            continue

        # Curated events are typically short bursts of text between think blocks
        patterns = [
            (r"(?:Investigating|Researching|Exploring|Analyzing)\s+.{10,}", "status"),
            (r"(?:Found|Discovered|Identified)\s+.{10,}", "finding"),
            (r"(?:Spawning|Branching|Following)\s+.{10,}", "branch"),
            (r"(?:Verifying|Cross-referencing|Checking)\s+.{10,}", "verify"),
            (r"\[.*?\]\s+.{10,}", "phase_message"),
        ]

        for pattern, evt_type in patterns:
            for match in re.finditer(pattern, content):
                curated.append({
                    "type": evt_type,
                    "text": match.group(0)[:300],
                    "elapsed": evt.get("elapsed", 0),
                    "wall_ts": evt.get("wall_ts", ""),
                })

    return curated


# ---------------------------------------------------------------------------
# Server-side Data Fetch
# ---------------------------------------------------------------------------

def extract_session_id(capture):
    """Extract req-XXXXXXXX session ID from the response ID.

    Response ID format: chatcmpl-pdr-XXXXXXXXXXXX
    Session ID format: req-XXXXXXXX (first 8 chars of the hash)
    """
    resp_id = capture.get("response_id", "")
    if not resp_id:
        return ""

    # Try extracting from chatcmpl-pdr-XXXX format
    match = re.search(r"chatcmpl-pdr-([a-f0-9]+)", resp_id)
    if match:
        hash_part = match.group(1)[:8]
        return f"req-{hash_part}"

    # Fallback: use the full ID
    return resp_id


def fetch_server_metrics(base_url, session_id):
    """Fetch structured metrics JSON from the server."""
    if not session_id:
        return {"error": "no session_id"}

    url = f"{base_url}/research/metrics/{session_id}"
    print(f"[metrics] GET {url}")
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[metrics] Got metrics: {len(json.dumps(data))} bytes")
            return data
        else:
            print(f"[metrics] HTTP {resp.status_code}: {resp.text[:200]}")
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:500]}
    except Exception as e:
        print(f"[metrics] Error: {e}")
        return {"error": str(e)}


def fetch_server_jsonl(ssh_host, ssh_port, session_id):
    """Fetch the JSONL event log from the server via SSH."""
    if not session_id:
        return []

    jsonl_path = f"/opt/persistent_research_logs/jsonl/{session_id}.jsonl"
    cmd = [
        "ssh", f"root@{ssh_host}",
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        f"cat {jsonl_path} 2>/dev/null || echo ''",
    ]
    print(f"[jsonl] Fetching {jsonl_path} via SSH")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split("\n")
        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"raw_line": line})
        print(f"[jsonl] Got {len(events)} events")
        return events
    except Exception as e:
        print(f"[jsonl] Error: {e}")
        return [{"error": str(e)}]


def fetch_server_html_report(ssh_host, ssh_port, session_id):
    """Fetch the server-generated HTML report via SSH."""
    if not session_id:
        return ""
    report_path = f"/opt/persistent_research_logs/reports/{session_id}.html"
    cmd = [
        "ssh", f"root@{ssh_host}",
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        f"cat {report_path} 2>/dev/null || echo ''",
    ]
    print(f"[report] Fetching {report_path} via SSH")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        html = result.stdout.strip()
        if html:
            print(f"[report] Got server HTML report: {len(html)} bytes")
        else:
            print("[report] No server HTML report found")
        return html
    except Exception as e:
        print(f"[report] Error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_stream(capture, phases, curated):
    """Comprehensive analysis of the captured stream data."""
    events = capture.get("events", [])
    content = capture.get("full_content", "")
    think = capture.get("think_content", "")

    # Timing analysis
    elapsed_values = [e["elapsed"] for e in events if "elapsed" in e]
    content_deltas = [e for e in events if e.get("content_delta")]

    # Inter-event timing (detect stalls)
    gaps = []
    for i in range(1, len(elapsed_values)):
        gap = elapsed_values[i] - elapsed_values[i - 1]
        if gap > 2.0:  # gaps > 2s are notable
            gaps.append({
                "from_elapsed": round(elapsed_values[i - 1], 2),
                "to_elapsed": round(elapsed_values[i], 2),
                "gap_secs": round(gap, 2),
                "event_index": i,
            })

    # Content metrics
    words = content.split()
    sentences = re.split(r"[.!?]+", content)
    paragraphs = [p for p in content.split("\n\n") if p.strip()]

    # Think block analysis
    think_words = think.split()
    think_phases_detected = len([p for p in phases if p["type"] == "phase"])

    # Curated event analysis
    curated_by_type = {}
    curated_lengths = []
    truncated_events = []
    for c in curated:
        t = c.get("type", "unknown")
        curated_by_type[t] = curated_by_type.get(t, 0) + 1
        text = c.get("text", "")
        curated_lengths.append(len(text))
        # Detect mid-word truncation
        if text and text[-1] not in ".!?)]}\"' " and len(text) > 100:
            truncated_events.append(text[-30:])

    # Quality signals
    issues = []
    if not content.strip():
        issues.append({"severity": "critical", "issue": "Empty final content"})
    if content and len(content) < 200:
        issues.append({"severity": "warning", "issue": f"Very short content: {len(content)} chars"})
    if capture.get("error"):
        issues.append({"severity": "critical", "issue": f"Stream error: {capture['error']}"})
    if truncated_events:
        issues.append({
            "severity": "warning",
            "issue": f"{len(truncated_events)} curated events appear truncated mid-word",
            "examples": truncated_events[:3],
        })
    if gaps:
        max_gap = max(g["gap_secs"] for g in gaps)
        issues.append({
            "severity": "info" if max_gap < 30 else "warning",
            "issue": f"{len(gaps)} notable gaps in stream (max {max_gap:.1f}s)",
        })
    if think_phases_detected == 0 and len(think) > 100:
        issues.append({
            "severity": "warning",
            "issue": "Think block has content but no phase markers detected",
        })

    # Compute throughput
    duration = capture.get("duration_secs", 1)
    events_per_sec = len(events) / max(duration, 0.01)
    chars_per_sec = len(content) / max(duration, 0.01)

    return {
        "timing": {
            "total_duration_secs": capture.get("duration_secs", 0),
            "start_time": capture.get("start_time", ""),
            "end_time": capture.get("end_time", ""),
            "time_to_first_content": (
                content_deltas[0]["elapsed"] if content_deltas else None
            ),
            "time_to_last_content": (
                content_deltas[-1]["elapsed"] if content_deltas else None
            ),
            "notable_gaps": gaps,
            "events_per_second": round(events_per_sec, 2),
            "chars_per_second": round(chars_per_sec, 2),
        },
        "content": {
            "total_chars": len(content),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "total_paragraphs": len(paragraphs),
            "think_chars": len(think),
            "think_words": len(think_words),
        },
        "events": {
            "total_sse_events": len(events),
            "content_bearing_events": len(content_deltas),
            "done_received": any(e.get("type") == "done" for e in events),
        },
        "phases": {
            "count": len(phases),
            "detected": [
                {"type": p["type"], "match": p["match"][:100]}
                for p in phases
            ],
        },
        "curated_events": {
            "total": len(curated),
            "by_type": curated_by_type,
            "avg_length": (
                round(sum(curated_lengths) / len(curated_lengths), 1)
                if curated_lengths else 0
            ),
            "max_length": max(curated_lengths) if curated_lengths else 0,
            "truncated_count": len(truncated_events),
        },
        "quality": {
            "issues": issues,
            "issue_count": len(issues),
            "critical_count": sum(1 for i in issues if i["severity"] == "critical"),
            "warning_count": sum(1 for i in issues if i["severity"] == "warning"),
        },
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(capture, analysis, server_metrics, server_jsonl):
    """Validate system invariants. Returns list of {name, passed, reason}."""
    results = []

    def check(name, condition, reason=""):
        results.append({
            "name": name,
            "passed": bool(condition),
            "reason": reason if not condition else "",
        })

    # 1. Stream completed without error
    check(
        "Stream completed without error",
        not capture.get("error"),
        capture.get("error", ""),
    )

    # 2. Got [DONE] marker
    check(
        "Received [DONE] marker",
        analysis["events"]["done_received"],
        "Stream ended without [DONE]",
    )

    # 3. Non-empty final content
    check(
        "Final answer is non-empty",
        analysis["content"]["total_chars"] > 0,
        f"Content length: {analysis['content']['total_chars']}",
    )

    # 4. Content has substance (>200 chars)
    check(
        "Final answer has substance (>200 chars)",
        analysis["content"]["total_chars"] > 200,
        f"Only {analysis['content']['total_chars']} chars",
    )

    # 5. Response ID was received
    check(
        "Response ID received",
        bool(capture.get("response_id")),
        "No response ID in stream",
    )

    # 6. Events were received
    check(
        "SSE events received (>10)",
        analysis["events"]["total_sse_events"] > 10,
        f"Only {analysis['events']['total_sse_events']} events",
    )

    # 7. Think block present
    check(
        "Think block present",
        analysis["content"]["think_chars"] > 0,
        "No think content detected",
    )

    # 8. Phases detected
    check(
        "At least one phase detected",
        analysis["phases"]["count"] > 0,
        "No phase markers found",
    )

    # 9. No critical issues
    check(
        "No critical quality issues",
        analysis["quality"]["critical_count"] == 0,
        f"{analysis['quality']['critical_count']} critical issues",
    )

    # 10. Curated events emitted
    check(
        "Curated events emitted",
        analysis["curated_events"]["total"] > 0,
        "No curated thought events detected",
    )

    # 11. No mid-word truncation in curated events
    check(
        "No mid-word truncation in curated events",
        analysis["curated_events"]["truncated_count"] == 0,
        f"{analysis['curated_events']['truncated_count']} truncated events",
    )

    # 12. Reasonable duration (not too fast = likely error, not too slow)
    dur = capture.get("duration_secs", 0)
    check(
        "Duration is reasonable (30s-1800s)",
        30 < dur < 1800,
        f"Duration: {dur:.1f}s",
    )

    # 13. Server metrics available
    check(
        "Server metrics endpoint responded",
        "error" not in server_metrics,
        server_metrics.get("error", ""),
    )

    # 14. JSONL log available
    has_jsonl = len(server_jsonl) > 0 and not (
        len(server_jsonl) == 1 and "error" in server_jsonl[0]
    )
    check(
        "Server JSONL log available",
        has_jsonl,
        "JSONL log empty or errored",
    )

    # 15. No TransferEncodingError (the bug the user reported)
    check(
        "No TransferEncodingError in stream",
        "TransferEncodingError" not in capture.get("error", ""),
        capture.get("error", ""),
    )

    # 16. Content does not contain moralizing boilerplate
    moral_patterns = [
        "I cannot", "I must emphasize", "Please consult",
        "ethical considerations", "I should note that",
    ]
    content_lower = capture.get("full_content", "").lower()
    moral_found = [p for p in moral_patterns if p.lower() in content_lower]
    check(
        "No moralizing boilerplate in output",
        len(moral_found) == 0,
        f"Found: {moral_found[:3]}",
    )

    return results


# ---------------------------------------------------------------------------
# LLM Report Builder
# ---------------------------------------------------------------------------

def build_llm_report(capture, analysis, assertions,
                     server_metrics, server_jsonl,
                     phases, curated):
    """Build comprehensive JSON report for LLM consumption.

    Every observable data point is included so an LLM can analyze
    the system performance and propose improvements.
    """
    return {
        "_meta": {
            "report_type": "live_integration_test",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_query": TEST_QUERY,
            "proxy_url": PROXY_URL or f"tunnel:{VAST_SSH_HOST}:{VAST_SSH_PORT}",
            "version": "1.0.0",
        },
        "stream_capture": {
            "response_id": capture.get("response_id", ""),
            "model": capture.get("model", ""),
            "duration_secs": capture.get("duration_secs", 0),
            "start_time": capture.get("start_time", ""),
            "end_time": capture.get("end_time", ""),
            "error": capture.get("error", ""),
            "total_events": len(capture.get("events", [])),
            "content_length_chars": len(capture.get("full_content", "")),
            "think_length_chars": len(capture.get("think_content", "")),
            "first_50_events": capture.get("events", [])[:50],
            "last_20_events": capture.get("events", [])[-20:],
            "content_preview": capture.get("full_content", "")[:2000],
            "think_preview": capture.get("think_content", "")[:2000],
        },
        "analysis": analysis,
        "phases": phases,
        "curated_events": curated,
        "assertions": {
            "results": assertions,
            "total": len(assertions),
            "passed": sum(1 for a in assertions if a["passed"]),
            "failed": sum(1 for a in assertions if not a["passed"]),
            "pass_rate": (
                round(
                    sum(1 for a in assertions if a["passed"])
                    / max(len(assertions), 1) * 100, 1
                )
            ),
        },
        "server_metrics": server_metrics,
        "server_jsonl": {
            "total_events": len(server_jsonl),
            "events": server_jsonl[:200],
            "event_types": _count_jsonl_types(server_jsonl),
        },
        "recommendations": _generate_recommendations(
            analysis, assertions, server_metrics
        ),
    }


def _count_jsonl_types(events):
    counts = {}
    for e in events:
        t = e.get("event", e.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1
    return counts


def _generate_recommendations(analysis, assertions, server_metrics):
    """Auto-generate improvement recommendations based on observed data."""
    recs = []

    # Timing recommendations
    timing = analysis.get("timing", {})
    ttfc = timing.get("time_to_first_content")
    if ttfc is not None and ttfc > 10:
        recs.append({
            "category": "latency",
            "severity": "high",
            "finding": f"Time to first content: {ttfc:.1f}s",
            "suggestion": (
                "Consider streaming a status message earlier or "
                "optimizing the retrieve phase"
            ),
        })

    gaps = timing.get("notable_gaps", [])
    if gaps:
        max_gap = max(g["gap_secs"] for g in gaps)
        if max_gap > 30:
            recs.append({
                "category": "latency",
                "severity": "high",
                "finding": f"Max gap between events: {max_gap:.1f}s",
                "suggestion": (
                    "Investigate what causes long pauses; consider more "
                    "frequent heartbeat emissions"
                ),
            })

    # Content quality
    content = analysis.get("content", {})
    if content.get("total_words", 0) < 100:
        recs.append({
            "category": "quality",
            "severity": "high",
            "finding": f"Final answer only {content.get('total_words', 0)} words",
            "suggestion": (
                "The synthesis phase may be too aggressive in summarizing; "
                "check the synthesis prompt"
            ),
        })

    # Curated events
    ce = analysis.get("curated_events", {})
    if ce.get("truncated_count", 0) > 0:
        recs.append({
            "category": "ux",
            "severity": "medium",
            "finding": f"{ce['truncated_count']} curated events appear truncated",
            "suggestion": (
                "Verify _truncate_at_word() is working; "
                "check max_len parameter"
            ),
        })

    if ce.get("total", 0) == 0:
        recs.append({
            "category": "ux",
            "severity": "medium",
            "finding": "No curated events detected in stream",
            "suggestion": (
                "The heartbeat/curated event system may not be emitting; "
                "check collector wiring"
            ),
        })

    # Assertion failures
    failed = [a for a in assertions if not a["passed"]]
    for a in failed:
        recs.append({
            "category": "assertion_failure",
            "severity": "high",
            "finding": f"FAILED: {a['name']}",
            "suggestion": a.get("reason", "Investigate this failure"),
        })

    # Server metrics recommendations
    if isinstance(server_metrics, dict) and "error" not in server_metrics:
        node_timings = server_metrics.get("node_timings", {})
        for node, timing_data in node_timings.items():
            if isinstance(timing_data, dict):
                dur = timing_data.get("duration_secs", 0)
                if dur > 120:
                    recs.append({
                        "category": "performance",
                        "severity": "medium",
                        "finding": f"Node '{node}' took {dur:.1f}s",
                        "suggestion": (
                            f"Investigate why {node} is slow; "
                            "consider parallelization or timeout"
                        ),
                    })

    return recs


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_integration_test():
    """Orchestrate the full live integration test."""
    print("=" * 70)
    print("LIVE INTEGRATION TEST")
    print(f"Query: {TEST_QUERY}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    tunnel = None
    base_url = PROXY_URL

    try:
        # 1. Establish connection
        if not base_url:
            print("\n--- Step 1: Opening SSH tunnel ---")
            tunnel = SSHTunnel(VAST_SSH_HOST, VAST_SSH_PORT,
                               PROXY_PORT, LOCAL_TUNNEL_PORT)
            base_url = tunnel.open()
        else:
            print(f"\n--- Step 1: Using provided PROXY_URL: {base_url} ---")

        # 2. Health check
        print("\n--- Step 2: Health check ---")
        try:
            hc = requests.get(f"{base_url}/health", timeout=10)
            print(f"[health] Status: {hc.status_code}, Body: {hc.text[:200]}")
        except Exception as e:
            print(f"[health] WARNING: Health check failed: {e}")
            print("[health] Proceeding anyway...")

        # 3. Capture SSE stream
        print("\n--- Step 3: Sending prompt and capturing SSE stream ---")
        capture = capture_sse_stream(base_url, TEST_QUERY)

        # 4. Extract session ID
        print("\n--- Step 4: Extracting session ID ---")
        session_id = extract_session_id(capture)
        print(f"[session] Response ID: {capture.get('response_id', 'none')}")
        print(f"[session] Session ID: {session_id or 'none'}")

        # 5. Detect phases and curated events
        print("\n--- Step 5: Analyzing stream ---")
        phases = detect_phases(capture)
        curated = detect_curated_events(capture)
        print(f"[analysis] {len(phases)} phases, {len(curated)} curated events")

        # 6. Fetch server-side data
        print("\n--- Step 6: Fetching server-side data ---")
        server_metrics = fetch_server_metrics(base_url, session_id)
        server_jsonl = fetch_server_jsonl(
            VAST_SSH_HOST, VAST_SSH_PORT, session_id
        )
        server_html = fetch_server_html_report(
            VAST_SSH_HOST, VAST_SSH_PORT, session_id
        )

        # 7. Full analysis
        print("\n--- Step 7: Running analysis ---")
        analysis = analyse_stream(capture, phases, curated)
        print(f"[analysis] Quality issues: {analysis['quality']['issue_count']}")
        print(
            f"[analysis] Content: {analysis['content']['total_chars']} chars, "
            f"{analysis['content']['total_words']} words"
        )

        # 8. Run assertions
        print("\n--- Step 8: Running assertions ---")
        assertions = run_assertions(
            capture, analysis, server_metrics, server_jsonl
        )
        passed = sum(1 for a in assertions if a["passed"])
        total = len(assertions)
        print(f"[assertions] {passed}/{total} passed")
        for a in assertions:
            status = "PASS" if a["passed"] else "FAIL"
            reason = f" -- {a['reason']}" if a.get("reason") else ""
            print(f"  [{status}] {a['name']}{reason}")

        # 9. Build LLM JSON report
        print("\n--- Step 9: Building LLM report ---")
        llm_report = build_llm_report(
            capture, analysis, assertions,
            server_metrics, server_jsonl,
            phases, curated,
        )

        # 10. Build HTML report
        print("\n--- Step 10: Building HTML report ---")
        from generate_flow_report import render_html_report
        html_report = render_html_report(
            capture=capture,
            analysis=analysis,
            assertions=assertions,
            phases=phases,
            curated_events=curated,
            server_metrics=server_metrics,
            server_jsonl=server_jsonl,
        )

        # 11. Write outputs
        print("\n--- Step 11: Writing reports ---")
        report_dir = Path(REPORT_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        html_path = report_dir / f"integration_report_{timestamp}.html"
        html_path.write_text(html_report)
        print(f"[output] HTML report: {html_path}")

        json_path = report_dir / f"integration_report_{timestamp}.json"
        json_path.write_text(json.dumps(llm_report, indent=2, default=str))
        print(f"[output] JSON report: {json_path}")

        # Also write latest copies
        latest_html = report_dir / "integration_report_latest.html"
        latest_json = report_dir / "integration_report_latest.json"
        latest_html.write_text(html_report)
        latest_json.write_text(json.dumps(llm_report, indent=2, default=str))
        print(f"[output] Latest HTML: {latest_html}")
        print(f"[output] Latest JSON: {latest_json}")

        # If server HTML report exists, save it too
        if server_html:
            server_html_path = report_dir / f"server_report_{timestamp}.html"
            server_html_path.write_text(server_html)
            print(f"[output] Server HTML report: {server_html_path}")

        # 12. Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Duration: {capture.get('duration_secs', 0):.1f}s")
        print(f"Events: {len(capture.get('events', []))}")
        print(
            f"Content: {analysis['content']['total_chars']} chars, "
            f"{analysis['content']['total_words']} words"
        )
        print(f"Phases: {len(phases)}")
        print(f"Curated events: {len(curated)}")
        print(
            f"Quality issues: {analysis['quality']['issue_count']} "
            f"({analysis['quality']['critical_count']} critical)"
        )
        print(f"Assertions: {passed}/{total} passed")
        print(
            f"Recommendations: "
            f"{len(llm_report.get('recommendations', []))}"
        )
        print(f"\nHTML report: {html_path}")
        print(f"JSON report: {json_path}")
        print("=" * 70)

        return {
            "success": passed == total,
            "passed": passed,
            "total": total,
            "html_path": str(html_path),
            "json_path": str(json_path),
            "llm_report": llm_report,
        }

    finally:
        if tunnel:
            tunnel.close()


if __name__ == "__main__":
    result = run_integration_test()
    sys.exit(0 if result.get("success") else 1)
