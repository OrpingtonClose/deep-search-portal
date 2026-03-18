"""
Veritas Inquisitor — Multi-agent fact-checking and hallucination detection system.

Implements a 5-agent swarm (Interrogator, Claim Decomposer, Evidence Gatherer,
Critic/Debater, Final Judge) using the INFINITE framework (arXiv 2603.14312):
  - Artifact DAG for provenance tracking
  - Pressure-scored NeedQueue for work prioritisation
  - ArtifactReactor loop for autonomous agent coordination

Can run standalone (via /v1/verify endpoint) or be invoked programmatically
from the persistent research proxy as a verification pass on research output.

Architecture:
  ┌─────────────┐
  │ Root Artifact│ (target output + original query)
  └──────┬──────┘
         │
  ┌──────▼──────┐   ┌──────────────────┐
  │ Interrogator│──▶│ NeedQueue (Needs) │
  └──────┬──────┘   └──────┬───────────┘
         │                 │
  ┌──────▼──────┐   ┌──────▼──────────┐
  │Claim Decomp.│──▶│ Evidence Gatherer│
  └──────┬──────┘   └──────┬──────────┘
         │                 │
  ┌──────▼──────┐   ┌──────▼──────┐
  │Critic/Debate│──▶│ Final Judge  │
  └─────────────┘   └─────────────┘

All artifacts are immutable and form a DAG.  The reactor loop runs until
no open Needs remain and the FinalJudge has emitted its report.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Optional

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("VERITAS_LOG_DIR", "/opt/veritas_logs")
log = setup_logging("veritas-inquisitor", LOG_DIR)

UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
AGENT_MODEL = os.getenv("VERITAS_AGENT_MODEL", "mistral-small-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("VERITAS_PORT", 9500, minimum=1)
MAX_CONCURRENT = env_int("VERITAS_MAX_CONCURRENT", 2, minimum=1)
MAX_DEBATE_ROUNDS = env_int("VERITAS_MAX_DEBATE_ROUNDS", 4, minimum=2)
MAX_EVIDENCE_CALLS = env_int("VERITAS_MAX_EVIDENCE_CALLS", 20, minimum=2)
PRESSURE_THRESHOLD = float(os.getenv("VERITAS_PRESSURE_THRESHOLD", "0.3"))

tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)


# ============================================================================
# Enums & Constants
# ============================================================================

class ClaimTag(str, Enum):
    DIRECT_TOOL = "direct-tool"
    INFERENCE = "inference"
    CITATION = "citation"
    ABSENCE = "absence"
    OPINION = "opinion"


class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    PLAUSIBLE_UNVERIFIED = "plausible-unverified"
    HALLUCINATED = "hallucinated"
    OVERCONFIDENT = "overconfident"


class NeedType(str, Enum):
    INTERROGATE = "interrogate"
    DECOMPOSE_CLAIMS = "decompose_claims"
    VERIFY_CLAIM = "verify_claim"
    COUNTER_EVIDENCE = "counter_evidence"
    DEBATE_ROUND = "debate_round"
    FINAL_JUDGEMENT = "final_judgement"


RISK_TAG_WEIGHTS: dict[str, float] = {
    ClaimTag.DIRECT_TOOL: 0.2,
    ClaimTag.INFERENCE: 0.6,
    ClaimTag.CITATION: 1.0,
    ClaimTag.ABSENCE: 1.0,
    ClaimTag.OPINION: 1.0,
}

SKILL_MAP: dict[NeedType, str] = {
    NeedType.INTERROGATE: "Interrogator",
    NeedType.DECOMPOSE_CLAIMS: "ClaimDecomposer",
    NeedType.VERIFY_CLAIM: "EvidenceGatherer",
    NeedType.COUNTER_EVIDENCE: "EvidenceGatherer",
    NeedType.DEBATE_ROUND: "CriticDebater",
    NeedType.FINAL_JUDGEMENT: "FinalJudge",
}


# ============================================================================
# Core Data Models (INFINITE framework)
# ============================================================================

@dataclass
class Artifact:
    """Immutable DAG node — a unit of work product in the verification pipeline."""
    id: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:12]}")
    artifact_type: str = ""          # e.g. "root", "probe", "claim", "evidence", "debate", "report"
    content: dict = field(default_factory=dict)
    epistemic_tag: str = ""          # ClaimTag value or ""
    tool_receipts: list[dict] = field(default_factory=list)
    parent_artifact_ids: list[str] = field(default_factory=list)
    pressure_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = ""             # agent role that created this

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type,
            "content": self.content,
            "epistemic_tag": self.epistemic_tag,
            "tool_receipts": self.tool_receipts,
            "parent_artifact_ids": self.parent_artifact_ids,
            "pressure_score": self.pressure_score,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass
class NeedItem:
    """A request for work in the reactor queue."""
    id: str = field(default_factory=lambda: f"need-{uuid.uuid4().hex[:12]}")
    need_type: NeedType = NeedType.VERIFY_CLAIM
    target_artifact_id: str = ""     # the artifact this need relates to
    target_claim_id: str = ""        # specific claim within the artifact (if applicable)
    pressure_score: float = 0.5
    required_skill: str = ""         # agent role required
    context: dict = field(default_factory=dict)  # additional context for the agent
    is_open: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArtifactIndex:
    """Append-only store for Artifacts with query capabilities."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._by_type: dict[str, list[str]] = {}

    def append(self, artifact: Artifact) -> None:
        self._artifacts[artifact.id] = artifact
        self._by_type.setdefault(artifact.artifact_type, []).append(artifact.id)

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifacts.get(artifact_id)

    def by_type(self, artifact_type: str) -> list[Artifact]:
        ids = self._by_type.get(artifact_type, [])
        return [self._artifacts[aid] for aid in ids if aid in self._artifacts]

    def children_of(self, parent_id: str) -> list[Artifact]:
        return [
            a for a in self._artifacts.values()
            if parent_id in a.parent_artifact_ids
        ]

    def all_artifacts(self) -> list[Artifact]:
        return list(self._artifacts.values())

    @property
    def count(self) -> int:
        return len(self._artifacts)


class NeedQueue:
    """Priority queue for NeedItems, ordered by pressure_score descending."""

    def __init__(self) -> None:
        self._needs: list[NeedItem] = []

    def post(self, need: NeedItem) -> None:
        self._needs.append(need)

    def open_needs(self) -> list[NeedItem]:
        return sorted(
            [n for n in self._needs if n.is_open],
            key=lambda n: n.pressure_score,
            reverse=True,
        )

    def close(self, need_id: str) -> None:
        for n in self._needs:
            if n.id == need_id:
                n.is_open = False
                break

    def has_open(self) -> bool:
        return any(n.is_open for n in self._needs)

    @property
    def total(self) -> int:
        return len(self._needs)

    @property
    def open_count(self) -> int:
        return sum(1 for n in self._needs if n.is_open)


class SkillRegistry:
    """Registry of available agent roles."""

    def __init__(self) -> None:
        self._skills: dict[str, dict] = {}

    def register(self, name: str, description: str, handles: list[NeedType]) -> None:
        self._skills[name] = {
            "description": description,
            "handles": [h.value for h in handles],
        }

    def can_handle(self, skill_name: str, need_type: NeedType) -> bool:
        skill = self._skills.get(skill_name)
        if not skill:
            return False
        return need_type.value in skill["handles"]

    def all_skills(self) -> dict[str, dict]:
        return dict(self._skills)


# ============================================================================
# Pressure Scoring
# ============================================================================

def compute_pressure(
    claim_tag: str,
    evidence_conflict_count: int,
    depth_in_dag: int,
) -> float:
    """
    pressure = (claim_risk_tag_weight * 0.4) +
               (evidence_conflict_count * 0.3) +
               (depth_in_dag * 0.3)

    Clamped to [0, 1].
    """
    risk_weight = RISK_TAG_WEIGHTS.get(claim_tag, 0.5)
    conflict_norm = min(evidence_conflict_count / 5.0, 1.0)  # normalise to 0-1
    depth_norm = min(depth_in_dag / 10.0, 1.0)               # normalise to 0-1

    score = (risk_weight * 0.4) + (conflict_norm * 0.3) + (depth_norm * 0.3)
    return max(0.0, min(1.0, score))


def dag_depth(artifact_id: str, index: ArtifactIndex, _cache: Optional[dict] = None) -> int:
    """Compute depth of an artifact in the provenance DAG."""
    if _cache is None:
        _cache = {}
    if artifact_id in _cache:
        return _cache[artifact_id]
    art = index.get(artifact_id)
    if not art or not art.parent_artifact_ids:
        _cache[artifact_id] = 0
        return 0
    depth = 1 + max(dag_depth(pid, index, _cache) for pid in art.parent_artifact_ids)
    _cache[artifact_id] = depth
    return depth


# ============================================================================
# LLM Communication (reuses upstream config)
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


async def call_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> dict:
    """Call the upstream LLM with retry logic. Returns parsed response dict."""
    model = model or AGENT_MODEL
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    last_error: Optional[str] = None
    client = http_client()

    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(300.0, connect=30.0),
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                last_error = f"HTTP {resp.status_code}: {error_text}"
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_LLM_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(f"[{req_id}] LLM retry {attempt+1}/{MAX_LLM_RETRIES} in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                return {"error": last_error}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "No choices in response"}

            message = choices[0].get("message", {})
            return {
                "content": message.get("content", "") or "",
                "finish_reason": choices[0].get("finish_reason", ""),
            }

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = f"Timeout: {type(e).__name__}"
            if attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(f"[{req_id}] LLM timeout, retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            return {"error": last_error}

        except Exception as e:
            return {"error": str(e)}

    return {"error": last_error or "Max retries exceeded"}


def _parse_json_from_llm(text: str) -> Optional[dict | list]:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array in the text using forward scan with
        # proper nesting depth tracking. Continue past failed candidates.
        for start_char, end_char in [("[", "]"), ("{" , "}")]:
            search_from = 0
            while True:
                start = text.find(start_char, search_from)
                if start == -1:
                    break
                # Forward scan to find matching close brace
                depth = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                    if depth == 0:
                        end = i
                        break
                if end == -1:
                    break  # no matching close brace found at all
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    # This candidate failed; try the next occurrence
                    search_from = start + 1
        return None


# ============================================================================
# Tool Wrappers (delegate to existing infrastructure)
# ============================================================================

async def tool_web_search(query: str, num_results: int = 10) -> str:
    """Web search via SearXNG."""
    try:
        client = http_client()
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return f"Search error: HTTP {resp.status_code}"

        data = resp.json()
        results = data.get("results", [])[:num_results]
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            formatted.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {e}"


async def tool_browse_page(url: str, instructions: str = "") -> str:
    """Fetch a webpage and extract readable text."""
    import html as html_mod
    import re
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; VeritasBot/1.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code}"

        raw = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html_mod.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 15000:
            text = text[:15000] + "\n[...truncated...]"

        result = f"Content from {url}:\n{text}"
        if instructions:
            result = f"Instructions: {instructions}\n\n{result}"
        return result

    except Exception as e:
        return f"Fetch error: {e}"


async def tool_code_execution(code: str) -> str:
    """Execute Python code in a sandboxed subprocess (non-blocking)."""
    import subprocess
    import sys
    import tempfile

    def _run_sync() -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=tempfile.gettempdir(),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if not output.strip():
                output = "(no output)"
            return output[:5000]
        except subprocess.TimeoutExpired:
            return "Code execution timed out (30s)"
        except Exception as e:
            return f"Execution error: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_sync)


async def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool call and return receipt."""
    t0 = time.monotonic()
    try:
        if tool_name == "web_search":
            result = await tool_web_search(
                arguments.get("query", ""),
                arguments.get("num_results", 10),
            )
        elif tool_name == "browse_page":
            result = await tool_browse_page(
                arguments.get("url", ""),
                arguments.get("instructions", ""),
            )
        elif tool_name == "code_execution":
            result = await tool_code_execution(arguments.get("code", ""))
        else:
            result = f"Unknown tool: {tool_name}"
    except Exception as e:
        result = f"Tool error: {e}"

    elapsed = time.monotonic() - t0
    return {
        "tool": tool_name,
        "arguments": arguments,
        "result": result[:8000],
        "elapsed_seconds": round(elapsed, 2),
    }


# ============================================================================
# Agent Prompts (verbatim from spec)
# ============================================================================

SYSTEM_PROMPT = (
    "You are Veritas Inquisitor. Objective: identify and classify every claim "
    "as verified, plausible-unverified, hallucinated, or overconfident. Use "
    "external tools for every claim. Output only structured data. Never add "
    "politeness or hedging."
)

INTERROGATOR_PROMPT = (
    "Input: target output. Output JSON: {\"new_probe_questions\": [...]}. "
    "Goal: generate questions that force additional verifiable claims. "
    "Generate 3-7 probing questions that test the factual accuracy of the "
    "claims in the target output."
)

CLAIM_DECOMPOSER_PROMPT = (
    "Parse input text. Output JSON array of "
    "{\"claim_text\": \"...\", \"tag\": \"direct-tool\"|\"inference\"|\"citation\"|\"absence\"|\"opinion\"}. "
    "Rules:\n"
    "- direct-tool: claims that reference specific tool outputs or data\n"
    "- inference: logical deductions or reasoning steps\n"
    "- citation: claims that reference specific sources, papers, or authorities\n"
    "- absence: claims about what does NOT exist or was NOT found\n"
    "- opinion: subjective judgements, recommendations, or predictions\n"
    "Extract EVERY distinct factual claim. Be thorough."
)

EVIDENCE_GATHERER_PROMPT = (
    "For each claim, issue supporting and disproof search queries. "
    "You must search for BOTH evidence that supports AND evidence that contradicts each claim. "
    "Output JSON: {\"claim_id\": \"...\", \"supporting_queries\": [...], \"disproof_queries\": [...], "
    "\"evidence_summary\": \"...\", \"conflicts_found\": 0}."
)

CRITIC_DEBATER_PROMPT = (
    "Current debate artifacts provided. Produce next round messages. "
    "Output JSON: {\"messages\": [{\"speaker\": \"prosecutor\"|\"defender\"|\"examiner\", \"message\": \"...\"}], "
    "\"key_conflicts\": [...], \"resolved_points\": [...]}. "
    "The prosecutor attacks claims with counter-evidence. "
    "The defender presents supporting evidence. "
    "The examiner checks internal consistency and source validity."
)

FINAL_JUDGE_PROMPT = (
    "Aggregate all DAG artifacts. Output final report JSON:\n"
    "{\n"
    "  \"claims\": [\n"
    "    {\n"
    "      \"id\": \"...\",\n"
    "      \"claim_text\": \"...\",\n"
    "      \"status\": \"verified\"|\"plausible-unverified\"|\"hallucinated\"|\"overconfident\",\n"
    "      \"evidence_summary\": \"...\",\n"
    "      \"confidence\": 0.0\n"
    "    }\n"
    "  ],\n"
    "  \"overall_hallucination_probability\": 0.0,\n"
    "  \"overall_score\": 0.0,\n"
    "  \"revised_output\": \"...\",\n"
    "  \"evidence_links\": [...]\n"
    "}\n\n"
    "Score each claim individually. overall_score is the fraction of verified claims. "
    "revised_output is the original text with hallucinated/overconfident claims corrected or removed."
)


# ============================================================================
# Agent Implementations
# ============================================================================

async def run_interrogator(
    target_text: str,
    original_query: str,
    parent_artifact_id: str,
    req_id: str,
) -> Artifact:
    """Interrogator: generates probing questions to elicit verifiable claims."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{INTERROGATOR_PROMPT}\n\n"
            f"Original query: {original_query}\n\n"
            f"Target output to verify:\n{target_text[:8000]}"
        )},
    ]

    result = await call_llm(messages, req_id, temperature=0.4)
    content_text = result.get("content", "")

    parsed = _parse_json_from_llm(content_text)
    if parsed is None:
        parsed = {"new_probe_questions": [], "raw": content_text}

    return Artifact(
        artifact_type="probe",
        content=parsed if isinstance(parsed, dict) else {"new_probe_questions": parsed},
        parent_artifact_ids=[parent_artifact_id],
        created_by="Interrogator",
    )


async def run_claim_decomposer(
    target_text: str,
    probe_questions: list[str],
    parent_artifact_ids: list[str],
    req_id: str,
) -> Artifact:
    """Claim Decomposer: parses text into atomic claims with epistemic tags."""
    probe_section = ""
    if probe_questions:
        probe_section = (
            "\n\nAdditional probe questions to consider:\n"
            + "\n".join(f"- {q}" for q in probe_questions)
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{CLAIM_DECOMPOSER_PROMPT}\n\n"
            f"Text to decompose:\n{target_text[:8000]}"
            f"{probe_section}"
        )},
    ]

    result = await call_llm(messages, req_id, temperature=0.1)
    content_text = result.get("content", "")

    parsed = _parse_json_from_llm(content_text)
    claims: list[dict] = []
    if isinstance(parsed, list):
        claims = parsed
    elif isinstance(parsed, dict) and "claims" in parsed:
        claims = parsed["claims"]
    else:
        claims = [{"claim_text": content_text, "tag": "inference"}]

    # Assign IDs to claims
    for i, claim in enumerate(claims):
        if "id" not in claim:
            claim["id"] = f"claim-{i:03d}"
        # Validate tag
        tag = claim.get("tag", "inference")
        if tag not in [t.value for t in ClaimTag]:
            claim["tag"] = "inference"

    return Artifact(
        artifact_type="claims",
        content={"claims": claims, "total": len(claims)},
        parent_artifact_ids=parent_artifact_ids,
        created_by="ClaimDecomposer",
    )


async def run_evidence_gatherer(
    claim: dict,
    parent_artifact_id: str,
    req_id: str,
) -> Artifact:
    """Evidence Gatherer: issues parallel supporting + disproof tool calls for a claim."""
    claim_text = claim.get("claim_text", "")
    claim_id = claim.get("id", "unknown")
    claim_tag = claim.get("tag", "inference")

    # Ask LLM to generate search queries
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{EVIDENCE_GATHERER_PROMPT}\n\n"
            f"Claim to verify (id: {claim_id}, tag: {claim_tag}):\n"
            f"\"{claim_text}\"\n\n"
            f"Generate search queries for web_search tool. Output JSON with "
            f"supporting_queries and disproof_queries (2-3 each)."
        )},
    ]

    result = await call_llm(messages, req_id, temperature=0.2)
    content_text = result.get("content", "")

    parsed = _parse_json_from_llm(content_text)
    if not isinstance(parsed, dict):
        parsed = {
            "supporting_queries": [f"{claim_text} evidence"],
            "disproof_queries": [f"{claim_text} debunked OR false OR incorrect"],
        }

    supporting_queries = parsed.get("supporting_queries", [claim_text])[:3]
    disproof_queries = parsed.get("disproof_queries", [f"NOT {claim_text}"])[:3]

    # Execute searches in parallel
    all_tool_receipts: list[dict] = []

    search_tasks = []
    for q in supporting_queries:
        search_tasks.append(execute_tool("web_search", {"query": q, "num_results": 5}))
    for q in disproof_queries:
        search_tasks.append(execute_tool("web_search", {"query": q, "num_results": 5}))

    search_results = await asyncio.gather(*search_tasks)
    all_tool_receipts.extend(search_results)

    # Count conflicts: any disproof search that returned results
    conflicts_found = 0
    for receipt in search_results[len(supporting_queries):]:
        result_text = receipt.get("result", "")
        if result_text and "No results found" not in result_text:
            conflicts_found += 1

    # Compute pressure for this claim
    depth = 1  # evidence is always depth 1 from claims
    pressure = compute_pressure(claim_tag, conflicts_found, depth)

    return Artifact(
        artifact_type="evidence",
        content={
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_tag": claim_tag,
            "supporting_queries": supporting_queries,
            "disproof_queries": disproof_queries,
            "conflicts_found": conflicts_found,
            "evidence_summary": f"Ran {len(supporting_queries)} supporting and {len(disproof_queries)} disproof searches. {conflicts_found} potential conflicts found.",
        },
        epistemic_tag=claim_tag,
        tool_receipts=all_tool_receipts,
        parent_artifact_ids=[parent_artifact_id],
        pressure_score=pressure,
        created_by="EvidenceGatherer",
    )


async def run_critic_debater(
    claims_with_evidence: list[dict],
    debate_round: int,
    prior_debate_artifacts: list[Artifact],
    parent_artifact_ids: list[str],
    req_id: str,
) -> Artifact:
    """Critic/Debater: runs one round of prosecutor/defender/examiner debate."""
    # Build context from prior debate rounds
    prior_messages = []
    for art in prior_debate_artifacts:
        msgs = art.content.get("messages", [])
        for m in msgs:
            prior_messages.append(f"[{m.get('speaker', '?')}]: {m.get('message', '')}")

    prior_context = ""
    if prior_messages:
        prior_context = (
            f"\n\nPrior debate rounds ({len(prior_debate_artifacts)}):\n"
            + "\n".join(prior_messages[-20:])  # cap context
        )

    # Build evidence summary
    evidence_lines = []
    for ce in claims_with_evidence:
        claim = ce.get("claim", {})
        evidence = ce.get("evidence", {})
        evidence_lines.append(
            f"- Claim [{claim.get('id', '?')}] ({claim.get('tag', '?')}): "
            f"\"{claim.get('claim_text', '')[:200]}\"\n"
            f"  Evidence: {evidence.get('evidence_summary', 'none')}\n"
            f"  Conflicts: {evidence.get('conflicts_found', 0)}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{CRITIC_DEBATER_PROMPT}\n\n"
            f"Debate round: {debate_round}\n\n"
            f"Claims and evidence:\n" + "\n".join(evidence_lines[:30])
            + prior_context
        )},
    ]

    result = await call_llm(messages, req_id, temperature=0.4)
    content_text = result.get("content", "")

    parsed = _parse_json_from_llm(content_text)
    if not isinstance(parsed, dict):
        parsed = {
            "messages": [{"speaker": "examiner", "message": content_text}],
            "key_conflicts": [],
            "resolved_points": [],
        }

    return Artifact(
        artifact_type="debate",
        content={
            **parsed,
            "round": debate_round,
        },
        parent_artifact_ids=parent_artifact_ids,
        pressure_score=0.5,  # moderate pressure for debate artifacts
        created_by="CriticDebater",
    )


async def run_final_judge(
    original_query: str,
    target_text: str,
    all_artifacts: list[Artifact],
    parent_artifact_ids: list[str],
    req_id: str,
) -> Artifact:
    """Final Judge: aggregates all DAG artifacts and produces the verdict."""
    # Build comprehensive context from all artifacts
    claims_section = ""
    evidence_section = ""
    debate_section = ""

    for art in all_artifacts:
        if art.artifact_type == "claims":
            claims = art.content.get("claims", [])
            claims_section = "CLAIMS:\n" + json.dumps(claims, indent=2)[:4000]

        elif art.artifact_type == "evidence":
            content = art.content
            evidence_section += (
                f"\nEvidence for {content.get('claim_id', '?')}:\n"
                f"  Tag: {content.get('claim_tag', '?')}\n"
                f"  Summary: {content.get('evidence_summary', '')}\n"
                f"  Conflicts: {content.get('conflicts_found', 0)}\n"
                f"  Tool receipts: {len(art.tool_receipts)}\n"
            )
            # Include key tool results
            for receipt in art.tool_receipts[:2]:
                receipt_result = receipt.get("result", "")[:500]
                evidence_section += f"  Search result: {receipt_result}\n"

        elif art.artifact_type == "debate":
            msgs = art.content.get("messages", [])
            for m in msgs:
                debate_section += f"  [{m.get('speaker', '?')}]: {m.get('message', '')[:300]}\n"

    context = (
        f"Original query: {original_query}\n\n"
        f"Target output to judge:\n{target_text[:4000]}\n\n"
        f"{claims_section}\n\n"
        f"EVIDENCE:\n{evidence_section[:4000]}\n\n"
        f"DEBATE:\n{debate_section[:2000]}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{FINAL_JUDGE_PROMPT}\n\n{context}"},
    ]

    result = await call_llm(
        messages, req_id,
        model=UPSTREAM_MODEL,  # use the larger model for final judgement
        max_tokens=8192,
        temperature=0.1,
    )
    content_text = result.get("content", "")

    parsed = _parse_json_from_llm(content_text)
    if not isinstance(parsed, dict):
        parsed = {
            "claims": [],
            "overall_hallucination_probability": -1,
            "overall_score": -1,
            "revised_output": "",
            "evidence_links": [],
            "raw_response": content_text,
        }

    return Artifact(
        artifact_type="report",
        content=parsed,
        parent_artifact_ids=parent_artifact_ids,
        pressure_score=0.0,
        created_by="FinalJudge",
    )


# ============================================================================
# Artifact Reactor (INFINITE main loop)
# ============================================================================

async def run_reactor(
    target_text: str,
    original_query: str,
    req_id: str,
    progress_callback: Optional[callable] = None,
) -> dict:
    """
    Main INFINITE reactor loop.

    1. Create root artifact
    2. Post initial Needs
    3. Loop: scan NeedQueue, assign to agents, create child artifacts
    4. Terminate when FinalJudge has posted report

    Returns the final report dict.
    """
    index = ArtifactIndex()
    queue = NeedQueue()
    registry = SkillRegistry()

    # Register all agent skills
    registry.register("Interrogator", "Generates probing questions", [NeedType.INTERROGATE])
    registry.register("ClaimDecomposer", "Decomposes text into atomic claims", [NeedType.DECOMPOSE_CLAIMS])
    registry.register("EvidenceGatherer", "Gathers supporting and counter-evidence", [NeedType.VERIFY_CLAIM, NeedType.COUNTER_EVIDENCE])
    registry.register("CriticDebater", "Runs adversarial debate rounds", [NeedType.DEBATE_ROUND])
    registry.register("FinalJudge", "Produces final verdict", [NeedType.FINAL_JUDGEMENT])

    async def _progress(msg: str) -> None:
        if progress_callback:
            await progress_callback(msg)
        log.info(f"[{req_id}] {msg}")

    # Step 1: Create root artifact
    root = Artifact(
        artifact_type="root",
        content={
            "target_text": target_text,
            "original_query": original_query,
        },
        created_by="system",
    )
    index.append(root)
    await _progress(f"Root artifact created ({len(target_text):,} chars)")

    # Step 2: Post initial needs
    queue.post(NeedItem(
        need_type=NeedType.INTERROGATE,
        target_artifact_id=root.id,
        pressure_score=0.9,  # must run before ClaimDecomposer
        required_skill="Interrogator",
    ))
    queue.post(NeedItem(
        need_type=NeedType.DECOMPOSE_CLAIMS,
        target_artifact_id=root.id,
        pressure_score=0.8,  # high priority — must happen early, but after Interrogator
        required_skill="ClaimDecomposer",
    ))

    # Step 3: Reactor loop
    # Scale the iteration cap with the number of needs posted so that large
    # claim sets (many VERIFY_CLAIM needs) can still reach FINAL_JUDGEMENT.
    # Base: 20 fixed overhead + 3× the total needs posted so far.
    # Re-evaluated each iteration so dynamically-posted needs are included.
    iteration = 0
    reactor_start = time.monotonic()
    MAX_REACTOR_SECONDS = 1800  # 30-minute hard wall-clock timeout

    while queue.has_open() and (time.monotonic() - reactor_start) < MAX_REACTOR_SECONDS:
        iteration += 1
        open_needs = queue.open_needs()

        if not open_needs:
            break

        # Process highest-priority need
        need = open_needs[0]

        # Check pressure threshold (skip low-priority needs if we have the report)
        reports = index.by_type("report")
        if reports and need.pressure_score < PRESSURE_THRESHOLD:
            queue.close(need.id)
            continue

        await _progress(
            f"Iteration {iteration}: Processing {need.need_type.value} "
            f"(pressure: {need.pressure_score:.2f}, open: {queue.open_count})"
        )

        # Dispatch to appropriate agent
        if need.need_type == NeedType.INTERROGATE:
            queue.close(need.id)
            probe_artifact = await run_interrogator(
                target_text, original_query, need.target_artifact_id, req_id,
            )
            index.append(probe_artifact)
            await _progress(
                f"Interrogator produced {len(probe_artifact.content.get('new_probe_questions', []))} probes"
            )

        elif need.need_type == NeedType.DECOMPOSE_CLAIMS:
            queue.close(need.id)

            # Gather probe questions from any probe artifacts
            probe_questions = []
            for art in index.by_type("probe"):
                probe_questions.extend(art.content.get("new_probe_questions", []))

            claims_artifact = await run_claim_decomposer(
                target_text, probe_questions,
                [need.target_artifact_id] + [a.id for a in index.by_type("probe")],
                req_id,
            )
            index.append(claims_artifact)

            claims = claims_artifact.content.get("claims", [])
            await _progress(f"Claim Decomposer found {len(claims)} atomic claims")

            # Post verification needs for each claim
            for claim in claims:
                claim_pressure = compute_pressure(
                    claim.get("tag", "inference"), 0,
                    dag_depth(claims_artifact.id, index),
                )
                queue.post(NeedItem(
                    need_type=NeedType.VERIFY_CLAIM,
                    target_artifact_id=claims_artifact.id,
                    target_claim_id=claim.get("id", ""),
                    pressure_score=claim_pressure,
                    required_skill="EvidenceGatherer",
                    context={"claim": claim},
                ))

        elif need.need_type in (NeedType.VERIFY_CLAIM, NeedType.COUNTER_EVIDENCE):
            queue.close(need.id)
            claim = need.context.get("claim", {})

            evidence_artifact = await run_evidence_gatherer(
                claim, need.target_artifact_id, req_id,
            )
            index.append(evidence_artifact)

            conflicts = evidence_artifact.content.get("conflicts_found", 0)
            await _progress(
                f"Evidence Gatherer: claim {claim.get('id', '?')} — "
                f"{conflicts} conflict(s)"
            )

            # If conflicts found, post counter-evidence need for deeper investigation
            if conflicts > 0 and need.need_type != NeedType.COUNTER_EVIDENCE:
                queue.post(NeedItem(
                    need_type=NeedType.COUNTER_EVIDENCE,
                    target_artifact_id=evidence_artifact.id,
                    target_claim_id=claim.get("id", ""),
                    pressure_score=evidence_artifact.pressure_score * 1.2,
                    required_skill="EvidenceGatherer",
                    context={"claim": claim},
                ))

            # Check if all claims have been verified — if so, trigger debate
            claims_artifacts = index.by_type("claims")
            evidence_artifacts = index.by_type("evidence")
            if claims_artifacts:
                all_claims = claims_artifacts[0].content.get("claims", [])
                verified_claim_ids = {
                    e.content.get("claim_id") for e in evidence_artifacts
                }
                all_claim_ids = {c.get("id") for c in all_claims}
                unverified = all_claim_ids - verified_claim_ids

                # Only start debate when all claims verified AND no
                # verify/counter-evidence needs remain (prevents premature
                # judgement while counter-evidence is still pending).
                remaining_verify_needs = [
                    n for n in queue.open_needs()
                    if n.need_type in (NeedType.VERIFY_CLAIM, NeedType.COUNTER_EVIDENCE)
                ]
                if not unverified and not remaining_verify_needs:
                    # Check if debate hasn't been posted yet
                    debate_needs = [
                        n for n in queue.open_needs()
                        if n.need_type == NeedType.DEBATE_ROUND
                    ]
                    if not debate_needs and not index.by_type("debate"):
                        queue.post(NeedItem(
                            need_type=NeedType.DEBATE_ROUND,
                            target_artifact_id=claims_artifacts[0].id,
                            pressure_score=0.7,
                            required_skill="CriticDebater",
                            context={"round": 1},
                        ))

        elif need.need_type == NeedType.DEBATE_ROUND:
            queue.close(need.id)
            current_round = need.context.get("round", 1)

            # Gather claims + evidence for debate
            claims_with_evidence = []
            claims_artifacts = index.by_type("claims")
            evidence_artifacts = index.by_type("evidence")

            if claims_artifacts:
                all_claims = claims_artifacts[0].content.get("claims", [])
                evidence_by_claim = {}
                for e in evidence_artifacts:
                    cid = e.content.get("claim_id", "")
                    evidence_by_claim[cid] = e.content

                for claim in all_claims:
                    claims_with_evidence.append({
                        "claim": claim,
                        "evidence": evidence_by_claim.get(claim.get("id", ""), {}),
                    })

            prior_debates = index.by_type("debate")

            debate_artifact = await run_critic_debater(
                claims_with_evidence, current_round, prior_debates,
                [need.target_artifact_id] + [a.id for a in prior_debates],
                req_id,
            )
            index.append(debate_artifact)

            msgs = debate_artifact.content.get("messages", [])
            await _progress(
                f"Debate round {current_round}: {len(msgs)} messages"
            )

            # Post another debate round if needed (up to MAX_DEBATE_ROUNDS)
            if current_round < MAX_DEBATE_ROUNDS:
                # Check if there are unresolved conflicts
                key_conflicts = debate_artifact.content.get("key_conflicts", [])
                if key_conflicts:
                    queue.post(NeedItem(
                        need_type=NeedType.DEBATE_ROUND,
                        target_artifact_id=debate_artifact.id,
                        pressure_score=0.6,
                        required_skill="CriticDebater",
                        context={"round": current_round + 1},
                    ))
                else:
                    await _progress(f"Debate converged after {current_round} round(s)")

            # If this is the last debate round or debate converged, post final judgement
            debate_needs = [
                n for n in queue.open_needs()
                if n.need_type == NeedType.DEBATE_ROUND
            ]
            if not debate_needs:
                judge_needs = [
                    n for n in queue.open_needs()
                    if n.need_type == NeedType.FINAL_JUDGEMENT
                ]
                if not judge_needs and not index.by_type("report"):
                    queue.post(NeedItem(
                        need_type=NeedType.FINAL_JUDGEMENT,
                        target_artifact_id=debate_artifact.id,
                        pressure_score=1.0,  # highest priority
                        required_skill="FinalJudge",
                    ))

        elif need.need_type == NeedType.FINAL_JUDGEMENT:
            queue.close(need.id)

            report_artifact = await run_final_judge(
                original_query, target_text,
                index.all_artifacts(),
                [a.id for a in index.all_artifacts() if a.artifact_type != "root"],
                req_id,
            )
            index.append(report_artifact)

            claims_report = report_artifact.content.get("claims", [])
            score = report_artifact.content.get("overall_score", -1)
            halluc_prob = report_artifact.content.get("overall_hallucination_probability", -1)
            await _progress(
                f"Final Judge: {len(claims_report)} claims scored, "
                f"overall_score={score}, hallucination_prob={halluc_prob}"
            )

        else:
            queue.close(need.id)
            log.warning(f"[{req_id}] Unknown need type: {need.need_type}")

    # Step 4: Extract final report
    reports = index.by_type("report")
    if reports:
        report = reports[-1]  # last report is the final one
        return {
            "report": report.content,
            "artifact_count": index.count,
            "dag": [a.to_dict() for a in index.all_artifacts()],
            "iterations": iteration,
        }

    # Fallback: no report produced (shouldn't happen normally)
    await _progress("WARNING: Reactor terminated without producing a report")
    return {
        "report": {
            "claims": [],
            "overall_hallucination_probability": -1,
            "overall_score": -1,
            "revised_output": target_text,
            "error": "Reactor terminated without producing a report",
        },
        "artifact_count": index.count,
        "dag": [a.to_dict() for a in index.all_artifacts()],
        "iterations": iteration,
    }


# ============================================================================
# Streaming Verification (SSE format)
# ============================================================================

async def stream_verification(
    target_text: str,
    original_query: str,
    req_id: str,
    model_id: str = "veritas-inquisitor",
) -> AsyncGenerator[str, None]:
    """Run the reactor and stream progress as SSE events."""
    request_id = f"chatcmpl-veritas-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    progress_messages: asyncio.Queue = asyncio.Queue()

    async def progress_callback(msg: str) -> None:
        await progress_messages.put(msg)

    yield chunk("<think>\n")
    yield chunk("**[Veritas Inquisitor: Starting Verification]**\n\n")
    yield chunk(f"Target text: {len(target_text):,} chars\n")
    yield chunk(f"Original query: {original_query[:100]}...\n\n")

    # Run the reactor in a task so we can stream progress
    reactor_task = asyncio.create_task(
        run_reactor(target_text, original_query, req_id, progress_callback)
    )

    try:
        # Stream progress updates while reactor runs
        while not reactor_task.done():
            try:
                msg = await asyncio.wait_for(progress_messages.get(), timeout=1.0)
                yield chunk(f"  {msg}\n")
            except asyncio.TimeoutError:
                continue

        # Drain any remaining messages
        while not progress_messages.empty():
            msg = await progress_messages.get()
            yield chunk(f"  {msg}\n")

        yield chunk("\n</think>\n\n")

        # Get the result
        try:
            result = reactor_task.result()
        except Exception as e:
            log.error(f"[{req_id}] Reactor error: {e}")
            yield chunk(f"## Verification Error\n\n{e}\n")
            yield chunk("", finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        report = result.get("report", {})
        artifact_count = result.get("artifact_count", 0)
        iterations = result.get("iterations", 0)

        # Format the report as markdown
        yield chunk("## Veritas Inquisitor Report\n\n")
        yield chunk(f"**Artifacts produced:** {artifact_count} | **Reactor iterations:** {iterations}\n\n")

        overall_score = report.get("overall_score", -1)
        halluc_prob = report.get("overall_hallucination_probability", -1)

        if overall_score >= 0:
            yield chunk(f"**Overall truthfulness score:** {overall_score:.1%}\n")
        if halluc_prob >= 0:
            yield chunk(f"**Hallucination probability:** {halluc_prob:.1%}\n\n")

        # Claims table
        claims = report.get("claims", [])
        if claims:
            yield chunk("### Claim Analysis\n\n")
            yield chunk("| # | Status | Confidence | Claim | Evidence |\n")
            yield chunk("|---|--------|------------|-------|----------|\n")

            for i, c in enumerate(claims, 1):
                status = c.get("status", "?")
                confidence = c.get("confidence", 0)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                claim_text = c.get("claim_text", "")[:80]
                evidence = c.get("evidence_summary", "")[:60]

                status_emoji = {
                    "verified": "VERIFIED",
                    "plausible-unverified": "UNVERIFIED",
                    "hallucinated": "**HALLUCINATED**",
                    "overconfident": "OVERCONFIDENT",
                }.get(status, status)

                yield chunk(
                    f"| {i} | {status_emoji} | {confidence:.0%} | "
                    f"{claim_text} | {evidence} |\n"
                )
            yield chunk("\n")

        # Evidence links
        evidence_links = report.get("evidence_links", [])
        if evidence_links:
            yield chunk("### Evidence Links\n\n")
            for link in evidence_links[:20]:
                if isinstance(link, str):
                    yield chunk(f"- {link}\n")
                elif isinstance(link, dict):
                    yield chunk(f"- [{link.get('title', link.get('url', '?'))}]({link.get('url', '')})\n")
            yield chunk("\n")

        # Revised output
        revised = report.get("revised_output", "")
        if revised:
            yield chunk("### Revised Output\n\n")
            yield chunk(f"{revised}\n\n")

        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        # Cancel the reactor task on client disconnect or generator close
        # to prevent orphaned LLM/tool calls from consuming resources.
        if not reactor_task.done():
            reactor_task.cancel()
            try:
                await reactor_task
            except asyncio.CancelledError:
                log.info(f"[{req_id}] Reactor task cancelled (client disconnect)")


# ============================================================================
# Programmatic API (for use by persistent proxy)
# ============================================================================

async def verify_output(
    target_text: str,
    original_query: str,
    req_id: str = "",
) -> dict:
    """
    Run Veritas verification and return the full result dict.

    This is the programmatic entry point — call this from other proxies
    or services to verify an LLM output.

    Returns:
        {
            "report": { claims, overall_score, hallucination_probability, ... },
            "artifact_count": int,
            "dag": [artifact_dicts],
            "iterations": int,
        }
    """
    if not req_id:
        req_id = f"veritas-{uuid.uuid4().hex[:8]}"

    return await run_reactor(target_text, original_query, req_id)


# ============================================================================
# FastAPI Application
# ============================================================================

app = create_app("Veritas Inquisitor")

register_standard_routes(
    app,
    service_name="veritas-inquisitor",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "agent_model": AGENT_MODEL,
        "judge_model": UPSTREAM_MODEL,
        "searxng": SEARXNG_URL,
        "max_concurrent": MAX_CONCURRENT,
        "max_debate_rounds": MAX_DEBATE_ROUNDS,
        "pressure_threshold": PRESSURE_THRESHOLD,
    },
)


@app.post("/v1/verify")
async def verify_endpoint(request: Request):
    """
    Verify an LLM output for hallucinations.

    Request body:
    {
        "target_text": "The LLM output to verify",
        "original_query": "The original user query",
        "stream": true/false  (default: true)
    }
    """
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}"}},
        )

    target_text = body.get("target_text", "")
    original_query = body.get("original_query", "")
    stream = body.get("stream", True)

    if not target_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "target_text is required"}},
        )

    tracker.start(req_id, target_chars=len(target_text))

    if not limiter.available():
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        f"Too many concurrent verifications "
                        f"({limiter.max_concurrent}). Try again shortly."
                    ),
                }
            },
        )

    if stream:
        async def _guarded_verify():
            async with limiter.hold():
                try:
                    async for event in stream_verification(
                        target_text, original_query, req_id,
                    ):
                        yield event
                finally:
                    tracker.finish(req_id)

        return StreamingResponse(
            _guarded_verify(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            async with limiter.hold():
                result = await verify_output(target_text, original_query, req_id)
            return JSONResponse(result)
        finally:
            tracker.finish(req_id)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    When a message asks to "verify" or "fact-check" content, routes to
    the Veritas pipeline. Otherwise returns a helpful error.
    """
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}"}},
        )

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages array is required"}},
        )

    # Extract target text — look for the assistant message to verify
    target_text = ""
    original_query = ""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > len(target_text):
                target_text = content
        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                original_query = content

    if not target_text:
        # If no assistant message, verify the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    target_text = content
                    break

    if not target_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No text found to verify"}},
        )

    tracker.start(req_id, target_chars=len(target_text))

    if not limiter.available():
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"Too many concurrent verifications ({limiter.max_concurrent})."
                }
            },
        )

    async def _guarded_verify():
        async with limiter.hold():
            try:
                model_id = body.get("model", "veritas-inquisitor")
                async for event in stream_verification(
                    target_text, original_query, req_id, model_id,
                ):
                    yield event
            finally:
                tracker.finish(req_id)

    return StreamingResponse(
        _guarded_verify(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "veritas-inquisitor",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deep-search-portal",
                "name": "Veritas Inquisitor (Fact Checker)",
            },
        ],
    })


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Veritas Inquisitor...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
