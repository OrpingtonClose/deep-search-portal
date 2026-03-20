#!/usr/bin/env python3
"""
Mistral (Real) Proxy — Verified LLM Responses via Inline Fact-Checking.

An OpenAI-compatible proxy that produces **verified** responses by combining
Mistral Large's chain-of-thought reasoning with the Veritas Inquisitor
fact-checking pipeline.  Every response goes through:

  1. **Draft generation** — Mistral Large (thinking) produces a reasoned draft.
  2. **Claim verification** — The Veritas reactor decomposes the draft into
     atomic claims, gathers evidence for/against each one, debates
     contradictions, and scores overall truthfulness.
  3. **Revision** — If the hallucination probability exceeds a configurable
     threshold, Mistral Large rewrites the response incorporating the
     evidence and corrections from the Veritas report.

The user sees the thinking/verification progress inside ``<think>`` tags
(rendered natively by Open WebUI), followed by the final verified answer.

Architecture (LangGraph)::

    START -> generate_draft -> verify_draft -> [route]
        -> (clean)         -> format_output -> END
        -> (needs_revision) -> revise_response -> format_output -> END

Runs as a FastAPI app under uvicorn.
"""

import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid
from typing import Annotated, Any, AsyncGenerator, Optional, TypedDict

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
)
from research_metrics import MetricsCollector, ResearchMetricsCallback
import langfuse_config

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("MISTRAL_REAL_LOG_DIR", "/opt/mistral_real_logs")
log = setup_logging("mistral-real", LOG_DIR)

UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
AGENT_MODEL = os.getenv("MISTRAL_REAL_AGENT_MODEL", "mistral-small-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("MISTRAL_REAL_PORT", 9600, minimum=1)
MAX_CONCURRENT = env_int("MISTRAL_REAL_MAX_CONCURRENT", 2, minimum=1)

# Hallucination threshold: if Veritas reports hallucination probability above
# this value, the draft is revised.  Set to 0.0 to always revise; 1.0 to
# never revise (just annotate).
HALLUCINATION_THRESHOLD = float(
    os.getenv("MISTRAL_REAL_HALLUCINATION_THRESHOLD", "0.3")
)

# Maximum number of revision rounds (generate -> verify -> revise cycles).
MAX_REVISION_ROUNDS = env_int("MISTRAL_REAL_MAX_REVISIONS", 1, minimum=0)

log.info(
    f"Config: upstream={UPSTREAM_BASE}, model={UPSTREAM_MODEL}, "
    f"agent_model={AGENT_MODEL}, hallucination_threshold={HALLUCINATION_THRESHOLD}, "
    f"max_revisions={MAX_REVISION_ROUNDS}, port={LISTEN_PORT}"
)

tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)


# ============================================================================
# LLM Communication
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


def _get_real_llm(
    model: str = "",
    *,
    max_tokens: int = 8192,
    temperature: float = 0.3,
    timeout: float = 300.0,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance for Mistral (Real).

    Note: We pass max_tokens via extra_body instead of the native parameter
    because langchain-openai >=1.0 converts max_tokens to
    max_completion_tokens, which the Mistral API rejects with a 422.
    """
    return ChatOpenAI(
        model=model or UPSTREAM_MODEL,
        api_key=UPSTREAM_KEY,
        base_url=UPSTREAM_BASE,
        temperature=temperature,
        timeout=timeout,
        extra_body={"max_tokens": max_tokens},
    )


def _dicts_to_lc_messages(
    messages: list[dict],
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Convert OpenAI-format message dicts to LangChain message objects."""
    lc: list[SystemMessage | HumanMessage | AIMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    return lc


# Per-request LangGraph callback config, keyed by req_id.
_real_request_configs: dict[str, dict] = {}


async def call_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.3,
    stream: bool = False,
) -> dict:
    """Call the upstream LLM via LangChain ChatOpenAI (fires callbacks).

    Returns dict with keys: content, finish_reason, usage (or error).
    The `stream` parameter is accepted for backward compat but ignored;
    LangChain handles streaming separately via astream().
    """
    llm = _get_real_llm(
        model=model, max_tokens=max_tokens, temperature=temperature,
    )
    lc_messages = _dicts_to_lc_messages(messages)
    config = _real_request_configs.get(req_id, {})

    last_error: Optional[str] = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            ai_msg: AIMessage = await llm.ainvoke(lc_messages, config=config)
            usage = ai_msg.response_metadata.get("token_usage", {})
            return {
                "content": ai_msg.content or "",
                "finish_reason": ai_msg.response_metadata.get(
                    "finish_reason", "stop"
                ),
                "usage": usage,
            }

        except Exception as e:
            err_str = str(e)
            _codes_pattern = "|".join(str(c) for c in RETRYABLE_STATUS_CODES)
            retryable = bool(
                re.search(rf"\b({_codes_pattern})\b", err_str)
            ) or isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout))

            last_error = f"{err_str[:500]}"

            if retryable and attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(
                    f"[{req_id}] LLM retry {attempt+1}/{MAX_LLM_RETRIES} "
                    f"in {wait}s: {err_str[:200]}"
                )
                await asyncio.sleep(wait)
                continue

            return {"error": last_error}

    return {"error": last_error or "Max retries exceeded"}


async def stream_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the upstream LLM via LangChain astream."""
    llm = _get_real_llm(
        model=model, max_tokens=max_tokens, temperature=temperature,
    )
    lc_messages = _dicts_to_lc_messages(messages)
    config = _real_request_configs.get(req_id, {})

    try:
        async for chunk in llm.astream(lc_messages, config=config):
            token = chunk.content
            if token:
                yield token
    except Exception as e:
        log.error(f"[{req_id}] Stream LLM exception: {e}")
        yield f"[Error: {e}]"


# ============================================================================
# Draft Generation Prompt
# ============================================================================

DRAFT_SYSTEM_PROMPT = """\
You are Mistral (Real), a factual-accuracy-focused AI assistant.

IMPORTANT: Structure every response in two clearly labelled sections:

<THINKING>
[Your step-by-step reasoning. Be thorough: consider multiple angles, evaluate \
what you know vs what you are unsure about, note any claims that would benefit \
from external verification. Flag specific facts, names, dates, numbers, and \
citations that you are less than fully confident in.]
</THINKING>

<ANSWER>
[Your final, comprehensive response. Be precise with facts. Clearly \
distinguish between what is well-established, what is your reasoning/inference, \
and what you are uncertain about. Cite sources where possible.]
</ANSWER>

Always include both sections. THINKING first, then ANSWER.\
"""

REVISION_PROMPT = """\
You are Mistral (Real), revising a draft response based on fact-checking results.

The Veritas fact-checking system has analysed the draft below and found issues.
Your job is to produce a CORRECTED version that:

1. Removes or corrects any claims marked as **hallucinated**.
2. Adds caveats to claims marked as **overconfident** or **plausible-unverified**.
3. Preserves claims that are **verified** — do not weaken them.
4. Incorporates the evidence and corrections from the Veritas report.
5. Maintains the same overall structure and helpfulness of the original.

Do NOT add a disclaimer about being fact-checked.  Just produce the best, \
most accurate response you can.

Output ONLY the corrected response text (no THINKING/ANSWER tags for revision).\
"""


# ============================================================================
# LangGraph State
# ============================================================================

def _append_log(left: list[str], right: list[str]) -> list[str]:
    return left + right


class MistralRealState(TypedDict):
    """LangGraph state for the Mistral (Real) pipeline."""
    req_id: str
    messages: list[dict]
    original_body: dict

    # Draft generation
    draft_text: str
    draft_thinking: str

    # Veritas verification
    veritas_report: dict
    veritas_iterations: int
    veritas_artifact_count: int
    hallucination_probability: float
    overall_score: float
    claims_summary: list[dict]

    # Revision tracking
    revision_round: int
    needs_revision: bool
    revised_text: str

    # Final output
    final_answer: str

    # Pipeline metadata
    progress_log: Annotated[list[str], _append_log]
    phase: str
    elapsed: float
    error: str


# ============================================================================
# LangGraph Nodes
# ============================================================================

async def node_generate_draft(state: MistralRealState) -> dict:
    """Generate a draft response using Mistral Large with thinking."""
    req_id = state["req_id"]
    messages = state["messages"]

    # Inject the draft system prompt
    augmented = [m.copy() for m in messages]
    has_system = False
    for i, m in enumerate(augmented):
        if m.get("role") == "system":
            augmented[i]["content"] = m["content"] + "\n\n" + DRAFT_SYSTEM_PROMPT
            has_system = True
            break
    if not has_system:
        augmented.insert(0, {"role": "system", "content": DRAFT_SYSTEM_PROMPT})

    log.info(f"[{req_id}] Generating draft with {UPSTREAM_MODEL}")

    result = await call_llm(
        augmented, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    if "error" in result:
        return {
            "error": result["error"],
            "phase": "error",
            "progress_log": [f"Draft generation failed: {result['error']}"],
        }

    content = result.get("content", "")

    # Parse THINKING and ANSWER sections
    thinking = ""
    answer = ""

    if "<THINKING>" in content and "</THINKING>" in content:
        think_start = content.index("<THINKING>") + len("<THINKING>")
        think_end = content.index("</THINKING>")
        thinking = content[think_start:think_end].strip()

    if "<ANSWER>" in content and "</ANSWER>" in content:
        ans_start = content.index("<ANSWER>") + len("<ANSWER>")
        ans_end = content.index("</ANSWER>")
        answer = content[ans_start:ans_end].strip()
    elif "</THINKING>" in content:
        # Answer is everything after </THINKING> if no explicit tags
        answer = content[content.index("</THINKING>") + len("</THINKING>"):].strip()

    if not answer:
        answer = content  # fallback: use entire response

    log.info(
        f"[{req_id}] Draft generated: {len(answer)} chars answer, "
        f"{len(thinking)} chars thinking"
    )

    return {
        "draft_text": answer,
        "draft_thinking": thinking,
        "phase": "verify",
        "progress_log": [
            f"Draft generated ({len(answer):,} chars)",
        ],
    }


async def node_verify_draft(state: MistralRealState) -> dict:
    """Run the Veritas reactor on the draft to verify all claims."""
    req_id = state["req_id"]
    draft_text = state["draft_text"]

    if not draft_text:
        return {
            "phase": "format_output",
            "veritas_report": {},
            "hallucination_probability": 0.0,
            "overall_score": 1.0,
            "needs_revision": False,
            "progress_log": ["No draft to verify — skipping verification"],
        }

    # Extract original user query from messages
    original_query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                original_query = content
            elif isinstance(content, list):
                original_query = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    log.info(f"[{req_id}] Starting Veritas verification on {len(draft_text):,} chars")

    # Import and run the Veritas reactor
    import veritas_inquisitor
    result = await veritas_inquisitor.run_reactor(
        draft_text, original_query, f"{req_id}-veritas",
    )

    report = result.get("report", {})
    artifact_count = result.get("artifact_count", 0)
    iterations = result.get("iterations", 0)

    halluc_prob = report.get("overall_hallucination_probability", 0.0)
    overall_score = report.get("overall_score", 1.0)

    # Normalise: if Veritas returned -1 (error), treat as unknown
    if halluc_prob < 0:
        halluc_prob = 0.5
    if overall_score < 0:
        overall_score = 0.5

    # Summarise claims for progress/revision
    claims = report.get("claims", [])
    claims_summary = []
    hallucinated_count = 0
    overconfident_count = 0
    verified_count = 0
    for c in claims:
        status = c.get("status", "unknown")
        claims_summary.append({
            "claim": c.get("claim_text", "")[:120],
            "status": status,
            "confidence": c.get("confidence", 0),
            "evidence": c.get("evidence_summary", "")[:200],
        })
        if status == "hallucinated":
            hallucinated_count += 1
        elif status == "overconfident":
            overconfident_count += 1
        elif status == "verified":
            verified_count += 1

    needs_revision = (
        halluc_prob > HALLUCINATION_THRESHOLD
        and state["revision_round"] < MAX_REVISION_ROUNDS
    )

    progress = [
        f"Veritas complete: {iterations} iterations, {artifact_count} artifacts",
        f"Claims: {verified_count} verified, {hallucinated_count} hallucinated, "
        f"{overconfident_count} overconfident, {len(claims) - verified_count - hallucinated_count - overconfident_count} other",
        f"Score: {overall_score:.0%} truthfulness, {halluc_prob:.0%} hallucination probability",
    ]
    if needs_revision:
        progress.append(
            f"Hallucination probability ({halluc_prob:.0%}) exceeds threshold "
            f"({HALLUCINATION_THRESHOLD:.0%}) — revising response"
        )
    else:
        progress.append("Response passes verification threshold")

    log.info(
        f"[{req_id}] Verification complete: score={overall_score:.2f}, "
        f"halluc={halluc_prob:.2f}, needs_revision={needs_revision}"
    )

    return {
        "veritas_report": report,
        "veritas_iterations": iterations,
        "veritas_artifact_count": artifact_count,
        "hallucination_probability": halluc_prob,
        "overall_score": overall_score,
        "claims_summary": claims_summary,
        "needs_revision": needs_revision,
        "phase": "revise" if needs_revision else "format_output",
        "progress_log": progress,
    }


async def node_revise_response(state: MistralRealState) -> dict:
    """Revise the draft incorporating Veritas corrections."""
    req_id = state["req_id"]
    draft_text = state["draft_text"]
    report = state["veritas_report"]
    claims_summary = state["claims_summary"]

    # Build context for revision
    claims_text = ""
    for i, c in enumerate(claims_summary, 1):
        claims_text += (
            f"{i}. [{c['status'].upper()}] {c['claim']}\n"
            f"   Evidence: {c['evidence']}\n\n"
        )

    revised_output = report.get("revised_output", "")

    revision_messages = [
        {"role": "system", "content": REVISION_PROMPT},
        {"role": "user", "content": (
            f"## Original Draft\n\n{draft_text}\n\n"
            f"## Veritas Fact-Check Results\n\n"
            f"Overall truthfulness: {state['overall_score']:.0%}\n"
            f"Hallucination probability: {state['hallucination_probability']:.0%}\n\n"
            f"### Claim Analysis\n\n{claims_text}\n"
            f"### Veritas Suggested Revision\n\n{revised_output or '(none provided)'}\n\n"
            f"## Task\n\n"
            f"Produce a corrected version of the original draft that fixes all "
            f"hallucinated claims and adds appropriate caveats to unverified ones. "
            f"Keep verified claims intact."
        )},
    ]

    log.info(f"[{req_id}] Revising response (round {state['revision_round'] + 1})")

    result = await call_llm(
        revision_messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.2,
    )

    if "error" in result:
        log.warning(f"[{req_id}] Revision failed: {result['error']}, using Veritas revised output")
        # Fall back to Veritas's own revised output, or the original draft
        revised = revised_output or draft_text
    else:
        revised = result.get("content", "") or draft_text

    return {
        "revised_text": revised,
        "draft_text": revised,  # update draft for potential re-verification
        "revision_round": state["revision_round"] + 1,
        "phase": "format_output",
        "progress_log": [
            f"Revision round {state['revision_round'] + 1} complete ({len(revised):,} chars)"
        ],
    }


async def node_format_output(state: MistralRealState) -> dict:
    """Assemble the final answer from draft/revised text."""
    if state.get("revised_text"):
        final = state["revised_text"]
    else:
        final = state.get("draft_text", "")

    return {
        "final_answer": final,
        "phase": "done",
        "progress_log": ["Output finalised"],
    }


# ============================================================================
# LangGraph: conditional routing after verification
# ============================================================================

def route_after_verify(state: MistralRealState) -> str:
    """Route to revision or directly to output based on verification results."""
    if state.get("needs_revision"):
        return "revise_response"
    return "format_output"


# ============================================================================
# Build the LangGraph
# ============================================================================

def build_mistral_real_graph() -> Any:
    """Build the Mistral (Real) LangGraph pipeline.

    Graph topology::

        START -> generate_draft -> verify_draft -> [route_after_verify]
            -> (clean)          -> format_output -> END
            -> (needs_revision) -> revise_response -> format_output -> END
    """
    graph = StateGraph(MistralRealState)

    graph.add_node("generate_draft", node_generate_draft)
    graph.add_node("verify_draft", node_verify_draft)
    graph.add_node("revise_response", node_revise_response)
    graph.add_node("format_output", node_format_output)

    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "verify_draft")
    graph.add_conditional_edges(
        "verify_draft",
        route_after_verify,
        {"revise_response": "revise_response", "format_output": "format_output"},
    )
    graph.add_edge("revise_response", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


_mistral_real_graph = build_mistral_real_graph()


# ============================================================================
# Streaming Response Generator
# ============================================================================

_STREAM_DONE = object()


async def _pipeline_producer(
    initial_state: dict[str, Any],
    config: dict,
    output_queue: asyncio.Queue,
    chunk_fn,
    req_id: str,
) -> None:
    """Run the LangGraph pipeline and push SSE chunks to the output queue."""
    last_progress_idx = 0
    final_state = initial_state

    try:
        async for state_update in _mistral_real_graph.astream(
            initial_state, config=config, stream_mode="values",
        ):
            final_state = state_update

            # Emit new progress messages
            progress_list = state_update.get("progress_log", [])
            for msg in progress_list[last_progress_idx:]:
                await output_queue.put(chunk_fn(f"  {msg}\n"))
            last_progress_idx = len(progress_list)

            # Stream phase transitions
            phase = state_update.get("phase", "")
            if phase == "verify":
                await output_queue.put(chunk_fn(
                    "\n**[Phase 2: Veritas Fact-Check]**\n"
                    "Running claim decomposition, evidence gathering, debate, "
                    "and final judgement...\n\n"
                ))
            elif phase == "revise":
                await output_queue.put(chunk_fn(
                    "\n**[Phase 3: Revision]**\n"
                    "Correcting hallucinated/overconfident claims...\n\n"
                ))

        # Check for error state
        error = final_state.get("error", "")
        if error:
            await output_queue.put(chunk_fn(f"\nError: {error}\n"))
            await output_queue.put(chunk_fn("\n</think>\n\n"))
            await output_queue.put(chunk_fn(
                f"**Mistral (Real) Error**\n\nAn error occurred: {error}"
            ))
            await output_queue.put(chunk_fn("", finish_reason="stop"))
            await output_queue.put("data: [DONE]\n\n")
            return

        # Emit verification summary before closing think tags
        report = final_state.get("veritas_report", {})
        if report:
            score = final_state.get("overall_score", -1)
            halluc = final_state.get("hallucination_probability", -1)
            claims = final_state.get("claims_summary", [])
            revision_round = final_state.get("revision_round", 0)

            await output_queue.put(chunk_fn("\n---\n"))
            await output_queue.put(chunk_fn(
                f"**Verification Summary:** "
                f"truthfulness={score:.0%}, "
                f"hallucination_risk={halluc:.0%}, "
                f"claims_checked={len(claims)}"
            ))
            if revision_round > 0:
                await output_queue.put(chunk_fn(
                    f", revision_rounds={revision_round}"
                ))
            await output_queue.put(chunk_fn("\n"))

            # Show claim statuses
            for c in claims:
                status = c["status"]
                marker = {
                    "verified": "PASS",
                    "plausible-unverified": "UNVERIFIED",
                    "hallucinated": "FAIL",
                    "overconfident": "WARN",
                }.get(status, status.upper())
                await output_queue.put(chunk_fn(
                    f"  [{marker}] {c['claim'][:100]}\n"
                ))

        # Close thinking tags
        await output_queue.put(chunk_fn("\n</think>\n\n"))

        # Stream the final answer
        final_answer = final_state.get("final_answer", "(No answer generated)")
        for i in range(0, len(final_answer), 200):
            await output_queue.put(chunk_fn(final_answer[i:i + 200]))

        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Pipeline error: {e}\n{tb}")
        await output_queue.put(chunk_fn(f"\nError: {e}\n"))
        await output_queue.put(chunk_fn("\n</think>\n\n"))
        await output_queue.put(chunk_fn(
            f"**Mistral (Real) Error**\n\nAn error occurred during "
            f"verified response generation: {e}"
        ))
        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    finally:
        await output_queue.put(_STREAM_DONE)


async def run_mistral_real(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Orchestrate the full Mistral (Real) pipeline via LangGraph.

    Produces SSE chunks with:
      - ``<think>`` section showing draft generation + verification progress
      - Final verified answer after ``</think>``
    """
    model_id = original_body.get("model", "mistral-real")
    request_id = f"chatcmpl-real-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # Extract user query for logging
    user_query = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content[:100]
            break

    log.info(f"[{req_id}] Starting Mistral (Real) pipeline: {user_query}")

    initial_state: dict[str, Any] = {
        "req_id": req_id,
        "messages": user_messages,
        "original_body": original_body,
        "draft_text": "",
        "draft_thinking": "",
        "veritas_report": {},
        "veritas_iterations": 0,
        "veritas_artifact_count": 0,
        "hallucination_probability": 0.0,
        "overall_score": 1.0,
        "claims_summary": [],
        "revision_round": 0,
        "needs_revision": False,
        "revised_text": "",
        "final_answer": "",
        "progress_log": [],
        "phase": "generate_draft",
        "elapsed": 0.0,
        "error": "",
    }

    output_queue: asyncio.Queue = asyncio.Queue()

    # --- Langfuse tracing: generate trace URL and emit as first message ---
    langfuse_trace_id = langfuse_config.create_trace_id(req_id)
    langfuse_trace_url = langfuse_config.get_trace_url(langfuse_trace_id)
    langfuse_handler = langfuse_config.create_callback_handler(
        trace_id=langfuse_trace_id,
        session_id=req_id,
        tags=["mistral-real"],
    )

    if langfuse_trace_url:
        yield chunk(f"[Langfuse trace]({langfuse_trace_url})\n\n")

    yield chunk("<think>\n")
    yield chunk("**[Phase 1: Draft Generation]**\n")
    yield chunk(f"Using {UPSTREAM_MODEL} to generate a reasoned draft...\n\n")

    # Wire LangChain callbacks so metrics fire for every LLM/tool call
    metrics_collector = MetricsCollector(session_id=req_id, query=user_query)
    metrics_callback = ResearchMetricsCallback(metrics_collector)
    callbacks = [metrics_callback]
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)
    config = {
        "configurable": {"thread_id": req_id},
        "callbacks": callbacks,
    }
    _real_request_configs[req_id] = config

    pipeline_task = asyncio.create_task(
        _pipeline_producer(initial_state, config, output_queue, chunk, req_id)
    )

    try:
        while True:
            try:
                item = await asyncio.wait_for(output_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if item is _STREAM_DONE:
                break

            yield item

    except asyncio.CancelledError:
        log.info(f"[{req_id}] Client disconnected, cancelling pipeline")
        pipeline_task.cancel()
        raise

    finally:
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass
        _real_request_configs.pop(req_id, None)
        langfuse_config.flush()
        tracker.finish(req_id)


# ============================================================================
# FastAPI App
# ============================================================================

app = create_app("Mistral (Real) Proxy")

register_standard_routes(
    app,
    service_name="mistral-real-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "upstream_model": UPSTREAM_MODEL,
        "agent_model": AGENT_MODEL,
        "searxng": SEARXNG_URL,
        "hallucination_threshold": HALLUCINATION_THRESHOLD,
        "max_revisions": MAX_REVISION_ROUNDS,
        "max_concurrent": MAX_CONCURRENT,
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "mistral-real",
            "object": "model",
            "created": 1700000000,
            "owned_by": "mistral-real-proxy",
            "name": "Mistral (Real)",
        }],
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests with inline fact-checking."""
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages array is required and must not be empty",
                    "type": "invalid_request",
                }
            },
        )

    utility = is_utility_request(messages)
    log.info(
        f"[{req_id}] New request: messages={len(messages)}, "
        f"model={body.get('model', '?')}, utility={utility}"
    )

    tracker.start(req_id, utility=utility, messages=len(messages), phase="init")

    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH (utility request)")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "mistral-real"),
            tracker=tracker,
            log=log,
        )
    else:
        if not limiter.available():
            tracker.finish(req_id)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": (
                            f"Too many concurrent verified sessions "
                            f"({limiter.max_concurrent}). Try again shortly."
                        ),
                        "type": "rate_limit",
                    }
                },
            )

        log.info(f"[{req_id}] Routing to MISTRAL REAL pipeline")

        async def _guarded_real():
            async with limiter.hold():
                async for event in run_mistral_real(messages, body, req_id):
                    yield event

        generator = _guarded_real()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/verify")
async def verify_endpoint(request: Request):
    """Direct verification endpoint — delegates to Veritas Inquisitor."""
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

    if not target_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "target_text is required"}},
        )

    tracker.start(req_id, phase="verify", target_chars=len(target_text))

    if not limiter.available():
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        f"Too many concurrent verified sessions "
                        f"({limiter.max_concurrent}). Try again shortly."
                    ),
                    "type": "rate_limit",
                }
            },
        )

    try:
        async with limiter.hold():
            import veritas_inquisitor
            result = await veritas_inquisitor.verify_output(
                target_text, original_query, req_id,
            )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e)}},
        )
    finally:
        tracker.finish(req_id)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Mistral (Real) Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
