#!/usr/bin/env python3
"""
Persistent Deep Research Proxy for Open WebUI.

An advanced research proxy implementing the full architecture from the design
document, extending the base MiroFlow deep research loop with:

  - **Subagent map-reduce**: A planning agent decomposes the query into multiple
    research angles, parallel subagents each conduct independent research, and a
    synthesis agent combines all findings into a comprehensive answer.
  - **Atom of Thoughts (AoT) state contraction**: Each subagent compresses its
    findings into atomic conditions (fact + source + confidence) rather than
    accumulating raw tool output, preventing context-window overflow.
  - **AoT Reflection**: Validates decomposition quality and triggers
    redecomposition when contraction quality is poor.
  - **Persistent memory**: Atomic conditions are stored in Neo4j (via the
    Knowledge Engine) so future queries can retrieve relevant prior findings.
    JSONL flat files provide local archival logging.
  - **Knowledge Graph**: Entity extraction from findings, relationship storage,
    cross-domain bridge edges, and graph-aware retrieval via Neo4j fulltext search.
  - **Dual-model architecture**: A small, fast model (e.g. Mistral Small) handles
    planning and subagent research; the large model handles final synthesis.
  - **Serendipity-aware exploration**: The planning agent generates cross-domain
    "bridge queries" alongside standard research angles, with serendipity scoring
    (relevance x novelty x surprise).
  - **Dynamic Saturation Detection**: Novelty tracking per turn with adaptive
    turn allocation -- expand when novelty > 0.3, stop when < 0.05.
  - **Trust Scoring**: URL domain-based confidence (.edu/.gov = 0.9, news = 0.6,
    forums/Reddit = 0.3) with contradiction penalties.
  - **Citation Verification**: Verify claims against sources, detect and resolve
    contradictions, adjust confidence scores.
  - **Draft-Synthesis-Revision Loop**: Draft -> critic review -> targeted
    micro-research on weak sections -> final polish.
  - **Recursive subagent spawning**: Subagents can spawn sub-subagents for
    emerging rabbit holes (depth-limited).
  - **Expanded Tool Ecosystem**: arxiv_search, wayback_fetch, wikidata_query
    alongside the existing searxng_search, fetch_webpage, python_exec.

Architecture:
  User Query
      |
      v
  [Retrieve Prior Knowledge] -- Neo4j fulltext + Knowledge Graph lookup
      |
      v
  [Planning Agent] (small model) -- decomposes into N angles + bridge queries
      |
      +--- Subagent 1: angle_1 (small model, AoT loop) ---+
      +--- Subagent 2: angle_2 (small model, AoT loop) ---+
      +--- ...                                             +--- progress queue
      +--- Subagent N: angle_N (small model, AoT loop) ---+
      |          (each may spawn recursive sub-subagents)
      v
  [Entity Extraction + Knowledge Graph Update]
      |
      v
  [Citation Verification] -- verify claims, detect contradictions
      |
      v
  [Store Conditions] -- persist to Neo4j + JSONL archive
      |
      v
  [Draft Synthesis] (large model) -- cross-reference, produce draft
      |
      v
  [Critic Agent] (small model) -- review draft, mark weak sections
      |
      v
  [Revision Agent] -- targeted micro-research on weak sections
      |
      v
  [Final Synthesis] (large model) -- polished final answer
      |
      v
  Streamed SSE response

This module is now a thin orchestration layer. All research logic has been
decomposed into the ``tools/`` package for independent testability.
"""

import asyncio
import html
import json
import os
import re
import time
import traceback
import uuid
from typing import Any, AsyncGenerator, Optional

# Sentinel used to signal the SSE output queue that the pipeline is done.
_STREAM_DONE = object()

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

import knowledge_client
import social_media_scrapers
import veritas_inquisitor
from research_metrics import (
    MetricsCollector,
    ResearchMetricsCallback,
    list_available_reports,
    load_metrics,
)
import langfuse_config

from shared import (
    ConcurrencyLimiter,
    RequestTracker,
    all_throttler_stats,
    create_app,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    setup_logging,
    stream_passthrough,
)

# ---------------------------------------------------------------------------
# Import everything from the decomposed tools/ package.
# The tools/ package is the canonical source of truth for all research logic.
# Names are re-exported at module scope for backward compatibility.
# ---------------------------------------------------------------------------
from tools.config import (
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    UPSTREAM_MODEL,
    SUBAGENT_MODEL,
    SEARXNG_URL,
    LISTEN_PORT,
    PORTAL_PUBLIC_URL,
    OWUI_INTERNAL_URL,
    MAX_SUBAGENT_TURNS,
    MAX_CONCURRENT,
    RESEARCH_NAMESPACE,
    JSONL_LOG_DIR,
    WEBPAGE_MAX_CHARS,
    PYTHON_TIMEOUT,
    PYTHON_OUTPUT_MAX,
    MAX_PRIOR_CONDITIONS,
    NOVELTY_EXPAND_THRESHOLD,
    NOVELTY_STOP_THRESHOLD,
    MAX_SUBAGENTS,
    MAX_RECURSIVE_DEPTH,
    TREE_MAX_CONCURRENT,
    TREE_MAX_DEPTH,
    TREE_MAX_NODES,
    TREE_PRESSURE_THRESHOLD,
    TREE_WORKER_IDLE_TIMEOUT,
    APIFY_API_KEY,
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_HOST,
    BRIGHT_DATA_CUSTOMER_ID,
    BRIGHT_DATA_ZONE,
    BRIGHT_DATA_SERP_ZONE,
    OXYLABS_USERNAME,
    OXYLABS_PASSWORD,
    VERITAS_VERIFY_ENABLED,
    VERITAS_MIN_CONDITIONS,
    VERITAS_HALLUCINATION_THRESHOLD,
    COMMERCIAL_SEARCH_ENABLED,
    MODERATION_MODEL,
    PLAYWRIGHT_AVAILABLE,
    SELENIUM_AVAILABLE,
    LOG_DIR,
    log,
)

from tools.models import AtomicCondition, SubagentResult, ResearchNode
from tools.scoring import trust_score_url, serendipity_score

from tools.llm import (
    _get_llm,
    _get_synthesis_llm,
    _get_subagent_llm,
    _dicts_to_langchain_messages,
    call_llm,
    _request_configs,
)

from tools.persistence import (
    _ensure_jsonl_dir,
    _append_jsonl,
    _log_conditions_jsonl,
    _log_entities_jsonl,
    _store_conditions_neo4j,
    _store_entities_neo4j,
    _retrieve_related,
    _retrieve_graph_neighbors,
    _is_large_document,
    run_document_ingestion,
)

from tools.tool_defs import NATIVE_TOOLS, LANGCHAIN_TOOLS

from tools.moderation import (
    _get_moderation_llm,
    moderate_query,
    _search_bright_data_serp,
    _search_oxylabs_serp,
    _commercial_search,
)

from tools.web_fetch import (
    _strip_html,
    _is_censored_response,
    _CENSORSHIP_KEYWORDS,
    _ERROR_PREFIXES,
    _fetch_via_httpx,
    _fetch_via_playwright,
    _fetch_via_selenium,
    _fetch_via_bright_data,
    _fetch_via_oxylabs,
    _fetch_via_wayback_cdx,
    enhanced_web_fetch,
    _tool_fetch_webpage_direct,
    tool_fetch_webpage,
)

from tools.search_tools import (
    _format_search_results,
    _searxng_query,
    _has_news_intent,
    tool_searxng_search,
    tool_news_search,
    tool_4plebs_search,
    tool_b4k_search,
    tool_warosu_search,
    tool_twitter_search,
    _twitter_via_bright_data,
    _twitter_via_oxylabs,
    _twitter_via_nitter,
    tool_python_exec,
    tool_arxiv_search,
    tool_wayback_fetch,
    tool_wikidata_query,
    tool_web_search,
    tool_hackernews_search,
    tool_stackexchange_search,
    tool_pubmed_search,
    tool_wikipedia_search,
    tool_archiveorg_search,
    tool_forum_search,
    tool_scholar_search,
    tool_substack_search,
    tool_knowledge_graph_search,
    tool_knowledge_discover,
)

from tools.tool_executor import (
    _retry_tool_call,
    _extract_pdf_text,
    _execute_tool_inner,
    execute_tool,
    execute_tools_parallel,
)

from tools.verification import (
    extract_entities_from_conditions,
    verify_conditions,
    verify_conditions_with_veritas,
    _fuzzy_match_claim_to_condition,
)

from tools.planning import plan_research, reflect_on_conditions
from tools.subagent import run_subagent, _parse_conditions

from tools.tree_reactor import (
    _compute_pressure,
    _spawn_sub_questions,
    _research_single_node,
    tree_research_reactor,
)

from tools.heartbeat import (
    LiveFindingsCollector,
    _generate_heartbeat_message,
    _heartbeat_loop,
    _format_curated_event_llm,
    _format_curated_event_fallback,
)

from tools.synthesis import (
    synthesize_with_revision,
    relevance_gate,
    strip_moralizing,
)

from tools.pipeline import (
    PersistentResearchState,
    build_persistent_research_graph,
    _live_collectors,
    _curated_queues,
    _metrics_collectors,
    _pdr_append_log,
    pdr_node_retrieve,
    pdr_node_tree_research,
    pdr_node_entities,
    pdr_node_verify,
    pdr_node_reflect,
    pdr_node_persist,
    pdr_node_synthesize,
)

# Backward-compat aliases for names used with underscore prefix in monolith
_PLAYWRIGHT_AVAILABLE = PLAYWRIGHT_AVAILABLE
_SELENIUM_AVAILABLE = SELENIUM_AVAILABLE

# ---------------------------------------------------------------------------
# Patch-propagation for decomposed tools/ submodules.
#
# Tests do ``patch.object(proxy, "some_func", mock)`` which only sets the
# attribute on *this* module.  After decomposition the actual callers live in
# tools/ submodules and look up names in their own globals.  We replace this
# module's class with a custom subclass whose ``__setattr__`` propagates
# every attribute change to every tools submodule that has the same name,
# so ``unittest.mock.patch.object`` automatically patches the real callers.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types


class _PropagatingModule(_types.ModuleType):
    """Module subclass that propagates setattr to tools submodules."""

    _tools_submodules: list[_types.ModuleType] = []

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for submod in self._tools_submodules:
            if name in submod.__dict__:
                submod.__dict__[name] = value


# Replace this module's class so setattr() propagation works.
_this = _sys.modules[__name__]
_this.__class__ = _PropagatingModule

# Collect all tools submodules for propagation.
_tools_mod_names = [
    "tools.config", "tools.models", "tools.scoring", "tools.llm",
    "tools.persistence", "tools.tool_defs", "tools.moderation",
    "tools.web_fetch", "tools.search_tools", "tools.tool_executor",
    "tools.verification", "tools.planning", "tools.subagent",
    "tools.tree_reactor", "tools.heartbeat", "tools.synthesis",
    "tools.pipeline",
]
_this._tools_submodules = [
    _sys.modules[mn] for mn in _tools_mod_names if mn in _sys.modules
]


# ---------------------------------------------------------------------------
# OWUI token validation -- protects dashboard/report endpoints
# ---------------------------------------------------------------------------

_owui_auth_cache: dict[str, float] = {}  # token -> expiry timestamp
_OWUI_CACHE_TTL = 300  # cache valid tokens for 5 minutes
_OWUI_CACHE_MAX_SIZE = 1000  # max entries before forced cleanup


def _evict_expired_tokens() -> None:
    """Remove expired entries from the auth cache to prevent unbounded growth."""
    now = time.monotonic()
    expired = [k for k, v in _owui_auth_cache.items() if v <= now]
    for k in expired:
        del _owui_auth_cache[k]


async def _validate_owui_token(request: Request) -> bool:
    """Validate that the request carries a valid Open WebUI session token.

    Checks the Authorization header (Bearer) or the ``token`` cookie, then
    verifies against OWUI's ``/api/v1/auths/`` endpoint.  Results are cached
    for 5 minutes to avoid hammering OWUI on every request.
    """
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("token", "").strip()
    if not token:
        return False

    # Evict expired entries to prevent unbounded memory growth
    if len(_owui_auth_cache) > _OWUI_CACHE_MAX_SIZE:
        _evict_expired_tokens()

    # Check cache
    now = time.monotonic()
    if token in _owui_auth_cache and _owui_auth_cache[token] > now:
        return True

    # Validate against OWUI
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{OWUI_INTERNAL_URL}/api/v1/auths/",
                headers={"Authorization": f"Bearer {token}"},
            )
        if resp.status_code == 200:
            _owui_auth_cache[token] = now + _OWUI_CACHE_TTL
            return True
        return False
    except Exception:
        # If OWUI is unreachable, deny access
        return False


def _auth_denied() -> JSONResponse:
    """Return a 401 response for unauthenticated dashboard requests."""
    return JSONResponse(
        {"error": "Authentication required. Please log in via the portal."},
        status_code=401,
    )


# --- Logging ---
log.info(
    f"Config: synthesis_model={UPSTREAM_MODEL}, subagent_model={SUBAGENT_MODEL}, "
    f"upstream={UPSTREAM_BASE}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, "
    f"tree_concurrent={TREE_MAX_CONCURRENT}, tree_depth={TREE_MAX_DEPTH}, "
    f"tree_nodes={TREE_MAX_NODES}, subagent_turns={MAX_SUBAGENT_TURNS}, "
    f"research_ns={RESEARCH_NAMESPACE}, "
    f"apify={'yes' if APIFY_API_KEY else 'no'}, "
    f"bright_data={'yes' if BRIGHT_DATA_API_KEY else 'no'}, "
    f"oxylabs={'yes' if OXYLABS_USERNAME else 'no'}, "
    f"playwright={'yes' if PLAYWRIGHT_AVAILABLE else 'no'}, "
    f"selenium={'yes' if SELENIUM_AVAILABLE else 'no'}"
)

# --- Shared state ---
tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)

# Initialise JSONL log directory
try:
    _ensure_jsonl_dir()
    log.info(f"JSONL log directory ready: {JSONL_LOG_DIR}")
except Exception as e:
    log.warning(f"Failed to create JSONL log directory: {e}")

# Build the LangGraph pipeline (compiled graph)
_persistent_research_graph = build_persistent_research_graph()


# ============================================================================
# Main Orchestrator
# ============================================================================

async def _pipeline_producer(
    initial_state: dict[str, Any],
    config: dict,
    output_queue: asyncio.Queue,
    chunk_fn,
    req_id: str,
) -> None:
    """Run the LangGraph pipeline and push SSE chunks to the output queue.

    This runs as a background task so the heartbeat can interleave its
    updates into the same queue.
    """
    last_progress_idx = 0
    final_state = initial_state

    try:
        async for state_update in _persistent_research_graph.astream(
            initial_state, config=config, stream_mode="values",
        ):
            final_state = state_update
            progress_list = state_update.get("progress_log", [])
            for msg in progress_list[last_progress_idx:]:
                await output_queue.put(chunk_fn(msg))
            last_progress_idx = len(progress_list)

        # Pipeline done — emit closing think tag, links header, and final answer
        await output_queue.put(chunk_fn("\n</think>\n\n"))

        # Emit report + trace links as the first visible lines
        report_url = final_state.get("report_url", "")
        metrics_url = final_state.get("metrics_url", "")
        langfuse_url = initial_state.get("_langfuse_trace_url", "")
        link_lines = []
        if report_url:
            link_lines.append(f"**[Full Report]({report_url})**")
        if metrics_url:
            link_lines.append(f"[Metrics JSON]({metrics_url})")
        if langfuse_url:
            link_lines.append(f"[Langfuse Trace]({langfuse_url})")
        if link_lines:
            await output_queue.put(chunk_fn(" | ".join(link_lines) + "\n\n"))

        final_answer = final_state.get("final_answer", "(No answer generated)")
        for i in range(0, len(final_answer), 200):
            await output_queue.put(chunk_fn(final_answer[i:i + 200]))
        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    except Exception as e:
        start_time = initial_state.get("start_time", 0)
        elapsed = time.monotonic() - start_time if start_time else 0
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Persistent research error after {elapsed:.2f}s: {e}\n{tb}")
        await output_queue.put(chunk_fn(f"\nError: {str(e)}\n"))
        await output_queue.put(chunk_fn("\n</think>\n\n"))
        await output_queue.put(chunk_fn(f"**Deep Research Error**\n\nAn error occurred during research: {str(e)}"))
        await output_queue.put(chunk_fn("", finish_reason="stop"))
        await output_queue.put("data: [DONE]\n\n")

    finally:
        await output_queue.put(_STREAM_DONE)


async def run_persistent_research(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Orchestrate the full persistent deep research pipeline via LangGraph.

    Uses an asyncio.Queue so the pipeline, heartbeat task, and keepalive
    comments can all push SSE chunks into a single ordered stream.
    """
    model_id = original_body.get("model", "persistent-miroflow")
    request_id = f"chatcmpl-pdr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    user_query = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if not user_query:
        yield chunk("Error: No user message found.")
        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    log.info(f"[{req_id}] Starting persistent deep research: {user_query[:100]}")

    # --- Langfuse tracing: generate trace URL early so it goes into initial_state ---
    langfuse_trace_id = langfuse_config.create_trace_id(req_id)
    langfuse_trace_url = langfuse_config.get_trace_url(langfuse_trace_id)
    langfuse_handler = langfuse_config.create_callback_handler(
        trace_id=langfuse_trace_id,
        session_id=req_id,
        tags=["persistent-research"],
    )

    initial_state: dict[str, Any] = {
        "req_id": req_id,
        "user_query": user_query,
        "start_time": start_time,
        "prior_conditions": [],
        "graph_neighbors": [],
        "subagent_results": [],
        "all_conditions": [],
        "total_turns": 0,
        "total_tools": 0,
        "total_children": 0,
        "nodes_explored": 0,
        "reflection": {},
        "final_answer": "",
        "progress_log": [],
        "phase": "retrieve",
        "report_url": "",
        "metrics_url": "",
        "_langfuse_trace_url": langfuse_trace_url or "",
    }

    # Create the shared output queue, live findings collector, and curated queue
    output_queue: asyncio.Queue = asyncio.Queue()
    collector = LiveFindingsCollector(user_query=user_query)
    _live_collectors[req_id] = collector
    curated_queue: asyncio.Queue = asyncio.Queue()
    _curated_queues[req_id] = curated_queue

    # Create metrics collector for this session
    metrics_collector = MetricsCollector(session_id=req_id, query=user_query)
    _metrics_collectors[req_id] = metrics_collector
    metrics_callback = ResearchMetricsCallback(metrics_collector)

    yield chunk("<think>\n")

    callbacks = [metrics_callback]
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    config = {
        "configurable": {"thread_id": req_id},
        "callbacks": callbacks,
    }

    # Register the config so call_llm and execute_tool can look it up
    # by req_id and fire callbacks on every LLM/tool invocation.
    _request_configs[req_id] = config

    # Start the pipeline producer as a background task
    pipeline_task = asyncio.create_task(
        _pipeline_producer(initial_state, config, output_queue, chunk, req_id)
    )

    # Start the heartbeat task with curated queue for tree reactor updates
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            output_queue, collector, chunk, req_id,
            interval=8.0, curated_queue=curated_queue,
        )
    )

    try:
        # Consume from the output queue and yield to the SSE response
        while True:
            try:
                item = await asyncio.wait_for(output_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # No items for 5s — emit invisible keepalive to prevent timeouts
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
        # Stop the heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Ensure the pipeline task is done
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

        # Clean up the live collector, curated queue, metrics collector, and config
        _live_collectors.pop(req_id, None)
        _curated_queues.pop(req_id, None)
        _metrics_collectors.pop(req_id, None)
        _request_configs.pop(req_id, None)
        langfuse_config.flush()
        tracker.finish(req_id)


# ============================================================================
# FastAPI App
# ============================================================================

app = create_app("Persistent Deep Research Proxy")

register_standard_routes(
    app,
    service_name="persistent-deep-research-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "upstream": UPSTREAM_BASE,
        "synthesis_model": UPSTREAM_MODEL,
        "subagent_model": SUBAGENT_MODEL,
        "searxng": SEARXNG_URL,
        "max_subagents": MAX_SUBAGENTS,
        "max_subagent_turns": MAX_SUBAGENT_TURNS,
        "max_recursive_depth": MAX_RECURSIVE_DEPTH,
        "tree_max_concurrent": TREE_MAX_CONCURRENT,
        "tree_max_depth": TREE_MAX_DEPTH,
        "tree_max_nodes": TREE_MAX_NODES,
        "research_namespace": RESEARCH_NAMESPACE,
        "jsonl_log_dir": JSONL_LOG_DIR,
        "tools": [t["function"]["name"] for t in NATIVE_TOOLS],
    },
)


@app.get("/v1/throttle_stats")
async def throttle_stats():
    """Return current throttling statistics for all external API providers."""
    return JSONResponse({"throttlers": all_throttler_stats()})


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "persistent-miroflow",
            "object": "model",
            "created": 1700000000,
            "owned_by": "persistent-deep-research-proxy",
            "name": "Persistent MiroFlow",
        }]
    })


@app.get("/knowledge/stats")
async def knowledge_stats():
    """Return statistics about the persistent knowledge base (from Neo4j)."""
    try:
        stats = await knowledge_client.research_stats(namespace=RESEARCH_NAMESPACE)
        return JSONResponse({
            "total_conditions": stats.get("total_conditions", 0),
            "total_sessions": stats.get("total_sessions", 0),
            "unique_queries": stats.get("total_queries", 0),
            "total_entities": stats.get("total_entities", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "storage_backend": "neo4j",
            "namespace": RESEARCH_NAMESPACE,
            "social_scraper_costs": social_media_scrapers.get_cost_tracker().get_session_stats(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/research/reports")
async def get_research_reports(request: Request):
    """List all available research reports with metadata."""
    if not await _validate_owui_token(request):
        return _auth_denied()
    try:
        reports = list_available_reports()
        return JSONResponse({"reports": reports, "count": len(reports)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)




_SAFE_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _render_markdown_page(md_text: str) -> str:
    """Convert Markdown text to a clean, self-contained HTML page.

    Uses a lightweight regex-based converter — no external dependencies.
    Handles headings, bold, italic, links, lists, horizontal rules, and code.
    """
    import re as _re

    body = html.escape(md_text)

    # Horizontal rules
    body = _re.sub(r"^---+$", "<hr>", body, flags=_re.MULTILINE)

    # Headings (### before ## before #)
    body = _re.sub(r"^### (.+)$", r"<h3>\1</h3>", body, flags=_re.MULTILINE)
    body = _re.sub(r"^## (.+)$", r"<h2>\1</h2>", body, flags=_re.MULTILINE)
    body = _re.sub(r"^# (.+)$", r"<h1>\1</h1>", body, flags=_re.MULTILINE)

    # Bold and italic
    body = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", body)

    # Links: [text](url)
    body = _re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        body,
    )

    # Unordered list items (- item)
    body = _re.sub(r"^- (.+)$", r"<li>\1</li>", body, flags=_re.MULTILINE)
    # Wrap consecutive <li> in <ul>
    body = _re.sub(
        r"((?:<li>.*?</li>\n?)+)",
        r"<ul>\1</ul>",
        body,
    )

    # Paragraphs: convert double newlines to paragraph breaks
    body = _re.sub(r"\n{2,}", "\n<br><br>\n", body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fafafa;
  }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid #2563eb; padding-bottom: 0.4rem; }}
  h2 {{ font-size: 1.3rem; color: #1e40af; margin-top: 1.5rem; }}
  h3 {{ font-size: 1.1rem; color: #374151; margin-top: 1.2rem; }}
  hr {{ border: none; border-top: 1px solid #d1d5db; margin: 1.5rem 0; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin-bottom: 0.5rem; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  em {{ color: #6b7280; }}
</style>
</head>
<body>
{body}
</body>
</html>"""




@app.get("/research/report/{session_id}")
async def get_research_report(session_id: str, request: Request):
    """Serve the research report for a session.

    Reports are stored as Markdown.  The endpoint renders them as a
    clean HTML page for browser viewing.  Pass `?raw=1` to get the
    raw Markdown text instead.
    """
    if not await _validate_owui_token(request):
        return _auth_denied()
    if not _SAFE_SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)

    from fastapi.responses import HTMLResponse, PlainTextResponse

    reports_dir = os.getenv(
        "RESEARCH_REPORTS_DIR", "/opt/persistent_research_logs/reports"
    )

    # Try .md first (new format), fall back to .html (legacy)
    md_path = os.path.join(reports_dir, f"{session_id}.md")
    html_path = os.path.join(reports_dir, f"{session_id}.html")

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        # Render Markdown as a simple HTML page for browser viewing
        return HTMLResponse(content=_render_markdown_page(md_content))
    except FileNotFoundError:
        pass

    # Legacy HTML fallback
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return JSONResponse(
            {"error": f"Report not found for session {session_id}"},
            status_code=404,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/research/metrics/{session_id}")
async def get_research_metrics(session_id: str, request: Request):
    """Serve the metrics JSON for a research session.

    Designed for LLM consumption — structured data for performance analysis.
    """
    if not await _validate_owui_token(request):
        return _auth_denied()
    if not _SAFE_SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)

    metrics = load_metrics(session_id)
    if metrics is None:
        return JSONResponse(
            {"error": f"Metrics not found for session {session_id}"},
            status_code=404,
        )
    return JSONResponse(metrics)


@app.get("/research/dashboard")
async def research_dashboard(request: Request):
    """Serve the observability dashboard for the research pipeline.

    Queries Langfuse Metrics API (if configured) and local metrics files
    to render a comprehensive HTML dashboard showing latency, cost, model
    usage, error rates, and per-session research statistics.

    Query params:
      ?days=N  — lookback window (default: 7)
    """
    from fastapi.responses import HTMLResponse

    if not await _validate_owui_token(request):
        return _auth_denied()
    days = 7
    try:
        days_param = request.query_params.get("days", "7")
        days = max(1, min(90, int(days_param)))
    except (ValueError, TypeError):
        pass

    try:
        from langfuse_dashboards import render_dashboard_html
        html_content = await asyncio.to_thread(render_dashboard_html, days=days)
        return HTMLResponse(content=html_content)
    except Exception as exc:
        log.error("Failed to render dashboard: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": f"Dashboard rendering failed: {exc}"},
            status_code=500,
        )


@app.get("/research/dashboard/data")
async def research_dashboard_data(request: Request):
    """Return dashboard data as JSON for programmatic consumption.

    Same data as the HTML dashboard but in machine-readable format.

    Query params:
      ?days=N  — lookback window (default: 7)
    """
    if not await _validate_owui_token(request):
        return _auth_denied()
    days = 7
    try:
        days_param = request.query_params.get("days", "7")
        days = max(1, min(90, int(days_param)))
    except (ValueError, TypeError):
        pass

    try:
        from langfuse_dashboards import (
            _langfuse_configured,
            aggregate_local_metrics,
            query_cost_over_time,
            query_error_rates,
            query_model_usage,
            query_observation_latency_by_name,
            query_trace_latency,
            query_trace_volume,
        )

        def _build_dashboard_data():
            langfuse_available = _langfuse_configured()
            data = {
                "langfuse_configured": langfuse_available,
                "days": days,
                "local_metrics": aggregate_local_metrics(),
            }
            if langfuse_available:
                data["langfuse"] = {
                    "trace_volume": query_trace_volume(days),
                    "model_usage": query_model_usage(days),
                    "observation_latency": query_observation_latency_by_name(days),
                    "errors": query_error_rates(days),
                    "cost_over_time": query_cost_over_time(days),
                    "trace_latency": query_trace_latency(days),
                }
            return data

        data = await asyncio.to_thread(_build_dashboard_data)
        return JSONResponse(data)
    except Exception as exc:
        log.error("Failed to get dashboard data: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": f"Dashboard data failed: {exc}"},
            status_code=500,
        )


@app.post("/v1/verify")
async def verify_research(request: Request):
    """Verify an LLM output using the Veritas Inquisitor swarm.

    Request body:
    {
        "target_text": "text to verify",
        "original_query": "original user query",
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

    log.info(f"[{req_id}] Verify request: {len(target_text)} chars")
    tracker.start(req_id, phase="verify", target_chars=len(target_text))

    if not limiter.available():
        tracker.finish(req_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        f"Too many concurrent persistent research sessions "
                        f"({limiter.max_concurrent}). Try again shortly."
                    ),
                    "type": "rate_limit",
                }
            },
        )

    if stream:
        async def _stream_verify():
            async with limiter.hold():
                try:
                    async for event in veritas_inquisitor.stream_verification(
                        target_text, original_query, req_id,
                    ):
                        yield event
                finally:
                    tracker.finish(req_id)

        return StreamingResponse(
            _stream_verify(),
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
                result = await veritas_inquisitor.verify_output(
                    target_text, original_query, req_id,
                )
            return JSONResponse(result)
        finally:
            tracker.finish(req_id)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
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
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", "persistent-miroflow"),
            tracker=tracker,
            log=log,
        )
    else:
        # Check if the last user message is a large document for ingestion
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    user_text = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                break

        if _is_large_document(user_text):
            log.info(
                f"[{req_id}] Routing to DOCUMENT INGESTION "
                f"({len(user_text):,} chars)"
            )

            async def _guarded_ingest():
                async with limiter.hold():
                    try:
                        async for event in run_document_ingestion(
                            user_text, body, req_id
                        ):
                            yield event
                    finally:
                        tracker.finish(req_id)

            generator = _guarded_ingest()
        else:
            if not limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": (
                                f"Too many concurrent persistent research sessions "
                                f"({limiter.max_concurrent}). Try again shortly."
                            ),
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Routing to PERSISTENT DEEP RESEARCH")

            async def _guarded_research():
                async with limiter.hold():
                    async for event in run_persistent_research(messages, body, req_id):
                        yield event

            generator = _guarded_research()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Persistent Deep Research Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
