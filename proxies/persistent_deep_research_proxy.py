#!/usr/bin/env python3
"""
Persistent Deep Research Proxy for Open WebUI.

Thin orchestration layer — all research logic lives in the ``tools/``
sub-package.  This file provides:

  * Backward-compatible re-exports so existing tests and imports continue
    to work (``import persistent_deep_research_proxy as pdr; pdr.call_llm``).
  * OWUI token validation for dashboard/report endpoints.
  * The FastAPI application with all HTTP routes.

See ``tools/`` for the actual research pipeline implementation.
"""

import asyncio
import html
import os
import re
import time
import uuid

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

import knowledge_client
import social_media_scrapers
import veritas_inquisitor
from research_metrics import list_available_reports, load_metrics

import langfuse_config

from shared import (
    ConcurrencyLimiter,  # noqa: F401
    RequestTracker,  # noqa: F401
    all_throttler_stats,
    create_app,
    env_int,  # noqa: F401
    get_throttler,  # noqa: F401
    http_client,  # noqa: F401
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,  # noqa: F401
    setup_logging,  # noqa: F401
    stream_passthrough,
)

# ---------------------------------------------------------------------------
# Re-export everything from tools/ at module level for backward compat.
# Tests do ``import persistent_deep_research_proxy as pdr; pdr.AtomicCondition``
# so every public name must be importable from this module.
# ---------------------------------------------------------------------------

from tools.config import (  # noqa: F401
    LOG_DIR,
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    UPSTREAM_MODEL,
    SUBAGENT_MODEL,
    SEARXNG_URL,
    LISTEN_PORT,
    PORTAL_PUBLIC_URL,
    OWUI_INTERNAL_URL,
    _get_llm,
    _get_synthesis_llm,
    _get_subagent_llm,
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
    OXYLABS_USERNAME,
    OXYLABS_PASSWORD,
    _PLAYWRIGHT_AVAILABLE,
    _SELENIUM_AVAILABLE,
    VERITAS_VERIFY_ENABLED,
    VERITAS_MIN_CONDITIONS,
    VERITAS_HALLUCINATION_THRESHOLD,
    COMMERCIAL_SEARCH_ENABLED,
    BRIGHT_DATA_SERP_ZONE,
    MODERATION_MODEL,
    tracker,
    limiter,
    _STREAM_DONE,
    _live_collectors,
    _curated_queues,
    _metrics_collectors,
    log,
)

from tools.models import (  # noqa: F401
    CrossRef,
    AtomicCondition,
    SubagentResult,
    ResearchNode,
)

from tools.scoring import trust_score_url, serendipity_score  # noqa: F401

from tools.persistence import (  # noqa: F401
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

from tools.tool_defs import NATIVE_TOOLS, LANGCHAIN_TOOLS  # noqa: F401

from tools.moderation import (  # noqa: F401
    _get_moderation_llm,
    classify_query,
    moderate_query,
    _search_bright_data_serp,
    _search_oxylabs_serp,
    _commercial_search,
)

from tools.search_tools import (  # noqa: F401
    _format_search_results,
    _searxng_query,
    _has_news_intent,
    tool_searxng_search,
    tool_news_search,
    tool_fetch_webpage,
)

from tools.web_fetch import (  # noqa: F401
    _strip_html,
    _is_censored_response,
    _CENSORSHIP_KEYWORDS,
    _fetch_via_httpx,
    _fetch_via_playwright,
    _fetch_via_selenium,
    _fetch_via_bright_data,
    _fetch_via_oxylabs,
    _fetch_via_wayback_cdx,
    enhanced_web_fetch,
    tool_4plebs_search,
    tool_b4k_search,
    tool_warosu_search,
)

from tools.search_tools2 import (  # noqa: F401
    _QWEN_OMNI_BASE,
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
    tool_youtube_search,
    _extract_video_id,
    tool_youtube_transcript,
    tool_youtube_video_metadata,
    tool_youtube_video_analyze,
)

from tools.tool_executor import (  # noqa: F401
    _simplify_query,
    _retry_tool_call,
    _retry_rare_tool,
    _extract_pdf_text,
    _execute_tool_inner,
    execute_tool,
    tool_knowledge_graph_search,
    tool_knowledge_discover,
    execute_tools_parallel,
)

from tools.llm import (  # noqa: F401
    _dicts_to_langchain_messages,
    call_llm,
    _request_configs,
)

from tools.pipeline import (  # noqa: F401
    QueryComprehension,
    AdmissionResult,
    ConditionStore,
    comprehend_query,
    _validate_source_url,
    _jaccard_similarity,
    _compute_topic_buckets,
)

from tools.planning import (  # noqa: F401
    route_research_question,
    extract_entities_from_conditions,
    verify_conditions,
    _fuzzy_match_claim_to_condition,
    verify_conditions_with_veritas,
)

from tools.subagent import (  # noqa: F401
    plan_research,
    reflect_on_conditions,
    run_subagent,
    _parse_conditions,
)

from tools.tree_reactor import (  # noqa: F401
    _compute_pressure,
    _spawn_sub_questions,
    _extract_entities_for_verification,
    _spawn_verification_nodes,
    _research_single_node,
    tree_research_reactor,
)
# Backward-compat alias used in synthesis.py
run_tree_research_reactor = tree_research_reactor

from tools.synthesis import (  # noqa: F401
    LiveFindingsCollector,
    PersistentResearchState,
    _build_context_aware_status,
    _generate_heartbeat_message,
    _heartbeat_loop,
    _format_curated_event_llm,
    _format_curated_event_fallback,
    _pdr_append_log,
    pdr_node_retrieve,
    _tree_sub_init,
    _tree_sub_explore,
    _build_tree_research_subgraph,
    pdr_node_entities,
    pdr_node_verify,
    pdr_node_persist,
    pdr_node_reflect,
    pdr_node_synthesize,
    build_persistent_research_graph,
    run_persistent_research,
)

# Also re-export the YouTube loader for test compatibility
try:
    from langchain_community.document_loaders import YoutubeLoader  # noqa: F401
except ImportError:
    YoutubeLoader = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# OWUI token validation — protects dashboard/report endpoints
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
    """Validate that the request carries a valid Open WebUI session token."""
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("token", "").strip()
    if not token:
        return False

    if len(_owui_auth_cache) > _OWUI_CACHE_MAX_SIZE:
        _evict_expired_tokens()

    now = time.monotonic()
    if token in _owui_auth_cache and _owui_auth_cache[token] > now:
        return True

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
        return False


def _auth_denied() -> JSONResponse:
    """Return a 401 response for unauthenticated dashboard requests."""
    return JSONResponse(
        {"error": "Authentication required. Please log in via the portal."},
        status_code=401,
    )

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


# ---------------------------------------------------------------------------
# Patch-forwarding module wrapper
# ---------------------------------------------------------------------------
# Tests do ``patch.object(proxy, "http_client", mock)`` which sets the
# attribute on *this* module.  After decomposition the real functions live in
# tools/ sub-modules and look up names in their own module globals, so the
# patch never reaches them.
#
# We replace this module in ``sys.modules`` with a thin wrapper whose
# ``__setattr__`` also propagates the write to the tools sub-module that
# actually owns the name.  This makes all existing test mocks work without
# any test changes.
# ---------------------------------------------------------------------------

import sys as _sys
import types as _types

# Map of name -> list of fully-qualified tools modules that hold the name.
# A name may live in multiple modules (the defining module + every module that
# imported it).  When a test patches the proxy's attribute we must forward the
# write to *all* of them so that every internal call site sees the mock.
_NAME_TO_TOOLS_MODULES: dict[str, list[str]] = {
    # --- config (env vars, constants, shared state) ---
    "APIFY_API_KEY": ["tools.config", "tools.web_fetch"],
    "BRIGHT_DATA_API_KEY": ["tools.config", "tools.moderation", "tools.search_tools2", "tools.web_fetch"],
    "BRIGHT_DATA_CUSTOMER_ID": ["tools.config", "tools.web_fetch"],
    "BRIGHT_DATA_HOST": ["tools.config", "tools.web_fetch"],
    "BRIGHT_DATA_SERP_ZONE": ["tools.config", "tools.moderation"],
    "BRIGHT_DATA_ZONE": ["tools.config", "tools.web_fetch"],
    "COMMERCIAL_SEARCH_ENABLED": ["tools.config", "tools.moderation", "tools.search_tools2"],
    "JSONL_LOG_DIR": ["tools.config", "tools.persistence"],
    "LISTEN_PORT": ["tools.config"],
    "LOG_DIR": ["tools.config"],
    "MAX_CONCURRENT": ["tools.config"],
    "MAX_PRIOR_CONDITIONS": ["tools.config", "tools.subagent"],
    "MAX_RECURSIVE_DEPTH": ["tools.config", "tools.subagent"],
    "MAX_SUBAGENTS": ["tools.config", "tools.subagent"],
    "MAX_SUBAGENT_TURNS": ["tools.config", "tools.subagent"],
    "MODERATION_MODEL": ["tools.config", "tools.moderation"],
    "NOVELTY_EXPAND_THRESHOLD": ["tools.config", "tools.tree_reactor"],
    "NOVELTY_STOP_THRESHOLD": ["tools.config", "tools.tree_reactor"],
    "OWUI_INTERNAL_URL": ["tools.config"],
    "OXYLABS_PASSWORD": ["tools.config", "tools.moderation", "tools.search_tools2", "tools.web_fetch"],
    "OXYLABS_USERNAME": ["tools.config", "tools.moderation", "tools.search_tools2", "tools.web_fetch"],
    "PORTAL_PUBLIC_URL": ["tools.config"],
    "PYTHON_OUTPUT_MAX": ["tools.config", "tools.search_tools2"],
    "PYTHON_TIMEOUT": ["tools.config", "tools.search_tools2"],
    "RESEARCH_NAMESPACE": ["tools.config", "tools.persistence", "tools.synthesis", "tools.tool_executor"],
    "SEARXNG_URL": ["tools.config", "tools.search_tools", "tools.search_tools2"],
    "SUBAGENT_MODEL": ["tools.config", "tools.pipeline", "tools.planning", "tools.subagent", "tools.synthesis", "tools.tool_executor", "tools.tree_reactor"],
    "TREE_MAX_CONCURRENT": ["tools.config", "tools.tree_reactor"],
    "TREE_MAX_DEPTH": ["tools.config", "tools.tree_reactor"],
    "TREE_MAX_NODES": ["tools.config", "tools.tree_reactor"],
    "TREE_PRESSURE_THRESHOLD": ["tools.config", "tools.tree_reactor"],
    "TREE_WORKER_IDLE_TIMEOUT": ["tools.config", "tools.tree_reactor"],
    "UPSTREAM_BASE": ["tools.config", "tools.moderation"],
    "UPSTREAM_KEY": ["tools.config", "tools.moderation"],
    "UPSTREAM_MODEL": ["tools.config", "tools.llm", "tools.synthesis"],
    "VERITAS_HALLUCINATION_THRESHOLD": ["tools.config", "tools.planning", "tools.synthesis"],
    "VERITAS_MIN_CONDITIONS": ["tools.config", "tools.planning", "tools.synthesis"],
    "VERITAS_VERIFY_ENABLED": ["tools.config", "tools.planning", "tools.synthesis"],
    "WEBPAGE_MAX_CHARS": ["tools.config", "tools.search_tools", "tools.search_tools2", "tools.web_fetch"],
    "_STREAM_DONE": ["tools.config", "tools.synthesis"],
    "_curated_queues": ["tools.config", "tools.synthesis"],
    "_live_collectors": ["tools.config", "tools.synthesis"],
    "_metrics_collectors": ["tools.config", "tools.synthesis", "tools.tool_executor"],
    "http_client": ["tools.config", "tools.moderation", "tools.persistence", "tools.search_tools", "tools.search_tools2", "tools.synthesis", "tools.tool_executor", "tools.web_fetch"],
    "get_throttler": ["tools.config", "tools.llm", "tools.moderation", "tools.persistence", "tools.search_tools", "tools.search_tools2", "tools.synthesis", "tools.tool_executor", "tools.web_fetch"],
    "make_sse_chunk": ["tools.config", "tools.persistence", "tools.synthesis"],
    "limiter": ["tools.config"],
    "log": ["tools.config", "tools.llm", "tools.moderation", "tools.persistence", "tools.pipeline", "tools.planning", "tools.search_tools", "tools.search_tools2", "tools.subagent", "tools.synthesis", "tools.tool_executor", "tools.tree_reactor", "tools.web_fetch"],
    "tracker": ["tools.config", "tools.synthesis"],
    "_get_llm": ["tools.config", "tools.llm", "tools.synthesis"],
    "_get_subagent_llm": ["tools.config", "tools.synthesis"],
    "_get_synthesis_llm": ["tools.config", "tools.synthesis"],
    # --- models ---
    "AtomicCondition": ["tools.models", "tools.persistence", "tools.pipeline", "tools.planning", "tools.search_tools2", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    "CrossRef": ["tools.models", "tools.pipeline"],
    "ResearchNode": ["tools.models", "tools.tree_reactor"],
    "SubagentResult": ["tools.models", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    # --- scoring ---
    "serendipity_score": ["tools.scoring", "tools.pipeline", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    "trust_score_url": ["tools.scoring", "tools.pipeline", "tools.search_tools", "tools.search_tools2", "tools.subagent", "tools.synthesis"],
    # --- persistence ---
    "_append_jsonl": ["tools.persistence", "tools.synthesis"],
    "_ensure_jsonl_dir": ["tools.persistence"],
    "_is_large_document": ["tools.persistence"],
    "_log_conditions_jsonl": ["tools.persistence", "tools.synthesis"],
    "_log_entities_jsonl": ["tools.persistence", "tools.synthesis"],
    "_retrieve_graph_neighbors": ["tools.persistence", "tools.synthesis"],
    "_retrieve_related": ["tools.persistence", "tools.synthesis"],
    "_store_conditions_neo4j": ["tools.persistence", "tools.synthesis"],
    "_store_entities_neo4j": ["tools.persistence", "tools.synthesis"],
    "run_document_ingestion": ["tools.persistence"],
    # --- tool_defs ---
    "LANGCHAIN_TOOLS": ["tools.tool_defs"],
    "NATIVE_TOOLS": ["tools.tool_defs", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    # --- moderation ---
    "_commercial_search": ["tools.moderation", "tools.search_tools", "tools.search_tools2"],
    "_get_moderation_llm": ["tools.moderation"],
    "_search_bright_data_serp": ["tools.moderation"],
    "_search_oxylabs_serp": ["tools.moderation"],
    "classify_query": ["tools.moderation", "tools.search_tools2"],
    "moderate_query": ["tools.moderation"],
    # --- search_tools ---
    "_format_search_results": ["tools.search_tools", "tools.search_tools2"],
    "_has_news_intent": ["tools.search_tools"],
    "_searxng_query": ["tools.search_tools", "tools.search_tools2"],
    "tool_fetch_webpage": ["tools.search_tools", "tools.search_tools2", "tools.tool_executor"],
    "tool_news_search": ["tools.search_tools", "tools.tool_executor"],
    "tool_searxng_search": ["tools.search_tools", "tools.search_tools2", "tools.synthesis", "tools.tool_executor"],
    # --- web_fetch ---
    "_CENSORSHIP_KEYWORDS": ["tools.web_fetch"],
    "_PLAYWRIGHT_AVAILABLE": ["tools.web_fetch"],
    "_SELENIUM_AVAILABLE": ["tools.web_fetch"],
    "_fetch_via_bright_data": ["tools.web_fetch"],
    "_fetch_via_httpx": ["tools.web_fetch"],
    "_fetch_via_oxylabs": ["tools.web_fetch"],
    "_fetch_via_playwright": ["tools.web_fetch"],
    "_fetch_via_selenium": ["tools.web_fetch"],
    "_fetch_via_wayback_cdx": ["tools.web_fetch"],
    "_is_censored_response": ["tools.web_fetch"],
    "_strip_html": ["tools.web_fetch", "tools.search_tools2"],
    "enhanced_web_fetch": ["tools.web_fetch", "tools.search_tools", "tools.search_tools2", "tools.tool_executor"],
    "tool_4plebs_search": ["tools.web_fetch", "tools.tool_executor"],
    "tool_b4k_search": ["tools.web_fetch", "tools.tool_executor"],
    "tool_warosu_search": ["tools.web_fetch", "tools.tool_executor"],
    # --- search_tools2 ---
    "_QWEN_OMNI_BASE": ["tools.search_tools2"],
    "_extract_video_id": ["tools.search_tools2"],
    "_twitter_via_bright_data": ["tools.search_tools2"],
    "_twitter_via_nitter": ["tools.search_tools2"],
    "_twitter_via_oxylabs": ["tools.search_tools2"],
    "tool_archiveorg_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_arxiv_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_forum_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_hackernews_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_pubmed_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_python_exec": ["tools.search_tools2", "tools.tool_executor"],
    "tool_scholar_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_stackexchange_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_substack_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_twitter_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_wayback_fetch": ["tools.search_tools2", "tools.tool_executor"],
    "tool_web_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_wikidata_query": ["tools.search_tools2", "tools.tool_executor"],
    "tool_wikipedia_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_youtube_search": ["tools.search_tools2", "tools.tool_executor"],
    "tool_youtube_transcript": ["tools.search_tools2", "tools.tool_executor"],
    "tool_youtube_video_analyze": ["tools.search_tools2", "tools.tool_executor"],
    "tool_youtube_video_metadata": ["tools.search_tools2", "tools.tool_executor"],
    # --- tool_executor ---
    "_execute_tool_inner": ["tools.tool_executor"],
    "_extract_pdf_text": ["tools.tool_executor"],
    "_retry_rare_tool": ["tools.tool_executor"],
    "_retry_tool_call": ["tools.tool_executor"],
    "_simplify_query": ["tools.tool_executor"],
    "execute_tool": ["tools.tool_executor", "tools.synthesis", "tools.tree_reactor"],
    "execute_tools_parallel": ["tools.tool_executor", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    "tool_knowledge_discover": ["tools.tool_executor"],
    "tool_knowledge_graph_search": ["tools.tool_executor"],
    # --- llm ---
    "_dicts_to_langchain_messages": ["tools.llm", "tools.synthesis"],
    "_request_configs": ["tools.llm", "tools.synthesis"],
    "call_llm": ["tools.llm", "tools.pipeline", "tools.planning", "tools.subagent", "tools.synthesis", "tools.tree_reactor"],
    # --- pipeline ---
    "AdmissionResult": ["tools.pipeline"],
    "ConditionStore": ["tools.pipeline", "tools.synthesis"],
    "QueryComprehension": ["tools.pipeline"],
    "_compute_topic_buckets": ["tools.pipeline"],
    "_jaccard_similarity": ["tools.pipeline"],
    "_validate_source_url": ["tools.pipeline"],
    "comprehend_query": ["tools.pipeline", "tools.tree_reactor"],
    # --- planning ---
    "_fuzzy_match_claim_to_condition": ["tools.planning"],
    "extract_entities_from_conditions": ["tools.planning", "tools.synthesis"],
    "route_research_question": ["tools.planning", "tools.subagent"],
    "verify_conditions": ["tools.planning", "tools.synthesis"],
    "verify_conditions_with_veritas": ["tools.planning", "tools.synthesis"],
    # --- subagent ---
    "_parse_conditions": ["tools.subagent"],
    "plan_research": ["tools.subagent"],
    "reflect_on_conditions": ["tools.subagent", "tools.synthesis"],
    "run_subagent": ["tools.subagent", "tools.tree_reactor"],
    # --- tree_reactor ---
    "_compute_pressure": ["tools.tree_reactor"],
    "_extract_entities_for_verification": ["tools.tree_reactor"],
    "_research_single_node": ["tools.tree_reactor"],
    "_spawn_sub_questions": ["tools.tree_reactor"],
    "_spawn_verification_nodes": ["tools.tree_reactor"],
    "tree_research_reactor": ["tools.tree_reactor", "tools.synthesis"],
    # --- synthesis ---
    "LiveFindingsCollector": ["tools.synthesis"],
    "PersistentResearchState": ["tools.synthesis"],
    "_build_context_aware_status": ["tools.synthesis"],
    "_format_curated_event_fallback": ["tools.synthesis"],
    "_format_curated_event_llm": ["tools.synthesis"],
    "_generate_heartbeat_message": ["tools.synthesis"],
    "_heartbeat_loop": ["tools.synthesis"],
    "_pdr_append_log": ["tools.synthesis"],
    "build_persistent_research_graph": ["tools.synthesis"],
    "pdr_node_entities": ["tools.synthesis"],
    "pdr_node_persist": ["tools.synthesis"],
    "pdr_node_reflect": ["tools.synthesis"],
    "pdr_node_retrieve": ["tools.synthesis"],
    "pdr_node_synthesize": ["tools.synthesis"],
    "_tree_sub_init": ["tools.synthesis"],
    "_tree_sub_explore": ["tools.synthesis"],
    "_build_tree_research_subgraph": ["tools.synthesis"],
    "pdr_node_verify": ["tools.synthesis"],
    "run_persistent_research": ["tools.synthesis"],
}


class _PatchForwardingModule(_types.ModuleType):
    """Module wrapper that forwards ``setattr`` to tools sub-modules.

    When ``unittest.mock.patch.object(proxy, "http_client", mock)`` is
    called, ``setattr(proxy, "http_client", mock)`` fires.  This override
    also applies the write to the tools sub-module that owns the name so
    that the actual function code (which resolves names via its own module
    globals) sees the mock.

    On ``delattr`` (patch cleanup) the original value is restored on the
    tools sub-module as well.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        _targets = _NAME_TO_TOOLS_MODULES.get(name)
        if _targets is not None:
            for _fq in _targets:
                _mod = _sys.modules.get(_fq)
                if _mod is not None:
                    _types.ModuleType.__setattr__(_mod, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        _targets = _NAME_TO_TOOLS_MODULES.get(name)
        if _targets is not None:
            for _fq in _targets:
                _mod = _sys.modules.get(_fq)
                if _mod is not None:
                    try:
                        _types.ModuleType.__delattr__(_mod, name)
                    except AttributeError:
                        pass


# Replace the module in sys.modules with the forwarding wrapper.
_self = _sys.modules[__name__]
_wrapper = _PatchForwardingModule(__name__)
_wrapper.__dict__.update(_self.__dict__)
_wrapper.__file__ = _self.__file__
_wrapper.__loader__ = _self.__loader__
_wrapper.__spec__ = _self.__spec__
_wrapper.__path__ = getattr(_self, "__path__", [])
_wrapper.__package__ = _self.__package__
_sys.modules[__name__] = _wrapper


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Persistent Deep Research Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
