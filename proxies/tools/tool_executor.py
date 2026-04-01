"""
Tool execution: retry wrappers, PDF text extraction, execute_tool dispatcher,
knowledge graph tools, and parallel tool execution.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from typing import Optional

import httpx

import langfuse_config
import social_media_scrapers

from shared import http_client

from .config import (
    log,
)
from .llm import _request_configs
from .rate_governor import governed_request
from .search_cache import cache_get, cache_put
from .tool_health import get_monitor, record_outcome
from .search_tools import (
    tool_news_search,
)
from .web_fetch import (
    enhanced_web_fetch,
    tool_4plebs_search,
    tool_b4k_search,
    tool_warosu_search,
)
from .search_tools2 import (
    tool_twitter_search,
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
    tool_telegram_search,
    tool_onion_fetch,
    tool_darknet_market_search,
    tool_facebook_search,
    tool_discord_search,
    tool_signal_search,
    tool_whatsapp_search,
    tool_crunchbase_search,
    tool_trustpilot_search,
    tool_whois_lookup,
    tool_youtube_search,
    tool_youtube_transcript,
    tool_youtube_video_metadata,
    tool_youtube_video_analyze,
)
from .grok_search import tool_grok_deep_search
from .search_gateway import gateway_search
from .sicry_tools import (
    tool_sicry_search,
    tool_sicry_fetch,
    tool_sicry_check_tor,
    tool_sicry_renew_identity,
)


# ============================================================================
# Retry Wrapper for Tool Execution
# ============================================================================

def _simplify_query(query: str) -> str:
    """Strip a long query down to its core keywords for retry.

    Community/underground search APIs often choke on long natural-language
    queries.  This extracts the 3-5 most significant words.
    """
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "about", "what", "which", "who", "whom",
        "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "and", "but", "or", "if",
    }
    words = [w for w in re.split(r'\W+', query.lower()) if w and w not in stop]
    return " ".join(words[:5]) if words else query


# Tools that benefit from retry with simplified queries
_RARE_SOURCE_TOOLS = {
    "chan_4plebs_search", "chan_b4k_search", "chan_warosu_search",
    "forum_search", "telegram_search", "darknet_market_search",
    "twitter_search", "substack_search", "youtube_search",
    "youtube_transcript", "youtube_video_metadata", "youtube_video_analyze",
    "sicry_search", "sicry_fetch",
}


async def _retry_tool_call(
    coro_factory,
    max_retries: int = 2,
    backoff_base: float = 1.0,
) -> str:
    """Retry a tool call with exponential backoff on transient failures.

    coro_factory is a zero-arg callable that returns a new coroutine each time.
    Only retries on timeout and server errors, not on valid empty results.
    """
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = await coro_factory()
            # Don't retry on valid "no results" — only on actual errors
            prefix = result.lower()[:80]
            if "error" not in prefix and "failed" not in prefix and "timed out" not in prefix:
                return result
            if attempt < max_retries:
                last_error = result
                await asyncio.sleep(backoff_base * (2 ** attempt))
                continue
            return result
        except httpx.TimeoutException:
            last_error = "request timed out"
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (2 ** attempt))
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (2 ** attempt))
                continue

    return f"Tool failed after {max_retries + 1} attempts: {last_error}"


async def _retry_rare_tool(
    tool_name: str,
    arguments: dict,
    req_id: str = "",
) -> str:
    """Retry a rare/community source tool with query simplification.

    For community/underground tools, if the first attempt returns no useful
    results, retry once with a simplified (keyword-only) query.  This handles
    the common case where long natural-language queries return empty on APIs
    that expect short keyword searches.
    """
    result = await execute_tool(tool_name, arguments, req_id)
    prefix = result.lower()[:120]

    # If result looks empty or error-ish, retry with simplified query
    is_empty = (
        "no results" in prefix
        or "0 results" in prefix
        or len(result.strip()) < 30
    )
    is_error = "error" in prefix or "failed" in prefix or "timed out" in prefix

    if (is_empty or is_error) and "query" in arguments:
        original_q = arguments["query"]
        simplified = _simplify_query(original_q)
        if simplified != original_q.lower().strip():
            log.info(
                f"[{req_id}] Retrying {tool_name} with simplified query: "
                f"'{original_q[:60]}' → '{simplified}'"
            )
            retry_args = {**arguments, "query": simplified}
            result2 = await execute_tool(tool_name, retry_args, req_id)
            prefix2 = result2.lower()[:120]
            if len(result2.strip()) > len(result.strip()):
                return result2

    return result


# ============================================================================
# PDF Text Extraction
# ============================================================================

async def _extract_pdf_text(url: str) -> Optional[str]:
    """Download and extract text from a PDF document.

    Uses PyMuPDF (fitz) if available, falls back to pdfplumber.
    Returns extracted text or None on failure.
    """
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
        )
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            return None

        pdf_bytes = resp.content

        # Try PyMuPDF first (fastest)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page_num in range(min(doc.page_count, 50)):  # Cap at 50 pages
                page = doc[page_num]
                pages_text.append(page.get_text())
            doc.close()
            text = "\n\n".join(pages_text).strip()
            if text:
                return text
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"PyMuPDF extraction failed for {url}: {e}")

        # Fallback: pdfplumber
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages[:50]:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n\n".join(pages_text).strip()
                if text:
                    return text
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"pdfplumber extraction failed for {url}: {e}")

        return None
    except Exception as e:
        log.debug(f"PDF download failed for {url}: {e}")
        return None


async def _execute_tool_inner(tool_name: str, arguments: dict) -> str:
    """Route and execute a tool call (inner implementation)."""
    if tool_name == "searxng_search":
        return await tool_web_search(arguments.get("query", ""))
    elif tool_name == "news_search":
        return await tool_news_search(
            arguments.get("query", ""),
            arguments.get("time_range", "week"),
        )
    elif tool_name == "fetch_webpage":
        # Use enhanced_web_fetch for the full fallback chain
        return await enhanced_web_fetch(
            arguments.get("url", ""),
            arguments.get("extract_info", ""),
        )
    elif tool_name == "python_exec":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool_python_exec, arguments.get("code", ""))
    elif tool_name == "arxiv_search":
        return await tool_arxiv_search(
            arguments.get("query", ""),
            arguments.get("max_results", 5),
        )
    elif tool_name == "wayback_fetch":
        return await tool_wayback_fetch(arguments.get("url", ""))
    elif tool_name == "wikidata_query":
        return await tool_wikidata_query(arguments.get("entity", ""))
    elif tool_name == "knowledge_graph_search":
        return await tool_knowledge_graph_search(arguments)
    elif tool_name == "knowledge_discover":
        return await tool_knowledge_discover(arguments)
    elif tool_name == "chan_4plebs_search":
        return await tool_4plebs_search(
            arguments.get("query", ""),
            arguments.get("board", "pol"),
        )
    elif tool_name == "chan_b4k_search":
        return await tool_b4k_search(arguments.get("query", ""))
    elif tool_name == "chan_warosu_search":
        return await tool_warosu_search(
            arguments.get("query", ""),
            arguments.get("board", "g"),
        )
    elif tool_name == "twitter_search":
        return await tool_twitter_search(arguments.get("query", ""))
    elif tool_name == "social_media_search":
        return await social_media_scrapers.tool_social_media_search(
            platform=arguments.get("platform", ""),
            query=arguments.get("query", ""),
            subreddit=arguments.get("subreddit", ""),
            result_type=arguments.get("result_type", "posts"),
        )
    elif tool_name == "reddit_search":
        return await social_media_scrapers.tool_reddit_search(
            query=arguments.get("query", ""),
            subreddit=arguments.get("subreddit", ""),
            sort=arguments.get("sort", "relevance"),
        )
    elif tool_name == "instagram_search":
        return await social_media_scrapers.tool_instagram_search(
            query=arguments.get("query", ""),
            result_type=arguments.get("result_type", "posts"),
        )
    elif tool_name == "tiktok_search":
        return await social_media_scrapers.tool_tiktok_search(
            query=arguments.get("query", ""),
            result_type=arguments.get("result_type", "posts"),
        )
    elif tool_name == "linkedin_search":
        return await social_media_scrapers.tool_linkedin_search(
            query=arguments.get("query", ""),
            result_type=arguments.get("result_type", "posts"),
        )
    elif tool_name == "youtube_search":
        return await social_media_scrapers.tool_youtube_search(
            query=arguments.get("query", ""),
            result_type=arguments.get("result_type", "videos"),
        )
    elif tool_name == "hackernews_search":
        return await tool_hackernews_search(
            arguments.get("query", ""),
            arguments.get("sort_by", "relevance"),
            arguments.get("time_range", ""),
        )
    elif tool_name == "stackexchange_search":
        return await tool_stackexchange_search(
            arguments.get("query", ""),
            arguments.get("site", "stackoverflow"),
            arguments.get("sort", "relevance"),
        )
    elif tool_name == "pubmed_search":
        return await tool_pubmed_search(
            arguments.get("query", ""),
            arguments.get("max_results", 10),
        )
    elif tool_name == "wikipedia_search":
        return await tool_wikipedia_search(
            arguments.get("query", ""),
            arguments.get("limit", 8),
        )
    elif tool_name == "archiveorg_search":
        return await tool_archiveorg_search(
            arguments.get("query", ""),
            arguments.get("media_type", ""),
            arguments.get("max_results", 10),
        )
    elif tool_name == "forum_search":
        return await tool_forum_search(
            arguments.get("query", ""),
            arguments.get("forum_url", ""),
        )
    elif tool_name == "scholar_search":
        return await tool_scholar_search(arguments.get("query", ""))
    elif tool_name == "substack_search":
        return await tool_substack_search(arguments.get("query", ""))
    elif tool_name == "telegram_search":
        return await tool_telegram_search(
            arguments.get("query", ""),
            arguments.get("platform", ""),
        )
    elif tool_name == "onion_fetch":
        return await tool_onion_fetch(arguments.get("url", ""))
    elif tool_name == "darknet_market_search":
        return await tool_darknet_market_search(arguments.get("query", ""))
    elif tool_name == "facebook_search":
        return await tool_facebook_search(
            arguments.get("query", ""),
            arguments.get("result_type", "posts"),
            arguments.get("platform", ""),
        )
    elif tool_name == "discord_search":
        return await tool_discord_search(arguments.get("query", ""))
    elif tool_name == "signal_search":
        return await tool_signal_search(arguments.get("query", ""))
    elif tool_name == "whatsapp_search":
        return await tool_whatsapp_search(arguments.get("query", ""))
    elif tool_name == "crunchbase_search":
        return await tool_crunchbase_search(arguments.get("query", ""))
    elif tool_name == "trustpilot_search":
        return await tool_trustpilot_search(arguments.get("query", ""))
    elif tool_name == "whois_lookup":
        return await tool_whois_lookup(
            domain=arguments.get("domain", ""),
            query=arguments.get("query", ""),
        )
    elif tool_name == "youtube_transcript":
        return await tool_youtube_transcript(
            arguments.get("url", ""),
            arguments.get("lang", "en"),
        )
    elif tool_name == "youtube_video_metadata":
        return await tool_youtube_video_metadata(arguments.get("url", ""))
    elif tool_name == "youtube_video_analyze":
        return await tool_youtube_video_analyze(
            arguments.get("url", ""),
            arguments.get("question", ""),
        )
    elif tool_name == "grok_deep_search":
        return await tool_grok_deep_search(
            arguments.get("query", ""),
            search_type=arguments.get("search_type", "both"),
            instructions=arguments.get("instructions", ""),
        )
    elif tool_name == "search_gateway":
        return await gateway_search(
            arguments.get("query", ""),
            sources=arguments.get("sources", "all"),
            search_type=arguments.get("search_type", "both"),
            max_results_per_source=arguments.get("max_results_per_source", 10),
            req_id="",
        )
    elif tool_name == "sicry_search":
        return await tool_sicry_search(
            arguments.get("query", ""),
            max_results=arguments.get("max_results", 20),
            engines=arguments.get("engines"),
        )
    elif tool_name == "sicry_fetch":
        return await tool_sicry_fetch(arguments.get("url", ""))
    elif tool_name == "sicry_check_tor":
        return await tool_sicry_check_tor()
    elif tool_name == "sicry_renew_identity":
        return await tool_sicry_renew_identity()
    else:
        return f"[TOOL_ERROR] Unknown tool: {tool_name}. This tool does not exist in the system."


# Tools whose results are cacheable (search-type tools)
_CACHEABLE_TOOLS = {
    "searxng_search", "web_search", "news_search", "arxiv_search",
    "wikidata_query", "hackernews_search", "stackexchange_search",
    "pubmed_search", "wikipedia_search", "archiveorg_search",
    "forum_search", "scholar_search", "substack_search",
    "youtube_search", "youtube_transcript", "youtube_video_metadata",
    "twitter_search", "telegram_search", "onion_fetch", "darknet_market_search",
    "facebook_search", "discord_search", "signal_search",
    "whatsapp_search", "crunchbase_search", "trustpilot_search",
    "whois_lookup", "wayback_fetch",
    "social_media_search", "reddit_search", "instagram_search",
    "tiktok_search", "linkedin_search",
    "chan_4plebs_search", "chan_b4k_search", "chan_warosu_search",
    "grok_deep_search", "search_gateway",
    "sicry_search", "sicry_fetch", "sicry_check_tor", "sicry_renew_identity",
}

# Tools that involve long-running local computation (e.g. WhisperX GPU
# transcription) and should NOT hold a global concurrency slot.  They are
# still cacheable but bypass the rate governor entirely.
_UNGOVERNED_HEAVY_TOOLS: set[str] = {
    "youtube_transcript",    # WhisperX can take up to 300s of local GPU work
    "youtube_video_analyze", # downloads + analyses video locally
}

# Tools that access the internet (governed by rate limiter).
# Excludes heavy local-compute tools that would starve the global semaphore.
_GOVERNED_TOOLS = (_CACHEABLE_TOOLS | {"fetch_webpage"}) - _UNGOVERNED_HEAVY_TOOLS


def _extract_query_for_cache(tool_name: str, arguments: dict) -> str:
    """Extract a normalized cache-relevant string from tool arguments.

    Normalizes the primary query field (case-folding, stop-word removal,
    word-order sorting) so that near-duplicate queries hit the cache,
    while preserving all other parameters (subreddit, platform, etc.)
    to avoid collisions.
    """
    from .search_cache import normalize_query

    # Keys that contain the primary natural-language query
    _QUERY_KEYS = ("query", "search_query", "question", "term", "keywords")

    normalized = dict(arguments)
    for key in _QUERY_KEYS:
        val = normalized.get(key)
        if isinstance(val, str) and val and not val.startswith("http"):
            normalized[key] = normalize_query(val)
    return json.dumps(normalized, sort_keys=True)


async def execute_tool(
    tool_name: str,
    arguments: dict,
    req_id: str = "",
) -> str:
    """Route and execute a tool call with rate limiting, caching, and health tracking.

    Wraps the inner tool execution with:
      1. Search result cache (check before, store after)
      2. Rate governor (per-provider throttling + global concurrency)
      3. Tool health monitor (track success/failure rates)
      4. LangChain callbacks for metrics
    """
    config = _request_configs.get(req_id, {}) if req_id else {}
    callbacks = config.get("callbacks", [])
    run_id = uuid.uuid4()
    serialized = {"name": tool_name}
    input_str = json.dumps(arguments)[:500]

    # Fire on_tool_start for all registered callbacks
    for cb in callbacks:
        try:
            cb.on_tool_start(serialized, input_str, run_id=run_id)
        except Exception:
            pass

    # --- Step 0: Open trace span ---
    tool_span = langfuse_config.start_span(
        req_id, f"tool:{tool_name}",
        input={"arguments": input_str},
    )

    # --- Step 1: Check cache FIRST — cached results bypass circuit breaker ---
    if tool_name in _CACHEABLE_TOOLS:
        query_str = _extract_query_for_cache(tool_name, arguments)
        cached = cache_get(tool_name, query_str)
        if cached is not None:
            log.debug(f"[{req_id}] Cache hit for {tool_name}: {query_str[:60]}")
            langfuse_config.end_span(tool_span, output={"cache": "hit", "result_len": len(cached)})
            for cb in callbacks:
                try:
                    cb.on_tool_end(f"[CACHED] {cached[:1000]}", run_id=run_id)
                except Exception:
                    pass
            return cached

    # --- Step 2: Circuit breaker — skip tools that are consistently failing ---
    # Runs AFTER cache check so cached results are still served for degraded tools.
    # Uses a half-open probe: after TOOL_HEALTH_RECOVERY_SECS of being blocked,
    # one call is allowed through.  If it succeeds, record_outcome(success=True)
    # resets consecutive_failures and the tool returns to healthy.
    _RECOVERY_SECS = int(os.environ.get("TOOL_HEALTH_RECOVERY_SECS", "120"))
    monitor = get_monitor()
    tool_status = monitor.get_tool_status(tool_name)
    if tool_status.get("status") == "degraded":
        last_err_t = tool_status.get("last_error_time") or 0.0
        elapsed = time.time() - last_err_t
        if elapsed < _RECOVERY_SECS:
            # Still in cooldown — block the call
            streak = tool_status.get("consecutive_failures", 0)
            blocked_msg = (
                f"[TOOL_BLOCKED] {tool_name} is currently degraded "
                f"({streak} consecutive failures, recovery probe in "
                f"{int(_RECOVERY_SECS - elapsed)}s). "
                f"Last error: {(tool_status.get('last_error') or 'unknown')[:200]}"
            )
            log.warning(f"[{req_id}] Circuit breaker blocked {tool_name}: {streak} failures, {int(elapsed)}s/{_RECOVERY_SECS}s cooldown")
            langfuse_config.end_span(tool_span, output={"circuit_breaker": "blocked", "streak": streak}, level="WARNING")
            for cb in callbacks:
                try:
                    cb.on_tool_end(blocked_msg, run_id=run_id)
                except Exception:
                    pass
            return blocked_msg
        else:
            # Cooldown expired — allow one probe call through
            log.info(f"[{req_id}] Circuit breaker half-open probe for {tool_name} after {int(elapsed)}s cooldown")

    # --- Step 3: Execute with rate governor ---
    try:
        if tool_name in _GOVERNED_TOOLS:
            async with governed_request(tool_name):
                result = await _execute_tool_inner(tool_name, arguments)
        else:
            result = await _execute_tool_inner(tool_name, arguments)
    except Exception as e:
        error_str = str(e)
        record_outcome(tool_name, success=False, error=error_str)
        langfuse_config.end_span(tool_span, output={"error": error_str[:200]}, level="ERROR")
        for cb in callbacks:
            try:
                cb.on_tool_error(e, run_id=run_id)
            except Exception:
                pass
        return f"Tool error ({tool_name}): {error_str}"

    # --- Step 4: Record health outcome ---
    result_prefix = result[:80]
    is_error = (
        result_prefix.lower().startswith("error")
        or result_prefix.lower().startswith("failed")
        or "search error:" in result_prefix.lower()
        or "timed out" in result_prefix.lower()
        or result.startswith("[TOOL_ERROR]")
        or result.startswith("Tool error")
        or result.startswith("Unknown tool:")
    )
    if is_error:
        record_outcome(tool_name, success=False, error=result[:500])
    else:
        record_outcome(tool_name, success=True)

    # --- Step 5: Store in cache ---
    if tool_name in _CACHEABLE_TOOLS and not is_error:
        query_str = _extract_query_for_cache(tool_name, arguments)
        cache_put(tool_name, query_str, result)

    langfuse_config.end_span(tool_span, output={
        "result_len": len(result),
        "is_error": is_error,
        "cached": tool_name in _CACHEABLE_TOOLS and not is_error,
    })

    # Fire on_tool_end for all registered callbacks
    for cb in callbacks:
        try:
            cb.on_tool_end(result[:1000], run_id=run_id)
        except Exception:
            pass

    return result


async def tool_knowledge_graph_search(arguments: dict) -> str:
    """Search the knowledge graph via the knowledge engine microservice."""
    try:
        import knowledge_client
        result = await knowledge_client.search(
            namespace=arguments.get("namespace", "default"),
            query=arguments.get("query", ""),
            mode=arguments.get("mode", "hybrid"),
            limit=min(arguments.get("limit", 10), 50),
        )
        results = result.get("results", [])
        if not results:
            return "No matching knowledge found in the graph."

        formatted = []
        for i, r in enumerate(results, 1):
            node_type = r.get("node_type", "")
            name = r.get("name", "")
            content = r.get("content", "")
            score = r.get("score", 0)
            props = r.get("properties", {})
            source_doc = r.get("source_doc", "")

            header = f"{i}. [{node_type}]"
            if name:
                header += f" **{name}**"
            if source_doc:
                header += f" (from: {source_doc})"
            header += f" [score: {score:.3f}]"

            body = content[:2000] if content else ""
            if props:
                prop_strs = []
                for k, v in props.items():
                    if k not in ("id",) and v is not None:
                        prop_strs.append(f"{k}: {v}")
                if prop_strs:
                    body += "\n  Properties: " + ", ".join(prop_strs[:5])

            formatted.append(f"{header}\n{body}" if body else header)

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Knowledge graph search error: {e}"


async def tool_knowledge_discover(arguments: dict) -> str:
    """Run graph discovery algorithms via the knowledge engine microservice."""
    try:
        import knowledge_client
        algorithm = arguments.get("algorithm", "")
        namespace = arguments.get("namespace", "default")
        seed_concepts = arguments.get("seed_concepts", [])
        limit = arguments.get("limit", 15)

        if algorithm == "spreading_activation":
            if not seed_concepts:
                return "Error: seed_concepts required for spreading_activation"
            result = await knowledge_client.spreading_activation(
                namespace=namespace,
                seed_concepts=seed_concepts,
                limit=limit,
            )
        elif algorithm == "swanson_abc":
            if not seed_concepts:
                return "Error: seed_concepts required for swanson_abc"
            result = await knowledge_client.swanson_abc(
                namespace=namespace,
                seed_concept=seed_concepts[0],
                limit=limit,
            )
        elif algorithm == "information_gaps":
            result = await knowledge_client.information_gaps(
                namespace=namespace,
                limit=limit,
            )
        else:
            return f"Unknown algorithm: {algorithm}. Use: spreading_activation, swanson_abc, information_gaps"

        discoveries = result.get("results", [])
        if not discoveries:
            return f"No discoveries from {algorithm}."

        formatted = [f"**{algorithm.replace('_', ' ').title()} Results:**\n"]
        for i, d in enumerate(discoveries, 1):
            parts = [f"{i}."]
            if "name" in d:
                parts.append(f"**{d['name']}**")
            elif "target_concept" in d:
                parts.append(f"**{d['target_concept']}**")
            if "activation" in d:
                parts.append(f"(activation: {d['activation']:.3f})")
            if "discovery_score" in d:
                parts.append(f"(discovery score: {d['discovery_score']:.3f})")
            if "gap_score" in d:
                parts.append(f"(gap score: {d['gap_score']:.3f})")
            if "bridge_count" in d:
                parts.append(f"via {d['bridge_count']} bridge concepts")
            if "top_bridges" in d:
                bridge_names = [b.get("name", "?") for b in d["top_bridges"][:3]]
                parts.append(f"[bridges: {', '.join(bridge_names)}]")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    except Exception as e:
        return f"Knowledge discovery error: {e}"


async def execute_tools_parallel(
    tool_calls_with_ids: list[tuple[str, str, dict]],
    req_id: str = "",
) -> list[tuple[str, str, str, float]]:
    """Execute multiple tool calls concurrently."""

    async def _run_one(tc_id: str, name: str, args: dict):
        t0 = time.monotonic()
        result = await execute_tool(name, args, req_id=req_id)
        return tc_id, name, result, time.monotonic() - t0

    tasks = [_run_one(tc_id, name, args) for tc_id, name, args in tool_calls_with_ids]
    return list(await asyncio.gather(*tasks))

