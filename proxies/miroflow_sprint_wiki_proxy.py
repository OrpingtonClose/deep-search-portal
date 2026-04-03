#!/usr/bin/env python3
"""
MiroFlow Sprint + Wiki — wiki-enabled variant of the MiroFlow Sprint proxy.

Same fast 2-3 round research as MiroFlow Sprint, but with the agentic
knowledge-wiki builder enabled: an LLM agent reads the live research state
and incrementally constructs an encyclopedia-style article, emitted as a
LibreChat Artifact in the side pane.

Designed to run as a separate process (separate screen session) on its own
port so the overrides do not affect the non-wiki sprint proxy.

Default port: 9900 (configurable via MIROFLOW_SPRINT_WIKI_PORT).
Model ID: miroflow-sprint-wiki
"""

import os

# ── Environment overrides — MUST happen before any tools/ import ──────────
# Enable the agentic wiki builder background task.
os.environ["WIKI_AGENT_ENABLED"] = "1"

# Sprint-mode research scope overrides (same as miroflow_sprint_proxy.py)
os.environ.setdefault("MAX_SUBAGENT_TURNS", "2")       # 2 turns per subagent (vs 15)
os.environ.setdefault("TREE_MAX_DEPTH", "2")            # shallow tree (vs 5)
os.environ.setdefault("TREE_MAX_NODES", "8")             # tiny budget (vs 50)
os.environ.setdefault("TREE_MAX_CONCURRENT", "5")        # fewer workers (vs 10)
os.environ.setdefault("RESEARCH_TIME_LIMIT", "90")       # 90s research phase (vs 300)
os.environ.setdefault("PIPELINE_HARD_TIMEOUT", "180")    # 3 min hard cap (vs 420)
os.environ.setdefault("MAX_RESEARCH_ITERATIONS", "1")    # no re-research loops
os.environ.setdefault("MAX_SUBAGENTS", "3")              # fewer legacy subagents
os.environ.setdefault("MAX_RECURSIVE_DEPTH", "1")        # minimal recursion

# Use a distinct port so it can run alongside other proxies.
_SPRINT_WIKI_PORT = int(os.environ.get("MIROFLOW_SPRINT_WIKI_PORT", "9900"))
os.environ["PERSISTENT_RESEARCH_PORT"] = str(_SPRINT_WIKI_PORT)

# Distinct log directory to avoid interleaving with other proxies.
os.environ.setdefault(
    "PERSISTENT_RESEARCH_LOG_DIR",
    "/opt/persistent_research_logs/sprint-wiki",
)
os.environ.setdefault(
    "JSONL_LOG_DIR",
    "/opt/persistent_research_logs/sprint-wiki/jsonl",
)

# ── Now safe to import everything from the tools/ package ─────────────────

import uuid  # noqa: E402

from fastapi import Request  # noqa: E402
from fastapi.responses import StreamingResponse, JSONResponse  # noqa: E402

from shared import (  # noqa: E402
    create_app,
    extract_user_text_with_attachments,
    is_utility_request,
    parse_attachments,
    register_standard_routes,
    stream_passthrough,
    utility_passthrough_json,
)

from tools.config import (  # noqa: E402
    LOG_DIR,
    UPSTREAM_BASE,
    UPSTREAM_KEY,
    UPSTREAM_MODEL,
    SUBAGENT_MODEL,
    SEARXNG_URL,
    MAX_SUBAGENT_TURNS,
    TREE_MAX_CONCURRENT,
    TREE_MAX_DEPTH,
    TREE_MAX_NODES,
    RESEARCH_NAMESPACE,
    JSONL_LOG_DIR,
    WIKI_AGENT_ENABLED,
    tracker,
    limiter,
    log,
)

from tools.tool_defs import NATIVE_TOOLS  # noqa: E402

from tools.persistence import (  # noqa: E402
    _is_large_document,
    run_document_ingestion,
)

from tools.conversation import (  # noqa: E402
    derive_conversation_id,
    get_conversation_store,
    merge_research_focus,
)

from tools.synthesis import (  # noqa: E402
    run_persistent_research,
)

import langfuse_config  # noqa: E402, F401  # imported for side-effects (trace registration)

# ── FastAPI Application ───────────────────────────────────────────────────

MODEL_ID = "miroflow-sprint-wiki"
MODEL_DISPLAY = "MiroFlow Sprint + Wiki"

app = create_app("MiroFlow Sprint Wiki Proxy")

register_standard_routes(
    app,
    service_name="miroflow-sprint-wiki-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "variant": "sprint-wiki",
        "upstream": UPSTREAM_BASE,
        "synthesis_model": UPSTREAM_MODEL,
        "subagent_model": SUBAGENT_MODEL,
        "searxng": SEARXNG_URL,
        "wiki_agent_enabled": WIKI_AGENT_ENABLED,
        "max_subagent_turns": MAX_SUBAGENT_TURNS,
        "tree_max_concurrent": TREE_MAX_CONCURRENT,
        "tree_max_depth": TREE_MAX_DEPTH,
        "tree_max_nodes": TREE_MAX_NODES,
        "research_namespace": RESEARCH_NAMESPACE,
        "jsonl_log_dir": JSONL_LOG_DIR,
        "tools": [t["function"]["name"] for t in NATIVE_TOOLS],
    },
)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 1700000000,
            "owned_by": "miroflow-sprint-wiki-proxy",
            "name": MODEL_DISPLAY,
        }],
    })


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
        f"[{req_id}] Sprint+Wiki request: messages={len(messages)}, "
        f"model={body.get('model', '?')}, utility={utility}"
    )

    tracker.start(req_id, utility=utility, messages=len(messages), phase="init")

    if utility:
        client_wants_stream = body.get("stream", True)
        if not client_wants_stream:
            log.info(f"[{req_id}] Routing to NON-STREAMING utility passthrough")
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=UPSTREAM_BASE,
                upstream_key=UPSTREAM_KEY,
                upstream_model=UPSTREAM_MODEL,
                log=log,
            )
            tracker.finish(req_id)
            return result
        log.info(f"[{req_id}] Routing to PASSTHROUGH (utility)")
        generator = stream_passthrough(
            messages, body,
            req_id=req_id,
            upstream_base=UPSTREAM_BASE,
            upstream_key=UPSTREAM_KEY,
            upstream_model=UPSTREAM_MODEL,
            model_id=body.get("model", MODEL_ID),
            tracker=tracker,
            log=log,
        )
    else:
        user_text = extract_user_text_with_attachments(messages)
        parsed = parse_attachments(user_text)

        # Prompt inheritance (same logic as the full proxy)
        prior_focus = ""
        conversation_id = ""
        try:
            conversation_id = derive_conversation_id(
                messages, chat_id=body.get("chat_id"),
            )
            store = get_conversation_store()
            latest = store.get_latest_turn(conversation_id)
            if latest is not None:
                prior_focus = latest.user_query
                log.info(
                    f"[{req_id}] Prior focus (conv={conversation_id}): "
                    f"{prior_focus[:80]!r}"
                )
        except Exception as e:
            log.warning(f"[{req_id}] Could not load prior focus: {e}")

        if parsed.has_attachments:
            doc_summary = ", ".join(
                f"{d.filename} ({len(d.content):,} chars)"
                for d in parsed.documents
            )
            log.info(
                f"[{req_id}] ATTACHMENT DETECTED: {len(parsed.documents)} "
                f"doc(s) [{doc_summary}]"
            )

            raw_prompt = parsed.prompt
            if raw_prompt and prior_focus:
                effective_prompt = await merge_research_focus(
                    prior_focus, raw_prompt, req_id,
                )
            elif raw_prompt:
                effective_prompt = raw_prompt
            elif prior_focus:
                effective_prompt = prior_focus
            else:
                effective_prompt = (
                    "Analyse the attached document(s) thoroughly."
                )

            augmented_messages = [
                msg for msg in messages
                if not (
                    msg.get("role") == "system"
                    and isinstance(msg.get("content", ""), str)
                    and msg["content"].lstrip().startswith("Attached document(s):")
                )
            ]
            for i in range(len(augmented_messages) - 1, -1, -1):
                if augmented_messages[i].get("role") == "user":
                    doc_system_msg = {
                        "role": "system",
                        "content": (
                            "The user has attached document(s) for research. "
                            "Treat as PRIMARY sources.\n\n"
                            "=== ATTACHED DOCUMENTS ===\n\n"
                            + parsed.all_document_text
                            + "\n\n=== END DOCUMENTS ==="
                        ),
                    }
                    augmented_messages.insert(i, doc_system_msg)
                    augmented_messages[i + 1] = {
                        **augmented_messages[i + 1],
                        "content": effective_prompt,
                    }
                    break

            if not limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": "Too many concurrent sprint+wiki sessions. Try again shortly.",
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Sprint+Wiki RESEARCH (with docs)")

            async def _guarded_research_with_docs():
                async with limiter.hold():
                    async for event in run_persistent_research(
                        augmented_messages, body, req_id,
                        conversation_id_override=conversation_id,
                    ):
                        yield event

            generator = _guarded_research_with_docs()

        elif _is_large_document(parsed.prompt or user_text):
            log.info(f"[{req_id}] Routing to DOCUMENT INGESTION")

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
            if prior_focus:
                effective_prompt = await merge_research_focus(
                    prior_focus, user_text, req_id,
                )
                if effective_prompt != user_text:
                    messages = list(messages)
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i] = {
                                **messages[i],
                                "content": effective_prompt,
                            }
                            break
                    log.info(
                        f"[{req_id}] Merged focus: "
                        f"{user_text[:60]!r} -> {effective_prompt[:80]!r}"
                    )

            if not limiter.available():
                tracker.finish(req_id)
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": "Too many concurrent sprint+wiki sessions. Try again shortly.",
                            "type": "rate_limit",
                        }
                    },
                )

            log.info(f"[{req_id}] Sprint+Wiki RESEARCH")

            async def _guarded_research():
                async with limiter.hold():
                    async for event in run_persistent_research(
                        messages, body, req_id,
                        conversation_id_override=conversation_id,
                    ):
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


# ── Uvicorn entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info(
        f"Starting {MODEL_DISPLAY} on port {_SPRINT_WIKI_PORT} "
        f"(wiki_agent={WIKI_AGENT_ENABLED}, turns={MAX_SUBAGENT_TURNS}, "
        f"depth={TREE_MAX_DEPTH}, nodes={TREE_MAX_NODES})"
    )
    uvicorn.run(app, host="0.0.0.0", port=_SPRINT_WIKI_PORT)
