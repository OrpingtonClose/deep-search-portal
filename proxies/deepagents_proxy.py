#!/usr/bin/env python3
"""
Deep Agents Proxy — LangGraph deepagents SDK wrapped as an OpenAI-compatible
endpoint, routed through Venice AI's GLM-4.7 Flash Heretic model.

The deepagents SDK (https://github.com/OrpingtonClose/deepagents) provides a
pre-wired LangGraph "deep agent" with built-in planning (write_todos),
virtual filesystem (ls / read_file / write_file / edit_file / glob / grep),
and sub-agent delegation (task). We expose that agent as a drop-in OpenAI
chat-completions server so LibreChat can speak to it like any other model.

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes them through
    to Venice AI directly (bypassing the deep-agent graph)
  - For real chat requests:
      * Converts ChatML messages to LangChain BaseMessage objects
      * Invokes the compiled LangGraph agent via ``astream_events(version="v2")``
      * Streams ``on_chat_model_stream`` tokens as SSE ``content`` chunks
      * Streams ``on_tool_start`` / ``on_tool_end`` events as ``reasoning_content``
        so LibreChat renders them in a collapsible Thinking block
      * Non-streaming requests collect the final AI message via ``ainvoke``

Runs as a FastAPI app under uvicorn in a screen session (default port 8200).
"""

import asyncio
import base64
import json
import os
import shutil
import time
import traceback
import uuid
from typing import Any, AsyncGenerator, Optional
from urllib.parse import quote as _urlquote

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from deepagents import SubAgent, create_deep_agent

from shared import (
    RequestTracker,
    create_app,
    env_int,
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
    utility_passthrough_json,
)

# --- Logging ---
LOG_DIR = os.getenv("DEEPAGENTS_LOG_DIR", "/opt/deepagents_proxy_logs")
log = setup_logging("deepagents-proxy", LOG_DIR)

# --- Configuration ---
VENICE_API_BASE = os.getenv("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_API_KEY = require_env("VENICE_API_KEY")
DEEPAGENTS_MODEL = os.getenv(
    "DEEPAGENTS_MODEL", "olafangensan-glm-4.7-flash-heretic"
)
LISTEN_PORT = env_int("DEEPAGENTS_PORT", 8200, minimum=1)

# Model ID we advertise to LibreChat via /v1/models.
MODEL_ID = "deepagents-research"

# --- File downloads: persist agent-written files + expose public URLs ---
# The deepagents SDK's filesystem tools (write_file / edit_file) write into
# an in-memory LangGraph StateBackend — by default those files are discarded
# when the turn ends and the user sees path references like ``report.md`` in
# the narrative with nothing to click. We persist every file in the final
# state to disk under ``FILES_DIR/{turn_id}/`` and append a "Generated files"
# markdown section with public URLs to the assistant message, so the user can
# actually download what the agent produced.
FILES_DIR = os.getenv("DEEPAGENTS_FILES_DIR", os.path.join(LOG_DIR, "files"))
PUBLIC_BASE_URL = os.getenv("DEEPAGENTS_PUBLIC_BASE_URL", "").rstrip("/")
FILES_TTL_DAYS = env_int("DEEPAGENTS_FILES_TTL_DAYS", 7, minimum=1)
os.makedirs(FILES_DIR, exist_ok=True)

log.info(
    f"Config: venice_base={VENICE_API_BASE}, model={DEEPAGENTS_MODEL}, "
    f"port={LISTEN_PORT}, model_id={MODEL_ID}"
)
log.info(
    f"Files: dir={FILES_DIR}, public_base={PUBLIC_BASE_URL or '(unset)'}, "
    f"ttl={FILES_TTL_DAYS}d"
)

tracker = RequestTracker()


# ============================================================================
# System prompts
# ============================================================================

# NOTE: The deepagents SDK appends its own BASE_AGENT_PROMPT (planning,
# filesystem, sub-agent instructions) after our custom system prompt, so this
# prompt should ONLY contain the domain-specific research rules we want to
# layer on top.

SYSTEM_PROMPT = """\
You are a deep research agent operating inside the Deep Search Portal. Your job \
is to run thorough, evidence-based investigations and deliver accurate, \
unvarnished answers.

CARDINAL RULE — NEVER ANSWER FROM MEMORY ALONE
Every non-trivial factual claim must be grounded in information you have \
actively gathered during this session (delegated research, scratchpad files, \
or prior tool output in context). If you do not have grounding, say so and \
seek more information via your sub-agents or by writing a plan to get it.

ANTI-REFRAMING RULE
Honour the user's exact query. Do NOT silently narrow, soften, moralize, or \
substitute a safer-sounding question. If the user asks about a controversial \
topic, investigate that topic — not a sanitized version of it.

SOURCE DIVERSITY REQUIREMENT
Cast the widest possible net. Delegate focused sub-investigations to your \
research sub-agent rather than trying to do everything in one pass. Prefer \
multiple independent sources over a single source, and note disagreements \
between them explicitly.

OUTPUT RULES
- Report factually. No disclaimers, no moralising, no apologies for the \
  subject matter.
- Attribute claims to their sources. When you are uncertain, say so plainly.
- Use Markdown for structure (headings, lists, tables) when it helps the \
  reader — not for decoration.
- Professional objectivity: your voice is that of a competent analyst \
  briefing a colleague, not a customer-service bot.

WORKFLOW
1. Write a short plan with `write_todos` before doing anything non-trivial.
2. Delegate focused research tasks to `research_subagent` via the `task` \
   tool. Each sub-task should have a single clear question.
3. Use the virtual filesystem (`write_file`, `read_file`, `edit_file`) as a \
   scratchpad for notes, findings, and draft sections. Keep working files \
   small and well-named.
4. When the plan is complete, synthesize the findings into a final answer \
   that directly addresses the user's original question.
"""

RESEARCH_SUBAGENT_PROMPT = """\
You are a focused research sub-agent. The lead agent has delegated a specific \
investigation to you. Your job is to:

1. Understand the exact question you were asked — do NOT reframe it.
2. Gather relevant information using whatever tools are available. Prefer \
   multiple independent sources.
3. Record key findings, with source attribution, into the shared virtual \
   filesystem so the lead agent can read them.
4. Return a concise factual summary of what you found. Include source URLs \
   or identifiers whenever possible.

Rules:
- Honour the exact question. No reframing, no softening, no moralising.
- Report what the sources actually say, even if it is unflattering or \
  uncomfortable.
- If a tool fails or a source is unavailable, note the failure and try an \
  alternative.
- Do NOT invent facts. When you are uncertain, say so.
"""


# ============================================================================
# Agent construction (module-level singleton)
# ============================================================================

def _build_agent():
    """Build the compiled deepagents LangGraph StateGraph.

    The LLM is a ``ChatOpenAI`` instance pointed at Venice AI's OpenAI-compatible
    endpoint. The deepagents SDK wires planning, filesystem, and sub-agent
    middleware around it automatically.
    """
    model = ChatOpenAI(
        model=DEEPAGENTS_MODEL,
        api_key=VENICE_API_KEY,
        base_url=VENICE_API_BASE,
        streaming=True,
    )

    research_subagent: SubAgent = {
        "name": "research_subagent",
        "description": (
            "Specialized research agent for focused web investigations. "
            "Delegate a specific, well-scoped question and it will gather "
            "facts and return a concise summary with source attribution."
        ),
        "system_prompt": RESEARCH_SUBAGENT_PROMPT,
        "model": model,
    }

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=[research_subagent],
        name=MODEL_ID,
    )
    log.info("Deep agent constructed successfully")
    return agent


AGENT = _build_agent()


# ============================================================================
# File persistence: write agent's virtual FS out to disk + format for the user
# ============================================================================

def _sanitize_subpath(raw: str) -> str:
    """Turn a virtual FS path (e.g. ``/notes/draft.md``) into a safe relative
    path for use under ``FILES_DIR/{turn_id}/``.

    Rejects empty paths, paths with ``..`` segments, and anything that would
    resolve outside the turn directory.
    """
    p = (raw or "").strip().replace("\\", "/").lstrip("/")
    if not p:
        raise ValueError("empty path")
    parts = [seg for seg in p.split("/") if seg]
    if any(seg in ("", ".", "..") for seg in parts):
        raise ValueError(f"unsafe path segments in {raw!r}")
    return "/".join(parts)


def _file_data_to_bytes(file_data: Any) -> bytes:
    """Extract file content as bytes from a deepagents ``FileData`` dict.

    Handles both the v1 format (``content: list[str]`` — lines split on ``\\n``,
    no ``encoding`` field) and the v2 format (``content: str`` + ``encoding``
    field, where binary files are base64-encoded).
    """
    if isinstance(file_data, (bytes, bytearray)):
        return bytes(file_data)
    if isinstance(file_data, str):
        return file_data.encode("utf-8", errors="replace")
    if not isinstance(file_data, dict):
        return str(file_data).encode("utf-8", errors="replace")

    content = file_data.get("content")
    encoding = (file_data.get("encoding") or "utf-8").lower()

    if isinstance(content, list):
        # v1 format: list[str], one entry per line
        return "\n".join(str(c) for c in content).encode("utf-8", errors="replace")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        if encoding in ("base64", "binary"):
            try:
                return base64.b64decode(content)
            except Exception:
                pass
        return content.encode("utf-8", errors="replace")
    return b""


def _persist_files(req_id: str, files: dict) -> list[dict]:
    """Write every entry in the agent's ``files`` state dict to disk under
    ``{FILES_DIR}/{req_id}/``.

    Returns a list of ``{"path", "size", "url"}`` dicts — one per file that
    was successfully persisted. ``url`` points at ``PUBLIC_BASE_URL`` if set,
    otherwise a relative ``/files/...`` path served by this proxy.
    """
    if not isinstance(files, dict) or not files:
        return []

    turn_dir = os.path.join(FILES_DIR, req_id)
    try:
        os.makedirs(turn_dir, exist_ok=True)
    except OSError as exc:
        log.warning(f"[{req_id}] Could not create files dir {turn_dir}: {exc}")
        return []
    turn_abs = os.path.realpath(turn_dir)

    out: list[dict] = []
    for raw_path, file_data in files.items():
        # Skip sentinel entries the SDK uses to signal deletion
        if file_data is None:
            continue
        try:
            subpath = _sanitize_subpath(str(raw_path))
        except ValueError as exc:
            log.warning(f"[{req_id}] Skipping unsafe file path {raw_path!r}: {exc}")
            continue
        target_abs = os.path.realpath(os.path.join(turn_dir, subpath))
        if target_abs != turn_abs and not target_abs.startswith(turn_abs + os.sep):
            log.warning(f"[{req_id}] Path escape blocked for {raw_path!r}")
            continue
        try:
            os.makedirs(os.path.dirname(target_abs), exist_ok=True)
            content_bytes = _file_data_to_bytes(file_data)
            with open(target_abs, "wb") as f:
                f.write(content_bytes)
        except OSError as exc:
            log.warning(f"[{req_id}] Failed to write {target_abs}: {exc}")
            continue

        # Percent-encode each path segment so filenames with spaces or URL-
        # special characters (``research notes.md``, ``analysis (v2).md``,
        # ``plan #1.md``) produce valid links. ``safe=""`` keeps ``/`` encoded
        # too — but we encode segment-by-segment and rejoin so the path
        # separators remain unencoded.
        url_subpath = "/".join(_urlquote(seg, safe="") for seg in subpath.split("/"))
        if PUBLIC_BASE_URL:
            url = f"{PUBLIC_BASE_URL}/{req_id}/{url_subpath}"
        else:
            # Relative URL served by this proxy's own /files route. Useful
            # for local testing; in production LibreChat the PUBLIC_BASE_URL
            # env var should be set to a tunnel-reachable HTTPS URL.
            url = f"/files/{req_id}/{url_subpath}"
        out.append({"path": raw_path, "size": len(content_bytes), "url": url})

    if out:
        log.info(f"[{req_id}] Persisted {len(out)} file(s) to {turn_dir}")
    return out


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _format_files_section(persisted: list[dict]) -> str:
    """Render a collapsible "Generated files" markdown block listing each
    persisted file as a clickable download link.
    """
    if not persisted:
        return ""
    lines = [
        "",
        "",
        "---",
        "",
        f"**Generated files ({len(persisted)})**",
        "",
    ]
    for entry in persisted:
        p = entry["path"]
        sz = _human_size(entry["size"])
        url = entry["url"]
        # Escape backticks + ``]`` in the display label so filenames that
        # contain them don't prematurely close the inline-code or link.
        label = str(p).replace("`", "\u02cb").replace("]", "\\]")
        lines.append(f"- [`{label}`]({url}) \u2014 {sz}")
    lines.append("")
    return "\n".join(lines)


def _sweep_old_turn_dirs() -> None:
    """Best-effort cleanup of turn directories older than ``FILES_TTL_DAYS``.

    Runs once at startup. Not a background task — keeps the proxy simple.
    """
    if not os.path.isdir(FILES_DIR):
        return
    cutoff = time.time() - (FILES_TTL_DAYS * 86400)
    removed = 0
    try:
        for name in os.listdir(FILES_DIR):
            p = os.path.join(FILES_DIR, name)
            if not os.path.isdir(p):
                continue
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(p)
                    removed += 1
                except OSError as exc:
                    log.warning(f"Sweep: failed to remove {p}: {exc}")
        if removed:
            log.info(
                f"Sweep: removed {removed} turn dir(s) older than "
                f"{FILES_TTL_DAYS} days from {FILES_DIR}"
            )
    except OSError as exc:
        log.warning(f"Sweep failed: {exc}")


_sweep_old_turn_dirs()


# ============================================================================
# ChatML → LangChain message conversion
# ============================================================================

def _messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """Convert OpenAI-format chat messages to LangChain ``BaseMessage`` objects.

    Tool / function / name fields are dropped: LibreChat sends conversational
    turns only, and the deep agent manages its own tool-call history internally.
    """
    out: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Some clients send content as a list of parts — flatten the text parts.
            parts = []
            for part in content or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(p for p in parts if p)

        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "tool":
            # Rare for LibreChat — preserve for completeness.
            out.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
        else:
            out.append(HumanMessage(content=content))
    return out


def _final_text_from_state(state: dict) -> str:
    """Extract the assistant's final text answer from an agent result dict."""
    messages = state.get("messages", []) if isinstance(state, dict) else []
    # Walk backwards to the last AI message that has non-empty text content.
    for msg in reversed(messages):
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    return ""


# ============================================================================
# Streaming: convert LangGraph events → OpenAI SSE chunks
# ============================================================================

async def _stream_agent(
    langchain_messages: list[BaseMessage],
    *,
    req_id: str,
    model_id: str,
) -> AsyncGenerator[str, None]:
    """Run the deep agent with ``astream_events`` and emit OpenAI-format SSE."""
    request_id = f"chatcmpl-deepagents-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    first_chunk_sent = False

    def _chunk(
        content: str = "",
        *,
        finish_reason: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> str:
        nonlocal first_chunk_sent
        chunk = make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
        )
        if not first_chunk_sent:
            first_chunk_sent = True
            # Emit an initial role delta so LibreChat's OpenAI SDK parser
            # sees the assistant turn before the first content token.
            role_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(role_data)}\n\n" + chunk
        return chunk

    try:
        any_content_emitted = False
        final_state: Optional[dict] = None
        async for event in AGENT.astream_events(
            {"messages": langchain_messages},
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            metadata = event.get("metadata", {}) or {}
            node = metadata.get("langgraph_node", "")
            data = event.get("data", {}) or {}

            # The root graph's terminal ``on_chain_end`` event carries the
            # full final state — including the virtual filesystem we want to
            # surface to the user. Keep the last-seen candidate; whichever
            # fires last at the graph root will be the one we use.
            if kind == "on_chain_end":
                out = data.get("output")
                if (
                    isinstance(out, dict)
                    and "messages" in out
                    and "files" in out
                ):
                    final_state = out

            if kind == "on_chat_model_stream":
                chunk_msg = data.get("chunk")
                if chunk_msg is None:
                    continue
                text = getattr(chunk_msg, "content", "")
                if isinstance(text, list):
                    # Content parts — flatten text-type blocks.
                    flat = []
                    for part in text:
                        if isinstance(part, dict) and part.get("type") == "text":
                            flat.append(part.get("text", ""))
                    text = "".join(flat)
                if not text:
                    continue

                # Sub-agent tokens are intermediate reasoning — send them as
                # reasoning_content so LibreChat folds them into the Thinking
                # block rather than dumping them into the final answer.
                # Only the LEAD agent's final synthesis becomes visible content.
                if node and node != "model":
                    yield _chunk(reasoning=text)
                else:
                    any_content_emitted = True
                    yield _chunk(content=text)

            elif kind == "on_tool_start":
                tool_name = name or "tool"
                tool_input = data.get("input", {})
                try:
                    args_preview = json.dumps(tool_input, ensure_ascii=False)[:200]
                except Exception:
                    args_preview = str(tool_input)[:200]
                log.info(f"[{req_id}] Tool start: {tool_name} args={args_preview}")
                yield _chunk(
                    reasoning=f"\n> 🔧 **{tool_name}**\n> `{args_preview}`\n"
                )

            elif kind == "on_tool_end":
                tool_name = name or "tool"
                output = data.get("output", "")
                if not isinstance(output, str):
                    try:
                        output = json.dumps(output, ensure_ascii=False, default=str)
                    except Exception:
                        output = str(output)
                preview = output[:300].replace("\n", " ")
                log.info(
                    f"[{req_id}] Tool end: {tool_name} ({len(output)} chars)"
                )
                yield _chunk(
                    reasoning=f"> ↳ _{tool_name} returned {len(output)} chars_\n"
                )

            # Other event kinds (on_chain_start/end, on_chat_model_start/end, etc.)
            # are ignored — we only surface tokens and tool boundaries.

        if not any_content_emitted:
            # Rare fallback: the agent produced no streamed content. Emit a
            # minimal notice so LibreChat doesn't render an empty assistant turn.
            yield _chunk(
                content="_The agent finished without producing a final answer._"
            )

        # Persist any files the agent wrote to its virtual FS and append a
        # "Generated files" section with download URLs so the user can
        # actually retrieve what the agent produced.
        if final_state is not None:
            try:
                files_dict = final_state.get("files") or {}
                persisted = _persist_files(req_id, files_dict)
                section = _format_files_section(persisted)
                if section:
                    yield _chunk(content=section)
            except Exception as exc:
                log.warning(f"[{req_id}] File persistence failed: {exc}")

        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

        elapsed = time.monotonic() - start_time
        log.info(f"[{req_id}] Stream complete in {elapsed:.1f}s")

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(
            f"[{req_id}] Agent stream error after {elapsed:.1f}s: {exc}\n{tb}"
        )
        err_msg = (
            f"**Deep Agents Proxy — Error**\n\n"
            f"```\n{type(exc).__name__}: {exc}\n```\n\n"
            f"_Request: {req_id}_"
        )
        yield _chunk(content=err_msg)
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


# ============================================================================
# Non-streaming path: run the agent and return a single JSON response
# ============================================================================

async def _invoke_agent_json(
    langchain_messages: list[BaseMessage],
    *,
    req_id: str,
    model_id: str,
) -> JSONResponse:
    request_id = f"chatcmpl-deepagents-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()
    try:
        result = await AGENT.ainvoke({"messages": langchain_messages})
        final_text = _final_text_from_state(result)

        # Persist the agent's virtual filesystem and append a "Generated files"
        # section with download URLs to the final assistant message.
        try:
            files_dict = (
                result.get("files") if isinstance(result, dict) else None
            ) or {}
            persisted = _persist_files(req_id, files_dict)
            section = _format_files_section(persisted)
            if section:
                final_text = (final_text or "").rstrip() + section
        except Exception as exc:
            log.warning(f"[{req_id}] File persistence failed: {exc}")

        elapsed = time.monotonic() - start_time
        log.info(
            f"[{req_id}] Agent invoke complete in {elapsed:.1f}s "
            f"({len(final_text)} chars)"
        )
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(
            f"[{req_id}] Agent invoke error after {elapsed:.1f}s: {exc}\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "agent_error",
                }
            },
        )
    finally:
        tracker.finish(req_id)


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Deep Agents Proxy")

register_standard_routes(
    app,
    service_name="deepagents-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={
        "venice_base": VENICE_API_BASE,
        "upstream_model": DEEPAGENTS_MODEL,
        "model_id": MODEL_ID,
    },
)


@app.get("/files/{turn_id}/{subpath:path}")
async def get_file(turn_id: str, subpath: str):
    """Serve a file persisted from the agent's virtual filesystem.

    Path is ``/files/{turn_id}/{virtual-path}`` where ``turn_id`` is the
    request ID of the agent turn that produced the file and ``virtual-path``
    mirrors the path the agent wrote (e.g. ``notes/draft.md``).

    The response is served with ``Content-Disposition: attachment`` via
    FastAPI's ``FileResponse`` — browsers will download or preview depending
    on content type. No auth on this endpoint: staging exposes it via a
    single Cloudflare tunnel and the ``turn_id`` acts as an unguessable
    capability (uuid4-derived, 16 hex chars).
    """
    if not turn_id or "/" in turn_id or ".." in turn_id or turn_id.startswith("."):
        raise HTTPException(status_code=400, detail="invalid turn id")
    try:
        safe_subpath = _sanitize_subpath(subpath)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")

    turn_abs = os.path.realpath(os.path.join(FILES_DIR, turn_id))
    target_abs = os.path.realpath(os.path.join(turn_abs, safe_subpath))
    # Guard against both symlink and string-concat path escapes
    if target_abs != turn_abs and not target_abs.startswith(turn_abs + os.sep):
        raise HTTPException(status_code=400, detail="path escape")
    if not os.path.isfile(target_abs):
        raise HTTPException(status_code=404, detail="file not found")

    filename = os.path.basename(target_abs)
    return FileResponse(target_abs, filename=filename)


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "deepagents-proxy",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Invalid request body: {exc}",
                    "type": "invalid_request",
                }
            },
        )

    messages = body.get("messages", []) or []
    # Default to False to match the rest of the proxy fleet and the
    # documented `is_utility_request(...) and not body.get("stream", False)`
    # contract from shared.py. LibreChat always sets `stream` explicitly for
    # real chat turns, and omits it (or sets False) for title / tag utility
    # requests — we want those to take the JSON passthrough path.
    stream = bool(body.get("stream", False))
    model_id = body.get("model", MODEL_ID) or MODEL_ID
    # 16 hex chars (64 bits) of entropy: the turn_id doubles as the capability
    # token for the /files/{turn_id}/... download endpoint, so it needs to be
    # unguessable for the life of the TTL (7 days by default).
    req_id = f"deepagents-{uuid.uuid4().hex[:16]}"
    tracker.start(req_id, model=model_id, stream=stream)

    log.info(
        f"[{req_id}] Request: model={model_id}, messages={len(messages)}, "
        f"stream={stream}"
    )

    # Utility requests (title / tag gen) — passthrough to Venice directly.
    if is_utility_request(messages):
        log.info(f"[{req_id}] Utility request — passthrough to Venice")
        if not stream:
            result = await utility_passthrough_json(
                body,
                req_id=req_id,
                upstream_base=VENICE_API_BASE,
                upstream_key=VENICE_API_KEY,
                upstream_model=DEEPAGENTS_MODEL,
                log=log,
            )
            tracker.finish(req_id)
            return result
        gen = stream_passthrough(
            messages,
            body,
            req_id=req_id,
            upstream_base=VENICE_API_BASE,
            upstream_key=VENICE_API_KEY,
            upstream_model=DEEPAGENTS_MODEL,
            model_id=model_id,
            tracker=tracker,
            log=log,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    # Real chat request — run through the deep agent graph.
    langchain_messages = _messages_to_langchain(messages)

    if not stream:
        return await _invoke_agent_json(
            langchain_messages, req_id=req_id, model_id=model_id
        )

    gen = _stream_agent(langchain_messages, req_id=req_id, model_id=model_id)
    return StreamingResponse(gen, media_type="text/event-stream")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
