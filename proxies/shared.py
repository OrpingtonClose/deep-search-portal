"""
Shared utilities for Deep Search Portal proxies.

Provides common infrastructure used by both the Thinking Proxy and the
Deep Research Proxy: logging, configuration, SSE helpers, HTTP client
management, passthrough streaming, utility-request detection, and
standard FastAPI endpoints (health, logs).
"""

import asyncio
import json
import logging
import logging.handlers
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ============================================================================
# Utility-request detection
# ============================================================================

UTILITY_PATTERNS = [
    "generate a concise",
    "generate 1-3 broad tags",
    "generate a title",
    "### task:\ngenerate",
    "create a concise title",
    "generate a search query",
    "autocomplete",
]


def is_utility_request(messages: list[dict]) -> bool:
    """Detect automated utility requests from LibreChat (title/tag gen, etc.)."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            for pattern in UTILITY_PATTERNS:
                if pattern in content_lower:
                    return True
    return False


# ============================================================================
# LibreChat file-attachment parsing
# ============================================================================

@dataclass
class AttachedDocument:
    """A single file extracted from a LibreChat "Upload as Text" message."""
    filename: str
    content: str


@dataclass
class ParsedMessage:
    """Result of separating LibreChat file attachments from the user prompt.

    *documents* — zero or more attached files (from "Upload as Text").
    *prompt*    — the user's actual text after removing the attachment block.
    *raw*       — the original, unparsed message text.
    """
    documents: list[AttachedDocument] = field(default_factory=list)
    prompt: str = ""
    raw: str = ""

    @property
    def has_attachments(self) -> bool:
        return len(self.documents) > 0

    @property
    def all_document_text(self) -> str:
        """Concatenate all attached document content into a single string."""
        return "\n\n---\n\n".join(
            f"# {doc.filename}\n{doc.content}" for doc in self.documents
        )


# Regex to detect the opening marker of a LibreChat "Attached document(s):" block.
# The actual closing marker is found via nesting-aware scanning (see
# ``_find_attachment_block`` below) so that code fences inside documents
# AND code fences in the user's prompt are both handled correctly.
#
# LibreChat v0.8.x may or may not emit a newline between the opening
# fence (` ```md`) and the first file header (`# "file.txt"`), so the
# pattern uses `\n?` to accept both variants.
_ATTACHMENT_OPEN_RE = re.compile(
    r"^Attached document\(s\):\s*```md\n?",
    re.MULTILINE,
)


def _find_attachment_block(text: str) -> tuple[str, str] | None:
    """Locate the LibreChat attachment block using nesting-aware fence matching.

    Returns ``(block_content, after_block)`` or *None* if no block is found.

    Strategy:
    1. Find the opening ``Attached document(s): ```md`` marker.
    2. Scan line-by-line tracking *only* language-tagged fence nesting
       (e.g. ` ```python ` … ` ``` `).
    3. The **first** bare ` ``` ` encountered at depth 0 is the closing
       marker of the attachment block.

    This deliberately does NOT attempt to handle bare (un-tagged) code
    fences *inside* the uploaded document — doing so requires an
    unbounded look-ahead that inevitably scans into the user's prompt
    text (which may itself contain code fences), causing misparsing.
    Language-tagged fences (` ```python `, ` ```bash `, etc.) inside
    documents are handled correctly via depth tracking.
    """
    m = _ATTACHMENT_OPEN_RE.search(text)
    if m is None:
        return None

    body_start = m.end()          # first char after the opening ```md\n
    lines = text[body_start:].split("\n")
    depth = 0                     # nesting depth for language-tagged fences
    closing_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_fence = (
            stripped.startswith("```")
            and not stripped.startswith("````")
        )
        if not is_fence:
            continue

        has_lang = len(stripped) > 3 and stripped[3:].strip() != ""
        if has_lang:
            if depth == 0:
                # Opening a language-tagged inner code block
                depth += 1
            else:
                # Could be another nested opener OR a language-tagged
                # closer (rare but valid, e.g. ```python closing as
                # ```python).  Treat as additional nesting for safety.
                depth += 1
        else:
            # Bare ``` line
            if depth > 0:
                # Close a language-tagged inner code block
                depth -= 1
            else:
                # depth == 0 — this is the attachment-closing marker.
                closing_idx = i
                break

    if closing_idx is None:
        return None

    block_content = "\n".join(lines[:closing_idx])
    after_block = "\n".join(lines[closing_idx + 1:])
    return block_content, after_block

# Within the block, each file starts with # "filename"
_FILE_HEADER_RE = re.compile(
    r'^# "([^"]+)"',
    re.MULTILINE,
)


def parse_attachments(text: str) -> ParsedMessage:
    """Parse a LibreChat user message that may contain "Upload as Text" files.

    LibreChat's ``extractFileContext`` produces a format like::

        Attached document(s):
        ```md
        # "report.pdf"
        <extracted text>

        ---

        # "notes.txt"
        <more text>
        ```
        <user's actual prompt here>

    .. note::

       In LibreChat v0.8.x the attachment block is delivered as a
       **system** message while the user's typed prompt is a separate
       *user* message.  Use ``extract_user_text_with_attachments()``
       to reassemble them into the single-string format this function
       expects.  The opening fence may or may not have a newline
       between ` ```md` and the first ``# "filename"`` header.

    This function separates the attachment block from the prompt and returns
    a ``ParsedMessage`` with structured documents and the clean prompt.

    If no attachment block is detected the full text is returned as the prompt.
    """
    if not text:
        return ParsedMessage(raw=text, prompt=text)

    result = _find_attachment_block(text)
    if result is None:
        return ParsedMessage(raw=text, prompt=text.strip())

    block_content, after_block = result
    after_block = after_block.strip()

    # Split block into individual documents by the --- separator
    documents: list[AttachedDocument] = []

    # Find all file headers and their positions
    headers = list(_FILE_HEADER_RE.finditer(block_content))

    if headers:
        for i, header_match in enumerate(headers):
            filename = header_match.group(1)
            # Content starts after the header line
            content_start = header_match.end()
            # Content ends at the next file's --- separator or end of block
            if i + 1 < len(headers):
                # Find the --- separator between this file and the next
                next_header_pos = headers[i + 1].start()
                # Look backwards from next header for ---
                chunk = block_content[content_start:next_header_pos]
                # Strip trailing separator
                chunk = re.sub(r"\n\n---\s*\n\n?$", "", chunk)
            else:
                chunk = block_content[content_start:]

            content = chunk.strip()
            if content:
                documents.append(AttachedDocument(
                    filename=filename,
                    content=content,
                ))
    elif block_content.strip():
        # No file headers found but there's content — treat as single unnamed doc
        documents.append(AttachedDocument(
            filename="document",
            content=block_content.strip(),
        ))

    return ParsedMessage(
        documents=documents,
        prompt=after_block,
        raw=text,
    )


def _msg_text(msg: dict) -> str:
    """Return the text content of a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def extract_user_text(messages: list[dict]) -> str:
    """Extract the last user message text from a messages array.

    Handles both string content and OpenAI multipart content arrays.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _msg_text(msg)
    return ""


def extract_user_text_with_attachments(messages: list[dict]) -> str:
    """Build a combined text that includes any LibreChat attachment context.

    LibreChat v0.8.x sends "Upload as Text" file content in a **system**
    message whose text starts with ``Attached document(s):`` while the
    user's typed prompt is in a separate *user* message.  To let
    ``parse_attachments()`` work on a single string that mirrors the
    original LibreChat wire format, this helper locates the attachment
    system message (if any) and concatenates it with the user prompt.

    If no attachment system message exists the result is identical to
    ``extract_user_text()``.
    """
    # 1) Find the last user message text.
    user_text = extract_user_text(messages)

    # 2) Look for a system message whose content starts with the
    #    LibreChat attachment marker — but ONLY in the *current turn*.
    #    In multi-turn conversations the full history is sent with each
    #    request, so an attachment system message from turn 1 would still
    #    be present in turns 2, 3, etc.  Scoping to messages after the
    #    last assistant reply prevents stale attachments from being
    #    re-injected on every follow-up.
    attachment_prefix = "Attached document(s):"
    last_asst_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_asst_idx = idx
            break
    current_turn = messages[last_asst_idx + 1:]
    for msg in reversed(current_turn):
        if msg.get("role") != "system":
            continue
        text = _msg_text(msg)
        if text.startswith(attachment_prefix):
            # Re-assemble into the single-string format that
            # ``parse_attachments`` expects:
            #   <attachment block>\n<user prompt>
            if user_text:
                return text + "\n" + user_text
            return text

    return user_text


# ============================================================================
# Logging
# ============================================================================

def setup_logging(service_name: str, log_dir: str) -> logging.Logger:
    """
    Configure root logging with console (INFO) and rotating file (DEBUG)
    handlers, then return a named logger for *service_name*.

    Safe to call once per process — guards against duplicate handlers when
    the module is re-imported.
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.root
    root.setLevel(logging.DEBUG)

    # Only add handlers if none are attached yet (prevents duplicates on reimport).
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(console)

        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "proxy.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(fh)

    return logging.getLogger(service_name)


# ============================================================================
# Configuration helpers
# ============================================================================

def require_env(name: str) -> str:
    """Return an environment variable or raise with a clear message."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return val


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an env var as int, clamping to *minimum*."""
    raw = os.getenv(name, "")
    if not raw:
        return max(default, minimum)
    try:
        return max(int(raw), minimum)
    except ValueError:
        logging.getLogger("config").warning(
            f"Invalid integer for {name}={raw!r}, using default {default}"
        )
        return max(default, minimum)


# ============================================================================
# Shared HTTP client (connection-pooled, created once per process)
# ============================================================================

# Populated by the lifespan context manager — see ``create_app``.
_http_client: Optional[httpx.AsyncClient] = None


def http_client() -> httpx.AsyncClient:
    """Return the shared httpx client.

    When called inside a running FastAPI app the lifespan-managed client is
    returned.  When called outside the app (e.g. standalone test scripts),
    a default ``AsyncClient`` is lazily created so that search tools remain
    usable without spinning up the full server.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
            follow_redirects=True,
        )
    return _http_client


# ============================================================================
# SSE helpers
# ============================================================================

def make_sse_chunk(
    content: str,
    *,
    request_id: str,
    created: int,
    model_id: str,
    finish_reason: Optional[str] = None,
    reasoning_content: Optional[str] = None,
) -> str:
    """Build a single SSE ``data:`` line in OpenAI chat-completion chunk format.

    When *reasoning_content* is provided the chunk carries a
    ``reasoning_content`` delta field (the standard OpenAI format used by
    o1/o3/DeepSeek reasoning models).  LibreChat's ``SplitStreamHandler``
    consumes this field natively to render a collapsible "Thinking" block.
    """
    delta: dict = {}
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if content:
        delta["content"] = content

    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(data)}\n\n"


# ============================================================================
# Active-request tracking
# ============================================================================

class RequestTracker:
    """Thread-safe (single-event-loop) tracker for in-flight requests."""

    def __init__(self) -> None:
        self._active: dict[str, dict] = {}

    def start(self, req_id: str, **meta: object) -> None:
        self._active[req_id] = {
            "started": datetime.now(timezone.utc).isoformat(),
            **meta,
        }

    def update(self, req_id: str, **fields: object) -> None:
        if req_id in self._active:
            self._active[req_id].update(fields)

    def finish(self, req_id: str) -> None:
        self._active.pop(req_id, None)

    @property
    def count(self) -> int:
        return len(self._active)

    @property
    def details(self) -> dict[str, dict]:
        return dict(self._active)


# ============================================================================
# Concurrency limiter
# ============================================================================

class ConcurrencyLimiter:
    """Async semaphore wrapper for limiting expensive concurrent operations."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def available(self) -> bool:
        """Return True if at least one slot is free (non-blocking check)."""
        return self._sem._value > 0  # noqa: SLF001

    @asynccontextmanager
    async def hold(self):
        """Acquire a slot for the duration of the ``async with`` block."""
        async with self._sem:
            yield


# ============================================================================
# Passthrough streaming (shared between both proxies)
# ============================================================================

async def stream_passthrough(
    messages: list[dict],
    original_body: dict,
    *,
    req_id: str,
    upstream_base: str,
    upstream_key: str,
    upstream_model: str,
    model_id: str,
    tracker: RequestTracker,
    log: logging.Logger,
    extra_headers: Optional[dict[str, str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Forward a request to the upstream LLM without any agent / thinking logic.
    Used for utility requests (title generation, tag generation, etc.).
    Always streams the response as SSE.
    """
    request_id = f"chatcmpl-pass-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def _chunk(content: str, finish_reason: Optional[str] = None) -> str:
        return make_sse_chunk(
            content,
            request_id=request_id,
            created=created,
            model_id=model_id,
            finish_reason=finish_reason,
        )

    # Build upstream body — strip Open-WebUI-specific keys
    upstream_body = {
        **original_body,
        "model": upstream_model,
        "messages": messages,
        "stream": True,
    }
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    log.info(f"[{req_id}] PASSTHROUGH upstream: model={upstream_model}, msgs={len(messages)}")

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        client = http_client()
        async with client.stream(
            "POST",
            f"{upstream_base}/chat/completions",
            json=upstream_body,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:1000]
                log.error(f"[{req_id}] Passthrough upstream error {resp.status_code}: {error_text}")
                yield _chunk(f"Error: {error_text[:200]}", finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(payload)
                        chunk["model"] = model_id
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        pass

            # Stream ended without [DONE] — send it ourselves
            yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Passthrough error after {elapsed:.2f}s: {e}")
        yield _chunk(f"Error: {str(e)[:200]}", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        tracker.finish(req_id)


async def utility_passthrough_json(
    body: dict,
    *,
    req_id: str,
    upstream_base: str,
    upstream_key: str,
    upstream_model: str,
    log: logging.Logger,
    extra_headers: Optional[dict[str, str]] = None,
) -> JSONResponse:
    """
    Handle utility requests (title/tag generation) as non-streaming JSON.

    LibreChat sends ``stream=false`` for these and expects a single JSON
    response — NOT an SSE stream.  Every proxy should call this when
    ``is_utility_request(messages) and not body.get("stream", False)``.
    """
    upstream_body = {
        **body,
        "model": upstream_model,
        "stream": False,
    }
    for key in ("user", "chat_id", "tools", "tool_choice",
                "functions", "function_call"):
        upstream_body.pop(key, None)

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        client = http_client()
        resp = await client.post(
            f"{upstream_base}/chat/completions",
            json=upstream_body,
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            log.info(f"[{req_id}] Utility JSON response OK")
            return JSONResponse(content=data)
        error_text = resp.text[:500]
        log.error(
            f"[{req_id}] Utility upstream error {resp.status_code}: {error_text}"
        )
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": {"message": error_text, "type": "upstream_error"}},
        )
    except Exception as e:
        log.error(f"[{req_id}] Utility request failed: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "proxy_error"}},
        )


# ============================================================================
# Standard FastAPI endpoints (health + logs)
# ============================================================================

def register_standard_routes(
    app: FastAPI,
    *,
    service_name: str,
    log_dir: str,
    tracker: RequestTracker,
    health_extras: Optional[dict] = None,
) -> None:
    """
    Register ``/health`` and ``/logs`` endpoints on *app*.

    *health_extras* is a dict of additional key/value pairs to include
    in the health response (e.g. upstream URL, model, searxng URL).
    """

    @app.get("/health")
    async def health():
        info = {
            "status": "ok",
            "service": service_name,
            **(health_extras or {}),
            "active_requests": tracker.count,
            "active_details": tracker.details,
        }
        return JSONResponse(info)

    @app.get("/logs")
    async def get_logs(lines: int = 100):
        log_path = os.path.join(log_dir, "proxy.log")
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
                return JSONResponse({
                    "total_lines": len(all_lines),
                    "returned": min(lines, len(all_lines)),
                    "lines": all_lines[-lines:],
                })
        except FileNotFoundError:
            return JSONResponse({"error": "Log file not found"}, status_code=404)


# ============================================================================
# Text Ingestion Infrastructure
# ============================================================================

INGEST_DB_PATH = os.getenv("INGEST_DB", "/opt/ingested_texts/ingest.db")
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "2000"))
INGEST_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))


def _init_ingest_db(db_path: str) -> None:
    """Create the ingestion SQLite database and tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT DEFAULT '',
                total_chunks INTEGER DEFAULT 0,
                total_chars INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES ingested_documents(id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id)
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(content, doc_title, source, content=document_chunks, content_rowid=rowid)
        """)

        # Triggers to keep FTS5 in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON document_chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, doc_title, source)
                SELECT new.rowid, new.content,
                       d.title, d.source
                FROM ingested_documents d WHERE d.id = new.doc_id;
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON document_chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, doc_title, source)
                SELECT 'delete', old.rowid, old.content,
                       d.title, d.source
                FROM ingested_documents d WHERE d.id = old.doc_id;
            END
        """)

        conn.commit()
    finally:
        conn.close()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_document(
    db_path: str,
    title: str,
    text: str,
    source: str = "",
    chunk_size: int = INGEST_CHUNK_SIZE,
    chunk_overlap: int = INGEST_CHUNK_OVERLAP,
) -> dict:
    """Chunk and store a large text document. Returns metadata about the ingestion."""
    _init_ingest_db(db_path)
    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    chunks = _chunk_text(text, chunk_size, chunk_overlap)

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO ingested_documents (id, title, source, total_chunks, total_chars, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (doc_id, title, source, len(chunks), len(text), now),
        )
        rows = [
            (f"chunk-{uuid.uuid4().hex[:12]}", doc_id, i, chunk, now)
            for i, chunk in enumerate(chunks)
        ]
        conn.executemany(
            """INSERT INTO document_chunks (id, doc_id, chunk_index, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "doc_id": doc_id,
        "title": title,
        "source": source,
        "total_chunks": len(chunks),
        "total_chars": len(text),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def search_ingested_text(db_path: str, query: str, limit: int = 10) -> list[dict]:
    """Search ingested documents using FTS5. Returns matching chunks ranked by relevance."""
    _init_ingest_db(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        tokens = [t.strip() for t in query.split() if len(t.strip()) > 1]
        if not tokens:
            return []
        fts_query = " OR ".join(tokens[:15])

        cursor = conn.execute(
            """SELECT dc.doc_id, dc.chunk_index, dc.content,
                      d.title, d.source,
                      rank
               FROM chunks_fts
               JOIN document_chunks dc ON chunks_fts.rowid = dc.rowid
               JOIN ingested_documents d ON dc.doc_id = d.id
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r[0],
                "chunk_index": r[1],
                "content": r[2],
                "doc_title": r[3],
                "source": r[4],
                "rank": r[5],
            }
            for r in rows
        ]
    except Exception as e:
        logging.getLogger("ingest").warning(f"FTS5 search error: {e}")
        return []
    finally:
        conn.close()


def list_ingested_documents(db_path: str) -> list[dict]:
    """Return metadata for all ingested documents."""
    _init_ingest_db(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cursor = conn.execute(
            """SELECT id, title, source, total_chunks, total_chars, created_at
               FROM ingested_documents
               ORDER BY created_at DESC"""
        )
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r[0],
                "title": r[1],
                "source": r[2],
                "total_chunks": r[3],
                "total_chars": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]
    finally:
        conn.close()


def delete_ingested_document(db_path: str, doc_id: str) -> bool:
    """Delete a document and all its chunks. Returns True if the document existed."""
    _init_ingest_db(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        existing = conn.execute(
            "SELECT id FROM ingested_documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not existing:
            return False
        # The FTS delete trigger fires automatically for each chunk row
        conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM ingested_documents WHERE id = ?", (doc_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def register_ingest_routes(
    app: FastAPI,
    db_path: str,
    log: logging.Logger,
) -> None:
    """
    Register text-ingestion REST endpoints on *app*:

    - ``POST /v1/ingest`` — ingest a large text document
    - ``GET  /v1/documents`` — list ingested documents
    - ``DELETE /v1/documents/{doc_id}`` — remove a document
    """

    @app.post("/v1/ingest")
    async def ingest_text(request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Invalid JSON body: {e}", "type": "invalid_request"}},
            )

        title = body.get("title", "").strip()
        text = body.get("text", "").strip()
        source = body.get("source", "").strip()

        if not title:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "'title' is required", "type": "invalid_request"}},
            )
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "'text' is required and must not be empty", "type": "invalid_request"}},
            )

        log.info(f"Ingesting document: title={title!r}, chars={len(text)}, source={source!r}")

        try:
            result = ingest_document(db_path, title, text, source)
        except Exception as e:
            log.error(f"Ingestion failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"Ingestion failed: {e}", "type": "internal_error"}},
            )

        log.info(
            f"Ingested document {result['doc_id']}: "
            f"{result['total_chunks']} chunks, {result['total_chars']} chars"
        )
        return JSONResponse(result)

    @app.get("/v1/documents")
    async def list_documents():
        docs = list_ingested_documents(db_path)
        return JSONResponse({"documents": docs, "total": len(docs)})

    @app.delete("/v1/documents/{doc_id}")
    async def remove_document(doc_id: str):
        deleted = delete_ingested_document(db_path, doc_id)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"Document {doc_id} not found", "type": "not_found"}},
            )
        log.info(f"Deleted document: {doc_id}")
        return JSONResponse({"deleted": True, "doc_id": doc_id})


# ============================================================================
# API Throttling — token-bucket rate limiter per external provider
# ============================================================================

class TokenBucketThrottler:
    """Async token-bucket rate limiter.

    Each bucket refills at *rate* tokens per second up to *capacity*.
    ``acquire()`` waits until a token is available.  Multiple callers
    are served in FIFO order so no request starves.

    Also enforces a maximum number of concurrent in-flight requests via
    *max_concurrent* (set to 0 to disable the concurrency cap).
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        max_concurrent: int = 0,
        name: str = "",
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self.name = name or "unnamed"
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: int = 0
        self._total_acquired: int = 0
        self._total_waited: float = 0.0
        self._concurrent_sem: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )
        self._active: int = 0

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self) -> float:
        """Wait for a token.  Returns the number of seconds waited."""
        waited = 0.0
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                sleep_time = deficit / self.rate
                self._waiters += 1
                # Release lock while sleeping so other callers can queue
                self._lock.release()
                try:
                    await asyncio.sleep(sleep_time)
                finally:
                    await self._lock.acquire()
                    self._waiters -= 1
                waited += sleep_time
                self._refill()
            self._tokens -= 1.0
            self._total_acquired += 1
            self._total_waited += waited

        # Concurrency cap (outside the token-bucket lock)
        if self._concurrent_sem is not None:
            await self._concurrent_sem.acquire()
            self._active += 1

        return waited

    def release(self) -> None:
        """Release a concurrency slot.  Call after the request completes."""
        if self._concurrent_sem is not None:
            self._concurrent_sem.release()
            self._active = max(0, self._active - 1)

    @asynccontextmanager
    async def throttle(self):
        """Context manager: acquire a token, yield, then release the slot."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def stats(self) -> dict:
        """Return current throttler statistics."""
        return {
            "name": self.name,
            "rate_per_sec": self.rate,
            "capacity": self.capacity,
            "tokens_available": round(self._tokens, 2),
            "waiters": self._waiters,
            "active": self._active,
            "total_acquired": self._total_acquired,
            "total_waited_sec": round(self._total_waited, 2),
        }


def _env_float(name: str, default: float) -> float:
    """Read an env var as float, returning *default* on missing/invalid."""
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Default rate limits per provider (requests per second).
# Override via environment variables: THROTTLE_{PROVIDER}_RPS
# and THROTTLE_{PROVIDER}_BURST for bucket capacity.
_PROVIDER_DEFAULTS: dict[str, tuple[float, int, int]] = {
    # (rps, burst_capacity, max_concurrent)
    "mistral":       (5.0,  10,  10),
    "bright_data":   (3.0,   5,   5),
    "oxylabs":       (3.0,   5,   5),
    "searxng":       (8.0,  15,   0),
    "arxiv":         (1.0,   3,   3),
    "wayback":       (2.0,   5,   5),
    "wikidata":      (3.0,   5,   5),
    "imageboard":    (2.0,   5,   5),
    "nitter":        (2.0,   3,   3),
    "apify":         (3.0,   5,   5),
    "web_fetch":     (15.0, 30,  20),
    "knowledge_engine": (10.0, 20, 0),
}

# Singleton registry — created lazily on first access.
_throttlers: dict[str, TokenBucketThrottler] = {}
_registry_lock: Optional[asyncio.Lock] = None


def _get_registry_lock() -> asyncio.Lock:
    """Return the module-level registry lock, creating it if needed."""
    global _registry_lock
    if _registry_lock is None:
        _registry_lock = asyncio.Lock()
    return _registry_lock


def get_throttler(provider: str) -> TokenBucketThrottler:
    """Return the throttler for *provider*, creating it from defaults/env on first call."""
    if provider in _throttlers:
        return _throttlers[provider]

    upper = provider.upper()
    defaults = _PROVIDER_DEFAULTS.get(provider, (5.0, 10, 0))
    rps = _env_float(f"THROTTLE_{upper}_RPS", defaults[0])
    burst = int(_env_float(f"THROTTLE_{upper}_BURST", defaults[1]))
    max_conc = int(_env_float(f"THROTTLE_{upper}_MAX_CONCURRENT", defaults[2]))

    throttler = TokenBucketThrottler(
        rate=rps,
        capacity=burst,
        max_concurrent=max_conc,
        name=provider,
    )
    _throttlers[provider] = throttler
    return throttler


def all_throttler_stats() -> list[dict]:
    """Return stats for every registered throttler."""
    return [t.stats() for t in _throttlers.values()]


# ============================================================================
# App factory with lifespan (manages shared httpx client)
# ============================================================================

def create_app(title: str) -> FastAPI:
    """
    Create a FastAPI app with a lifespan that manages the shared httpx client.
    The client uses connection pooling (max 20 keepalive, 100 total connections)
    and generous timeouts suited for LLM API calls.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _http_client
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
            follow_redirects=True,
        )
        yield
        await _http_client.aclose()
        _http_client = None

    return FastAPI(title=title, lifespan=lifespan)
