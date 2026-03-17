#!/usr/bin/env python3
"""
Persistent Deep Research Proxy for Open WebUI.

An advanced research proxy that extends the base MiroFlow deep research loop with:
  - **Subagent map-reduce**: A planning agent decomposes the query into multiple
    research angles, parallel subagents each conduct independent research, and a
    synthesis agent combines all findings into a comprehensive answer.
  - **Atom of Thoughts (AoT) state contraction**: Each subagent compresses its
    findings into atomic conditions (fact + source + confidence) rather than
    accumulating raw tool output, preventing context-window overflow.
  - **Persistent memory**: Atomic conditions are stored in a local SQLite database
    so future queries can retrieve relevant prior findings.
  - **Dual-model architecture**: A small, fast model (e.g. Mistral Small) handles
    planning and subagent research; the large model handles final synthesis.
  - **Serendipity-aware exploration**: The planning agent generates cross-domain
    "bridge queries" alongside standard research angles.

Architecture:
  User Query
      |
      v
  [Retrieve Prior Knowledge] -- SQLite FTS5 lookup
      |
      v
  [Planning Agent] (small model) -- decomposes into N angles + bridge queries
      |
      +--- Subagent 1: angle_1 (small model, AoT loop) ---+
      +--- Subagent 2: angle_2 (small model, AoT loop) ---+
      +--- ...                                             +--- progress queue
      +--- Subagent N: angle_N (small model, AoT loop) ---+
      |
      v
  [Store Conditions] -- persist to SQLite
      |
      v
  [Synthesis Agent] (large model) -- cross-reference, resolve, produce answer
      |
      v
  Streamed SSE response
"""

import asyncio
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    is_utility_request,
    make_sse_chunk,
    register_standard_routes,
    require_env,
    setup_logging,
    stream_passthrough,
)

# --- Logging ---
LOG_DIR = os.getenv("PERSISTENT_RESEARCH_LOG_DIR", "/opt/persistent_research_logs")
log = setup_logging("persistent-research", LOG_DIR)

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = require_env("UPSTREAM_KEY")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "mistral-small-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = env_int("PERSISTENT_RESEARCH_PORT", 9300, minimum=1)
MAX_SUBAGENTS = env_int("MAX_SUBAGENTS", 5, minimum=1)
MAX_SUBAGENT_TURNS = env_int("MAX_SUBAGENT_TURNS", 10, minimum=1)
MAX_CONCURRENT = env_int("MAX_CONCURRENT_PERSISTENT", 2, minimum=1)
PERSISTENCE_DB = os.getenv("PERSISTENCE_DB", "/opt/persistent_research_logs/knowledge.db")
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000
MAX_PRIOR_CONDITIONS = 20  # max prior conditions to seed context with

log.info(
    f"Config: synthesis_model={UPSTREAM_MODEL}, subagent_model={SUBAGENT_MODEL}, "
    f"upstream={UPSTREAM_BASE}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, "
    f"max_subagents={MAX_SUBAGENTS}, max_subagent_turns={MAX_SUBAGENT_TURNS}, "
    f"persistence_db={PERSISTENCE_DB}"
)

# --- Shared state ---
tracker = RequestTracker()
limiter = ConcurrencyLimiter(MAX_CONCURRENT)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class AtomicCondition:
    """A single compressed research finding."""
    fact: str
    source_url: str = ""
    confidence: float = 0.5
    angle: str = ""
    domain: str = ""
    is_serendipitous: bool = False

    def to_text(self) -> str:
        parts = [f"- {self.fact}"]
        if self.source_url:
            parts[0] += f" [source: {self.source_url}]"
        if self.confidence != 0.5:
            parts[0] += f" (confidence: {self.confidence:.1f})"
        if self.is_serendipitous:
            parts[0] += " [SERENDIPITOUS]"
        return parts[0]


@dataclass
class SubagentResult:
    """Result from a single subagent's research."""
    angle: str
    conditions: list[AtomicCondition] = field(default_factory=list)
    turns_used: int = 0
    tool_calls_made: int = 0
    error: str = ""


# ============================================================================
# Persistent Storage (SQLite + FTS5)
# ============================================================================

def _init_db(db_path: str) -> None:
    """Create the SQLite database and tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atomic_conditions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                angle TEXT DEFAULT '',
                fact TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                domain TEXT DEFAULT '',
                is_serendipitous INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        # FTS5 virtual table for full-text search on facts
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conditions_fts
            USING fts5(fact, query, angle, content=atomic_conditions, content_rowid=rowid)
        """)
        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS conditions_ai AFTER INSERT ON atomic_conditions BEGIN
                INSERT INTO conditions_fts(rowid, fact, query, angle)
                VALUES (new.rowid, new.fact, new.query, new.angle);
            END
        """)
        conn.commit()
    finally:
        conn.close()


def _store_conditions_sync(
    db_path: str,
    session_id: str,
    query: str,
    conditions: list[AtomicCondition],
) -> int:
    """Store atomic conditions in the database. Returns count stored.

    This is a synchronous function; call via ``run_in_executor``.
    Thread safety is handled by SQLite's built-in WAL-mode locking.
    """
    if not conditions:
        return 0
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                f"cond-{uuid.uuid4().hex[:12]}",
                session_id,
                query,
                c.angle,
                c.fact,
                c.source_url,
                c.confidence,
                c.domain,
                1 if c.is_serendipitous else 0,
                now,
            )
            for c in conditions
        ]
        conn.executemany(
            """INSERT INTO atomic_conditions
               (id, session_id, query, angle, fact, source_url, confidence,
                domain, is_serendipitous, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _retrieve_related_sync(db_path: str, query: str, limit: int = 20) -> list[dict]:
    """Retrieve prior conditions related to the query using FTS5.

    Synchronous; call via ``run_in_executor``.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        # Tokenise the query for FTS5 — use OR matching for broad recall
        tokens = [t.strip() for t in query.split() if len(t.strip()) > 2]
        if not tokens:
            return []
        fts_query = " OR ".join(tokens[:10])  # cap at 10 tokens
        cursor = conn.execute(
            """SELECT ac.fact, ac.source_url, ac.confidence, ac.angle,
                      ac.is_serendipitous, ac.query, ac.created_at
               FROM conditions_fts
               JOIN atomic_conditions ac ON conditions_fts.rowid = ac.rowid
               WHERE conditions_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "fact": r[0],
                "source_url": r[1],
                "confidence": r[2],
                "angle": r[3],
                "is_serendipitous": bool(r[4]),
                "original_query": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
    except Exception as e:
        log.warning(f"FTS5 retrieval error: {e}")
        return []
    finally:
        conn.close()


# Initialise the database at import time
try:
    _init_db(PERSISTENCE_DB)
    log.info(f"Persistence DB initialised: {PERSISTENCE_DB}")
except Exception as e:
    log.error(f"Failed to initialise persistence DB: {e}")


# ============================================================================
# Native Tool Definitions (OpenAI function-calling format)
# ============================================================================

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": (
                "Search the web using SearXNG. Returns top results with titles, "
                "URLs, and snippets. Use this to find information, verify facts, "
                "discover sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch a webpage and extract its readable text content. Use this "
                "to read articles, documentation, or any web page found via search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "extract_info": {
                        "type": "string",
                        "description": "Optional: specific information to look for",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute Python code for calculations, data processing, or analysis. "
                "Code runs in a sandboxed subprocess with a 30-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ============================================================================
# Tool Execution
# ============================================================================

async def tool_searxng_search(query: str) -> str:
    """Execute a SearXNG search and return formatted results."""
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
        results = data.get("results", [])[:10]

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")

        return "\n\n".join(formatted)

    except httpx.TimeoutException:
        return "Search error: request timed out after 20s"
    except Exception as e:
        return f"Search error: {str(e)}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage and extract readable text."""
    try:
        client = http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code}"

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Non-text content type: {content_type}"

        raw_html = resp.text

        text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        if len(text) > WEBPAGE_MAX_CHARS:
            text = text[:WEBPAGE_MAX_CHARS] + "\n\n[... content truncated ...]"

        if not text.strip():
            return "Page returned no readable text content."

        result = f"**Content from {url}:**\n\n{text}"
        if extract_info:
            result = f"**Looking for: {extract_info}**\n\n{result}"
        return result

    except httpx.ReadTimeout:
        return f"Fetch error: Timeout reading {url}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


def tool_python_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > PYTHON_OUTPUT_MAX:
            output = output[:PYTHON_OUTPUT_MAX] + "\n[... output truncated ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


async def execute_tool(tool_name: str, arguments: dict) -> str:
    """Route and execute a tool call."""
    if tool_name == "searxng_search":
        return await tool_searxng_search(arguments.get("query", ""))
    elif tool_name == "fetch_webpage":
        return await tool_fetch_webpage(
            arguments.get("url", ""),
            arguments.get("extract_info", ""),
        )
    elif tool_name == "python_exec":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool_python_exec, arguments.get("code", ""))
    else:
        return f"Unknown tool: {tool_name}"


async def execute_tools_parallel(
    tool_calls_with_ids: list[tuple[str, str, dict]],
) -> list[tuple[str, str, str, float]]:
    """Execute multiple tool calls concurrently."""

    async def _run_one(tc_id: str, name: str, args: dict):
        t0 = time.monotonic()
        result = await execute_tool(name, args)
        return tc_id, name, result, time.monotonic() - t0

    tasks = [_run_one(tc_id, name, args) for tc_id, name, args in tool_calls_with_ids]
    return list(await asyncio.gather(*tasks))


# ============================================================================
# LLM Communication
# ============================================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]


async def call_llm(
    messages: list[dict],
    req_id: str,
    *,
    model: str = "",
    include_tools: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict:
    """
    Call the upstream LLM. Supports both the large synthesis model and the
    small subagent model via the *model* parameter.
    """
    model = model or UPSTREAM_MODEL
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if include_tools:
        body["tools"] = NATIVE_TOOLS
        body["tool_choice"] = "auto"

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
                last_error = f"[LLM Error: HTTP {resp.status_code}] {error_text}"

                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_LLM_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    log.warning(
                        f"[{req_id}] Retryable error {resp.status_code}, "
                        f"waiting {wait}s (attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue

                return {"error": last_error}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "[LLM Error: No choices in response]"}

            message = choices[0].get("message", {})
            return {
                "message": message,
                "content": message.get("content", "") or "",
                "tool_calls": message.get("tool_calls", None),
                "finish_reason": choices[0].get("finish_reason", ""),
            }

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = f"[LLM Error: {type(e).__name__}]"
            if attempt < MAX_LLM_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.warning(
                    f"[{req_id}] Timeout, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{MAX_LLM_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            return {"error": last_error}

        except Exception as e:
            return {"error": f"[LLM Error: {str(e)}]"}

    return {"error": last_error or "[LLM Error: Max retries exceeded]"}


# ============================================================================
# Planning Agent
# ============================================================================

PLANNING_PROMPT = """You are a research planning agent. Your job is to decompose a user's question into distinct research angles that can be investigated independently and in parallel.

Given the user's query, produce a JSON object with exactly this structure:
{
  "angles": [
    {"title": "short angle title", "query": "specific search query for this angle", "description": "what this angle investigates"},
    ...
  ],
  "bridge_queries": [
    {"query": "cross-domain search query", "domains": ["domain1", "domain2"], "rationale": "why this unexpected connection might be useful"}
  ]
}

Rules:
1. Generate 3-7 angles covering: factual/technical, historical/context, contrarian/alternative views, practical/applied, and recent developments.
2. Generate 1-2 bridge queries that look for unexpected cross-domain connections (serendipity). These should connect the topic to a seemingly unrelated field.
3. Each angle should be independent enough to research separately.
4. Make search queries specific and actionable.
5. Output ONLY valid JSON, no markdown fences or commentary."""


async def plan_research(
    user_query: str,
    prior_conditions: list[dict],
    req_id: str,
) -> dict:
    """
    Use the small model to decompose the query into research angles.
    Returns {"angles": [...], "bridge_queries": [...]}.
    """
    messages = [{"role": "system", "content": PLANNING_PROMPT}]

    user_content = f"User query: {user_query}"
    if prior_conditions:
        prior_text = "\n".join(
            f"- {c['fact']} [from prior research on: {c['original_query']}]"
            for c in prior_conditions[:10]
        )
        user_content += f"\n\nPrior knowledge from previous research sessions:\n{prior_text}"
        user_content += "\n\nConsider these prior findings when planning angles. Avoid redundant research."

    messages.append({"role": "user", "content": user_content})

    result = await call_llm(messages, req_id, model=SUBAGENT_MODEL, max_tokens=2048, temperature=0.4)

    if "error" in result:
        log.error(f"[{req_id}] Planning agent error: {result['error']}")
        # Fallback: create a single angle from the query itself
        return {
            "angles": [{"title": "General research", "query": user_query, "description": "Direct research"}],
            "bridge_queries": [],
        }

    content = result.get("content", "")

    # Try to parse JSON from the response
    try:
        # Strip markdown fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        plan = json.loads(cleaned)

        angles = plan.get("angles", [])
        bridge_queries = plan.get("bridge_queries", [])

        # Validate
        if not angles:
            raise ValueError("No angles in plan")

        # Cap at MAX_SUBAGENTS
        angles = angles[:MAX_SUBAGENTS]

        # Add bridge queries as additional angles
        for bq in bridge_queries[:2]:
            if len(angles) < MAX_SUBAGENTS + 2:  # allow up to 2 extra for bridges
                angles.append({
                    "title": f"Bridge: {bq.get('domains', ['?'])[0]} x {bq.get('domains', ['?', '?'])[1] if len(bq.get('domains', [])) > 1 else '?'}",
                    "query": bq.get("query", ""),
                    "description": bq.get("rationale", "Cross-domain exploration"),
                    "is_bridge": True,
                })

        return {"angles": angles, "bridge_queries": bridge_queries}

    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[{req_id}] Planning agent returned invalid JSON: {e}, content={content[:200]}")
        # Fallback: extract angles heuristically or use the query directly
        return {
            "angles": [
                {"title": "General research", "query": user_query, "description": "Direct research on the topic"},
                {"title": "Recent developments", "query": f"{user_query} recent news 2024 2025", "description": "Latest developments"},
                {"title": "Expert analysis", "query": f"{user_query} expert analysis review", "description": "Expert perspectives"},
            ],
            "bridge_queries": [],
        }


# ============================================================================
# Subagent Research (with AoT State Contraction)
# ============================================================================

SUBAGENT_PROMPT_TEMPLATE = """You are a focused research subagent. Today is: {date}

Your assigned research angle: {angle_title}
Description: {angle_description}
Initial search query: {angle_query}

**INSTRUCTIONS:**
1. Use tools to research this specific angle thoroughly.
2. After EACH tool result, extract the key facts as atomic conditions.
3. Search from multiple sub-angles within your assigned topic.
4. Read actual web pages, don't just rely on search snippets.
5. Be thorough but focused on your assigned angle.

**ATOMIC CONDITION FORMAT:**
After gathering information, you must output your findings as atomic conditions.
When you are done researching, output your findings in this exact JSON format:
```json
{{"conditions": [
    {{"fact": "clear factual statement", "source_url": "url", "confidence": 0.8}},
    ...
]}}
```

**TOOL USAGE:**
- You MUST use tools. Never answer from training data alone.
- Do NOT repeat the same search query or fetch the same URL twice.
- If a tool call fails, try a different approach.

**WHEN TO STOP:**
- You have found 3-8 distinct facts about your angle
- Additional searches return information you already have
- You have verified key claims across sources"""

CONDITION_EXTRACTION_PROMPT = """Based on the research you've done so far, extract all key findings as atomic conditions.

Output ONLY a JSON object with this structure:
{"conditions": [
    {"fact": "clear factual statement supported by your research", "source_url": "the URL source", "confidence": 0.9},
    ...
]}

Rules:
- Each fact should be a single, clear, verifiable statement
- Confidence: 0.9 for well-sourced facts, 0.7 for partially verified, 0.5 for single-source, 0.3 for uncertain
- Include the most relevant source URL for each fact
- Output 3-10 conditions maximum
- Output ONLY valid JSON, no markdown fences"""


async def run_subagent(
    angle: dict,
    subagent_index: int,
    progress_queue: asyncio.Queue,
    req_id: str,
    user_query: str,
) -> SubagentResult:
    """
    Run a single subagent's research loop on one angle.
    Uses AoT-style state contraction: after tool results, compress to conditions.
    """
    angle_title = angle.get("title", f"Angle {subagent_index + 1}")
    angle_query = angle.get("query", user_query)
    angle_desc = angle.get("description", "Research this angle")
    is_bridge = angle.get("is_bridge", False)
    sa_id = f"{req_id}-sa{subagent_index}"

    log.info(f"[{sa_id}] Starting subagent: {angle_title}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = SUBAGENT_PROMPT_TEMPLATE.format(
        date=today,
        angle_title=angle_title,
        angle_description=angle_desc,
        angle_query=angle_query,
    )

    agent_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research this angle thoroughly: {angle_query}"},
    ]

    result = SubagentResult(angle=angle_title)
    used_queries: set[str] = set()
    consecutive_errors = 0

    try:
        for turn in range(1, MAX_SUBAGENT_TURNS + 1):
            await progress_queue.put({
                "type": "progress",
                "subagent": subagent_index,
                "text": f"  [{angle_title}] Turn {turn}/{MAX_SUBAGENT_TURNS}\n",
            })

            llm_result = await call_llm(
                agent_messages, sa_id,
                model=SUBAGENT_MODEL,
                include_tools=True,
                max_tokens=4096,
                temperature=0.3,
            )

            if "error" in llm_result:
                consecutive_errors += 1
                log.warning(f"[{sa_id}] Turn {turn}: Error: {llm_result['error']}")
                if consecutive_errors >= 3:
                    result.error = llm_result["error"]
                    break
                agent_messages.append({"role": "assistant", "content": llm_result["error"]})
                agent_messages.append({"role": "user", "content": "Error occurred. Try a different approach."})
                continue

            consecutive_errors = 0
            content = llm_result.get("content", "")
            tool_calls = llm_result.get("tool_calls")

            # No tool calls — model wants to stop
            if not tool_calls:
                result.turns_used = turn
                # Try to extract conditions from the final content
                conditions = _parse_conditions(content, angle_title, is_bridge)
                if conditions:
                    result.conditions.extend(conditions)
                break

            # Process tool calls
            assistant_msg: dict = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
            agent_messages.append(assistant_msg)

            calls_to_run: list[tuple[str, str, dict]] = []
            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments_str = func.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}

                query_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if query_key in used_queries:
                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "Duplicate call skipped. Try a different query.",
                    })
                    continue

                used_queries.add(query_key)
                calls_to_run.append((tc_id, tool_name, arguments))

            if calls_to_run:
                tool_results = await execute_tools_parallel(calls_to_run)
                result.tool_calls_made += len(tool_results)

                for tc_id, tool_name, tool_result, duration in tool_results:
                    await progress_queue.put({
                        "type": "tool",
                        "subagent": subagent_index,
                        "text": f"  [{angle_title}] {tool_name} ({duration:.1f}s)\n",
                    })

                    # Truncate long results for context management (AoT principle)
                    truncated = tool_result
                    if len(tool_result) > 8000:
                        truncated = tool_result[:6000] + "\n[...truncated...]\n" + tool_result[-1500:]

                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": truncated,
                    })

            # AoT State Contraction: periodically compress context
            # Every 3 turns, ask the model to extract conditions and reset context
            if turn > 0 and turn % 3 == 0 and turn < MAX_SUBAGENT_TURNS:
                contraction_msgs = agent_messages + [
                    {"role": "user", "content": CONDITION_EXTRACTION_PROMPT}
                ]
                extract_result = await call_llm(
                    contraction_msgs, sa_id,
                    model=SUBAGENT_MODEL,
                    max_tokens=2048,
                    temperature=0.1,
                )
                if "error" not in extract_result:
                    mid_conditions = _parse_conditions(
                        extract_result.get("content", ""), angle_title, is_bridge
                    )
                    if mid_conditions:
                        result.conditions.extend(mid_conditions)
                        # Contract: replace full history with summary
                        conditions_text = "\n".join(c.to_text() for c in mid_conditions)
                        agent_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": (
                                f"Continue researching: {angle_query}\n\n"
                                f"Findings so far (from previous turns):\n{conditions_text}\n\n"
                                f"Find NEW information that is NOT covered above. "
                                f"Search for different sub-angles, deeper details, or verification."
                            )},
                        ]
                        log.info(
                            f"[{sa_id}] Turn {turn}: AoT contraction - "
                            f"compressed {len(mid_conditions)} conditions, reset context"
                        )

            result.turns_used = turn

        # Final condition extraction if we used all turns
        if result.turns_used >= MAX_SUBAGENT_TURNS and not result.conditions:
            agent_messages.append({"role": "user", "content": CONDITION_EXTRACTION_PROMPT})
            final_extract = await call_llm(
                agent_messages, sa_id,
                model=SUBAGENT_MODEL,
                max_tokens=2048,
                temperature=0.1,
            )
            if "error" not in final_extract:
                conditions = _parse_conditions(
                    final_extract.get("content", ""), angle_title, is_bridge
                )
                if conditions:
                    result.conditions.extend(conditions)

    except Exception as e:
        log.error(f"[{sa_id}] Subagent error: {e}\n{traceback.format_exc()}")
        result.error = str(e)

    # Deduplicate conditions
    seen_facts: set[str] = set()
    unique_conditions: list[AtomicCondition] = []
    for c in result.conditions:
        key = c.fact.lower().strip()[:100]
        if key not in seen_facts:
            seen_facts.add(key)
            unique_conditions.append(c)
    result.conditions = unique_conditions

    await progress_queue.put({
        "type": "done",
        "subagent": subagent_index,
        "angle": angle_title,
        "conditions_count": len(result.conditions),
    })

    log.info(
        f"[{sa_id}] Subagent complete: {len(result.conditions)} conditions, "
        f"{result.turns_used} turns, {result.tool_calls_made} tool calls"
    )
    return result


def _parse_conditions(content: str, angle: str, is_bridge: bool) -> list[AtomicCondition]:
    """Try to parse atomic conditions from LLM output."""
    if not content:
        return []

    # Try JSON parsing
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        data = json.loads(cleaned)
        conditions_data = data.get("conditions", [])
        return [
            AtomicCondition(
                fact=c.get("fact", ""),
                source_url=c.get("source_url", ""),
                confidence=float(c.get("confidence", 0.5)),
                angle=angle,
                is_serendipitous=is_bridge,
            )
            for c in conditions_data
            if c.get("fact")
        ]
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Fallback: try to find JSON embedded in text
    json_match = re.search(r'\{[^{}]*"conditions"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return [
                AtomicCondition(
                    fact=c.get("fact", ""),
                    source_url=c.get("source_url", ""),
                    confidence=float(c.get("confidence", 0.5)),
                    angle=angle,
                    is_serendipitous=is_bridge,
                )
                for c in data.get("conditions", [])
                if c.get("fact")
            ]
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: treat the whole content as a single condition
    if len(content.strip()) > 20:
        return [
            AtomicCondition(
                fact=content.strip()[:500],
                angle=angle,
                confidence=0.3,
                is_serendipitous=is_bridge,
            )
        ]

    return []


# ============================================================================
# Synthesis Agent
# ============================================================================

SYNTHESIS_PROMPT_TEMPLATE = """You are an expert synthesis agent. Today is: {date}

You have received atomic research conditions from {n_subagents} parallel research subagents investigating different angles of the user's question. Your job is to synthesize these into a comprehensive, well-structured answer.

**SYNTHESIS RULES:**
1. Cross-reference conditions across angles. Where multiple sources agree, note the consensus.
2. Where conditions contradict, explicitly note the contradiction and explain which is more reliable (based on confidence scores and source quality).
3. Highlight any serendipitous findings (marked [SERENDIPITOUS]) as "unexpected connections."
4. Structure the answer with clear headings and logical flow.
5. Cite sources with URLs where available.
6. Be thorough but clear. Write in plain, accessible language.
7. Do NOT add unsolicited warnings, ethical disclaimers, or safety preambles. Just answer the question directly.
8. Treat the user as an intelligent adult who deserves a straight answer.

**RESEARCH CONDITIONS BY ANGLE:**
{conditions_text}

{prior_knowledge_text}"""


async def synthesize_findings(
    user_query: str,
    subagent_results: list[SubagentResult],
    prior_conditions: list[dict],
    req_id: str,
) -> str:
    """
    Use the large model to synthesize all subagent findings into a final answer.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build conditions text grouped by angle
    conditions_by_angle: dict[str, list[str]] = {}
    total_conditions = 0
    for sr in subagent_results:
        if sr.conditions:
            angle_conditions = [c.to_text() for c in sr.conditions]
            conditions_by_angle[sr.angle] = angle_conditions
            total_conditions += len(angle_conditions)

    if not conditions_by_angle:
        return "No research findings were gathered. The subagents could not find relevant information."

    conditions_text = ""
    for angle, conds in conditions_by_angle.items():
        conditions_text += f"\n### {angle}\n"
        conditions_text += "\n".join(conds) + "\n"

    prior_text = ""
    if prior_conditions:
        prior_text = "\n**PRIOR KNOWLEDGE (from previous sessions):**\n"
        prior_text += "\n".join(
            f"- {c['fact']} [prior research: {c['original_query']}]"
            for c in prior_conditions[:10]
        )

    system_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        date=today,
        n_subagents=len(subagent_results),
        conditions_text=conditions_text,
        prior_knowledge_text=prior_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Based on the {total_conditions} research conditions gathered from "
            f"{len(subagent_results)} research angles, provide a comprehensive, "
            f"well-structured answer to the original question:\n\n{user_query}"
        )},
    ]

    result = await call_llm(
        messages, req_id,
        model=UPSTREAM_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    if "error" in result:
        return f"Synthesis error: {result['error']}"

    return result.get("content", "(No synthesis generated)")


# ============================================================================
# Main Orchestrator
# ============================================================================

async def run_persistent_research(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Orchestrate the full persistent deep research pipeline:
    1. Retrieve prior knowledge
    2. Planning (decompose into angles)
    3. Parallel subagent research (with AoT contraction)
    4. Persist findings
    5. Synthesis
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

    # Extract user query from the last user message
    user_query = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                # multimodal: extract text parts
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

    # --- Open thinking block ---
    yield chunk("<think>\n")

    try:
        # ================================================================
        # Phase 1: Retrieve Prior Knowledge
        # ================================================================
        yield chunk("**[Phase 1: Retrieving Prior Knowledge]**\n")

        loop = asyncio.get_running_loop()
        prior_conditions = await loop.run_in_executor(
            None, _retrieve_related_sync, PERSISTENCE_DB, user_query, MAX_PRIOR_CONDITIONS
        )

        if prior_conditions:
            yield chunk(f"Found {len(prior_conditions)} relevant prior findings:\n")
            for pc in prior_conditions[:5]:
                yield chunk(f"  - {pc['fact'][:100]}...\n")
            if len(prior_conditions) > 5:
                yield chunk(f"  ... and {len(prior_conditions) - 5} more\n")
        else:
            yield chunk("No prior knowledge found (first research on this topic).\n")

        # ================================================================
        # Phase 2: Planning
        # ================================================================
        yield chunk("\n**[Phase 2: Planning Research Angles]**\n")
        yield chunk(f"Using {SUBAGENT_MODEL} for planning...\n")

        plan = await plan_research(user_query, prior_conditions, req_id)
        angles = plan["angles"]

        yield chunk(f"Decomposed into {len(angles)} research angles:\n")
        for i, angle in enumerate(angles, 1):
            bridge_tag = " [BRIDGE]" if angle.get("is_bridge") else ""
            yield chunk(f"  {i}. **{angle['title']}**{bridge_tag}: {angle.get('description', '')[:80]}\n")

        # ================================================================
        # Phase 3: Parallel Subagent Research
        # ================================================================
        n_agents = len(angles)
        yield chunk(f"\n**[Phase 3: Launching {n_agents} Parallel Subagents]** (model: {SUBAGENT_MODEL})\n")

        progress_queue: asyncio.Queue = asyncio.Queue()

        # Launch all subagents concurrently
        subagent_tasks = [
            asyncio.create_task(
                run_subagent(angle, i, progress_queue, req_id, user_query)
            )
            for i, angle in enumerate(angles)
        ]

        # Stream progress updates as subagents work.
        # Use a task-monitoring approach so that crashed subagents
        # (which never post "done") don't hang the orchestrator.
        pending_tasks = set(subagent_tasks)
        completed_count = 0

        while completed_count < n_agents and pending_tasks:
            # Check if any tasks have finished (crashed or completed)
            done_tasks, _ = await asyncio.wait(
                pending_tasks, timeout=0, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done_tasks:
                pending_tasks.discard(t)
                if t.exception() is not None:
                    # Subagent crashed without posting "done"
                    completed_count += 1
                    idx = subagent_tasks.index(t)
                    log.error(f"[{req_id}] Subagent {idx} crashed: {t.exception()}")
                    yield chunk(f"  \u26a0\ufe0f Subagent {idx + 1} failed: {t.exception()}\n")

            # Drain progress messages (non-blocking)
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # Check if all tasks are done even though queue is empty
                if all(t.done() for t in subagent_tasks):
                    break
                continue

            if msg["type"] == "progress":
                yield chunk(msg["text"])
            elif msg["type"] == "tool":
                yield chunk(msg["text"])
            elif msg["type"] == "done":
                completed_count += 1
                yield chunk(
                    f"  \u2705 Subagent {msg['subagent'] + 1} ({msg['angle']}) complete: "
                    f"{msg['conditions_count']} atomic conditions\n"
                )

        # Collect all results
        subagent_results: list[SubagentResult] = []
        for task in subagent_tasks:
            if task.done() and task.exception() is None:
                try:
                    subagent_results.append(task.result())
                except Exception as e:
                    log.error(f"[{req_id}] Failed to collect subagent result: {e}")
            elif task.done() and task.exception() is not None:
                log.error(f"[{req_id}] Subagent task exception: {task.exception()}")
            else:
                # Still running — cancel and skip
                task.cancel()
                log.warning(f"[{req_id}] Cancelled hanging subagent task")

        # Summary
        all_conditions: list[AtomicCondition] = []
        total_turns = 0
        total_tools = 0
        for sr in subagent_results:
            all_conditions.extend(sr.conditions)
            total_turns += sr.turns_used
            total_tools += sr.tool_calls_made

        yield chunk(
            f"\n**Research Summary:** {len(all_conditions)} atomic conditions, "
            f"{total_turns} total turns, {total_tools} tool calls\n"
        )

        # ================================================================
        # Phase 4: Persist Findings
        # ================================================================
        if all_conditions:
            yield chunk("\n**[Phase 4: Persisting Knowledge]**\n")
            stored = await loop.run_in_executor(
                None, _store_conditions_sync, PERSISTENCE_DB, req_id, user_query, all_conditions
            )
            yield chunk(f"Stored {stored} conditions to persistent knowledge base.\n")

        # ================================================================
        # Phase 5: Synthesis
        # ================================================================
        yield chunk(f"\n**[Phase 5: Synthesizing Answer]** (model: {UPSTREAM_MODEL})\n")
        yield chunk("Cross-referencing findings across all angles...\n")

        final_answer = await synthesize_findings(
            user_query, subagent_results, prior_conditions, req_id
        )

        elapsed = time.monotonic() - start_time
        yield chunk(
            f"\n\u2705 Research complete in {elapsed:.1f}s "
            f"({len(all_conditions)} conditions from {n_agents} subagents)\n"
        )

        # --- Close thinking, stream answer ---
        yield chunk("\n</think>\n\n")

        # Stream the final answer in chunks
        for i in range(0, len(final_answer), 200):
            yield chunk(final_answer[i:i + 200])

        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Persistent research error after {elapsed:.2f}s: {e}\n{tb}")
        yield chunk(f"\n\u26a0\ufe0f Error: {str(e)}\n")
        yield chunk("\n</think>\n\n")
        yield chunk(f"**Deep Research Error**\n\nAn error occurred during research: {str(e)}")
        yield chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
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
        "persistence_db": PERSISTENCE_DB,
    },
)


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
    """Return statistics about the persistent knowledge base."""
    try:
        conn = sqlite3.connect(PERSISTENCE_DB)
        try:
            total = conn.execute("SELECT COUNT(*) FROM atomic_conditions").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM atomic_conditions").fetchone()[0]
            queries = conn.execute("SELECT COUNT(DISTINCT query) FROM atomic_conditions").fetchone()[0]
            recent = conn.execute(
                "SELECT query, COUNT(*) as cnt FROM atomic_conditions "
                "GROUP BY query ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            return JSONResponse({
                "total_conditions": total,
                "total_sessions": sessions,
                "unique_queries": queries,
                "recent_queries": [{"query": r[0], "conditions": r[1]} for r in recent],
            })
        finally:
            conn.close()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
