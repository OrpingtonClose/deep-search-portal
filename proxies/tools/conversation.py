"""
Conversation state persistence — stores and restores research context
across multiple turns in a single Open WebUI chat thread.

Each chat thread gets a stable ``conversation_id`` derived from the
message history.  After each research run we snapshot the key outputs
(conditions, comprehension, final answer, entities) so that follow-up
queries can inherit prior context without re-researching from scratch.

Storage backend: a single SQLite file alongside the LangGraph checkpoint
DB.  Each row is a JSON blob keyed by (conversation_id, turn_index).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import log
from .llm import call_llm
from .config import SUBAGENT_MODEL

# ---------------------------------------------------------------------------
# Conversation ID derivation
# ---------------------------------------------------------------------------

def derive_conversation_id(messages: list[dict], chat_id: str | None = None) -> str:
    """Derive a stable conversation ID from the message history.

    If *chat_id* is provided (from Open WebUI's request body) we use it
    directly — it uniquely identifies the chat thread and avoids
    cross-user collisions when two users happen to open with the same
    first message.

    Fallback strategy: hash only the first user message content.  This
    gives a stable ID across all turns because the opening user message
    never changes within a chat thread.
    """
    if chat_id:
        return hashlib.sha256(chat_id.encode()).hexdigest()[:16]
    first_user = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "user" and not first_user:
            first_user = content.strip()[:500]
            break

    if not first_user:
        # No user messages — unlikely but handle gracefully
        return hashlib.sha256(json.dumps(messages[:3]).encode()).hexdigest()[:16]

    # Always hash only the first user message to keep the ID stable
    # across all turns.  Including the assistant response would cause
    # the ID to change between turn 1 (no assistant yet) and turn 2+.
    seed = first_user
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def count_user_turns(messages: list[dict]) -> int:
    """Count the number of user messages in the conversation history."""
    return sum(1 for m in messages if m.get("role") == "user")


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

_FOLLOWUP_DETECT_PROMPT = """You are a conversation analyst. Given the previous research context and a new user message, classify the new message.

Previous research topic: {prev_topic}
Previous research summary (first 500 chars): {prev_summary}

New user message: {new_message}

Is this new message:
A) A FOLLOW-UP to the previous research (asking for more detail, a related angle, clarification, or building on the prior findings)
B) A completely NEW TOPIC unrelated to the previous research

Output ONLY "A" or "B"."""


async def detect_followup(
    new_query: str,
    prev_query: str,
    prev_summary: str,
    req_id: str,
) -> bool:
    """Detect whether a new query is a follow-up to prior research.

    Returns True if the new query builds on the prior conversation.
    Falls back to keyword overlap heuristic if LLM call fails.
    """
    # Fast path: if queries are very similar, it's clearly a follow-up
    if _keyword_overlap(new_query, prev_query) > 0.5:
        return True

    prompt = (
        _FOLLOWUP_DETECT_PROMPT
        .replace("{prev_topic}", prev_query[:300])
        .replace("{prev_summary}", prev_summary[:500])
        .replace("{new_message}", new_query[:500])
    )

    try:
        result = await call_llm(
            [{"role": "user", "content": prompt}],
            req_id,
            model=SUBAGENT_MODEL,
            max_tokens=5,
            temperature=0.0,
        )
        if "error" not in result:
            answer = result.get("content", "").strip().upper()
            if answer.startswith("A"):
                return True
            if answer.startswith("B"):
                return False
    except Exception as e:
        log.debug(f"[{req_id}] Follow-up detection LLM failed: {e}")

    # Fallback: keyword overlap heuristic
    return _keyword_overlap(new_query, prev_query) > 0.25


def _keyword_overlap(query_a: str, query_b: str) -> float:
    """Compute keyword overlap ratio between two queries (Jaccard)."""
    words_a = set(re.split(r'\W+', query_a.lower())) - {"", "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "with", "this", "that", "it", "be", "as", "by", "from", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "shall", "about", "what", "how", "why", "when", "where", "which", "who", "whom"}
    words_b = set(re.split(r'\W+', query_b.lower())) - {"", "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "with", "this", "that", "it", "be", "as", "by", "from", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "shall", "about", "what", "how", "why", "when", "where", "which", "who", "whom"}
    # Only keep words with 3+ chars to avoid noise
    words_a = {w for w in words_a if len(w) >= 3}
    words_b = {w for w in words_b if len(w) >= 3}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ---------------------------------------------------------------------------
# Conversation state snapshot
# ---------------------------------------------------------------------------

@dataclass
class ConversationSnapshot:
    """Serialisable snapshot of a single research turn's output."""
    conversation_id: str
    turn_index: int
    user_query: str
    # Serialised condition facts (just the fact strings — lightweight)
    condition_facts: list[str] = field(default_factory=list)
    # Comprehension data dict (from QueryComprehension)
    comprehension_data: dict = field(default_factory=dict)
    # Final synthesised answer
    final_answer: str = ""
    # Report URL (if generated)
    report_url: str = ""
    # Timestamp
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "user_query": self.user_query,
            "condition_facts": self.condition_facts,
            "comprehension_data": self.comprehension_data,
            "final_answer": self.final_answer,
            "report_url": self.report_url,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSnapshot":
        return cls(
            conversation_id=data.get("conversation_id", ""),
            turn_index=data.get("turn_index", 0),
            user_query=data.get("user_query", ""),
            condition_facts=data.get("condition_facts", []),
            comprehension_data=data.get("comprehension_data", {}),
            final_answer=data.get("final_answer", ""),
            report_url=data.get("report_url", ""),
            created_at=data.get("created_at", 0.0),
        )


# ---------------------------------------------------------------------------
# SQLite-backed conversation store
# ---------------------------------------------------------------------------

_CONV_DB_PATH = os.getenv(
    "CONVERSATION_STATE_DB",
    "/opt/persistent_research_logs/conversation_state.sqlite3",
)


class ConversationStateStore:
    """Thread-safe SQLite store for conversation snapshots.

    Each conversation is identified by a ``conversation_id`` derived
    from the message history.  Multiple turns (snapshots) can be stored
    per conversation, ordered by ``turn_index``.
    """

    def __init__(self, db_path: str = _CONV_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
        try:
            db_dir = os.path.dirname(self._db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    conversation_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (conversation_id, turn_index)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_id
                ON conversation_turns (conversation_id)
                """
            )
            self._conn.commit()
            log.info(f"Conversation state store ready at {self._db_path}")
        except Exception as e:
            log.warning(f"Conversation state store init failed: {e}")
            self._conn = None

    def save_turn(self, snapshot: ConversationSnapshot) -> None:
        """Save or update a conversation turn snapshot."""
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO conversation_turns
                    (conversation_id, turn_index, data, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        snapshot.conversation_id,
                        snapshot.turn_index,
                        json.dumps(snapshot.to_dict(), default=str),
                        snapshot.created_at or time.time(),
                    ),
                )
                self._conn.commit()
        except Exception as e:
            log.warning(f"Failed to save conversation turn: {e}")

    def load_turns(self, conversation_id: str) -> list[ConversationSnapshot]:
        """Load all turns for a conversation, ordered by turn_index."""
        if self._conn is None:
            return []
        try:
            with self._lock:
                cursor = self._conn.execute(
                    """
                    SELECT data FROM conversation_turns
                    WHERE conversation_id = ?
                    ORDER BY turn_index ASC
                    """,
                    (conversation_id,),
                )
                rows = cursor.fetchall()
            return [
                ConversationSnapshot.from_dict(json.loads(row[0]))
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Failed to load conversation turns: {e}")
            return []

    def get_latest_turn(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        """Load the most recent turn for a conversation."""
        if self._conn is None:
            return None
        try:
            with self._lock:
                cursor = self._conn.execute(
                    """
                    SELECT data FROM conversation_turns
                    WHERE conversation_id = ?
                    ORDER BY turn_index DESC
                    LIMIT 1
                    """,
                    (conversation_id,),
                )
                row = cursor.fetchone()
            if row:
                return ConversationSnapshot.from_dict(json.loads(row[0]))
            return None
        except Exception as e:
            log.warning(f"Failed to load latest conversation turn: {e}")
            return None

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Module-level singleton store
_conversation_store: Optional[ConversationStateStore] = None


def get_conversation_store() -> ConversationStateStore:
    """Get (or create) the module-level conversation state store."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStateStore()
    return _conversation_store


# ---------------------------------------------------------------------------
# Context injection: build prior-context summary for follow-up queries
# ---------------------------------------------------------------------------

def build_followup_context(
    prior_turns: list[ConversationSnapshot],
    max_facts: int = 30,
) -> dict[str, Any]:
    """Build a context dict from prior conversation turns.

    Returns a dict with:
      - prior_condition_facts: list of fact strings from prior research
      - prior_queries: list of previously researched queries
      - prior_summary: the most recent final_answer (truncated)
      - prior_comprehension: the most recent comprehension_data
      - turn_count: how many prior turns exist
    """
    all_facts: list[str] = []
    prior_queries: list[str] = []
    prior_summary = ""
    prior_comprehension: dict = {}

    for turn in prior_turns:
        prior_queries.append(turn.user_query)
        all_facts.extend(turn.condition_facts)
        if turn.final_answer:
            prior_summary = turn.final_answer
        if turn.comprehension_data:
            prior_comprehension = turn.comprehension_data

    # Deduplicate and limit facts
    seen: set[str] = set()
    unique_facts: list[str] = []
    for fact in all_facts:
        key = fact.lower().strip()[:100]
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)

    return {
        "prior_condition_facts": unique_facts[-max_facts:],
        "prior_queries": prior_queries,
        "prior_summary": prior_summary[:3000],
        "prior_comprehension": prior_comprehension,
        "turn_count": len(prior_turns),
    }
