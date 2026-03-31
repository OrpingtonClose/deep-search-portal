"""Configuration for the Knowledge Engine service."""

import logging
import logging.handlers
import os

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

# --- LLM (Mistral) ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "mistral-small-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")

# --- Service ---
LISTEN_PORT = int(os.getenv("KNOWLEDGE_ENGINE_PORT", "9850"))
RAW_FILES_DIR = os.getenv("RAW_FILES_DIR", "/opt/knowledge_engine/raw")
LOG_DIR = os.getenv("LOG_DIR", "/opt/knowledge_engine/logs")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Extraction ---
MAX_EXTRACTION_CONCURRENCY = int(os.getenv("MAX_EXTRACTION_CONCURRENCY", "5"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))


def setup_logging(name: str = "knowledge-engine") -> logging.Logger:
    """Set up logging with file and console handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    return log
