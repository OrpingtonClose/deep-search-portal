"""Text chunking with configurable size and overlap."""

import logging
import re
import uuid

from .config import CHUNK_SIZE, CHUNK_OVERLAP

log = logging.getLogger("knowledge-engine")


def _uid() -> str:
    return f"chunk-{uuid.uuid4().hex[:12]}"


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Returns a list of dicts with keys: id, chunk_index, content.
    """
    if not text.strip():
        return []

    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if len(text) <= chunk_size:
        return [{"id": _uid(), "chunk_index": 0, "content": text.strip()}]

    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < len(text):
            search_start = start + int(chunk_size * 0.8)
            boundary = _find_sentence_boundary(text, search_start, end)
            if boundary > search_start:
                end = boundary

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunks.append({
                "id": _uid(),
                "chunk_index": idx,
                "content": chunk_text_content,
            })
            idx += 1

        # Advance with overlap
        if end >= len(text):
            break
        start = end - overlap
        if start <= (end - chunk_size):
            start = end  # Safety: prevent infinite loop

    log.debug(f"Chunked {len(text)} chars into {len(chunks)} chunks")
    return chunks


def _find_sentence_boundary(text: str, search_start: int, search_end: int) -> int:
    """Find the best sentence boundary in the given range.

    Looks for period/newline/paragraph boundaries, preferring paragraph > newline > period.
    """
    segment = text[search_start:search_end]

    # Prefer paragraph break
    para_idx = segment.rfind("\n\n")
    if para_idx >= 0:
        return search_start + para_idx + 2

    # Then newline
    nl_idx = segment.rfind("\n")
    if nl_idx >= 0:
        return search_start + nl_idx + 1

    # Then sentence-ending punctuation followed by space
    for pattern in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
        p_idx = segment.rfind(pattern)
        if p_idx >= 0:
            return search_start + p_idx + len(pattern)

    return search_end
