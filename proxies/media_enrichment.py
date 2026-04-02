#!/usr/bin/env python3
"""
Media Enrichment — append image and video results to any answer.

Searches SearXNG for images + videos, and optionally Filmot for
spoken-content matches in YouTube transcripts.  All sources run in
parallel; results are merged, deduplicated by video ID, and formatted
as structured metadata for the synthesis model.

Fail-safe: any exception returns an empty string so the main answer
is never blocked by media enrichment failures.
"""

import asyncio
import logging
import os
import re
from urllib.parse import parse_qs, urlparse

from shared import http_client

log = logging.getLogger("media-enrichment")

# ---------------------------------------------------------------------------
# Configuration (all toggleable via env vars)
# ---------------------------------------------------------------------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
MEDIA_ENRICHMENT_ENABLED = os.getenv("MEDIA_ENRICHMENT_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
MEDIA_ENRICHMENT_MAX_IMAGES = int(os.getenv("MEDIA_ENRICHMENT_MAX_IMAGES", "6"))
MEDIA_ENRICHMENT_MAX_VIDEOS = int(os.getenv("MEDIA_ENRICHMENT_MAX_VIDEOS", "4"))

# Filmot (spoken-content search inside YouTube transcripts)
# Get a key at https://rapidapi.com/Jopik1/api/filmot-tube-metadata-archive/
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
FILMOT_ENABLED = bool(RAPIDAPI_KEY)
FILMOT_HOST = "filmot-tube-metadata-archive.p.rapidapi.com"


# ---------------------------------------------------------------------------
# SearXNG query helper
# ---------------------------------------------------------------------------
async def _searxng_media_query(
    query: str, categories: str, max_results: int
) -> list[dict]:
    """Call SearXNG search endpoint and return raw result dicts."""
    client = http_client()
    params = {"q": query, "format": "json", "categories": categories}
    resp = await client.get(
        f"{SEARXNG_URL}/search", params=params, timeout=15.0
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return results[:max_results]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_image_results(results: list[dict], max_items: int) -> str:
    """Format SearXNG image results as a markdown section.

    SearXNG image results include ``img_src`` (direct image URL),
    ``url`` (source page), ``title``, and ``content``.
    Each is rendered as a clickable markdown image embed.
    Deduplicates by ``img_src``.
    """
    if not results:
        return ""

    seen: set[str] = set()
    lines: list[str] = []

    for item in results:
        img_src = item.get("img_src", "")
        if not img_src or img_src in seen:
            continue
        seen.add(img_src)

        title = item.get("title", "Image").strip() or "Image"
        page_url = item.get("url", img_src)
        caption = item.get("content", "").strip()

        line = f"[![{title}]({img_src})]({page_url})"
        if caption:
            line += f"  \n*{caption}*"
        lines.append(line)

        if len(lines) >= max_items:
            break

    if not lines:
        return ""

    return "\n\n### Visual References\n\n" + "\n\n".join(lines) + "\n"


def _format_video_results(results: list[dict], max_items: int) -> str:
    """Format SearXNG video results as a structured section for synthesis.

    Each YouTube video entry includes the video_id, thumbnail URL, title,
    and description so the synthesis model can judge relevance per-section
    and render both an artifact embed and a clickable thumbnail.
    """
    if not results:
        return ""

    lines: list[str] = []

    for item in results:
        url = item.get("url", "")
        if not url:
            continue

        title = item.get("title", "Video").strip() or "Video"
        content = item.get("content", "").strip()

        # Try to extract YouTube video ID
        video_id = _extract_youtube_id(url)
        source_tag = item.get("_source", "searxng")
        if video_id:
            thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            line = (
                f"- **Title:** {title}\n"
                f"  **Video ID:** {video_id}\n"
                f"  **Thumbnail:** {thumbnail}\n"
                f"  **URL:** {url}"
            )
            if content:
                # Filmot results have transcript snippets with timestamps;
                # SearXNG results have a generic description.
                label = "Spoken content" if source_tag == "filmot" else "Description"
                line += f"\n  **{label}:** {content}"
        else:
            line = f"- **Title:** {title}\n  **URL:** {url}"
            if content:
                line += f"\n  **Description:** {content}"

        lines.append(line)

        if len(lines) >= max_items:
            break

    if not lines:
        return ""

    return "\n\n### Available Videos\n\n" + "\n\n".join(lines) + "\n"


_YT_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)"
    r"([A-Za-z0-9_-]{11})"
)


def _extract_youtube_id(url: str) -> str:
    """Return the 11-char YouTube video ID from a URL, or empty string."""
    m = _YT_PATTERN.search(url)
    if m:
        return m.group(1)
    # Fallback: parse query string for 'v' parameter
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        v_list = qs.get("v", [])
        if v_list and len(v_list[0]) == 11:
            return v_list[0]
    return ""


# ---------------------------------------------------------------------------
# Filmot spoken-content search
# ---------------------------------------------------------------------------
async def _filmot_search(query: str, max_results: int) -> list[dict]:
    """Search YouTube transcripts via Filmot RapidAPI.

    Returns results normalised to the same shape as SearXNG video results
    so they can be merged seamlessly.
    """
    if not FILMOT_ENABLED:
        return []

    client = http_client()
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": FILMOT_HOST,
    }
    params = {
        "query": query,
        "lang": "en",
        "searchManualSubs": "true",
        "searchAutoSubs": "true",
    }
    resp = await client.get(
        f"https://{FILMOT_HOST}/getsearchresults",
        headers=headers,
        params=params,
        timeout=12.0,
    )
    resp.raise_for_status()
    data = resp.json()

    # Filmot returns a list of hit objects; normalise to SearXNG-like dicts
    normalised: list[dict] = []
    for hit in data if isinstance(data, list) else []:
        vid = hit.get("id") or hit.get("videoId") or ""
        if not vid:
            continue
        title = hit.get("title") or "Video"
        # Build a spoken-content snippet from the hits
        hits_data = hit.get("hits") or hit.get("hitsData") or []
        snippets: list[str] = []
        for h in hits_data[:3]:  # first 3 transcript matches
            text = h.get("text", "").strip()
            ts = h.get("start", h.get("timestamp", 0))
            if text:
                snippets.append(f"[{_fmt_ts(ts)}] {text}")
        content = " | ".join(snippets) if snippets else ""
        normalised.append({
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title,
            "content": content,
            "_source": "filmot",  # tag for dedup priority
        })
        if len(normalised) >= max_results:
            break
    return normalised


def _fmt_ts(seconds) -> str:
    """Format seconds (int/float/str) as M:SS for display."""
    try:
        s = int(float(seconds))
    except (ValueError, TypeError):
        return "0:00"
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Merge + deduplicate video results from multiple sources
# ---------------------------------------------------------------------------
def _merge_video_results(
    searxng: list[dict], filmot: list[dict]
) -> list[dict]:
    """Merge video results, preferring Filmot (richer transcript data).

    Deduplicates by YouTube video ID.  When the same video appears in
    both sources, the Filmot entry wins because it carries spoken-content
    snippets that help the synthesis model judge relevance.
    """
    seen_ids: set[str] = set()
    merged: list[dict] = []

    # Filmot first (higher quality — has transcript snippets)
    for item in filmot:
        vid = _extract_youtube_id(item.get("url", ""))
        key = vid or item.get("url", "")
        if key and key not in seen_ids:
            seen_ids.add(key)
            merged.append(item)

    # Then SearXNG (fills in non-duplicate results)
    for item in searxng:
        vid = _extract_youtube_id(item.get("url", ""))
        key = vid or item.get("url", "")
        if key and key not in seen_ids:
            seen_ids.add(key)
            merged.append(item)

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def enrich_with_media(user_query: str, req_id: str = "") -> str:
    """Run image + video + Filmot searches in parallel and return combined markdown.

    Fail-safe: any exception returns an empty string so the main answer
    is never blocked by media enrichment failures.
    """
    if not MEDIA_ENRICHMENT_ENABLED:
        return ""

    try:
        image_task = _searxng_media_query(
            user_query, "images", MEDIA_ENRICHMENT_MAX_IMAGES
        )
        video_task = _searxng_media_query(
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS * 2  # extra for dedup
        )
        filmot_task = _filmot_search(
            user_query, MEDIA_ENRICHMENT_MAX_VIDEOS
        )

        image_results, video_results, filmot_results = await asyncio.gather(
            image_task, video_task, filmot_task, return_exceptions=True
        )

        # If any search raised, treat its results as empty
        if isinstance(image_results, BaseException):
            log.warning(
                "[%s] Media enrichment image search failed: %s",
                req_id, image_results,
            )
            image_results = []
        if isinstance(video_results, BaseException):
            log.warning(
                "[%s] Media enrichment video search failed: %s",
                req_id, video_results,
            )
            video_results = []
        if isinstance(filmot_results, BaseException):
            log.warning(
                "[%s] Filmot spoken-content search failed: %s",
                req_id, filmot_results,
            )
            filmot_results = []

        # Merge video sources (Filmot wins on duplicates)
        all_videos = _merge_video_results(video_results, filmot_results)

        images_md = _format_image_results(image_results, MEDIA_ENRICHMENT_MAX_IMAGES)
        videos_md = _format_video_results(all_videos, MEDIA_ENRICHMENT_MAX_VIDEOS)

        return images_md + videos_md

    except Exception as exc:
        log.warning("[%s] Media enrichment failed: %s", req_id, exc)
        return ""
