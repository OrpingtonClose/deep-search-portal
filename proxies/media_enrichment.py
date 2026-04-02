#!/usr/bin/env python3
"""
Media Enrichment — append image and video results to any answer.

Calls SearXNG directly for images and videos in parallel, then formats
results as markdown to append to the tier chooser response.

Fail-safe: any exception returns an empty string so the answer is never blocked.
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
        if video_id:
            thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            line = (
                f"- **Title:** {title}\n"
                f"  **Video ID:** {video_id}\n"
                f"  **Thumbnail:** {thumbnail}\n"
                f"  **URL:** {url}"
            )
            if content:
                line += f"\n  **Description:** {content}"
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
# Main entry point
# ---------------------------------------------------------------------------
async def enrich_with_media(user_query: str, req_id: str = "") -> str:
    """Run image + video searches in parallel and return combined markdown.

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
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS
        )

        image_results, video_results = await asyncio.gather(
            image_task, video_task, return_exceptions=True
        )

        # If either search raised, treat its results as empty
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

        images_md = _format_image_results(image_results, MEDIA_ENRICHMENT_MAX_IMAGES)
        videos_md = _format_video_results(video_results, MEDIA_ENRICHMENT_MAX_VIDEOS)

        return images_md + videos_md

    except Exception as exc:
        log.warning("[%s] Media enrichment failed: %s", req_id, exc)
        return ""
