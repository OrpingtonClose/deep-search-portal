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

# Brave Search API (image search)
# Get a free key at https://brave.com/search/api/ (2000 queries/month)
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
BRAVE_SEARCH_ENABLED = bool(BRAVE_SEARCH_API_KEY)


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
# Brave Search API — image search
# ---------------------------------------------------------------------------
async def _brave_image_search(query: str, max_results: int) -> list[dict]:
    """Search for images via Brave Search API.

    Returns results normalised to the same shape as SearXNG image results
    (``img_src``, ``url``, ``title``, ``content``) so they can be merged
    seamlessly with :func:`_format_image_results`.
    """
    if not BRAVE_SEARCH_ENABLED:
        return []

    client = http_client()
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }
    params = {
        "q": query,
        "count": str(min(max_results * 2, 20)),  # request extra for dedup
        "safesearch": "off",
    }

    resp = await client.get(
        "https://api.search.brave.com/res/v1/images/search",
        headers=headers,
        params=params,
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    raw_results = data.get("results", [])

    normalised: list[dict] = []
    for item in raw_results:
        img_src = item.get("properties", {}).get("url", "") or item.get("thumbnail", {}).get("src", "")
        if not img_src:
            continue
        normalised.append({
            "img_src": img_src,
            "url": item.get("url", img_src),
            "title": item.get("title", "Image"),
            "content": item.get("description", ""),
            "_source": "brave",
        })
        if len(normalised) >= max_results:
            break

    return normalised


def _merge_image_results(
    searxng: list[dict], brave: list[dict], max_items: int
) -> list[dict]:
    """Merge image results from SearXNG and Brave, deduplicating by img_src.

    Brave results are preferred when both sources return the same image
    (Brave typically provides higher-quality metadata).
    """
    seen: set[str] = set()
    merged: list[dict] = []

    # Brave first (higher quality metadata)
    for item in brave:
        src = item.get("img_src", "")
        if src and src not in seen:
            seen.add(src)
            merged.append(item)

    # Then SearXNG (fills in non-duplicate results)
    for item in searxng:
        src = item.get("img_src", "")
        if src and src not in seen:
            seen.add(src)
            merged.append(item)

    return merged[:max_items]


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
    """Format SearXNG video results as a markdown section.

    For YouTube URLs, extracts the video ID and shows a thumbnail via
    ``https://img.youtube.com/vi/{id}/mqdefault.jpg``.
    Other platforms get a bold link with snippet.
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
            thumbnail = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            line = f"[![{title}]({thumbnail})]({url})"
        else:
            line = f"**[{title}]({url})**"
            if content:
                line += f"  \n{content}"

        lines.append(line)

        if len(lines) >= max_items:
            break

    if not lines:
        return ""

    return "\n\n### Related Videos\n\n" + "\n\n".join(lines) + "\n"


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
async def enrich_with_media_structured(
    user_query: str, req_id: str = "",
) -> list[dict]:
    """Run image + video searches and return structured media data.

    Returns a list of dicts, each with keys:
      - ``type``: ``"image"`` or ``"video"``
      - ``url``: source page URL
      - ``title``: human-readable title
      - ``description``: caption / snippet
      - ``img_src``: direct image URL (images only)
      - ``video_id``: YouTube video ID (videos only, may be empty)
      - ``thumbnail``: thumbnail URL (videos only)

    This structured format lets the synthesis model decide where to place
    each piece of media inline, instead of appending a block at the end.
    """
    if not MEDIA_ENRICHMENT_ENABLED:
        return []

    try:
        searxng_image_task = _searxng_media_query(
            user_query, "images", MEDIA_ENRICHMENT_MAX_IMAGES
        )
        video_task = _searxng_media_query(
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS
        )
        brave_image_task = _brave_image_search(
            user_query, MEDIA_ENRICHMENT_MAX_IMAGES
        )

        searxng_images, video_results, brave_images = await asyncio.gather(
            searxng_image_task, video_task, brave_image_task,
            return_exceptions=True,
        )

        if isinstance(searxng_images, BaseException):
            log.warning("[%s] SearXNG image search failed: %s", req_id, searxng_images)
            searxng_images = []
        if isinstance(video_results, BaseException):
            log.warning("[%s] Media enrichment video search failed: %s", req_id, video_results)
            video_results = []
        if isinstance(brave_images, BaseException):
            log.warning("[%s] Brave image search failed: %s", req_id, brave_images)
            brave_images = []

        all_images = _merge_image_results(
            searxng_images, brave_images, MEDIA_ENRICHMENT_MAX_IMAGES
        )

        media: list[dict] = []

        # Images
        seen_img: set[str] = set()
        for item in all_images:
            img_src = item.get("img_src", "")
            if not img_src or img_src in seen_img:
                continue
            seen_img.add(img_src)
            media.append({
                "type": "image",
                "url": item.get("url", img_src),
                "title": (item.get("title", "") or "Image").strip(),
                "description": (item.get("content", "") or "").strip(),
                "img_src": img_src,
            })

        # Videos
        for item in video_results:
            url = item.get("url", "")
            if not url:
                continue
            vid_id = _extract_youtube_id(url)
            thumbnail = (
                f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"
                if vid_id else ""
            )
            media.append({
                "type": "video",
                "url": url,
                "title": (item.get("title", "") or "Video").strip(),
                "description": (item.get("content", "") or "").strip(),
                "video_id": vid_id,
                "thumbnail": thumbnail,
            })

        return media

    except Exception as exc:
        log.warning("[%s] Media enrichment (structured) failed: %s", req_id, exc)
        return []


async def enrich_with_media(user_query: str, req_id: str = "") -> str:
    """Run image + video searches in parallel and return combined markdown.

    Searches SearXNG for images and videos, and optionally Brave Search
    API for images.  All sources run concurrently; image results from
    Brave and SearXNG are merged and deduplicated.

    Fail-safe: any exception returns an empty string so the main answer
    is never blocked by media enrichment failures.
    """
    if not MEDIA_ENRICHMENT_ENABLED:
        return ""

    try:
        searxng_image_task = _searxng_media_query(
            user_query, "images", MEDIA_ENRICHMENT_MAX_IMAGES
        )
        video_task = _searxng_media_query(
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS
        )
        brave_image_task = _brave_image_search(
            user_query, MEDIA_ENRICHMENT_MAX_IMAGES
        )

        searxng_images, video_results, brave_images = await asyncio.gather(
            searxng_image_task, video_task, brave_image_task,
            return_exceptions=True,
        )

        # If any search raised, treat its results as empty
        if isinstance(searxng_images, BaseException):
            log.warning(
                "[%s] SearXNG image search failed: %s",
                req_id, searxng_images,
            )
            searxng_images = []
        if isinstance(video_results, BaseException):
            log.warning(
                "[%s] Media enrichment video search failed: %s",
                req_id, video_results,
            )
            video_results = []
        if isinstance(brave_images, BaseException):
            log.warning(
                "[%s] Brave image search failed: %s",
                req_id, brave_images,
            )
            brave_images = []

        # Merge image results from both sources (Brave preferred on dupes)
        all_images = _merge_image_results(
            searxng_images, brave_images, MEDIA_ENRICHMENT_MAX_IMAGES
        )

        images_md = _format_image_results(all_images, MEDIA_ENRICHMENT_MAX_IMAGES)
        videos_md = _format_video_results(video_results, MEDIA_ENRICHMENT_MAX_VIDEOS)

        return images_md + videos_md

    except Exception as exc:
        log.warning("[%s] Media enrichment failed: %s", req_id, exc)
        return ""
