#!/usr/bin/env python3
"""
Media Enrichment — append image and video results to any answer.

Searches SearXNG for images + videos, and optionally TranscriptAPI for
YouTube search + transcript extraction.  All sources run in parallel;
results are merged, deduplicated by video ID, and formatted as
structured metadata for the synthesis model.

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
MEDIA_ENRICHMENT_MAX_IMAGES = int(os.getenv("MEDIA_ENRICHMENT_MAX_IMAGES", "10"))
MEDIA_ENRICHMENT_MAX_VIDEOS = int(os.getenv("MEDIA_ENRICHMENT_MAX_VIDEOS", "6"))

# Brave Search API (image search)
# Get a free key at https://brave.com/search/api/ (2000 queries/month)
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
BRAVE_SEARCH_ENABLED = bool(BRAVE_SEARCH_API_KEY)

# TranscriptAPI (YouTube search + transcript extraction)
# Get a key at https://transcriptapi.com/onboarding
TRANSCRIPTAPI_KEY = os.getenv("TRANSCRIPTAPI_KEY", "")
TRANSCRIPTAPI_ENABLED = bool(TRANSCRIPTAPI_KEY)
TRANSCRIPTAPI_BASE = "https://transcriptapi.com/api/v2"


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
    """Format video results as a markdown section.

    For YouTube URLs, extracts the video ID and shows a clickable thumbnail.
    TranscriptAPI results include spoken-content snippets with timestamps;
    SearXNG results have a generic description.
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
        source_tag = item.get("_source", "")

        # Try to extract YouTube video ID
        video_id = _extract_youtube_id(url)
        if video_id:
            thumbnail = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            line = (
                f"[![{title}]({thumbnail})]({url})\n"
                f"  **URL:** {url}"
            )
            if content:
                # TranscriptAPI results have transcript snippets with timestamps;
                # SearXNG results have a generic description.
                label = "Spoken content" if source_tag == "transcriptapi" else "Description"
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
# TranscriptAPI — YouTube search + transcript extraction
# ---------------------------------------------------------------------------
async def _transcriptapi_search(query: str, max_results: int) -> list[dict]:
    """Search YouTube via TranscriptAPI and fetch transcript snippets.

    Two-step process:
    1. Search YouTube for relevant videos via /youtube/search
    2. For top results with captions, fetch a transcript snippet
       via /youtube/transcript to get spoken-content data

    Returns results normalised to the same shape as SearXNG video results
    so they can be merged seamlessly.
    """
    if not TRANSCRIPTAPI_ENABLED:
        return []

    client = http_client()
    headers = {"Authorization": f"Bearer {TRANSCRIPTAPI_KEY}"}

    # Step 1: search YouTube
    resp = await client.get(
        f"{TRANSCRIPTAPI_BASE}/youtube/search",
        headers=headers,
        params={"q": query, "type": "video", "limit": str(max_results * 2)},
        timeout=12.0,
    )
    resp.raise_for_status()
    search_data = resp.json()
    results = search_data.get("results", [])
    if not results:
        return []

    # Step 2: for videos with captions, fetch transcript snippets in parallel
    normalised: list[dict] = []
    transcript_tasks = []
    captioned_results = []
    for item in results:
        vid = item.get("videoId", "")
        if not vid:
            continue
        title = item.get("title", "Video")
        has_captions = item.get("hasCaptions", False)
        normalised_item = {
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title,
            "content": "",
            "_source": "transcriptapi",
        }
        if has_captions and len(transcript_tasks) < max_results:
            captioned_results.append((len(normalised), normalised_item))
            transcript_tasks.append(
                _fetch_transcript_snippet(client, headers, vid)
            )
        normalised.append(normalised_item)
        if len(normalised) >= max_results * 2:
            break

    # Fetch transcripts concurrently
    if transcript_tasks:
        snippets = await asyncio.gather(*transcript_tasks, return_exceptions=True)
        for (idx, _item), snippet in zip(captioned_results, snippets):
            if isinstance(snippet, str) and snippet:
                normalised[idx]["content"] = snippet

    # Return only items up to max_results, preferring those with transcripts
    with_transcript = [r for r in normalised if r.get("content")]
    without_transcript = [r for r in normalised if not r.get("content")]
    final = (with_transcript + without_transcript)[:max_results]
    return final


async def _fetch_transcript_snippet(
    client, headers: dict, video_id: str, max_chars: int = 300
) -> str:
    """Fetch transcript for a video and return a short spoken-content snippet."""
    try:
        resp = await client.get(
            f"{TRANSCRIPTAPI_BASE}/youtube/transcript",
            headers=headers,
            params={
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
                "include_timestamp": "true",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        segments = data.get("transcript", [])
        if not segments:
            return ""
        # Build a snippet from the first few segments
        snippets: list[str] = []
        total_chars = 0
        for seg in segments:
            text = seg.get("text", "").strip()
            ts = seg.get("start", 0)
            if text:
                entry = f"[{_fmt_ts(ts)}] {text}"
                snippets.append(entry)
                total_chars += len(entry)
                if total_chars >= max_chars:
                    break
        return " | ".join(snippets)
    except Exception:
        return ""


def _fmt_ts(seconds) -> str:
    """Format seconds as ``MM:SS`` or ``H:MM:SS``."""
    try:
        s = int(float(seconds))
    except (ValueError, TypeError):
        return "0:00"
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Merge + deduplicate video results from multiple sources
# ---------------------------------------------------------------------------
def _merge_video_results(
    searxng: list[dict], transcript_results: list[dict]
) -> list[dict]:
    """Merge video results, preferring TranscriptAPI (richer transcript data).

    Deduplicates by YouTube video ID.  When the same video appears in
    both sources, the TranscriptAPI entry wins because it carries
    spoken-content snippets that help the synthesis model judge relevance.
    """
    seen_ids: set[str] = set()
    merged: list[dict] = []

    # TranscriptAPI first (higher quality — has transcript snippets)
    for item in transcript_results:
        vid = _extract_youtube_id(item.get("url", ""))
        key = vid or item.get("url", "")
        if key and key not in seen_ids:
            seen_ids.add(key)
            merged.append(item)

    # Then SearXNG (fills in remaining slots)
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
async def enrich_with_media_structured(
    user_query: str, req_id: str = "",
) -> list[dict]:
    """Run image + video + TranscriptAPI searches and return structured media data.

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
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS * 2
        )
        brave_image_task = _brave_image_search(
            user_query, MEDIA_ENRICHMENT_MAX_IMAGES
        )
        transcript_task = _transcriptapi_search(
            user_query, MEDIA_ENRICHMENT_MAX_VIDEOS
        )

        searxng_images, video_results, brave_images, transcript_results = (
            await asyncio.gather(
                searxng_image_task, video_task, brave_image_task,
                transcript_task, return_exceptions=True,
            )
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
        if isinstance(transcript_results, BaseException):
            log.warning("[%s] TranscriptAPI search failed: %s", req_id, transcript_results)
            transcript_results = []

        all_images = _merge_image_results(
            searxng_images, brave_images, MEDIA_ENRICHMENT_MAX_IMAGES
        )
        # Merge video sources (TranscriptAPI wins on duplicates)
        all_videos = _merge_video_results(video_results, transcript_results)

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

        # Videos (merged from SearXNG + TranscriptAPI, capped to configured max)
        video_count = 0
        for item in all_videos:
            if video_count >= MEDIA_ENRICHMENT_MAX_VIDEOS:
                break
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
            video_count += 1

        return media

    except Exception as exc:
        log.warning("[%s] Media enrichment (structured) failed: %s", req_id, exc)
        return []


async def enrich_with_media(user_query: str, req_id: str = "") -> str:
    """Run image + video + TranscriptAPI searches in parallel and return combined markdown.

    Searches SearXNG for images and videos, Brave Search API for images,
    and TranscriptAPI for YouTube spoken-content.  All sources run
    concurrently; results are merged and deduplicated.

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
            user_query, "videos", MEDIA_ENRICHMENT_MAX_VIDEOS * 2  # extra for dedup
        )
        brave_image_task = _brave_image_search(
            user_query, MEDIA_ENRICHMENT_MAX_IMAGES
        )
        transcript_task = _transcriptapi_search(
            user_query, MEDIA_ENRICHMENT_MAX_VIDEOS
        )

        image_results, video_results, brave_images, transcript_results = (
            await asyncio.gather(
                searxng_image_task, video_task, brave_image_task,
                transcript_task, return_exceptions=True,
            )
        )

        # If any search raised, treat its results as empty
        if isinstance(image_results, BaseException):
            log.warning(
                "[%s] SearXNG image search failed: %s",
                req_id, image_results,
            )
            image_results = []
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
        if isinstance(transcript_results, BaseException):
            log.warning(
                "[%s] TranscriptAPI search failed: %s",
                req_id, transcript_results,
            )
            transcript_results = []

        # Merge image results from both sources (Brave preferred on dupes)
        all_images = _merge_image_results(
            image_results, brave_images, MEDIA_ENRICHMENT_MAX_IMAGES
        )
        # Merge video sources (TranscriptAPI wins on duplicates)
        all_videos = _merge_video_results(video_results, transcript_results)

        images_md = _format_image_results(all_images, MEDIA_ENRICHMENT_MAX_IMAGES)
        videos_md = _format_video_results(all_videos, MEDIA_ENRICHMENT_MAX_VIDEOS)

        return images_md + videos_md

    except Exception as exc:
        log.warning("[%s] Media enrichment failed: %s", req_id, exc)
        return ""
