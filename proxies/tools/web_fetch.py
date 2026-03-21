"""Multi-tier web fetch: httpx, Playwright, Selenium, Bright Data, Oxylabs, Wayback.

Extracted from persistent_deep_research_proxy.py lines 1881-2297.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from shared import get_throttler

from . import _get_http_client

from .config import (
    WEBPAGE_MAX_CHARS,
    BRIGHT_DATA_API_KEY,
    BRIGHT_DATA_CUSTOMER_ID,
    BRIGHT_DATA_ZONE,
    OXYLABS_USERNAME,
    OXYLABS_PASSWORD,
    PLAYWRIGHT_AVAILABLE,
    SELENIUM_AVAILABLE,
)

log = logging.getLogger("persistent-research")

# Conditional imports for JS rendering
if PLAYWRIGHT_AVAILABLE:
    from playwright.async_api import async_playwright

if SELENIUM_AVAILABLE:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService

_CENSORSHIP_KEYWORDS = [
    "access denied", "403 forbidden", "blocked", "not available in your",
    "enable javascript", "captcha", "verify you are human", "cf-browser",
    "just a moment", "checking your browser", "ray id",
]

_ERROR_PREFIXES = (
    "Fetch error", "Non-text content", "PDF document", "No readable text",
    "Search error", "Page returned no readable",
)


async def _tool_fetch_webpage_direct(url: str, extract_info: str = "") -> str:
    """Direct fetch — original implementation."""
    # (identical to old tool_fetch_webpage, now an internal helper)
    try:
        client = _get_http_client()
        resp = await client.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/2.0)"},
        )
        if resp.status_code != 200:
            return f"Fetch error: HTTP {resp.status_code} for {url}"

        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            return f"PDF document at {url} (binary content, cannot extract text directly)"
        if ("text/html" not in content_type and "text/plain" not in content_type
                and "text/xml" not in content_type and "application/json" not in content_type):
            return f"Non-text content type: {content_type} at {url}"

        raw = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return f"No readable text content found at {url}"

        if len(text) > WEBPAGE_MAX_CHARS:
            text = text[:WEBPAGE_MAX_CHARS] + "\n[...truncated...]\n"

        result = f"Content from {url}:\n{text}"
        if extract_info:
            result = f"Instructions: {extract_info}\n\n{result}"
        return result

    except Exception as e:
        return f"Fetch error for {url}: {e}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage with enhanced scraping fallback chain."""
    return await enhanced_web_fetch(url, extract_info)


def _strip_html(raw_html: str) -> str:
    """Extract readable text from HTML, stripping scripts/styles/tags."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text


def _is_censored_response(text: str) -> bool:
    """Detect if a response looks censored/blocked rather than real content."""
    if not text or text.startswith(_ERROR_PREFIXES):
        return False
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    lower = stripped.lower()
    matches = sum(1 for kw in _CENSORSHIP_KEYWORDS if kw in lower)
    return matches >= 2


async def _fetch_via_httpx(url: str) -> str:
    """Tier 0: Fast HTTP fetch via httpx (no JS rendering)."""
    client = _get_http_client()
    resp = await client.get(
        url,
        timeout=20.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if resp.status_code in (403, 404, 410, 451):
        return f"Fetch error: HTTP {resp.status_code} for {url}"
    if resp.status_code != 200:
        return f"Fetch error: HTTP {resp.status_code}"

    content_type = resp.headers.get("content-type", "")
    if "pdf" in content_type.lower():
        return f"PDF document at {url} (binary content, cannot extract text directly)"
    if ("text/html" not in content_type and "text/plain" not in content_type
            and "text/xml" not in content_type and "application/json" not in content_type):
        return f"Non-text content type: {content_type}"

    return _strip_html(resp.text)


async def _fetch_via_playwright(url: str) -> Optional[str]:
    """Tier 1: Headless Playwright for JS-rendered pages.

    Returns None if Playwright is not available.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 720},
                )
                page = await ctx.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                body_text = await page.inner_text("body")
                if body_text and len(body_text.strip()) > 50:
                    return body_text.strip()
                return None
            finally:
                await browser.close()
    except Exception as e:
        log.debug(f"Playwright fetch failed for {url}: {e}")
        return None


async def _fetch_via_selenium(url: str) -> Optional[str]:
    """Tier 1 fallback: Headless Selenium/ChromeDriver for JS-rendered pages.

    Returns None if Selenium is not available.
    """
    if not SELENIUM_AVAILABLE:
        return None
    try:
        loop = asyncio.get_running_loop()

        def _sync_fetch():
            opts = ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-gpu")
            opts.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            driver = webdriver.Chrome(options=opts)
            try:
                driver.set_page_load_timeout(30)
                driver.get(url)
                body = driver.find_element("tag name", "body")
                return body.text.strip() if body else None
            finally:
                driver.quit()

        result = await loop.run_in_executor(None, _sync_fetch)
        if result and len(result) > 50:
            return result
        return None
    except Exception as e:
        log.debug(f"Selenium fetch failed for {url}: {e}")
        return None


async def _fetch_via_bright_data(url: str) -> Optional[str]:
    """Tier 2: Bright Data Web Unlocker for geo-blocked/protected pages.

    Returns None if Bright Data is not configured or the request fails.
    """
    if not BRIGHT_DATA_API_KEY:
        return None
    try:
        async with get_throttler("bright_data").throttle():
            proxy_url = (
                f"https://brd-customer-{BRIGHT_DATA_CUSTOMER_ID}-zone-{BRIGHT_DATA_ZONE}"
                f":{BRIGHT_DATA_API_KEY}@brd.superproxy.io:33335"
            )
            async with httpx.AsyncClient(
                proxy=proxy_url,
                verify=False,
                timeout=httpx.Timeout(45.0, connect=15.0),
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
                if resp.status_code != 200:
                    return None
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return None
                text = _strip_html(resp.text)
                return text if text and len(text.strip()) > 50 else None
    except Exception as e:
        log.debug(f"Bright Data fetch failed for {url}: {e}")
        return None


async def _fetch_via_oxylabs(url: str) -> Optional[str]:
    """Tier 2 fallback: Oxylabs Web Scraper for protected pages.

    Returns None if Oxylabs is not configured or the request fails.
    """
    if not OXYLABS_USERNAME or not OXYLABS_PASSWORD:
        return None
    try:
        async with get_throttler("oxylabs").throttle():
            async with httpx.AsyncClient(
                proxy=f"https://{OXYLABS_USERNAME}:{OXYLABS_PASSWORD}@unblock.oxylabs.io:60000",
                verify=False,
                timeout=httpx.Timeout(45.0, connect=15.0),
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
                if resp.status_code != 200:
                    return None
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return None
                text = _strip_html(resp.text)
                return text if text and len(text.strip()) > 50 else None
    except Exception as e:
        log.debug(f"Oxylabs fetch failed for {url}: {e}")
        return None


async def _fetch_via_wayback_cdx(url: str) -> Optional[str]:
    """Archive cascade: Wayback Machine CDX lookup for dead/blocked URLs.

    Checks the Wayback Machine for the most recent snapshot and fetches it.
    Returns None if no archive is found.
    """
    try:
        async with get_throttler("wayback").throttle():
            client = _get_http_client()
            # CDX API returns the most recent successful capture
            cdx_resp = await client.get(
                "https://web.archive.org/cdx/search/cdx",
                params={
                    "url": url,
                    "output": "json",
                    "limit": 1,
                    "fl": "timestamp,statuscode",
                    "filter": "statuscode:200",
                    "sort": "reverse",
                },
                timeout=15.0,
            )
            if cdx_resp.status_code != 200:
                return None

            rows = cdx_resp.json()
            # First row is header, second is data
            if len(rows) < 2:
                return None

            timestamp = rows[1][0]
            archive_url = f"https://web.archive.org/web/{timestamp}id_/{url}"

            archive_resp = await client.get(
                archive_url,
                timeout=20.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            if archive_resp.status_code != 200:
                return None

            text = _strip_html(archive_resp.text)
            if text and len(text.strip()) > 50:
                return f"[ARCHIVED — Wayback Machine snapshot from {timestamp}]\n\n{text}"
            return None
    except Exception as e:
        log.debug(f"Wayback CDX fetch failed for {url}: {e}")
        return None


async def enhanced_web_fetch(url: str, extract_info: str = "") -> str:
    """Multi-tier web fetch with JS rendering, proxy fallback, and archive cascade.

    Fallback chain:
      Tier 0: httpx fast path (plain HTML)
      Tier 1: Playwright headless (JS rendering) or Selenium fallback
      Tier 2: Bright Data Web Unlocker / Oxylabs (geo-blocks, anti-bot)
      Tier 3: Wayback CDX archive (dead URLs, 404/403)

    Censorship detection: if any tier returns content that looks
    censored/blocked, the next tier is tried.  If all tiers fail,
    we return whatever we got with a censorship warning.
    """
    # Tier -1: PDF extraction (if URL looks like a PDF)
    if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
        from .tool_executor import _extract_pdf_text  # lazy to avoid circular import
        pdf_text = await _extract_pdf_text(url)
        if pdf_text:
            if len(pdf_text) > WEBPAGE_MAX_CHARS:
                pdf_text = pdf_text[:WEBPAGE_MAX_CHARS] + "\n\n[... PDF content truncated ...]"
            result = f"**PDF content from {url}:**\n\n{pdf_text}"
            if extract_info:
                result = f"**Looking for: {extract_info}**\n\n{result}"
            return result

    # Tier 0: httpx fast path
    try:
        direct = await _fetch_via_httpx(url)
    except Exception as e:
        direct = f"Fetch error: {str(e)}"

    is_error = direct.startswith(_ERROR_PREFIXES)
    # 404/410 = content truly gone (skip JS rendering AND proxies)
    is_url_gone = any(
        direct.startswith(f"Fetch error: HTTP {c}") for c in (404, 410)
    )
    # 403/451 = access blocked (skip JS rendering but TRY proxies)
    is_access_blocked = any(
        direct.startswith(f"Fetch error: HTTP {c}") for c in (403, 451)
    )

    # If fast path got good, non-empty content, use it
    if not is_error and not _is_censored_response(direct) and len(direct.strip()) > 50:
        text = direct
    else:
        text = None

        # Tier 1: JS rendering (Playwright → Selenium fallback)
        # Skip for server-side blocks (403/451) and dead URLs — JS rendering won't help
        if text is None and not is_url_gone and not is_access_blocked:
            rendered = await _fetch_via_playwright(url)
            if rendered is None:
                rendered = await _fetch_via_selenium(url)
            if rendered and not _is_censored_response(rendered):
                text = rendered

        # Tier 2: Commercial proxies (Bright Data → Oxylabs)
        # Skip only for truly dead URLs — proxies CAN bypass 403/451
        if text is None and not is_url_gone:
            proxied = await _fetch_via_bright_data(url)
            if proxied is None:
                proxied = await _fetch_via_oxylabs(url)
            if proxied and not _is_censored_response(proxied):
                text = proxied

        # Tier 3: Archive cascade for dead URLs or total failure
        if text is None:
            archived = await _fetch_via_wayback_cdx(url)
            if archived:
                text = archived

        # Final fallback: return whatever we got
        if text is None:
            text = direct
            # Append censorship warning only for actual page content,
            # not for error messages from _fetch_via_httpx
            if _is_censored_response(direct):
                text += (
                    "\n\n[WARNING: This result may be incomplete or blocked. "
                    "The page may require JavaScript rendering, authentication, "
                    "or be geo-restricted. Treat 'no results found' with skepticism "
                    "and try alternative sources.]"
                )

    # Truncate
    if len(text) > WEBPAGE_MAX_CHARS:
        text = text[:WEBPAGE_MAX_CHARS] + "\n\n[... content truncated ...]"

    if not text.strip():
        return "Page returned no readable text content."

    # If all tiers failed and we're returning the original error from httpx,
    # return it bare (without "Content from" wrapper) to preserve the tool
    # output contract that the LLM relies on to detect fetch failures.
    if is_error and text is direct:
        return text

    result = f"**Content from {url}:**\n\n{text}"
    if extract_info:
        result = f"**Looking for: {extract_info}**\n\n{result}"
    return result

