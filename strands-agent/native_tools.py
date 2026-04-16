"""Native @tool search functions — uncensored, no MCP server needed.

These go FIRST in the researcher's tool list so the LLM naturally
prefers them over API-key-gated MCP tools.

DuckDuckGo: free, no API key, uncensored results.
Jina Reader: URL → clean markdown (free tier, no key needed for reader).
Mojeek: independent crawler, needs API key but uncensored.
"""

from __future__ import annotations

import logging
import os

import httpx
from strands import tool

log = logging.getLogger("native-tools")


@tool
def duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo. Free, uncensored, no API key needed.

    Returns top results with titles, URLs, and snippets. Use this as your
    FIRST search tool — it's uncensored and always available.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 10).
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"


@tool
def duckduckgo_news(query: str, max_results: int = 10) -> str:
    """Search DuckDuckGo News. Free, uncensored, good for recent events.

    Args:
        query: The news search query.
        max_results: Maximum number of results (default 10).
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        if not results:
            return "No news results found."
        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', '')}** ({r.get('date', '')})\n"
                f"{r.get('url', '')}\n{r.get('body', '')}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"DuckDuckGo news error: {e}"


@tool
def jina_read_url(url: str) -> str:
    """Fetch a URL and extract clean markdown text using Jina Reader.

    Free tier available (no API key needed for basic usage).
    Use this to read the full content of any web page, PDF, or document.

    Args:
        url: The URL to read.
    """
    jina_api_key = os.environ.get("JINA_API_KEY", "")
    headers = {"Accept": "text/markdown"}
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"
    try:
        resp = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text[:15000]  # cap at 15K chars
        return text if text.strip() else "Page returned empty content."
    except Exception as e:
        return f"Jina Reader error: {e}"


@tool
def mojeek_search(query: str, max_results: int = 10) -> str:
    """Search using Mojeek — an independent web crawler with its own index.

    Uncensored, independent from Google/Bing. Requires MOJEEK_API_KEY env var.

    Args:
        query: The search query.
        max_results: Maximum results (default 10).
    """
    api_key = os.environ.get("MOJEEK_API_KEY", "")
    if not api_key:
        return "Mojeek API key not configured."
    try:
        resp = httpx.get(
            "https://api.mojeek.com/search",
            params={
                "q": query,
                "fmt": "json",
                "t": max_results,
                "api_key": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("response", {}).get("results", [])
        if not results:
            return "No Mojeek results found."
        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', '')}**\n{r.get('url', '')}\n{r.get('desc', '')}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Mojeek search error: {e}"


def get_native_tools() -> list:
    """Return all native @tool functions.

    These are ordered uncensored-first so the LLM naturally prefers them.
    """
    tools = [duckduckgo_search, duckduckgo_news, jina_read_url]
    # Only include Mojeek if API key is configured
    if os.environ.get("MOJEEK_API_KEY"):
        tools.append(mojeek_search)
    return tools
