"""Shared Pydantic models for MCP server search results."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """Unified search result returned by all MCP servers."""

    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.0
    published_date: str = ""
