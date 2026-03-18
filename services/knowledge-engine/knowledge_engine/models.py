"""Pydantic models for the Knowledge Engine API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class SearchMode(str, Enum):
    hybrid = "hybrid"
    keyword = "keyword"
    semantic = "semantic"
    graph = "graph"
    spreading_activation = "spreading_activation"
    swanson_abc = "swanson_abc"


class JobStatus(str, Enum):
    pending = "pending"
    chunking = "chunking"
    extracting = "extracting"
    resolving = "resolving"
    embedding = "embedding"
    loading = "loading"
    computing_graph = "computing_graph"
    completed = "completed"
    failed = "failed"


# --- Request Models ---

class IngestRequest(BaseModel):
    """Request to ingest a text corpus into a namespace."""
    namespace: str = Field(..., description="Conversation/context namespace")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Full text content to ingest")
    source: str = Field("", description="Source URL or identifier")
    rebuild: bool = Field(
        True,
        description=(
            "If True, clear all existing data in this namespace before "
            "ingesting. If False, append to existing namespace data."
        ),
    )


class SearchRequest(BaseModel):
    """Request to search the knowledge graph."""
    namespace: str = Field(..., description="Namespace to search within")
    query: str = Field(..., description="Search query")
    mode: SearchMode = Field(SearchMode.hybrid, description="Search mode")
    limit: int = Field(10, ge=1, le=50, description="Max results")
    cross_namespace: bool = Field(
        False, description="Search across all namespaces"
    )


class SpreadingActivationRequest(BaseModel):
    """Request for spreading activation algorithm."""
    namespace: str
    seed_concepts: list[str] = Field(
        ..., description="Concept names to seed activation from"
    )
    hops: int = Field(3, ge=1, le=6, description="Number of propagation hops")
    decay: float = Field(0.7, gt=0.0, lt=1.0, description="Decay per hop")
    threshold: float = Field(0.01, gt=0.0, description="Min activation")
    limit: int = Field(20, ge=1, le=100)


class SwansonABCRequest(BaseModel):
    """Request for Swanson ABC literature-based discovery."""
    namespace: str
    seed_concept: str = Field(
        ..., description="Starting concept A"
    )
    limit: int = Field(20, ge=1, le=100)


# --- Response Models ---

class IngestResponse(BaseModel):
    job_id: str
    namespace: str
    title: str
    status: JobStatus
    total_chars: int
    total_chunks: int = 0
    message: str = ""


class IngestJobStatus(BaseModel):
    job_id: str
    namespace: str
    status: JobStatus
    progress: str = ""
    stats: dict = Field(default_factory=dict)
    error: Optional[str] = None


class NamespaceInfo(BaseModel):
    namespace: str
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    claim_count: int = 0
    anomaly_count: int = 0


class SearchResult(BaseModel):
    node_type: str
    name: str = ""
    content: str = ""
    properties: dict = Field(default_factory=dict)
    score: float = 0.0
    source_doc: str = ""
    chunk_index: int = -1


class SearchResponse(BaseModel):
    query: str
    mode: str
    namespace: str
    results: list[SearchResult]
    total: int


class GraphStatsResponse(BaseModel):
    namespace: str
    nodes: dict = Field(default_factory=dict)
    relationships: dict = Field(default_factory=dict)
    communities: int = 0
    top_entities: list[dict] = Field(default_factory=list)
