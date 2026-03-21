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


# --- Research Condition Models ---

class ResearchConditionInput(BaseModel):
    """A single research condition (atomic finding) to store."""
    fact: str = Field(..., description="The research finding text")
    source_url: str = Field("", description="Source URL")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    trust_score: float = Field(0.5, ge=0.0, le=1.0)
    angle: str = Field("", description="Research angle that produced this")
    domain: str = Field("", description="Source domain")
    is_serendipitous: bool = Field(False)
    serendipity_score: float = Field(0.0, ge=0.0, le=1.0)
    publication_date: str = Field("", description="Publication date of source")
    author: str = Field("", description="Author of source")
    content_type: str = Field("", description="Content type (article, paper, etc.)")
    source_type: str = Field("", description="Source type (web, academic, etc.)")


class StoreConditionsRequest(BaseModel):
    """Request to store research conditions from a research session."""
    namespace: str = Field("research", description="Namespace for research data")
    session_id: str = Field(..., description="Research session ID")
    query: str = Field(..., description="Original research query")
    conditions: list[ResearchConditionInput] = Field(
        ..., description="List of atomic conditions to store"
    )


class EntityInput(BaseModel):
    """An entity extracted from research."""
    name: str
    type: str = "concept"


class RelationshipInput(BaseModel):
    """A relationship between two entities."""
    entity1: str
    entity2: str
    type: str = "RELATED_TO"
    is_bridge: bool = False


class StoreEntitiesRequest(BaseModel):
    """Request to store entities and relationships from research."""
    namespace: str = Field("research", description="Namespace for research data")
    session_id: str = Field(..., description="Research session ID")
    entities: list[EntityInput] = Field(default_factory=list)
    relationships: list[RelationshipInput] = Field(default_factory=list)


class SearchConditionsRequest(BaseModel):
    """Request to search prior research conditions."""
    namespace: str = Field("research")
    query: str = Field(..., description="Search query")
    limit: int = Field(20, ge=1, le=100)


class GraphNeighborsRequest(BaseModel):
    """Request to find conditions related to given entities via graph."""
    namespace: str = Field("research")
    entity_names: list[str] = Field(..., description="Entity names to start from")
    max_hops: int = Field(2, ge=1, le=4)
    limit: int = Field(20, ge=1, le=100)


class ResearchConditionResult(BaseModel):
    """A research condition returned from search."""
    fact: str
    source_url: str = ""
    confidence: float = 0.0
    angle: str = ""
    is_serendipitous: bool = False
    query: str = ""
    created_at: str = ""
    trust_score: float = 0.0
    serendipity_score: float = 0.0


class ResearchStatsResponse(BaseModel):
    """Statistics about the persistent research knowledge base."""
    total_conditions: int = 0
    total_sessions: int = 0
    total_queries: int = 0
    total_entities: int = 0
    total_relationships: int = 0
