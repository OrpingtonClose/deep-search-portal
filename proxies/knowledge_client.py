"""Lightweight client for the Knowledge Engine microservice.

Used by all proxies to interact with the knowledge graph.
All calls are async and use httpx.
"""

import logging
import os

import httpx

log = logging.getLogger(__name__)

KNOWLEDGE_ENGINE_URL = os.getenv("KNOWLEDGE_ENGINE_URL", "http://localhost:9400")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=KNOWLEDGE_ENGINE_URL,
            timeout=300.0,  # Long timeout for large corpus ingestion
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

async def ingest(
    namespace: str,
    title: str,
    text: str,
    source: str = "",
    rebuild: bool = True,
) -> dict:
    """Submit a corpus for ingestion into the knowledge graph."""
    client = _get_client()
    resp = await client.post(
        "/v1/ingest",
        json={
            "namespace": namespace,
            "title": title,
            "text": text,
            "source": source,
            "rebuild": rebuild,
        },
    )
    resp.raise_for_status()
    return resp.json()


async def ingest_status(job_id: str) -> dict:
    """Check status of an ingest job."""
    client = _get_client()
    resp = await client.get(f"/v1/ingest/{job_id}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def search(
    namespace: str,
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    cross_namespace: bool = False,
) -> dict:
    """Search the knowledge graph."""
    client = _get_client()
    resp = await client.post(
        "/v1/search",
        json={
            "namespace": namespace,
            "query": query,
            "mode": mode,
            "limit": limit,
            "cross_namespace": cross_namespace,
        },
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

async def spreading_activation(
    namespace: str,
    seed_concepts: list[str],
    hops: int = 3,
    decay: float = 0.7,
    limit: int = 20,
) -> dict:
    """Run spreading activation from seed concepts."""
    client = _get_client()
    resp = await client.post(
        "/v1/algorithms/spreading-activation",
        json={
            "namespace": namespace,
            "seed_concepts": seed_concepts,
            "hops": hops,
            "decay": decay,
            "limit": limit,
        },
    )
    resp.raise_for_status()
    return resp.json()


async def swanson_abc(
    namespace: str,
    seed_concept: str,
    limit: int = 20,
) -> dict:
    """Run Swanson ABC literature-based discovery."""
    client = _get_client()
    resp = await client.post(
        "/v1/algorithms/swanson-abc",
        json={
            "namespace": namespace,
            "seed_concept": seed_concept,
            "limit": limit,
        },
    )
    resp.raise_for_status()
    return resp.json()


async def information_gaps(namespace: str, limit: int = 15) -> dict:
    """Find under-connected concepts."""
    client = _get_client()
    resp = await client.get(
        f"/v1/algorithms/information-gaps/{namespace}",
        params={"limit": limit},
    )
    resp.raise_for_status()
    return resp.json()


async def serendipity_beam(
    namespace: str,
    seed_concept: str,
    beam_width: int = 5,
    depth: int = 4,
    limit: int = 15,
) -> dict:
    """RNS-guided beam search for serendipitous connections."""
    client = _get_client()
    resp = await client.post(
        "/v1/algorithms/serendipity-beam",
        params={
            "namespace": namespace,
            "seed_concept": seed_concept,
            "beam_width": beam_width,
            "depth": depth,
            "limit": limit,
        },
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Namespace Management
# ---------------------------------------------------------------------------

async def list_namespaces() -> list[dict]:
    """List all namespaces with stats."""
    client = _get_client()
    resp = await client.get("/v1/namespaces")
    resp.raise_for_status()
    return resp.json()


async def delete_namespace(namespace: str) -> dict:
    """Delete a namespace and all its data."""
    client = _get_client()
    resp = await client.delete(f"/v1/namespaces/{namespace}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Graph Stats & Neighborhood
# ---------------------------------------------------------------------------

async def graph_stats(namespace: str) -> dict:
    """Get graph statistics for a namespace."""
    client = _get_client()
    resp = await client.get(f"/v1/graph/stats/{namespace}")
    resp.raise_for_status()
    return resp.json()


async def concept_neighborhood(
    namespace: str,
    concept_name: str,
    depth: int = 2,
    limit: int = 30,
) -> dict:
    """Get the neighborhood of a concept."""
    client = _get_client()
    resp = await client.get(
        f"/v1/graph/neighborhood/{namespace}/{concept_name}",
        params={"depth": depth, "limit": limit},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def health() -> dict:
    """Check knowledge engine health."""
    client = _get_client()
    try:
        resp = await client.get("/health")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
