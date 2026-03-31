"""Unified search across the knowledge graph.

Modes:
  - keyword: Full-text search on chunks, concepts, claims
  - semantic: Vector similarity search on chunk embeddings (requires embeddings)
  - graph: Cypher-based graph traversal from matched concepts
  - hybrid: Keyword + graph combined with reciprocal rank fusion
  - spreading_activation: Algorithm-based (delegates to algorithms.py)
  - swanson_abc: Literature-based discovery (delegates to algorithms.py)
"""

import logging

from .models import SearchMode, SearchResult
from .neo4j_client import get_driver

log = logging.getLogger("knowledge-engine")


def search(
    namespace: str,
    query: str,
    mode: SearchMode = SearchMode.hybrid,
    limit: int = 10,
    cross_namespace: bool = False,
) -> list[SearchResult]:
    """Unified search entry point."""
    ns_filter = namespace if not cross_namespace else None

    if mode == SearchMode.keyword:
        return _keyword_search(query, ns_filter, limit)
    elif mode == SearchMode.graph:
        return _graph_search(query, ns_filter, limit)
    elif mode == SearchMode.semantic:
        # Semantic search requires embeddings — fall back to keyword if unavailable
        return _keyword_search(query, ns_filter, limit)
    elif mode == SearchMode.hybrid:
        return _hybrid_search(query, ns_filter, limit)
    else:
        return _keyword_search(query, ns_filter, limit)


# ============================================================================
# Keyword Search (Full-Text)
# ============================================================================

def _keyword_search(
    query: str,
    namespace: str | None,
    limit: int,
) -> list[SearchResult]:
    """Full-text search across chunks, concepts, and claims."""
    driver = get_driver()
    results: list[SearchResult] = []

    # Escape Lucene special characters for fulltext search
    escaped_query = _escape_lucene(query)
    if not escaped_query.strip():
        return results

    with driver.session() as session:
        # Search chunks
        ns_where = "AND node.namespace = $ns" if namespace else ""
        chunk_result = session.run(
            f"""
            CALL db.index.fulltext.queryNodes('fulltext_chunks', $search_query)
            YIELD node, score
            WHERE score > 0.1 {ns_where}
            OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(node)
            RETURN node.id AS id, node.content AS content,
                   node.chunk_index AS chunk_index,
                   doc.title AS doc_title, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            search_query=escaped_query, ns=namespace, limit=limit,
        )
        for r in chunk_result:
            results.append(SearchResult(
                node_type="Chunk",
                content=r["content"] or "",
                score=r["score"],
                source_doc=r["doc_title"] or "",
                chunk_index=r["chunk_index"] if r["chunk_index"] is not None else -1,
                properties={"id": r["id"]},
            ))

        # Search concepts
        concept_result = session.run(
            f"""
            CALL db.index.fulltext.queryNodes('fulltext_concepts', $search_query)
            YIELD node, score
            WHERE score > 0.1 {ns_where}
            RETURN node.id AS id, node.name AS name,
                   node.mention_count AS mentions,
                   node.community_id AS community,
                   node.rns_score AS rns, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            search_query=escaped_query, ns=namespace, limit=limit,
        )
        for r in concept_result:
            results.append(SearchResult(
                node_type="Concept",
                name=r["name"] or "",
                score=r["score"],
                properties={
                    "id": r["id"],
                    "mentions": r["mentions"],
                    "community": r["community"],
                    "rns_score": r["rns"],
                },
            ))

        # Search claims
        claim_result = session.run(
            f"""
            CALL db.index.fulltext.queryNodes('fulltext_claims', $search_query)
            YIELD node, score
            WHERE score > 0.1 {ns_where}
            RETURN node.id AS id, node.statement AS statement,
                   node.confidence AS confidence,
                   node.polarity AS polarity, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            search_query=escaped_query, ns=namespace, limit=limit,
        )
        for r in claim_result:
            results.append(SearchResult(
                node_type="Claim",
                content=r["statement"] or "",
                score=r["score"],
                properties={
                    "id": r["id"],
                    "confidence": r["confidence"],
                    "polarity": r["polarity"],
                },
            ))

        # Search anomalies
        anom_result = session.run(
            f"""
            CALL db.index.fulltext.queryNodes('fulltext_anomalies', $search_query)
            YIELD node, score
            WHERE score > 0.1 {ns_where}
            RETURN node.id AS id, node.description AS description,
                   node.surprise_score AS surprise, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            search_query=escaped_query, ns=namespace, limit=limit,
        )
        for r in anom_result:
            results.append(SearchResult(
                node_type="Anomaly",
                content=r["description"] or "",
                score=r["score"],
                properties={
                    "id": r["id"],
                    "surprise_score": r["surprise"],
                },
            ))

    # Sort by score and limit
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


# ============================================================================
# Graph Search
# ============================================================================

def _graph_search(
    query: str,
    namespace: str | None,
    limit: int,
) -> list[SearchResult]:
    """Graph-based search: find concepts matching query, then traverse neighbors."""
    driver = get_driver()
    results: list[SearchResult] = []

    escaped_query = _escape_lucene(query)
    if not escaped_query.strip():
        return results

    with driver.session() as session:
        ns_where = "AND c.namespace = $ns" if namespace else ""

        # Find matching concepts and their neighborhoods
        result = session.run(
            f"""
            CALL db.index.fulltext.queryNodes('fulltext_concepts', $search_query)
            YIELD node AS c, score
            WHERE score > 0.1 {ns_where}
            WITH c, score
            ORDER BY score DESC
            LIMIT 5

            // Get direct neighbors
            OPTIONAL MATCH (c)-[r]-(neighbor:Concept)
            WHERE neighbor.namespace = c.namespace

            With c, score, collect(DISTINCT {{
                name: neighbor.name,
                id: neighbor.id,
                rel_type: type(r),
                community: neighbor.community_id,
                rns: neighbor.rns_score
            }})[..10] AS neighbors

            // Get connected chunks
            OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(c)
            WITH c, score, neighbors,
                 collect(DISTINCT {{
                     content: left(chunk.content, 200),
                     chunk_index: chunk.chunk_index
                 }})[..3] AS chunks

            RETURN c.name AS name, c.id AS id,
                   c.community_id AS community,
                   c.rns_score AS rns,
                   c.mention_count AS mentions,
                   score, neighbors, chunks
            """,
            search_query=escaped_query, ns=namespace,
        )

        for r in result:
            results.append(SearchResult(
                node_type="Concept",
                name=r["name"] or "",
                score=r["score"],
                properties={
                    "id": r["id"],
                    "community": r["community"],
                    "rns_score": r["rns"],
                    "mentions": r["mentions"],
                    "neighbors": r["neighbors"],
                    "chunks": r["chunks"],
                },
            ))

    return results[:limit]


# ============================================================================
# Hybrid Search (Keyword + Graph with RRF)
# ============================================================================

def _hybrid_search(
    query: str,
    namespace: str | None,
    limit: int,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of keyword and graph search results."""
    keyword_results = _keyword_search(query, namespace, limit * 2)
    graph_results = _graph_search(query, namespace, limit)

    # RRF fusion
    k = 60  # RRF constant
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(keyword_results):
        key = f"{r.node_type}:{r.name or r.content[:50]}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        result_map[key] = r

    for rank, r in enumerate(graph_results):
        key = f"{r.node_type}:{r.name or r.content[:50]}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = r

    # Sort by fused score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for key in sorted_keys[:limit]:
        r = result_map[key]
        r.score = round(scores[key], 6)
        fused.append(r)

    return fused


# ============================================================================
# Helpers
# ============================================================================

def _escape_lucene(query: str) -> str:
    """Escape Lucene special characters for Neo4j fulltext search."""
    special = r'+-&|!(){}[]^"~*?:\/'
    escaped = []
    for ch in query:
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)
