"""Knowledge Engine — Neo4j-centric knowledge corpus ETL microservice.

Run with: uvicorn knowledge_engine.main:app --host 0.0.0.0 --port 9400
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from .algorithms import (
    concept_neighborhood,
    information_gaps,
    serendipity_beam_search,
    spreading_activation,
    swanson_abc,
)
from .config import LISTEN_PORT, setup_logging
from .models import (
    GraphNeighborsRequest,
    GraphStatsResponse,
    IngestJobStatus,
    IngestRequest,
    IngestResponse,
    JobStatus,
    NamespaceInfo,
    ResearchConditionResult,
    ResearchStatsResponse,
    SearchConditionsRequest,
    SearchRequest,
    SearchResponse,
    SpreadingActivationRequest,
    StoreConditionsRequest,
    StoreEntitiesRequest,
    SwansonABCRequest,
)
from .neo4j_client import close_driver, get_driver, get_session, init_schema, clear_namespace, run_query
from .ontology import batch_create_research_conditions, batch_create_research_entities
from .pipeline import get_job, list_jobs, run_ingest_pipeline
from .search import search as do_search

log = setup_logging()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop Neo4j driver and initialise schema."""
    log.info("Knowledge Engine starting up...")
    try:
        get_driver()
        init_schema()
        log.info("Neo4j schema ready")
    except Exception as e:
        log.error(f"Neo4j connection failed: {e}")
        log.warning("Service will start but Neo4j operations will fail until connection is available")
    yield
    close_driver()
    log.info("Knowledge Engine shut down")


app = FastAPI(
    title="Knowledge Engine",
    description="Neo4j-centric knowledge corpus ETL and graph query service",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        get_driver().verify_connectivity()
        return {"status": "healthy", "neo4j": "connected"}
    except Exception as e:
        return {"status": "degraded", "neo4j": str(e)}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a text corpus into the knowledge graph.

    When rebuild=True (default), clears the namespace first and runs
    the full ETL pipeline: chunk → multi-pass extract → entity resolution → load.
    """
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    total_chars = len(req.text)

    background_tasks.add_task(
        run_ingest_pipeline,
        job_id=job_id,
        namespace=req.namespace,
        title=req.title,
        text=req.text,
        source=req.source,
        rebuild=req.rebuild,
    )

    return IngestResponse(
        job_id=job_id,
        namespace=req.namespace,
        title=req.title,
        status=JobStatus.pending,
        total_chars=total_chars,
        message="Ingest job started. Poll /v1/ingest/{job_id} for status.",
    )


@app.get("/v1/ingest/{job_id}", response_model=IngestJobStatus)
async def ingest_status(job_id: str):
    """Check the status of an ingest job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@app.get("/v1/jobs")
async def list_all_jobs(namespace: str | None = None):
    """List all ingest jobs, optionally filtered by namespace."""
    return list_jobs(namespace)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/v1/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """Unified search across the knowledge graph."""
    results = do_search(
        namespace=req.namespace,
        query=req.query,
        mode=req.mode,
        limit=req.limit,
        cross_namespace=req.cross_namespace,
    )
    return SearchResponse(
        query=req.query,
        mode=req.mode.value,
        namespace=req.namespace,
        results=results,
        total=len(results),
    )


# ---------------------------------------------------------------------------
# Graph Algorithms
# ---------------------------------------------------------------------------

@app.post("/v1/algorithms/spreading-activation")
async def algo_spreading_activation(req: SpreadingActivationRequest):
    """Run spreading activation from seed concepts."""
    results = spreading_activation(
        namespace=req.namespace,
        seed_concepts=req.seed_concepts,
        hops=req.hops,
        decay=req.decay,
        threshold=req.threshold,
        limit=req.limit,
    )
    return {"algorithm": "spreading_activation", "results": results}


@app.post("/v1/algorithms/swanson-abc")
async def algo_swanson_abc(req: SwansonABCRequest):
    """Run Swanson ABC literature-based discovery."""
    results = swanson_abc(
        namespace=req.namespace,
        seed_concept=req.seed_concept,
        limit=req.limit,
    )
    return {"algorithm": "swanson_abc", "results": results}


@app.get("/v1/algorithms/information-gaps/{namespace}")
async def algo_information_gaps(namespace: str, limit: int = 15):
    """Find under-connected concepts (knowledge gaps)."""
    results = information_gaps(namespace=namespace, limit=limit)
    return {"algorithm": "information_gaps", "results": results}


@app.post("/v1/algorithms/serendipity-beam")
async def algo_serendipity_beam(
    namespace: str,
    seed_concept: str,
    beam_width: int = 5,
    depth: int = 4,
    limit: int = 15,
):
    """RNS-guided beam search for serendipitous connections."""
    results = serendipity_beam_search(
        namespace=namespace,
        seed_concept=seed_concept,
        beam_width=beam_width,
        depth=depth,
        limit=limit,
    )
    return {"algorithm": "serendipity_beam_search", "results": results}


@app.get("/v1/graph/neighborhood/{namespace}/{concept_name}")
async def graph_neighborhood(namespace: str, concept_name: str, depth: int = 2, limit: int = 30):
    """Get the neighborhood of a concept."""
    result = concept_neighborhood(
        namespace=namespace,
        concept_name=concept_name,
        depth=depth,
        limit=limit,
    )
    return result


# ---------------------------------------------------------------------------
# Namespace Management
# ---------------------------------------------------------------------------

@app.get("/v1/namespaces")
async def list_namespaces():
    """List all namespaces with statistics."""
    records = run_query(
        """
        MATCH (d:Document)
        WITH d.namespace AS ns, count(d) AS doc_count
        OPTIONAL MATCH (c:Chunk {namespace: ns})
        WITH ns, doc_count, count(c) AS chunk_count
        OPTIONAL MATCH (con:Concept {namespace: ns})
        WITH ns, doc_count, chunk_count, count(con) AS concept_count
        OPTIONAL MATCH (cl:Claim {namespace: ns})
        WITH ns, doc_count, chunk_count, concept_count, count(cl) AS claim_count
        OPTIONAL MATCH (a:Anomaly {namespace: ns})
        RETURN ns AS namespace, doc_count, chunk_count, concept_count,
               claim_count, count(a) AS anomaly_count
        ORDER BY ns
        """
    )
    return [
        NamespaceInfo(
            namespace=r["namespace"],
            document_count=r["doc_count"],
            chunk_count=r["chunk_count"],
            entity_count=r["concept_count"],
            claim_count=r["claim_count"],
            anomaly_count=r["anomaly_count"],
        )
        for r in records
    ]


@app.delete("/v1/namespaces/{namespace}")
async def delete_namespace(namespace: str):
    """Delete a namespace and all its data."""
    cleared = clear_namespace(namespace)
    return {"namespace": namespace, "deleted": cleared}


# ---------------------------------------------------------------------------
# Graph Stats
# ---------------------------------------------------------------------------

@app.get("/v1/graph/stats/{namespace}", response_model=GraphStatsResponse)
async def graph_stats(namespace: str):
    """Get statistics about the knowledge graph in a namespace."""
    # Node counts by label
    node_counts = {}
    for label in ["Document", "Chunk", "Concept", "Claim", "Hypothesis", "Anomaly", "Evidence", "Method"]:
        records = run_query(
            f"MATCH (n:{label} {{namespace: $ns}}) RETURN count(n) AS cnt",
            {"ns": namespace},
        )
        node_counts[label] = records[0]["cnt"] if records else 0

    # Relationship counts
    rel_records = run_query(
        """
        MATCH (a {namespace: $ns})-[r]->(b {namespace: $ns})
        RETURN type(r) AS rel_type, count(r) AS cnt
        ORDER BY cnt DESC
        """,
        {"ns": namespace},
    )
    rel_counts = {r["rel_type"]: r["cnt"] for r in rel_records}

    # Community count
    comm_records = run_query(
        """
        MATCH (c:Concept {namespace: $ns})
        WHERE c.community_id >= 0
        RETURN count(DISTINCT c.community_id) AS communities
        """,
        {"ns": namespace},
    )
    communities = comm_records[0]["communities"] if comm_records else 0

    # Top entities by RNS score
    top_records = run_query(
        """
        MATCH (c:Concept {namespace: $ns})
        RETURN c.name AS name, c.rns_score AS rns,
               c.mention_count AS mentions,
               c.community_id AS community,
               c.betweenness_centrality AS betweenness
        ORDER BY c.rns_score DESC
        LIMIT 10
        """,
        {"ns": namespace},
    )
    top_entities = [dict(r) for r in top_records]

    return GraphStatsResponse(
        namespace=namespace,
        nodes=node_counts,
        relationships=rel_counts,
        communities=communities,
        top_entities=top_entities,
    )


# ---------------------------------------------------------------------------
# Research Conditions (persistent research memory)
# ---------------------------------------------------------------------------

@app.post("/v1/research/conditions")
async def store_research_conditions(req: StoreConditionsRequest):
    """Store atomic conditions from a research session into Neo4j."""
    conditions_dicts = [c.model_dump() for c in req.conditions]
    with get_session() as session:
        count = session.execute_write(
            batch_create_research_conditions,
            namespace=req.namespace,
            session_id=req.session_id,
            query=req.query,
            conditions=conditions_dicts,
        )
    return {"stored": count, "session_id": req.session_id}


@app.post("/v1/research/entities")
async def store_research_entities(req: StoreEntitiesRequest):
    """Store entities and relationships from a research session."""
    entities_dicts = [e.model_dump() for e in req.entities]
    rels_dicts = [r.model_dump() for r in req.relationships]
    with get_session() as session:
        ent_count, rel_count = session.execute_write(
            batch_create_research_entities,
            namespace=req.namespace,
            session_id=req.session_id,
            entities=entities_dicts,
            relationships=rels_dicts,
        )
    return {"entities_stored": ent_count, "relationships_stored": rel_count}


@app.post("/v1/research/conditions/search", response_model=list[ResearchConditionResult])
async def search_research_conditions(req: SearchConditionsRequest):
    """Full-text search over prior research conditions."""
    tokens = [t.strip() for t in req.query.split() if len(t.strip()) > 2]
    if not tokens:
        return []
    # Build Lucene query for the fulltext index
    lucene_query = " OR ".join(tokens[:10])
    records = run_query(
        """
        CALL db.index.fulltext.queryNodes('fulltext_research_conditions', $q)
        YIELD node, score
        WHERE node.namespace = $ns
        RETURN node.fact AS fact, node.source_url AS source_url,
               node.confidence AS confidence, node.angle AS angle,
               node.is_serendipitous AS is_serendipitous,
               node.query AS query, node.created_at AS created_at,
               node.trust_score AS trust_score,
               node.serendipity_score AS serendipity_score,
               score
        ORDER BY score DESC
        LIMIT $limit
        """,
        {"q": lucene_query, "ns": req.namespace, "limit": req.limit},
    )
    return [
        ResearchConditionResult(
            fact=r["fact"],
            source_url=r.get("source_url", ""),
            confidence=r.get("confidence", 0.0),
            angle=r.get("angle", ""),
            is_serendipitous=bool(r.get("is_serendipitous", False)),
            query=r.get("query", ""),
            created_at=r.get("created_at", ""),
            trust_score=r.get("trust_score", 0.0),
            serendipity_score=r.get("serendipity_score", 0.0),
        )
        for r in records
    ]


@app.post("/v1/research/conditions/neighbors", response_model=list[ResearchConditionResult])
async def graph_neighbor_conditions(req: GraphNeighborsRequest):
    """Find research conditions related to given entities via graph traversal."""
    if not req.entity_names:
        return []
    seed_names = [n.strip().lower() for n in req.entity_names if n.strip()]
    if not seed_names:
        return []

    # Multi-hop expansion through Concept relationships
    records = run_query(
        """
        UNWIND $seeds AS seed_name
        MATCH (start:Concept {name: seed_name, namespace: $ns})
        CALL {
            WITH start
            MATCH (start)-[*1..""" + str(req.max_hops) + """]->(neighbor:Concept {namespace: $ns})
            RETURN DISTINCT neighbor.name AS entity_name
        }
        WITH collect(DISTINCT entity_name) + $seeds AS all_entities
        UNWIND all_entities AS ename
        MATCH (rc:ResearchCondition {namespace: $ns})
        WHERE toLower(rc.fact) CONTAINS ename
        RETURN DISTINCT rc.fact AS fact, rc.source_url AS source_url,
               rc.confidence AS confidence, rc.angle AS angle,
               rc.is_serendipitous AS is_serendipitous,
               rc.query AS query, rc.created_at AS created_at,
               rc.trust_score AS trust_score,
               rc.serendipity_score AS serendipity_score
        LIMIT $limit
        """,
        {"seeds": seed_names, "ns": req.namespace, "limit": req.limit},
    )
    return [
        ResearchConditionResult(
            fact=r["fact"],
            source_url=r.get("source_url", ""),
            confidence=r.get("confidence", 0.0),
            angle=r.get("angle", ""),
            is_serendipitous=bool(r.get("is_serendipitous", False)),
            query=r.get("query", ""),
            created_at=r.get("created_at", ""),
            trust_score=r.get("trust_score", 0.0),
            serendipity_score=r.get("serendipity_score", 0.0),
        )
        for r in records
    ]


@app.get("/v1/research/stats", response_model=ResearchStatsResponse)
async def research_stats(namespace: str = "research"):
    """Get statistics about the persistent research knowledge base."""
    cond_records = run_query(
        "MATCH (rc:ResearchCondition {namespace: $ns}) RETURN count(rc) AS cnt",
        {"ns": namespace},
    )
    total_conditions = cond_records[0]["cnt"] if cond_records else 0

    session_records = run_query(
        "MATCH (rc:ResearchCondition {namespace: $ns}) RETURN count(DISTINCT rc.session_id) AS cnt",
        {"ns": namespace},
    )
    total_sessions = session_records[0]["cnt"] if session_records else 0

    query_records = run_query(
        "MATCH (rc:ResearchCondition {namespace: $ns}) RETURN count(DISTINCT rc.query) AS cnt",
        {"ns": namespace},
    )
    total_queries = query_records[0]["cnt"] if query_records else 0

    ent_records = run_query(
        "MATCH (c:Concept {namespace: $ns}) RETURN count(c) AS cnt",
        {"ns": namespace},
    )
    total_entities = ent_records[0]["cnt"] if ent_records else 0

    rel_records = run_query(
        """
        MATCH (a:Concept {namespace: $ns})-[r]->(b:Concept {namespace: $ns})
        RETURN count(r) AS cnt
        """,
        {"ns": namespace},
    )
    total_relationships = rel_records[0]["cnt"] if rel_records else 0

    return ResearchStatsResponse(
        total_conditions=total_conditions,
        total_sessions=total_sessions,
        total_queries=total_queries,
        total_entities=total_entities,
        total_relationships=total_relationships,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
