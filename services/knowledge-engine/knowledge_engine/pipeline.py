"""ETL pipeline orchestrator.

Coordinates: chunk → multi-pass extract → entity resolution → load into Neo4j.
Supports full rebuild (clear namespace first) or append mode.
"""

import logging
import os
import uuid

from .chunker import chunk_text
from .config import RAW_FILES_DIR, EXTRACTION_MODEL
from .entity_resolver import resolve_entities
from .extractor import extract_all_chunks
from .models import IngestJobStatus, JobStatus
from .neo4j_client import clear_namespace, get_driver
from .ontology import (
    create_anomaly_node,
    create_chunk_node,
    create_claim_node,
    create_concept_node,
    create_document_node,
    create_evidence_node,
    create_extraction_run,
    create_hypothesis_node,
    create_method_node,
    create_relationship,
)

log = logging.getLogger("knowledge-engine")

# In-memory job tracker (lightweight — no Redis/Celery needed)
_jobs: dict[str, IngestJobStatus] = {}


def get_job(job_id: str) -> IngestJobStatus | None:
    return _jobs.get(job_id)


def list_jobs(namespace: str | None = None) -> list[IngestJobStatus]:
    if namespace:
        return [j for j in _jobs.values() if j.namespace == namespace]
    return list(_jobs.values())


async def run_ingest_pipeline(
    job_id: str,
    namespace: str,
    title: str,
    text: str,
    source: str = "",
    rebuild: bool = True,
) -> None:
    """Full ETL pipeline: chunk → extract → resolve → load.

    This runs as a background task. Progress is tracked in _jobs.
    """
    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    status = IngestJobStatus(
        job_id=job_id,
        namespace=namespace,
        status=JobStatus.pending,
        stats={},
    )
    _jobs[job_id] = status

    try:
        # --- Step 0: Save raw file to disk ---
        raw_path = _save_raw_file(namespace, doc_id, title, text)
        status.stats["raw_file"] = raw_path

        # --- Step 1: Clear namespace if rebuild ---
        if rebuild:
            status.status = JobStatus.chunking
            status.progress = "Clearing namespace for rebuild..."
            cleared = clear_namespace(namespace)
            status.stats["cleared"] = cleared
            log.info(f"[{job_id}] Rebuild: cleared namespace '{namespace}'")

        # --- Step 2: Chunk ---
        status.status = JobStatus.chunking
        status.progress = "Chunking text..."
        chunks = chunk_text(text)
        status.stats["total_chunks"] = len(chunks)
        status.stats["total_chars"] = len(text)
        log.info(f"[{job_id}] Chunked into {len(chunks)} chunks")

        # --- Step 3: Multi-pass extraction ---
        status.status = JobStatus.extracting
        status.progress = f"Extracting knowledge from {len(chunks)} chunks (3 passes)..."
        extraction_results = await extract_all_chunks(chunks)
        status.stats["extraction_results"] = len(extraction_results)
        log.info(f"[{job_id}] Extraction complete for {len(extraction_results)} chunks")

        # --- Step 4: Load into Neo4j ---
        status.status = JobStatus.loading
        status.progress = "Loading into Neo4j..."
        load_stats = _load_into_neo4j(
            namespace=namespace,
            doc_id=doc_id,
            title=title,
            source=source,
            chunks=chunks,
            extraction_results=extraction_results,
        )
        status.stats.update(load_stats)
        log.info(f"[{job_id}] Loaded into Neo4j: {load_stats}")

        # --- Step 5: Entity resolution ---
        status.status = JobStatus.resolving
        status.progress = "Resolving and deduplicating entities..."
        resolve_stats = resolve_entities(namespace)
        status.stats["entity_resolution"] = resolve_stats
        log.info(f"[{job_id}] Entity resolution: {resolve_stats}")

        # --- Step 6: Compute graph metrics ---
        status.status = JobStatus.computing_graph
        status.progress = "Computing graph metrics (community detection, centrality)..."
        _compute_graph_metrics(namespace)
        log.info(f"[{job_id}] Graph metrics computed")

        # --- Done ---
        status.status = JobStatus.completed
        status.progress = "Ingest complete"
        log.info(f"[{job_id}] Pipeline complete for '{title}' in namespace '{namespace}'")

    except Exception as e:
        log.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
        status.status = JobStatus.failed
        status.error = str(e)
        status.progress = f"Failed: {e}"


def _save_raw_file(namespace: str, doc_id: str, title: str, text: str) -> str:
    """Save the raw text to disk for archival / re-processing."""
    ns_dir = os.path.join(RAW_FILES_DIR, namespace)
    os.makedirs(ns_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
    file_path = os.path.join(ns_dir, f"{doc_id}_{safe_title}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def _load_into_neo4j(
    namespace: str,
    doc_id: str,
    title: str,
    source: str,
    chunks: list[dict],
    extraction_results: list[dict],
) -> dict:
    """Load chunks and extracted knowledge into Neo4j."""
    driver = get_driver()
    stats = {
        "documents": 1,
        "chunks_loaded": 0,
        "concepts": 0,
        "claims": 0,
        "hypotheses": 0,
        "anomalies": 0,
        "evidence": 0,
        "methods": 0,
        "relationships": 0,
    }

    with driver.session() as session:
        # Create extraction run record
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        tx = session.begin_transaction()
        try:
            create_extraction_run(tx, namespace, run_id, EXTRACTION_MODEL, "multi-pass", doc_id)

            # Create document node
            create_document_node(
                tx, namespace, doc_id, title, source,
                total_chunks=len(chunks), total_chars=sum(len(c["content"]) for c in chunks),
            )

            # Create chunk nodes
            prev_chunk_id = ""
            for chunk in chunks:
                create_chunk_node(
                    tx, namespace, chunk["id"], doc_id,
                    chunk["chunk_index"], chunk["content"],
                    prev_chunk_id=prev_chunk_id,
                )
                prev_chunk_id = chunk["id"]
                stats["chunks_loaded"] += 1

            tx.commit()
        except Exception:
            tx.rollback()
            raise

        # Load extraction results per chunk
        for i, result in enumerate(extraction_results):
            chunk_id = result.get("chunk_id", chunks[i]["id"] if i < len(chunks) else "")
            p1 = result.get("pass1", {})
            p2 = result.get("pass2", {})

            tx = session.begin_transaction()
            try:
                # --- Pass 1 results ---
                concept_ids: dict[str, str] = {}

                for concept in p1.get("concepts", []):
                    name = concept.get("name", "").strip()
                    if not name:
                        continue
                    cid = create_concept_node(
                        tx, namespace, name,
                        domains=concept.get("domains", []),
                        abstraction_level=concept.get("abstraction_level", "concrete"),
                        chunk_id=chunk_id,
                        extraction_run_id=run_id,
                    )
                    concept_ids[name.lower()] = cid
                    stats["concepts"] += 1

                for claim in p1.get("claims", []):
                    stmt = claim.get("statement", "").strip()
                    if not stmt:
                        continue
                    create_claim_node(
                        tx, namespace, stmt,
                        confidence=claim.get("confidence", 0.5),
                        polarity=claim.get("polarity", "positive"),
                        chunk_id=chunk_id,
                        extraction_run_id=run_id,
                    )
                    stats["claims"] += 1

                for ev in p1.get("evidence", []):
                    text_val = ev.get("text", "").strip()
                    if not text_val:
                        continue
                    create_evidence_node(
                        tx, namespace, text_val,
                        strength=ev.get("strength", "moderate"),
                        chunk_id=chunk_id,
                    )
                    stats["evidence"] += 1

                for method in p1.get("methods", []):
                    name = method.get("name", "").strip()
                    if not name:
                        continue
                    create_method_node(
                        tx, namespace, name,
                        domain=method.get("domain", ""),
                        transferable=method.get("transferable", False),
                        chunk_id=chunk_id,
                    )
                    stats["methods"] += 1

                # --- Pass 2 results ---
                for hyp in p2.get("hypotheses", []):
                    stmt = hyp.get("statement", "").strip()
                    if not stmt:
                        continue
                    create_hypothesis_node(
                        tx, namespace, stmt,
                        status=hyp.get("status", "open"),
                        abductive_origin=hyp.get("abductive_origin", ""),
                        chunk_id=chunk_id,
                    )
                    stats["hypotheses"] += 1

                for anom in p2.get("anomalies", []):
                    desc = anom.get("description", "").strip()
                    if not desc:
                        continue
                    create_anomaly_node(
                        tx, namespace, desc,
                        surprise_score=anom.get("surprise_score", 0.5),
                        chunk_id=chunk_id,
                    )
                    stats["anomalies"] += 1

                # Pass 2 analogies → ANALOGOUS_TO relationships
                for analogy in p2.get("analogies", []):
                    a_name = analogy.get("concept_a", "").strip().lower()
                    b_name = analogy.get("concept_b", "").strip().lower()
                    if a_name in concept_ids and b_name in concept_ids:
                        create_relationship(
                            tx,
                            concept_ids[a_name], concept_ids[b_name],
                            "Concept", "Concept",
                            "ANALOGOUS_TO",
                            {
                                "cross_domain": analogy.get("cross_domain", False),
                                "bridge_score": analogy.get("bridge_score", 0.5),
                            },
                        )
                        stats["relationships"] += 1

                # Pass 2 implicit relationships
                for rel in p2.get("implicit_relationships", []):
                    src = rel.get("source", "").strip().lower()
                    tgt = rel.get("target", "").strip().lower()
                    rel_type = rel.get("relationship", "RELATED_TO").upper().replace(" ", "_")
                    if src in concept_ids and tgt in concept_ids:
                        create_relationship(
                            tx,
                            concept_ids[src], concept_ids[tgt],
                            "Concept", "Concept",
                            rel_type,
                            {"confidence": rel.get("confidence", 0.5)},
                        )
                        stats["relationships"] += 1

                tx.commit()
            except Exception as e:
                tx.rollback()
                log.error(f"Failed loading chunk {i}: {e}")

        # --- Cross-chunk relationships (from pass 3) ---
        if extraction_results and "cross_relationships" in extraction_results[0]:
            tx = session.begin_transaction()
            try:
                for rel in extraction_results[0]["cross_relationships"]:
                    src_name = rel.get("source_name", "").strip()
                    tgt_name = rel.get("target_name", "").strip()
                    rel_type = rel.get("relationship_type", "RELATED_TO").upper().replace(" ", "_")
                    src_label = rel.get("source_type", "Concept")
                    tgt_label = rel.get("target_type", "Concept")

                    if not src_name or not tgt_name:
                        continue

                    # Find source and target by name
                    result = tx.run(
                        f"""
                        MATCH (a:{src_label} {{namespace: $ns}})
                        WHERE toLower(a.name) = toLower($src_name)
                           OR toLower(a.statement) = toLower($src_name)
                        RETURN a.id AS id LIMIT 1
                        """,
                        ns=namespace, src_name=src_name,
                    )
                    src_record = result.single()

                    result = tx.run(
                        f"""
                        MATCH (b:{tgt_label} {{namespace: $ns}})
                        WHERE toLower(b.name) = toLower($tgt_name)
                           OR toLower(b.statement) = toLower($tgt_name)
                        RETURN b.id AS id LIMIT 1
                        """,
                        ns=namespace, tgt_name=tgt_name,
                    )
                    tgt_record = result.single()

                    if src_record and tgt_record:
                        create_relationship(
                            tx,
                            src_record["id"], tgt_record["id"],
                            src_label, tgt_label,
                            rel_type,
                            {
                                "cross_domain": rel.get("cross_domain", False),
                                "confidence": rel.get("confidence", 0.5),
                            },
                        )
                        stats["relationships"] += 1

                tx.commit()
            except Exception as e:
                tx.rollback()
                log.error(f"Failed loading cross-chunk relationships: {e}")

    return stats


def _compute_graph_metrics(namespace: str) -> None:
    """Compute graph metrics using networkx (no GDS dependency).

    Computes: community detection (Louvain), betweenness centrality,
    cross-community edge count, and serendipity (RNS) scores.
    """
    try:
        import community as community_louvain
        import networkx as nx
    except ImportError:
        log.warning("networkx/community not available — skipping graph metrics")
        return

    driver = get_driver()

    with driver.session() as session:
        # Export concept graph to networkx
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            RETURN c.id AS id, c.name AS name, c.mention_count AS mentions
            """,
            ns=namespace,
        )
        nodes = [(r["id"], {"name": r["name"], "mentions": r["mentions"]}) for r in result]

        if len(nodes) < 2:
            return

        result = session.run(
            """
            MATCH (a:Concept {namespace: $ns})-[r]->(b:Concept {namespace: $ns})
            RETURN a.id AS src, b.id AS tgt, type(r) AS rel_type
            """,
            ns=namespace,
        )
        edges = [(r["src"], r["tgt"], {"rel_type": r["rel_type"]}) for r in result]

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if G.number_of_nodes() < 2:
        return

    # Community detection (Louvain)
    try:
        partition = community_louvain.best_partition(G)
    except Exception:
        partition = {n: 0 for n in G.nodes()}

    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    # Cross-community edges per node
    cross_community: dict[str, int] = {n: 0 for n in G.nodes()}
    for u, v in G.edges():
        if partition.get(u, -1) != partition.get(v, -1):
            cross_community[u] = cross_community.get(u, 0) + 1
            cross_community[v] = cross_community.get(v, 0) + 1

    # Write metrics back to Neo4j
    with driver.session() as session:
        for node_id in G.nodes():
            comm = partition.get(node_id, -1)
            bc = betweenness.get(node_id, 0.0)
            cc = cross_community.get(node_id, 0)

            # RNS score: Relevance (mentions) × Novelty (cross-community) × Surprise (betweenness)
            mentions = G.nodes[node_id].get("mentions", 1) or 1
            relevance = min(mentions / 10.0, 1.0)
            novelty = min(cc / 5.0, 1.0) if cc > 0 else 0.0
            surprise = min(bc * 10.0, 1.0)
            rns = relevance * 0.3 + novelty * 0.4 + surprise * 0.3

            session.run(
                """
                MATCH (c:Concept {id: $id})
                SET c.community_id = $community,
                    c.betweenness_centrality = $bc,
                    c.cross_community_edges = $cc,
                    c.rns_score = $rns
                """,
                id=node_id, community=comm, bc=bc, cc=cc, rns=rns,
            )
