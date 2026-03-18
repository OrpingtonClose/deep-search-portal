"""Epistemic ontology — node types, relationship types, and Neo4j loading.

The ontology has four layers:
  1. Epistemic nodes: Concept, Claim, Hypothesis, Anomaly, Evidence, Method,
     ResearchCondition
  2. Epistemic relationships: typed, directional, with confidence/strength
  3. Provenance: Document → Chunk → extracted nodes, ExtractionRun metadata
  4. Serendipity scoring: computed properties (betweenness, community, RNS)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("knowledge-engine")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ============================================================================
# Document & Chunk Layer (Provenance)
# ============================================================================

def create_document_node(
    tx: Any,
    namespace: str,
    doc_id: str,
    title: str,
    source: str,
    total_chunks: int,
    total_chars: int,
    file_path: str = "",
) -> None:
    """Create a :Document node."""
    tx.run(
        """
        MERGE (d:Document {id: $id})
        SET d.namespace = $namespace,
            d.title = $title,
            d.source = $source,
            d.total_chunks = $total_chunks,
            d.total_chars = $total_chars,
            d.file_path = $file_path,
            d.created_at = $now
        """,
        id=doc_id, namespace=namespace, title=title, source=source,
        total_chunks=total_chunks, total_chars=total_chars,
        file_path=file_path, now=_now(),
    )


def create_chunk_node(
    tx: Any,
    namespace: str,
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    content: str,
    prev_chunk_id: str = "",
) -> None:
    """Create a :Chunk node and link it to its :Document and previous :Chunk."""
    tx.run(
        """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.namespace = $namespace,
            c.chunk_index = $chunk_index,
            c.content = $content,
            c.created_at = $now
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        chunk_id=chunk_id, namespace=namespace, doc_id=doc_id,
        chunk_index=chunk_index, content=content, now=_now(),
    )

    # Link sequential chunks
    if prev_chunk_id:
        tx.run(
            """
            MATCH (prev:Chunk {id: $prev_id})
            MATCH (curr:Chunk {id: $curr_id})
            MERGE (prev)-[:NEXT]->(curr)
            """,
            prev_id=prev_chunk_id, curr_id=chunk_id,
        )


# ============================================================================
# Epistemic Node Layer
# ============================================================================

def create_concept_node(
    tx: Any,
    namespace: str,
    name: str,
    domains: list[str],
    abstraction_level: str = "concrete",
    chunk_id: str = "",
    extraction_run_id: str = "",
) -> str:
    """Create or merge a :Concept node. Returns the node ID."""
    concept_id = _uid("concept-")
    result = tx.run(
        """
        MERGE (c:Concept {name: $name, namespace: $ns})
        ON CREATE SET
            c.id = $id,
            c.domains = $domains,
            c.abstraction_level = $abstraction_level,
            c.mention_count = 1,
            c.created_at = $now,
            c.community_id = -1,
            c.cross_community_edges = 0,
            c.betweenness_centrality = 0.0,
            c.rns_score = 0.0,
            c.visit_count = 0
        ON MATCH SET
            c.mention_count = c.mention_count + 1,
            c.domains = [x IN c.domains + $domains WHERE x IS NOT NULL | x]
        RETURN c.id AS id
        """,
        id=concept_id, name=name, ns=namespace, domains=domains,
        abstraction_level=abstraction_level, now=_now(),
    )
    record = result.single()
    actual_id = record["id"] if record else concept_id

    if chunk_id:
        tx.run(
            """
            MATCH (c:Concept {id: $concept_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (ch)-[:MENTIONS]->(c)
            """,
            concept_id=actual_id, chunk_id=chunk_id,
        )

    if extraction_run_id:
        tx.run(
            """
            MATCH (c:Concept {id: $concept_id})
            MATCH (er:ExtractionRun {id: $er_id})
            MERGE (c)-[:EXTRACTED_BY]->(er)
            """,
            concept_id=actual_id, er_id=extraction_run_id,
        )

    return actual_id


def create_claim_node(
    tx: Any,
    namespace: str,
    statement: str,
    confidence: float = 0.5,
    polarity: str = "positive",
    chunk_id: str = "",
    extraction_run_id: str = "",
) -> str:
    """Create a :Claim node. Returns the node ID."""
    claim_id = _uid("claim-")
    tx.run(
        """
        CREATE (c:Claim {
            id: $id,
            namespace: $ns,
            statement: $statement,
            confidence: $confidence,
            polarity: $polarity,
            verified: false,
            created_at: $now
        })
        """,
        id=claim_id, ns=namespace, statement=statement,
        confidence=confidence, polarity=polarity, now=_now(),
    )

    if chunk_id:
        tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (cl)-[:DERIVED_FROM]->(ch)
            """,
            claim_id=claim_id, chunk_id=chunk_id,
        )

    if extraction_run_id:
        tx.run(
            """
            MATCH (cl:Claim {id: $claim_id})
            MATCH (er:ExtractionRun {id: $er_id})
            MERGE (cl)-[:EXTRACTED_BY]->(er)
            """,
            claim_id=claim_id, er_id=extraction_run_id,
        )

    return claim_id


def create_hypothesis_node(
    tx: Any,
    namespace: str,
    statement: str,
    status: str = "open",
    abductive_origin: str = "",
    chunk_id: str = "",
) -> str:
    """Create a :Hypothesis node."""
    hyp_id = _uid("hyp-")
    tx.run(
        """
        CREATE (h:Hypothesis {
            id: $id,
            namespace: $ns,
            statement: $statement,
            status: $status,
            abductive_origin: $abductive_origin,
            created_at: $now
        })
        """,
        id=hyp_id, ns=namespace, statement=statement,
        status=status, abductive_origin=abductive_origin, now=_now(),
    )

    if chunk_id:
        tx.run(
            """
            MATCH (h:Hypothesis {id: $hyp_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (h)-[:DERIVED_FROM]->(ch)
            """,
            hyp_id=hyp_id, chunk_id=chunk_id,
        )

    return hyp_id


def create_anomaly_node(
    tx: Any,
    namespace: str,
    description: str,
    surprise_score: float = 0.5,
    chunk_id: str = "",
) -> str:
    """Create an :Anomaly node — highest serendipity signal."""
    anom_id = _uid("anom-")
    tx.run(
        """
        CREATE (a:Anomaly {
            id: $id,
            namespace: $ns,
            description: $description,
            surprise_score: $surprise_score,
            created_at: $now
        })
        """,
        id=anom_id, ns=namespace, description=description,
        surprise_score=surprise_score, now=_now(),
    )

    if chunk_id:
        tx.run(
            """
            MATCH (a:Anomaly {id: $anom_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (a)-[:DERIVED_FROM]->(ch)
            """,
            anom_id=anom_id, chunk_id=chunk_id,
        )

    return anom_id


def create_evidence_node(
    tx: Any,
    namespace: str,
    text: str,
    strength: str = "moderate",
    chunk_id: str = "",
) -> str:
    """Create an :Evidence node."""
    ev_id = _uid("ev-")
    tx.run(
        """
        CREATE (e:Evidence {
            id: $id,
            namespace: $ns,
            text: $text,
            strength: $strength,
            created_at: $now
        })
        """,
        id=ev_id, ns=namespace, text=text, strength=strength, now=_now(),
    )

    if chunk_id:
        tx.run(
            """
            MATCH (e:Evidence {id: $ev_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (e)-[:DERIVED_FROM]->(ch)
            """,
            ev_id=ev_id, chunk_id=chunk_id,
        )

    return ev_id


def create_method_node(
    tx: Any,
    namespace: str,
    name: str,
    domain: str = "",
    transferable: bool = False,
    chunk_id: str = "",
) -> str:
    """Create a :Method node."""
    method_id = _uid("method-")
    result = tx.run(
        """
        MERGE (m:Method {name: $name, namespace: $ns})
        ON CREATE SET
            m.id = $id,
            m.domain = $domain,
            m.transferable = $transferable,
            m.mention_count = 1,
            m.created_at = $now
        ON MATCH SET
            m.mention_count = m.mention_count + 1
        RETURN m.id AS id
        """,
        id=method_id, name=name, ns=namespace, domain=domain,
        transferable=transferable, now=_now(),
    )
    record = result.single()
    actual_id = record["id"] if record else method_id

    if chunk_id:
        tx.run(
            """
            MATCH (m:Method {id: $method_id})
            MATCH (ch:Chunk {id: $chunk_id})
            MERGE (ch)-[:MENTIONS]->(m)
            """,
            method_id=actual_id, chunk_id=chunk_id,
        )

    return actual_id


def create_research_condition_node(
    tx: Any,
    namespace: str,
    session_id: str,
    query: str,
    fact: str,
    source_url: str = "",
    confidence: float = 0.5,
    trust_score: float = 0.5,
    angle: str = "",
    domain: str = "",
    is_serendipitous: bool = False,
    serendipity_score: float = 0.0,
) -> str:
    """Create a :ResearchCondition node — a finding from a live research session."""
    cond_id = _uid("rc-")
    tx.run(
        """
        CREATE (rc:ResearchCondition {
            id: $id,
            namespace: $ns,
            session_id: $session_id,
            query: $query,
            fact: $fact,
            source_url: $source_url,
            confidence: $confidence,
            trust_score: $trust_score,
            angle: $angle,
            domain: $domain,
            is_serendipitous: $is_serendipitous,
            serendipity_score: $serendipity_score,
            verified: false,
            created_at: $now
        })
        """,
        id=cond_id, ns=namespace, session_id=session_id, query=query,
        fact=fact, source_url=source_url, confidence=confidence,
        trust_score=trust_score, angle=angle, domain=domain,
        is_serendipitous=is_serendipitous, serendipity_score=serendipity_score,
        now=_now(),
    )
    return cond_id


def batch_create_research_conditions(
    tx: Any,
    namespace: str,
    session_id: str,
    query: str,
    conditions: list[dict],
) -> int:
    """Create multiple :ResearchCondition nodes in a single transaction.

    Each dict in *conditions* should have keys: fact, source_url, confidence,
    trust_score, angle, domain, is_serendipitous, serendipity_score.
    Returns the number of nodes created.
    """
    now = _now()
    count = 0
    for c in conditions:
        fact = c.get("fact", "").strip()
        if not fact:
            continue
        cond_id = _uid("rc-")
        tx.run(
            """
            CREATE (rc:ResearchCondition {
                id: $id,
                namespace: $ns,
                session_id: $session_id,
                query: $query,
                fact: $fact,
                source_url: $source_url,
                confidence: $confidence,
                trust_score: $trust_score,
                angle: $angle,
                domain: $domain,
                is_serendipitous: $is_serendipitous,
                serendipity_score: $serendipity_score,
                verified: false,
                created_at: $now
            })
            """,
            id=cond_id, ns=namespace, session_id=session_id, query=query,
            fact=fact,
            source_url=c.get("source_url", ""),
            confidence=float(c.get("confidence", 0.5)),
            trust_score=float(c.get("trust_score", 0.5)),
            angle=c.get("angle", ""),
            domain=c.get("domain", ""),
            is_serendipitous=bool(c.get("is_serendipitous", False)),
            serendipity_score=float(c.get("serendipity_score", 0.0)),
            now=now,
        )
        count += 1
    return count


def batch_create_research_entities(
    tx: Any,
    namespace: str,
    session_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> tuple[int, int]:
    """Create Concept nodes and relationships from research entity extraction.

    Returns (entities_created, relationships_created).
    """
    now = _now()
    ent_count = 0
    for ent in entities:
        name = ent.get("name", "").strip().lower()
        etype = ent.get("type", "concept")
        if not name or len(name) < 2:
            continue
        concept_id = _uid("concept-")
        tx.run(
            """
            MERGE (c:Concept {name: $name, namespace: $ns})
            ON CREATE SET
                c.id = $id,
                c.domains = [$etype],
                c.abstraction_level = 'concrete',
                c.mention_count = 1,
                c.created_at = $now,
                c.community_id = -1,
                c.cross_community_edges = 0,
                c.betweenness_centrality = 0.0,
                c.rns_score = 0.0,
                c.visit_count = 0,
                c.first_seen_session = $session_id
            ON MATCH SET
                c.mention_count = c.mention_count + 1
            """,
            id=concept_id, name=name, ns=namespace, etype=etype,
            session_id=session_id, now=now,
        )
        ent_count += 1

    rel_count = 0
    for rel in relationships:
        e1 = rel.get("entity1", "").strip().lower()
        e2 = rel.get("entity2", "").strip().lower()
        rtype = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
        is_bridge = bool(rel.get("is_bridge", False))
        if not e1 or not e2 or e1 == e2:
            continue
        tx.run(
            """
            MATCH (a:Concept {name: $e1, namespace: $ns})
            MATCH (b:Concept {name: $e2, namespace: $ns})
            MERGE (a)-[r:""" + rtype + """]->(b)
            SET r.is_bridge = $is_bridge,
                r.weight = coalesce(r.weight, 0.0) + 1.0,
                r.updated_at = $now
            """,
            e1=e1, e2=e2, ns=namespace, is_bridge=is_bridge, now=now,
        )
        rel_count += 1

    return ent_count, rel_count


def create_extraction_run(
    tx: Any,
    namespace: str,
    run_id: str,
    model: str,
    pass_name: str,
    doc_id: str,
) -> None:
    """Record an extraction run for provenance."""
    tx.run(
        """
        CREATE (er:ExtractionRun {
            id: $id,
            namespace: $ns,
            model: $model,
            pass_name: $pass_name,
            doc_id: $doc_id,
            created_at: $now
        })
        """,
        id=run_id, ns=namespace, model=model,
        pass_name=pass_name, doc_id=doc_id, now=_now(),
    )


# ============================================================================
# Epistemic Relationships
# ============================================================================

def create_relationship(
    tx: Any,
    source_id: str,
    target_id: str,
    source_label: str,
    target_label: str,
    rel_type: str,
    properties: dict | None = None,
) -> None:
    """Create a typed relationship between two epistemic nodes."""
    props = properties or {}
    props_str = ", ".join(f"r.{k} = ${k}" for k in props)
    set_clause = f"SET {props_str}" if props_str else ""

    cypher = f"""
        MATCH (a:{source_label} {{id: $source_id}})
        MATCH (b:{target_label} {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
    """
    params = {"source_id": source_id, "target_id": target_id, **props}
    tx.run(cypher, params)
