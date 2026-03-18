"""Neo4j connection management and schema initialization."""

import logging
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

log = logging.getLogger("knowledge-engine")

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """Get or create the Neo4j driver singleton."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
        )
        _driver.verify_connectivity()
        log.info(f"Connected to Neo4j at {NEO4J_URI}")
    return _driver


def close_driver() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        log.info("Neo4j driver closed")


def get_session() -> Session:
    """Get a new Neo4j session."""
    return get_driver().session()


def init_schema() -> None:
    """Create constraints, indexes, and vector indexes for the epistemic ontology."""
    driver = get_driver()

    constraints = [
        # Uniqueness constraints for each epistemic node type
        ("constraint_document_id", "CREATE CONSTRAINT constraint_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"),
        ("constraint_chunk_id", "CREATE CONSTRAINT constraint_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"),
        ("constraint_concept_id", "CREATE CONSTRAINT constraint_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE"),
        ("constraint_claim_id", "CREATE CONSTRAINT constraint_claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE"),
        ("constraint_hypothesis_id", "CREATE CONSTRAINT constraint_hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE"),
        ("constraint_anomaly_id", "CREATE CONSTRAINT constraint_anomaly_id IF NOT EXISTS FOR (a:Anomaly) REQUIRE a.id IS UNIQUE"),
        ("constraint_evidence_id", "CREATE CONSTRAINT constraint_evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE"),
        ("constraint_method_id", "CREATE CONSTRAINT constraint_method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE"),
        ("constraint_topic_id", "CREATE CONSTRAINT constraint_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE"),
        ("constraint_extraction_run_id", "CREATE CONSTRAINT constraint_extraction_run_id IF NOT EXISTS FOR (er:ExtractionRun) REQUIRE er.id IS UNIQUE"),
        ("constraint_research_condition_id", "CREATE CONSTRAINT constraint_research_condition_id IF NOT EXISTS FOR (rc:ResearchCondition) REQUIRE rc.id IS UNIQUE"),
    ]

    indexes = [
        # Namespace indexes for fast filtering
        "CREATE INDEX idx_document_ns IF NOT EXISTS FOR (d:Document) ON (d.namespace)",
        "CREATE INDEX idx_chunk_ns IF NOT EXISTS FOR (c:Chunk) ON (c.namespace)",
        "CREATE INDEX idx_concept_ns IF NOT EXISTS FOR (c:Concept) ON (c.namespace)",
        "CREATE INDEX idx_claim_ns IF NOT EXISTS FOR (c:Claim) ON (c.namespace)",
        "CREATE INDEX idx_hypothesis_ns IF NOT EXISTS FOR (h:Hypothesis) ON (h.namespace)",
        "CREATE INDEX idx_anomaly_ns IF NOT EXISTS FOR (a:Anomaly) ON (a.namespace)",
        "CREATE INDEX idx_evidence_ns IF NOT EXISTS FOR (e:Evidence) ON (e.namespace)",
        "CREATE INDEX idx_method_ns IF NOT EXISTS FOR (m:Method) ON (m.namespace)",
        # Name lookups
        "CREATE INDEX idx_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        "CREATE INDEX idx_claim_statement IF NOT EXISTS FOR (c:Claim) ON (c.statement)",
        # Community detection support
        "CREATE INDEX idx_concept_community IF NOT EXISTS FOR (c:Concept) ON (c.community_id, c.cross_community_edges)",
        # Full-text search indexes
        "CREATE FULLTEXT INDEX fulltext_chunks IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content]",
        "CREATE FULLTEXT INDEX fulltext_concepts IF NOT EXISTS FOR (c:Concept) ON EACH [c.name]",
        "CREATE FULLTEXT INDEX fulltext_claims IF NOT EXISTS FOR (c:Claim) ON EACH [c.statement]",
        "CREATE FULLTEXT INDEX fulltext_anomalies IF NOT EXISTS FOR (a:Anomaly) ON EACH [a.description]",
        # ResearchCondition indexes
        "CREATE INDEX idx_rc_ns IF NOT EXISTS FOR (rc:ResearchCondition) ON (rc.namespace)",
        "CREATE INDEX idx_rc_session IF NOT EXISTS FOR (rc:ResearchCondition) ON (rc.session_id)",
        "CREATE INDEX idx_rc_query IF NOT EXISTS FOR (rc:ResearchCondition) ON (rc.query)",
        "CREATE FULLTEXT INDEX fulltext_research_conditions IF NOT EXISTS FOR (rc:ResearchCondition) ON EACH [rc.fact, rc.query, rc.angle]",
    ]

    with driver.session() as session:
        for name, cypher in constraints:
            try:
                session.run(cypher)
                log.debug(f"Constraint ensured: {name}")
            except Exception as e:
                log.warning(f"Constraint {name}: {e}")

        for cypher in indexes:
            try:
                session.run(cypher)
                log.debug(f"Index ensured: {cypher[:60]}...")
            except Exception as e:
                log.warning(f"Index creation: {e}")

    log.info("Neo4j schema initialized")


def clear_namespace(namespace: str) -> dict:
    """Delete all nodes and relationships in a namespace. Returns counts."""
    driver = get_driver()
    counts: dict[str, int] = {}

    with driver.session() as session:
        # Delete in batches to avoid memory issues with large namespaces
        labels = [
            "Chunk", "Concept", "Claim", "Hypothesis", "Anomaly",
            "Evidence", "Method", "Topic", "ExtractionRun", "Document",
            "ResearchCondition",
        ]
        total_deleted = 0
        for label in labels:
            batch_deleted = 0
            while True:
                result = session.run(
                    f"""
                    MATCH (n:{label} {{namespace: $ns}})
                    WITH n LIMIT 1000
                    DETACH DELETE n
                    RETURN count(*) AS deleted
                    """,
                    ns=namespace,
                )
                deleted = result.single()["deleted"]
                batch_deleted += deleted
                total_deleted += deleted
                if deleted < 1000:
                    break
            if batch_deleted > 0:
                counts[label] = batch_deleted

    log.info(f"Cleared namespace '{namespace}': {total_deleted} nodes deleted")
    return counts


def run_query(cypher: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
    """Execute a read query and return results as a list of dicts."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


def run_write(cypher: str, params: Optional[dict] = None) -> Any:
    """Execute a write query."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return result.consume()
