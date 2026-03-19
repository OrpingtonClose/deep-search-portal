"""Entity resolution / deduplication.

Two-stage approach:
  1. Exact-match merging via MERGE semantics (handled in ontology.py)
  2. Fuzzy/semantic merging: string similarity for near-duplicate concepts
"""

import logging
from typing import Any

from .neo4j_client import get_driver

log = logging.getLogger("knowledge-engine")


def _normalise(name: str) -> str:
    """Normalise a concept name for comparison."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def resolve_entities(namespace: str) -> dict:
    """Run entity resolution on all Concept nodes in a namespace.

    Merges near-duplicate concepts using string similarity.
    Returns stats about how many merges were performed.
    """
    driver = get_driver()
    merges = 0

    with driver.session() as session:
        # Fetch all concept names in namespace
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            RETURN c.id AS id, c.name AS name, c.mention_count AS mentions
            ORDER BY c.mention_count DESC
            """,
            ns=namespace,
        )
        concepts = [(r["id"], r["name"], r["mentions"]) for r in result]

    if len(concepts) < 2:
        return {"merges": 0, "total_concepts": len(concepts)}

    # Build normalised name groups
    groups: dict[str, list[tuple[str, str, int]]] = {}
    for cid, name, mentions in concepts:
        norm = _normalise(name)
        groups.setdefault(norm, []).append((cid, name, mentions))

    # Merge groups with multiple entries
    with driver.session() as session:
        for norm_name, entries in groups.items():
            if len(entries) <= 1:
                continue

            # Keep the entry with highest mention count as canonical
            entries.sort(key=lambda x: x[2], reverse=True)
            canonical_id = entries[0][0]

            for dup_id, dup_name, _ in entries[1:]:
                _merge_concept_pair(session, canonical_id, dup_id)
                merges += 1

    # Also check for substring containment (e.g., "GPT-4" vs "GPT-4 model")
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            RETURN c.id AS id, c.name AS name, c.mention_count AS mentions
            ORDER BY size(c.name) ASC
            """,
            ns=namespace,
        )
        remaining = [(r["id"], r["name"], r["mentions"]) for r in result]

    # Check short names that are substrings of longer names
    absorbed: set[str] = set()
    for i, (short_id, short_name, short_mentions) in enumerate(remaining):
        if short_id in absorbed or len(short_name) < 3:
            continue
        short_norm = _normalise(short_name)

        for long_id, long_name, long_mentions in remaining[i + 1:]:
            if long_id in absorbed:
                continue
            long_norm = _normalise(long_name)

            # Only merge if short is a complete word within long
            # and the longer name is <=2x the shorter
            if (
                short_norm in long_norm
                and len(long_norm) <= len(short_norm) * 2
                and short_norm != long_norm
            ):
                # Keep the shorter, more canonical name
                with driver.session() as session:
                    _merge_concept_pair(session, short_id, long_id)
                    absorbed.add(long_id)
                    merges += 1

    log.info(
        f"Entity resolution for namespace '{namespace}': "
        f"{merges} merges, {len(concepts)} original concepts"
    )
    return {"merges": merges, "total_concepts": len(concepts) - merges}


def _merge_concept_pair(session: Any, keep_id: str, remove_id: str) -> None:
    """Merge two Concept nodes, transferring all relationships to the keeper."""
    # Transfer incoming relationships
    session.run(
        """
        MATCH (remove:Concept {id: $remove_id})<-[r]-(other)
        WHERE other.id <> $keep_id
        WITH other, type(r) AS rel_type, properties(r) AS props, remove
        MATCH (keep:Concept {id: $keep_id})
        CALL {
            WITH other, rel_type, props, keep
            WITH other, rel_type, props, keep
            WHERE rel_type = 'MENTIONS'
            MERGE (other)-[:MENTIONS]->(keep)
        }
        """,
        keep_id=keep_id, remove_id=remove_id,
    )

    # Transfer outgoing relationships
    session.run(
        """
        MATCH (remove:Concept {id: $remove_id})-[r]->(other)
        WHERE other.id <> $keep_id
        WITH other, type(r) AS rel_type, properties(r) AS props, remove
        MATCH (keep:Concept {id: $keep_id})
        CALL {
            WITH other, rel_type, props, keep
            WITH other, rel_type, props, keep
            WHERE rel_type = 'RELATED_TO'
            MERGE (keep)-[:RELATED_TO]->(other)
        }
        """,
        keep_id=keep_id, remove_id=remove_id,
    )

    # Update mention count on keeper
    session.run(
        """
        MATCH (keep:Concept {id: $keep_id})
        MATCH (remove:Concept {id: $remove_id})
        SET keep.mention_count = keep.mention_count + remove.mention_count
        """,
        keep_id=keep_id, remove_id=remove_id,
    )

    # Delete the duplicate
    session.run(
        "MATCH (c:Concept {id: $id}) DETACH DELETE c",
        id=remove_id,
    )
