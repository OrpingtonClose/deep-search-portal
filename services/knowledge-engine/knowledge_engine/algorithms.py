"""Graph algorithms implemented as Cypher queries + networkx fallbacks.

Algorithms:
  1. Spreading Activation — multi-hop activation propagation from seed concepts
  2. Swanson ABC — literature-based discovery (A→C via hidden B intermediaries)
  3. Community detection — Louvain via networkx (computed in pipeline.py)
  4. Information gap — find under-connected high-centrality nodes
  5. Serendipity beam search — RNS-guided traversal for unexpected connections
"""

import logging

from .neo4j_client import get_driver

log = logging.getLogger("knowledge-engine")


# ============================================================================
# 1. Spreading Activation
# ============================================================================

def spreading_activation(
    namespace: str,
    seed_concepts: list[str],
    hops: int = 3,
    decay: float = 0.7,
    threshold: float = 0.01,
    limit: int = 20,
) -> list[dict]:
    """Multi-hop spreading activation from seed concepts.

    Iteratively propagates activation energy through the concept graph.
    Each hop decays the energy. Returns activated nodes sorted by activation.

    This is pure Cypher — no GDS dependency.
    """
    driver = get_driver()

    with driver.session() as session:
        # Step 1: Find seed node IDs and initialise activation
        seed_result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            WHERE toLower(c.name) IN $seeds
            RETURN c.id AS id, c.name AS name
            """,
            ns=namespace,
            seeds=[s.lower() for s in seed_concepts],
        )
        seed_nodes = {r["id"]: {"name": r["name"], "activation": 1.0} for r in seed_result}

        if not seed_nodes:
            return []

        # Iterative activation spreading
        activated: dict[str, dict] = dict(seed_nodes)
        frontier = set(seed_nodes.keys())

        for hop in range(hops):
            hop_decay = decay ** (hop + 1)
            if hop_decay < threshold:
                break

            new_frontier: set[str] = set()

            for node_id in frontier:
                parent_activation = activated[node_id]["activation"]
                propagated = parent_activation * decay

                if propagated < threshold:
                    continue

                # Get neighbors
                result = session.run(
                    """
                    MATCH (a:Concept {id: $nid})-[r]-(b:Concept {namespace: $ns})
                    RETURN b.id AS id, b.name AS name, type(r) AS rel_type,
                           b.community_id AS community, b.betweenness_centrality AS bc
                    """,
                    nid=node_id, ns=namespace,
                )

                for record in result:
                    nid = record["id"]
                    # Cross-community bonus: activation gets a 1.5x boost
                    # when crossing community boundaries
                    boost = 1.0
                    src_comm = activated.get(node_id, {}).get("community", -1)
                    tgt_comm = record["community"]
                    if src_comm != tgt_comm and src_comm != -1 and tgt_comm != -1:
                        boost = 1.5

                    new_activation = propagated * boost

                    if nid in activated:
                        # Accumulate activation (convergent paths amplify)
                        activated[nid]["activation"] = max(
                            activated[nid]["activation"],
                            activated[nid]["activation"] + new_activation * 0.5,
                        )
                    else:
                        activated[nid] = {
                            "name": record["name"],
                            "activation": new_activation,
                            "community": record["community"],
                            "betweenness": record["bc"] or 0.0,
                            "hop": hop + 1,
                            "via_rel": record["rel_type"],
                        }
                        new_frontier.add(nid)

            frontier = new_frontier

    # Sort by activation and return top results (excluding seeds)
    results = [
        {
            "id": nid,
            "name": info["name"],
            "activation": round(info["activation"], 4),
            "community": info.get("community", -1),
            "betweenness": round(info.get("betweenness", 0.0), 4),
            "hop": info.get("hop", 0),
            "is_seed": nid in seed_nodes,
        }
        for nid, info in activated.items()
    ]
    results.sort(key=lambda x: x["activation"], reverse=True)
    return results[:limit]


# ============================================================================
# 2. Swanson ABC (Literature-Based Discovery)
# ============================================================================

def swanson_abc(
    namespace: str,
    seed_concept: str,
    limit: int = 20,
) -> list[dict]:
    """Swanson ABC literature-based discovery.

    Given concept A, find concepts C that are connected to A through
    intermediate concepts B, but where A and C are NOT directly connected.
    These A→B→C paths reveal hidden connections — the core of bisociation.

    Pure Cypher — no GDS dependency.
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            // Find A
            MATCH (a:Concept {namespace: $ns})
            WHERE toLower(a.name) = toLower($seed)

            // Find B (directly connected to A)
            MATCH (a)-[r1]-(b:Concept {namespace: $ns})

            // Find C (connected to B but NOT directly to A)
            MATCH (b)-[r2]-(c:Concept {namespace: $ns})
            WHERE NOT (a)-[]-(c)
              AND c.id <> a.id
              AND c.id <> b.id

            // Score by number of distinct B intermediaries and their properties
            WITH a, c,
                 collect(DISTINCT {
                     name: b.name,
                     rel_to_a: type(r1),
                     rel_to_c: type(r2),
                     community: b.community_id,
                     betweenness: b.betweenness_centrality
                 }) AS bridges,
                 count(DISTINCT b) AS bridge_count

            // Compute discovery score
            WITH c.name AS target_concept,
                 c.id AS target_id,
                 c.community_id AS target_community,
                 c.rns_score AS target_rns,
                 bridge_count,
                 bridges,
                 // Higher score for more bridges and cross-community paths
                 bridge_count * 1.0 +
                 CASE WHEN c.rns_score > 0.5 THEN 2.0 ELSE 0.0 END +
                 CASE WHEN c.cross_community_edges > 2 THEN 1.5 ELSE 0.0 END
                 AS discovery_score

            RETURN target_concept, target_id, target_community, target_rns,
                   bridge_count, bridges[..5] AS top_bridges, discovery_score
            ORDER BY discovery_score DESC
            LIMIT $limit
            """,
            ns=namespace, seed=seed_concept, limit=limit,
        )

        discoveries = []
        for record in result:
            discoveries.append({
                "target_concept": record["target_concept"],
                "target_id": record["target_id"],
                "target_community": record["target_community"],
                "target_rns": record["target_rns"],
                "bridge_count": record["bridge_count"],
                "top_bridges": record["top_bridges"],
                "discovery_score": round(record["discovery_score"], 3),
            })

    return discoveries


# ============================================================================
# 3. Information Gap Detection
# ============================================================================

def information_gaps(
    namespace: str,
    limit: int = 15,
) -> list[dict]:
    """Find concepts that should be better connected but aren't.

    Looks for high-mention, high-centrality concepts with few edges.
    These represent knowledge gaps — areas where the corpus mentions
    something important but hasn't connected it to the broader graph.
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            OPTIONAL MATCH (c)-[r]-(other:Concept {namespace: $ns})
            WITH c, count(r) AS edge_count
            WHERE c.mention_count > 1
            WITH c.name AS name,
                 c.id AS id,
                 c.mention_count AS mentions,
                 edge_count,
                 c.betweenness_centrality AS betweenness,
                 c.community_id AS community,
                 // Gap score: high mentions + low edges = under-connected
                 toFloat(c.mention_count) / (edge_count + 1) AS gap_score
            RETURN name, id, mentions, edge_count, betweenness, community, gap_score
            ORDER BY gap_score DESC
            LIMIT $limit
            """,
            ns=namespace, limit=limit,
        )

        gaps = []
        for record in result:
            gaps.append({
                "name": record["name"],
                "id": record["id"],
                "mentions": record["mentions"],
                "edge_count": record["edge_count"],
                "betweenness": round(record["betweenness"] or 0.0, 4),
                "community": record["community"],
                "gap_score": round(record["gap_score"], 3),
            })

    return gaps


# ============================================================================
# 4. Serendipity Beam Search
# ============================================================================

def serendipity_beam_search(
    namespace: str,
    seed_concept: str,
    beam_width: int = 5,
    depth: int = 4,
    limit: int = 15,
) -> list[dict]:
    """RNS-guided beam search for serendipitous connections.

    At each step, expand the top-k nodes by RNS score (not just proximity).
    This biases traversal toward surprising, cross-community, high-novelty paths.
    """
    driver = get_driver()

    with driver.session() as session:
        # Find seed
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            WHERE toLower(c.name) = toLower($seed)
            RETURN c.id AS id, c.name AS name, c.community_id AS comm
            """,
            ns=namespace, seed=seed_concept,
        )
        seed = result.single()
        if not seed:
            return []

        visited: set[str] = {seed["id"]}
        beam = [{"id": seed["id"], "name": seed["name"], "path": [seed["name"]], "score": 0.0}]
        all_discoveries: list[dict] = []

        for step in range(depth):
            candidates: list[dict] = []

            for node in beam:
                result = session.run(
                    """
                    MATCH (a:Concept {id: $nid})-[r]-(b:Concept {namespace: $ns})
                    WHERE NOT b.id IN $visited
                    RETURN b.id AS id, b.name AS name,
                           b.rns_score AS rns, b.community_id AS comm,
                           b.cross_community_edges AS cce,
                           b.betweenness_centrality AS bc,
                           type(r) AS rel_type
                    ORDER BY b.rns_score DESC
                    LIMIT 20
                    """,
                    nid=node["id"], ns=namespace, visited=list(visited),
                )

                for record in result:
                    rns = record["rns"] or 0.0
                    # Bonus for ANALOGOUS_TO and CONTRADICTS edges
                    rel_bonus = 0.0
                    if record["rel_type"] in ("ANALOGOUS_TO", "CONTRADICTS"):
                        rel_bonus = 0.3

                    score = rns + rel_bonus + (record["bc"] or 0.0) * 0.2

                    candidates.append({
                        "id": record["id"],
                        "name": record["name"],
                        "path": node["path"] + [record["name"]],
                        "score": score,
                        "community": record["comm"],
                        "rns": rns,
                        "via_rel": record["rel_type"],
                        "step": step + 1,
                    })

            if not candidates:
                break

            # Select top beam_width candidates
            candidates.sort(key=lambda x: x["score"], reverse=True)
            beam = candidates[:beam_width]

            for b in beam:
                visited.add(b["id"])
                all_discoveries.append(b)

    # Sort all discoveries by score
    all_discoveries.sort(key=lambda x: x["score"], reverse=True)
    return all_discoveries[:limit]


# ============================================================================
# 5. Concept Neighborhood
# ============================================================================

def concept_neighborhood(
    namespace: str,
    concept_name: str,
    depth: int = 2,
    limit: int = 30,
) -> dict:
    """Get the neighborhood of a concept up to N hops."""
    driver = get_driver()

    with driver.session() as session:
        # Find the concept
        result = session.run(
            """
            MATCH (c:Concept {namespace: $ns})
            WHERE toLower(c.name) = toLower($name)
            RETURN c.id AS id, c.name AS name, c.community_id AS community,
                   c.mention_count AS mentions, c.rns_score AS rns
            """,
            ns=namespace, name=concept_name,
        )
        center = result.single()
        if not center:
            return {"center": None, "nodes": [], "edges": []}

        # Get neighborhood
        result = session.run(
            """
            MATCH path = (c:Concept {id: $cid})-[*1..""" + str(depth) + """]->(other:Concept {namespace: $ns})
            WITH other, min(length(path)) AS dist
            RETURN other.id AS id, other.name AS name,
                   other.community_id AS community,
                   other.mention_count AS mentions,
                   other.rns_score AS rns,
                   dist
            ORDER BY dist, other.rns_score DESC
            LIMIT $limit
            """,
            cid=center["id"], ns=namespace, limit=limit,
        )
        nodes = [dict(r) for r in result]

        # Get edges between all these nodes
        all_ids = [center["id"]] + [n["id"] for n in nodes]
        result = session.run(
            """
            MATCH (a:Concept)-[r]->(b:Concept)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id AS source, b.id AS target, type(r) AS rel_type,
                   properties(r) AS props
            """,
            ids=all_ids,
        )
        edges = [dict(r) for r in result]

    return {
        "center": dict(center),
        "nodes": nodes,
        "edges": edges,
    }
