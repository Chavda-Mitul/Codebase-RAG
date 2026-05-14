"""Graph traversal retriever: expands seed nodes to their neighbors in Neo4j."""
from __future__ import annotations
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.models import RetrievedNode, node_dict_to_retrieved

# Cypher: given a list of seed node IDs, return 1-hop neighbors
_EXPAND_QUERY = """
UNWIND $ids AS seed_id
MATCH (seed {id: seed_id})-[r]-(neighbor)
WHERE neighbor.id IS NOT NULL
RETURN DISTINCT neighbor, type(r) AS rel_type
LIMIT $limit
"""

# Cypher: get the full context for a specific node (its file, class, siblings)
_CONTEXT_QUERY = """
MATCH (n {id: $node_id})
OPTIONAL MATCH (parent)-[:CONTAINS|DEFINES]->(n)
OPTIONAL MATCH (n)-[:IMPORTS]->(mod)
RETURN n, collect(DISTINCT parent) AS parents, collect(DISTINCT mod) AS modules
"""


def graph_expand(
    seed_ids: list[str],
    client: Neo4jClient,
    limit: int = 20,
) -> list[RetrievedNode]:
    """Expand seed node IDs to their graph neighbors for richer context."""
    if not seed_ids:
        return []

    with client.driver.session() as session:
        result = session.run(_EXPAND_QUERY, ids=seed_ids, limit=limit)
        rows = list(result)

    nodes: list[RetrievedNode] = []
    seen_ids: set[str] = set(seed_ids)

    for row in rows:
        neighbor = dict(row["neighbor"])
        nid = neighbor.get("id", "")
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        retrieved = node_dict_to_retrieved(neighbor, score=0.5, source="graph")
        retrieved.metadata["via_rel"] = row["rel_type"]
        nodes.append(retrieved)

    return nodes


def get_node_context(node_id: str, client: Neo4jClient) -> list[RetrievedNode]:
    """Return the immediate structural context for a node (parent file/class, imports)."""
    with client.driver.session() as session:
        result = session.run(_CONTEXT_QUERY, node_id=node_id)
        row = result.single()

    if not row:
        return []

    nodes: list[RetrievedNode] = []
    for parent in row["parents"]:
        if parent:
            nodes.append(node_dict_to_retrieved(dict(parent), score=0.4, source="graph"))
    for mod in row["modules"]:
        if mod:
            nodes.append(node_dict_to_retrieved(dict(mod), score=0.3, source="graph"))
    return nodes
