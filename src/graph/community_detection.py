"""Connected-component community detection over the Neo4j knowledge graph.

Uses union-find (disjoint set union) over all node relationships — requires no
GDS plugin, works on any Neo4j 5.x instance.
"""
from __future__ import annotations
from src.graph.neo4j_client import Neo4jClient


def _union_find(edges: list[dict]) -> dict[str, list[str]]:
    """Return {root_id: [member_ids]} from a list of {src, tgt} dicts."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for edge in edges:
        src, tgt = edge.get("src"), edge.get("tgt")
        if src and tgt:
            union(src, tgt)

    groups: dict[str, list[str]] = {}
    for node_id in parent:
        root = find(node_id)
        groups.setdefault(root, []).append(node_id)

    return groups


def detect_communities(
    client: Neo4jClient,
    min_size: int = 3,
    max_size: int = 50,
) -> list[list[str]]:
    """Detect communities as connected components of the knowledge graph.

    Returns a list of node-ID groups. Each group has between min_size and max_size
    members. Large components (> max_size) are split into chunks of max_size so
    LLM summarization stays tractable.
    """
    edges = client.run_read_query(
        "MATCH (a)-[r]-(b) "
        "WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
        "RETURN a.id AS src, b.id AS tgt "
        "LIMIT 20000"
    )

    groups = _union_find(edges)

    result: list[list[str]] = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        # Split oversized components
        for i in range(0, len(members), max_size):
            chunk = members[i : i + max_size]
            if len(chunk) >= min_size:
                result.append(chunk)

    return result
