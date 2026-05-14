"""Vector similarity search using Neo4j vector indexes."""
from __future__ import annotations
from src.graph.neo4j_client import Neo4jClient
from src.embeddings.embedder import embed_text
from src.retrieval.models import RetrievedNode, node_dict_to_retrieved

_SEARCHABLE_LABELS = ["File", "Class", "Function", "Concept"]


def vector_search(
    query: str,
    client: Neo4jClient,
    top_k: int = 5,
    labels: list[str] | None = None,
) -> list[RetrievedNode]:
    """Embed query and search across all node label indexes. Returns merged top-K results."""
    query_embedding = embed_text(query)
    labels_to_search = labels or _SEARCHABLE_LABELS

    all_results: list[RetrievedNode] = []
    for label in labels_to_search:
        try:
            raw = client.vector_search(query_embedding, label, top_k=top_k)
        except Exception:
            continue
        for r in raw:
            all_results.append(
                node_dict_to_retrieved(r["node"], score=float(r["score"]), source="vector")
            )

    # Deduplicate by node_id, keep highest score
    seen: dict[str, RetrievedNode] = {}
    for r in all_results:
        if r.node_id not in seen or r.score > seen[r.node_id].score:
            seen[r.node_id] = r

    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k * len(labels_to_search)]
