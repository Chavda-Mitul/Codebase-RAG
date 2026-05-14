"""Hybrid retriever: merges vector + BM25 + graph results using Reciprocal Rank Fusion."""
from __future__ import annotations
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.models import RetrievedNode, node_dict_to_retrieved
from src.retrieval.vector_retriever import vector_search
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.graph_retriever import graph_expand
from src.embeddings.embedder import embed_text

_RRF_K = 60  # standard RRF constant


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedNode]],
) -> list[tuple[str, float]]:
    """Apply RRF to multiple ranked lists. Returns (node_id, rrf_score) sorted desc."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, node in enumerate(ranked):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    client: Neo4jClient,
    top_k: int = 8,
    use_graph_expansion: bool = True,
    use_reranker: bool | None = None,
) -> list[RetrievedNode]:
    """
    Run vector + BM25 retrieval, merge with RRF, optionally expand via graph traversal,
    then optionally rerank with a cross-encoder. Returns top_k nodes.

    use_reranker defaults to settings.use_reranker when not explicitly set.
    """
    vec_results = vector_search(query, client, top_k=top_k)
    bm25_results = bm25_search(query, client, top_k=top_k)

    rrf_scores = _reciprocal_rank_fusion([vec_results, bm25_results])

    node_map: dict[str, RetrievedNode] = {}
    for node in vec_results + bm25_results:
        if node.node_id not in node_map:
            node_map[node.node_id] = node

    merged: list[RetrievedNode] = []
    for node_id, rrf_score in rrf_scores[:top_k]:
        if node_id in node_map:
            node = node_map[node_id].model_copy()
            node.score = rrf_score
            node.source = "hybrid"
            merged.append(node)

    if use_graph_expansion and vec_results:
        seed_ids = [n.node_id for n in vec_results[:3]]
        expanded = graph_expand(seed_ids, client, limit=top_k)
        seen_ids = {n.node_id for n in merged}
        for node in expanded:
            if node.node_id not in seen_ids:
                merged.append(node)
                seen_ids.add(node.node_id)

    # Cross-encoder reranking for precision improvement
    should_rerank = use_reranker if use_reranker is not None else settings.use_reranker
    if should_rerank and merged:
        from src.retrieval.reranker import rerank
        merged = rerank(query, merged, model_name=settings.reranker_model, top_k=top_k)
    else:
        merged = merged[:top_k]

    return merged


def community_retrieve(
    query: str,
    client: Neo4jClient,
    top_k: int = 3,
) -> list[RetrievedNode]:
    """Vector search restricted to Community nodes (high-level module summaries)."""
    query_embedding = embed_text(query)
    try:
        raw = client.vector_search(query_embedding, "Community", top_k=top_k)
    except Exception:
        return []
    results = []
    for r in raw:
        node = dict(r["node"])
        node["label"] = "Community"
        retrieved = node_dict_to_retrieved(node, score=float(r["score"]), source="community")
        if node.get("summary"):
            retrieved = retrieved.model_copy(
                update={"text": f"[Community] {node.get('name', '')} | {node['summary']}"}
            )
        results.append(retrieved)
    return results
