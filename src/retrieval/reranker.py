"""Cross-encoder reranker for improving retrieval precision after RRF fusion."""
from __future__ import annotations
from functools import lru_cache
from src.retrieval.models import RetrievedNode


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[RetrievedNode],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int | None = None,
) -> list[RetrievedNode]:
    """Score each (query, candidate.text) pair with a cross-encoder and re-sort.

    Replaces each node's .score with the cross-encoder logit score.
    Returns sorted by score desc, optionally trimmed to top_k.
    """
    if not candidates:
        return candidates

    encoder = _get_cross_encoder(model_name)
    pairs = [(query, c.text) for c in candidates]
    scores = encoder.predict(pairs)

    reranked = []
    for node, score in zip(candidates, scores):
        updated = node.model_copy()
        updated.score = float(score)
        reranked.append(updated)

    reranked.sort(key=lambda n: n.score, reverse=True)
    return reranked[:top_k] if top_k is not None else reranked
