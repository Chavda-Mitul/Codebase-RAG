"""BM25 keyword retriever over the Neo4j node corpus."""
from __future__ import annotations
import re
from functools import lru_cache
from rank_bm25 import BM25Okapi
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.models import RetrievedNode, node_dict_to_retrieved

_LOAD_QUERY = """
MATCH (n)
WHERE n.id IS NOT NULL AND (n.name IS NOT NULL OR n.path IS NOT NULL)
RETURN n
LIMIT 5000
"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class BM25Index:
    def __init__(self, nodes: list[dict]):
        self._nodes = nodes
        corpus = [_tokenize(node_dict_to_retrieved(n).text) for n in nodes]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedNode]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            node = self._nodes[idx]
            r = node_dict_to_retrieved(node, score=float(scores[idx]), source="bm25")
            results.append(r)
        return results


_index_cache: BM25Index | None = None


def build_bm25_index(client: Neo4jClient) -> BM25Index:
    """Load all nodes from Neo4j and build a BM25 index."""
    global _index_cache
    with client.driver.session() as session:
        result = session.run(_LOAD_QUERY)
        nodes = [dict(row["n"]) for row in result]
    _index_cache = BM25Index(nodes)
    return _index_cache


def get_bm25_index(client: Neo4jClient) -> BM25Index:
    """Return cached index, building it if needed."""
    global _index_cache
    if _index_cache is None:
        return build_bm25_index(client)
    return _index_cache


def bm25_search(query: str, client: Neo4jClient, top_k: int = 10) -> list[RetrievedNode]:
    index = get_bm25_index(client)
    return index.search(query, top_k=top_k)
