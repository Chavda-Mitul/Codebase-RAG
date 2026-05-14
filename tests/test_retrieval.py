"""Unit tests for retrieval components (no Neo4j required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.models import RetrievedNode, node_dict_to_text, node_dict_to_retrieved
from src.retrieval.bm25_retriever import BM25Index, _tokenize
from src.retrieval.hybrid_retriever import _reciprocal_rank_fusion


# --- model tests ---

def test_node_dict_to_text_function():
    node = {"label": "Function", "name": "detect_anomaly", "docstring": "Detects anomalies", "file_path": "src/model.py"}
    text = node_dict_to_text(node)
    assert "Function" in text
    assert "detect_anomaly" in text
    assert "Detects anomalies" in text


def test_node_dict_to_text_file():
    node = {"label": "File", "path": "src/main.py"}
    text = node_dict_to_text(node)
    assert "File" in text
    assert "src/main.py" in text


def test_node_dict_to_retrieved():
    node = {"id": "abc123", "label": "Class", "name": "AnomalyDetector", "docstring": "Detects stuff"}
    r = node_dict_to_retrieved(node, score=0.9, source="vector")
    assert r.node_id == "abc123"
    assert r.label == "Class"
    assert r.score == 0.9
    assert r.source == "vector"


# --- BM25 tests ---

def test_tokenize():
    tokens = _tokenize("AnomalyDetector fit_model")
    assert "AnomalyDetector".lower() in tokens or "anomalydetector" in tokens
    assert "fit_model" in tokens or "fit" in tokens


def test_bm25_index_search():
    nodes = [
        {"id": "1", "label": "Function", "name": "detect_anomaly", "docstring": "detects invoice anomalies"},
        {"id": "2", "label": "Class", "name": "InvoiceLoader", "docstring": "loads invoice data from CSV"},
        {"id": "3", "label": "Function", "name": "preprocess", "docstring": "preprocesses data features"},
        {"id": "4", "label": "Module", "name": "sklearn", "docstring": ""},
    ]
    index = BM25Index(nodes)
    # "detects anomalies" appears only in node 1 → IDF is high and score > 0
    results = index.search("detects anomalies", top_k=2)
    assert len(results) >= 1
    top_names = [r.name for r in results]
    assert any("anomaly" in n.lower() or "detect" in n.lower() for n in top_names)


def test_bm25_empty_query():
    nodes = [{"id": "1", "label": "Function", "name": "foo", "docstring": "bar"}]
    index = BM25Index(nodes)
    assert index.search("", top_k=5) == []


def test_bm25_no_match_returns_empty():
    nodes = [{"id": "1", "label": "Function", "name": "foo", "docstring": "bar"}]
    index = BM25Index(nodes)
    # xyzzy is not in corpus → score 0 → filtered out
    results = index.search("xyzzy qqqq zzzzz", top_k=5)
    assert results == []


# --- RRF tests ---

def _make_nodes(ids_scores: list[tuple[str, float]], source: str) -> list[RetrievedNode]:
    return [
        RetrievedNode(node_id=nid, label="Function", name=nid, text=nid, score=s, source=source)
        for nid, s in ids_scores
    ]


def test_rrf_merges_lists():
    vec = _make_nodes([("a", 0.9), ("b", 0.8), ("c", 0.6)], "vector")
    bm25 = _make_nodes([("b", 5.0), ("d", 4.0), ("a", 3.0)], "bm25")
    merged = _reciprocal_rank_fusion([vec, bm25])
    ids = [m[0] for m in merged]
    # "b" appears in both lists at rank 1 (bm25) and rank 1 (vec rank 2) — should score high
    assert "b" in ids[:2]


def test_rrf_deduplicates():
    vec = _make_nodes([("a", 0.9), ("a", 0.8)], "vector")  # duplicate
    bm25 = _make_nodes([("a", 5.0)], "bm25")
    merged = _reciprocal_rank_fusion([vec, bm25])
    ids = [m[0] for m in merged]
    assert ids.count("a") == 1
