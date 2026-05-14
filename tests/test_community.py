"""Unit tests for community detection and summarization."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.graph.community_detection import _union_find, detect_communities
from src.graph.schema import CommunityNode


# ---------------------------------------------------------------------------
# union-find tests
# ---------------------------------------------------------------------------

def test_union_find_simple_chain():
    edges = [{"src": "a", "tgt": "b"}, {"src": "b", "tgt": "c"}]
    groups = _union_find(edges)
    # All three should be in one group
    all_members = set()
    for members in groups.values():
        all_members.update(members)
    assert {"a", "b", "c"} <= all_members


def test_union_find_two_components():
    edges = [
        {"src": "a", "tgt": "b"},
        {"src": "c", "tgt": "d"},
    ]
    groups = _union_find(edges)
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [2, 2]


def test_union_find_ignores_null_edges():
    edges = [{"src": None, "tgt": "b"}, {"src": "a", "tgt": None}]
    groups = _union_find(edges)
    # No valid edges → either empty or singleton groups
    all_members = set()
    for members in groups.values():
        all_members.update(members)
    assert "a" not in all_members or "b" not in all_members


def test_union_find_self_loop():
    edges = [{"src": "a", "tgt": "a"}]
    groups = _union_find(edges)
    all_members = {m for g in groups.values() for m in g}
    assert "a" in all_members


def test_union_find_empty():
    groups = _union_find([])
    assert groups == {}


# ---------------------------------------------------------------------------
# detect_communities tests
# ---------------------------------------------------------------------------

def test_detect_communities_filters_min_size():
    client = MagicMock()
    # Two nodes connected (size 2, below min_size=3) + three nodes connected
    client.run_read_query.return_value = [
        {"src": "a", "tgt": "b"},
        {"src": "x", "tgt": "y"}, {"src": "y", "tgt": "z"},
    ]
    communities = detect_communities(client, min_size=3)
    assert len(communities) == 1
    assert set(communities[0]) == {"x", "y", "z"}


def test_detect_communities_splits_large_components():
    client = MagicMock()
    # Create a chain of 6 nodes: a-b-c-d-e-f
    edges = [{"src": chr(97 + i), "tgt": chr(97 + i + 1)} for i in range(5)]
    client.run_read_query.return_value = edges
    communities = detect_communities(client, min_size=2, max_size=3)
    # All 6 nodes in chunks of 3
    assert all(len(c) <= 3 for c in communities)
    all_seen = set(m for c in communities for m in c)
    assert len(all_seen) == 6


def test_detect_communities_returns_list_of_lists():
    client = MagicMock()
    client.run_read_query.return_value = [
        {"src": "a", "tgt": "b"}, {"src": "b", "tgt": "c"},
    ]
    result = detect_communities(client, min_size=3)
    assert isinstance(result, list)
    assert all(isinstance(c, list) for c in result)


def test_detect_communities_empty_graph():
    client = MagicMock()
    client.run_read_query.return_value = []
    assert detect_communities(client) == []


# ---------------------------------------------------------------------------
# community_summarizer tests
# ---------------------------------------------------------------------------

@patch("src.graph.community_summarizer.embed_text")
def test_summarize_community_returns_node(mock_embed):
    from src.graph.community_summarizer import summarize_community

    mock_embed.return_value = [0.1] * 384

    client = MagicMock()
    client.run_read_query.return_value = [
        {"label": "Class", "name": "AnomalyDetector", "docstring": "Detects anomalies", "description": None, "sig": None},
        {"label": "Function", "name": "fit", "docstring": "Fits model", "description": None, "sig": "def fit(self, X)"},
        {"label": "Function", "name": "predict", "docstring": "Makes predictions", "description": None, "sig": "def predict(self, X)"},
    ]

    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content="This module handles anomaly detection using a trained model."),
        MagicMock(content="Anomaly Detection Module"),
    ]

    node = summarize_community(["id1", "id2", "id3"], client, llm)

    assert node is not None
    assert isinstance(node, CommunityNode)
    assert node.label == "Community"
    assert len(node.members) == 3
    assert node.summary != ""
    assert node.name != ""
    assert node.embedding == [0.1] * 384


@patch("src.graph.community_summarizer.embed_text")
def test_summarize_community_empty_returns_none(mock_embed):
    from src.graph.community_summarizer import summarize_community

    client = MagicMock()
    llm = MagicMock()
    result = summarize_community([], client, llm)
    assert result is None


@patch("src.graph.community_summarizer.embed_text")
def test_summarize_community_no_neo4j_rows_returns_none(mock_embed):
    from src.graph.community_summarizer import summarize_community

    client = MagicMock()
    client.run_read_query.return_value = []
    llm = MagicMock()
    result = summarize_community(["id1"], client, llm)
    assert result is None


@patch("src.graph.community_summarizer.embed_text")
def test_summarize_community_llm_failure_uses_fallback(mock_embed):
    from src.graph.community_summarizer import summarize_community

    mock_embed.return_value = [0.0] * 384

    client = MagicMock()
    client.run_read_query.return_value = [
        {"label": "Function", "name": "foo", "docstring": None, "description": None, "sig": None},
        {"label": "Function", "name": "bar", "docstring": None, "description": None, "sig": None},
        {"label": "Function", "name": "baz", "docstring": None, "description": None, "sig": None},
    ]
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM unavailable")

    node = summarize_community(["id1", "id2", "id3"], client, llm)
    assert node is not None
    assert "3" in node.summary or "group" in node.summary.lower()


# ---------------------------------------------------------------------------
# community_id determinism
# ---------------------------------------------------------------------------

def test_community_id_is_deterministic():
    from src.graph.community_summarizer import _community_id
    ids = ["node_a", "node_b", "node_c"]
    assert _community_id(ids) == _community_id(ids)
    assert _community_id(ids) == _community_id(list(reversed(ids)))


def test_community_id_differs_for_different_members():
    from src.graph.community_summarizer import _community_id
    assert _community_id(["a", "b"]) != _community_id(["a", "c"])


# ---------------------------------------------------------------------------
# community_retrieve tests
# ---------------------------------------------------------------------------

@patch("src.retrieval.hybrid_retriever.embed_text")
def test_community_retrieve_returns_empty_on_neo4j_failure(mock_embed):
    from src.retrieval.hybrid_retriever import community_retrieve

    mock_embed.return_value = [0.0] * 384
    client = MagicMock()
    client.vector_search.side_effect = Exception("index not found")

    result = community_retrieve("test query", client)
    assert result == []


@patch("src.retrieval.hybrid_retriever.embed_text")
def test_community_retrieve_uses_summary_as_text(mock_embed):
    from src.retrieval.hybrid_retriever import community_retrieve

    mock_embed.return_value = [0.0] * 384
    client = MagicMock()
    client.vector_search.return_value = [{
        "node": {
            "id": "community_abc",
            "name": "Auth Module",
            "summary": "Handles authentication and authorization.",
            "label": "Community",
        },
        "score": 0.85,
    }]

    result = community_retrieve("authentication", client, top_k=1)
    assert len(result) == 1
    assert "Handles authentication" in result[0].text
    assert result[0].source == "community"
    assert result[0].score == pytest.approx(0.85)
