"""Unit tests for the cross-encoder reranker."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.retrieval.models import RetrievedNode
from src.retrieval.reranker import rerank


def _make_nodes(n: int) -> list[RetrievedNode]:
    return [
        RetrievedNode(
            node_id=f"node_{i}",
            label="Function",
            name=f"func_{i}",
            text=f"text content for node {i}",
            score=float(n - i),  # descending initial scores
            source="hybrid",
        )
        for i in range(n)
    ]


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_replaces_scores(mock_get_encoder):
    """Cross-encoder scores should replace original RRF scores."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.9, 0.1, 0.5]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(3)
    original_scores = [n.score for n in nodes]

    result = rerank("test query", nodes)

    # Original nodes should be unchanged (model_copy used)
    assert [n.score for n in nodes] == original_scores

    # Result scores come from encoder.predict
    result_scores = [n.score for n in result]
    assert result_scores == [0.9, 0.5, 0.1]  # sorted descending


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_sorts_descending(mock_get_encoder):
    """Results must be sorted by cross-encoder score descending."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.2, 0.8, 0.5, 0.95]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(4)
    result = rerank("query", nodes)

    scores = [n.score for n in result]
    assert scores == sorted(scores, reverse=True)


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_top_k_trims(mock_get_encoder):
    """top_k parameter should limit returned results."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(5)
    result = rerank("query", nodes, top_k=3)

    assert len(result) == 3
    assert result[0].score == 0.9


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_no_top_k_returns_all(mock_get_encoder):
    """Without top_k, all candidates are returned."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.3, 0.7, 0.1]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(3)
    result = rerank("query", nodes)

    assert len(result) == 3


def test_rerank_empty_input():
    """Empty candidate list should return empty list without calling encoder."""
    result = rerank("query", [])
    assert result == []


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_passes_correct_pairs(mock_get_encoder):
    """Encoder.predict should receive (query, text) pairs in order."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.5, 0.5]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(2)
    rerank("my query", nodes)

    call_args = encoder.predict.call_args[0][0]
    assert call_args == [("my query", nodes[0].text), ("my query", nodes[1].text)]


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_preserves_node_ids(mock_get_encoder):
    """Node IDs must be preserved through reranking."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.1, 0.9, 0.5]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(3)
    result = rerank("query", nodes)

    result_ids = {n.node_id for n in result}
    original_ids = {n.node_id for n in nodes}
    assert result_ids == original_ids


@patch("src.retrieval.reranker._get_cross_encoder")
def test_rerank_does_not_mutate_input(mock_get_encoder):
    """Input nodes should not be mutated (model_copy used)."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.1, 0.9]
    mock_get_encoder.return_value = encoder

    nodes = _make_nodes(2)
    original_scores = [n.score for n in nodes]

    rerank("query", nodes)

    assert [n.score for n in nodes] == original_scores
