"""Unit tests for CRAG graders and graph routing logic (no Neo4j, no LLM calls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crag.state import CRAGState
from src.crag.nodes import decide_after_grading, decide_after_hallucination, decide_after_answer_check
from src.retrieval.models import RetrievedNode


def _node(nid: str) -> RetrievedNode:
    return RetrievedNode(node_id=nid, label="Function", name=nid, text=f"text for {nid}", score=0.8)


def _base_state(**kwargs) -> CRAGState:
    base: CRAGState = {
        "question": "What does AnomalyDetector do?",
        "documents": [],
        "filtered_documents": [],
        "answer": "",
        "query_rewrites": [],
        "correction_triggered": False,
        "doc_grades": [],
        "hallucination_check": "",
        "answer_check": "",
        "iteration": 1,
    }
    base.update(kwargs)
    return base


# --- decide_after_grading ---

def test_decide_after_grading_no_correction_goes_to_generate():
    state = _base_state(correction_triggered=False)
    assert decide_after_grading(state) == "generate"


def test_decide_after_grading_correction_under_limit_goes_to_rewrite():
    state = _base_state(correction_triggered=True, iteration=1)
    assert decide_after_grading(state) == "rewrite"


def test_decide_after_grading_correction_at_max_goes_to_generate():
    # At max iterations, give up rewriting and generate anyway
    state = _base_state(correction_triggered=True, iteration=3)
    assert decide_after_grading(state) == "generate"


# --- decide_after_hallucination ---

def test_decide_grounded_goes_to_answer_check():
    state = _base_state(hallucination_check="grounded", iteration=1)
    assert decide_after_hallucination(state) == "check_answer_quality"


def test_decide_hallucinated_under_limit_retries_generate():
    state = _base_state(hallucination_check="hallucinated", iteration=1)
    assert decide_after_hallucination(state) == "generate"


def test_decide_hallucinated_at_max_goes_to_answer_check():
    # Accept imperfect answer at max iterations
    state = _base_state(hallucination_check="hallucinated", iteration=3)
    assert decide_after_hallucination(state) == "check_answer_quality"


# --- decide_after_answer_check ---

def test_decide_useful_ends():
    state = _base_state(answer_check="useful", iteration=1)
    assert decide_after_answer_check(state) == "end"


def test_decide_not_useful_under_limit_rewrites():
    state = _base_state(answer_check="not useful", iteration=1)
    assert decide_after_answer_check(state) == "rewrite"


def test_decide_not_useful_at_max_ends():
    state = _base_state(answer_check="not useful", iteration=3)
    assert decide_after_answer_check(state) == "end"


# --- state structure ---

def test_state_has_expected_keys():
    state = _base_state()
    required = ["question", "documents", "filtered_documents", "answer",
                "query_rewrites", "correction_triggered", "doc_grades",
                "hallucination_check", "answer_check", "iteration"]
    for key in required:
        assert key in state, f"Missing key: {key}"


def test_filtered_documents_subset():
    docs = [_node("a"), _node("b"), _node("c")]
    filtered = [_node("a"), _node("c")]
    state = _base_state(documents=docs, filtered_documents=filtered)
    assert len(state["filtered_documents"]) == 2
    assert all(d.node_id in ("a", "c") for d in state["filtered_documents"])
