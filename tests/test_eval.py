"""Unit tests for eval metrics and golden QA data (no LLM calls — all mocked)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock
from src.eval.golden_qa import GOLDEN_QA_PAIRS, get_by_difficulty, get_by_category
from src.eval.metrics import (
    EvalScores,
    FaithfulnessResult, ClaimVerdict,
    AnswerRelevanceResult,
    ContextRecallResult,
    ContextPrecisionResult,
    compute_faithfulness,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
)
from src.eval.runner import EvalReport, EvalResult


# --- Golden QA data ---

def test_golden_qa_has_entries():
    assert len(GOLDEN_QA_PAIRS) >= 15

def test_golden_qa_all_have_ids():
    ids = [q.id for q in GOLDEN_QA_PAIRS]
    assert len(ids) == len(set(ids)), "Duplicate QA IDs found"

def test_golden_qa_difficulties():
    easy = get_by_difficulty("easy")
    medium = get_by_difficulty("medium")
    hard = get_by_difficulty("hard")
    assert len(easy) >= 3
    assert len(medium) >= 3
    assert len(hard) >= 3

def test_golden_qa_categories():
    factual = get_by_category("factual")
    structural = get_by_category("structural")
    architectural = get_by_category("architectural")
    assert len(factual) >= 3
    assert len(structural) >= 2
    assert len(architectural) >= 2

def test_golden_qa_routes():
    routes = {q.expected_route for q in GOLDEN_QA_PAIRS}
    assert "simple" in routes
    assert "complex" in routes
    assert "conceptual" in routes


# --- EvalScores model ---

def test_eval_scores_average():
    s = EvalScores(faithfulness=0.8, answer_relevance=0.6, context_recall=0.7, context_precision=0.9)
    assert s.average() == round((0.8 + 0.6 + 0.7 + 0.9) / 4, 3)

def test_eval_scores_default_zeros():
    s = EvalScores()
    assert s.faithfulness == 0.0
    assert s.average() == 0.0


# --- Faithfulness (mocked LLM) ---

def _mock_llm_for_faithfulness(claim_list, verdicts):
    """Build a mock LLM that returns structured outputs for faithfulness grading."""
    mock = MagicMock()
    call_count = [0]

    class FakeClaimList:
        claims = claim_list

    class FakeBool:
        def __init__(self, val):
            self.supported = val

    invoke_results = [FakeClaimList()] + [FakeBool(v) for v in verdicts]

    def mock_invoke(messages):
        result = invoke_results[call_count[0]]
        call_count[0] += 1
        return result

    structured = MagicMock()
    structured.invoke = mock_invoke
    mock.with_structured_output.return_value = structured
    return mock


def test_faithfulness_all_supported():
    mock_llm = _mock_llm_for_faithfulness(
        claim_list=["Claim A", "Claim B"],
        verdicts=[True, True],
    )
    result = compute_faithfulness("The answer uses IsolationForest.", "Context with IsolationForest details.", llm=mock_llm)
    assert result.score == 1.0
    assert result.supported == 2
    assert result.total == 2


def test_faithfulness_partial():
    mock_llm = _mock_llm_for_faithfulness(
        claim_list=["Claim A", "Claim B", "Claim C"],
        verdicts=[True, False, True],
    )
    result = compute_faithfulness("answer", "context", llm=mock_llm)
    assert result.score == round(2 / 3, 3)
    assert result.supported == 2


def test_faithfulness_none_supported():
    mock_llm = _mock_llm_for_faithfulness(
        claim_list=["Claim A"],
        verdicts=[False],
    )
    result = compute_faithfulness("answer", "context", llm=mock_llm)
    assert result.score == 0.0


# --- Answer Relevance (mocked LLM) ---

def test_answer_relevance_high_score():
    class FakeScore:
        score = 0.95
        reason = "Fully answers the question"

    mock = MagicMock()
    mock.with_structured_output.return_value.invoke.return_value = FakeScore()
    result = compute_answer_relevance("What does X do?", "X does Y by calling Z.", llm=mock)
    assert result.score == 0.95

def test_answer_relevance_low_score():
    class FakeScore:
        score = 0.1
        reason = "Vague answer"

    mock = MagicMock()
    mock.with_structured_output.return_value.invoke.return_value = FakeScore()
    result = compute_answer_relevance("Question?", "I don't know.", llm=mock)
    assert result.score == 0.1


# --- Context Precision (mocked LLM) ---

def test_context_precision_all_relevant():
    class FakeRel:
        relevant = True

    mock = MagicMock()
    mock.with_structured_output.return_value.invoke.return_value = FakeRel()
    result = compute_context_precision("question", ["doc1", "doc2", "doc3"], llm=mock)
    assert result.score == 1.0
    assert result.relevant_count == 3

def test_context_precision_none_relevant():
    class FakeRel:
        relevant = False

    mock = MagicMock()
    mock.with_structured_output.return_value.invoke.return_value = FakeRel()
    result = compute_context_precision("question", ["doc1", "doc2"], llm=mock)
    assert result.score == 0.0
    assert result.relevant_count == 0

def test_context_precision_empty_docs():
    result = compute_context_precision("question", [], llm=MagicMock())
    assert result.score == 0.0


# --- Context Recall (mocked LLM) ---

def test_context_recall_all_attributed():
    call_count = [0]

    class FakeStatements:
        statements = ["stmt A", "stmt B"]

    class FakeAttribution:
        attributed = True

    results = [FakeStatements(), FakeAttribution(), FakeAttribution()]

    mock = MagicMock()
    def side_effect(messages):
        r = results[call_count[0]]
        call_count[0] += 1
        return r
    mock.with_structured_output.return_value.invoke = side_effect
    result = compute_context_recall("ground truth text", "context text", llm=mock)
    assert result.score == 1.0
    assert result.attributed == 2


# --- EvalReport ---

def _make_result(score_val: float) -> EvalResult:
    return EvalResult(
        qa_id="test",
        question="q",
        pipeline="naive",
        answer="a",
        ground_truth="gt",
        context="ctx",
        scores=EvalScores(faithfulness=score_val, answer_relevance=score_val,
                          context_recall=score_val, context_precision=score_val),
        latency_ms=100.0,
    )

def test_eval_report_avg_scores():
    report = EvalReport(pipeline="naive")
    report.results = [_make_result(0.8), _make_result(0.6)]
    avg = report.avg_scores()
    assert avg.faithfulness == 0.7
    assert avg.average() == 0.7

def test_eval_report_to_json():
    report = EvalReport(pipeline="crag")
    report.results = [_make_result(0.9)]
    json_str = report.to_json()
    assert "crag" in json_str
    assert "faithfulness" in json_str

def test_eval_report_empty():
    report = EvalReport(pipeline="naive")
    avg = report.avg_scores()
    assert avg.average() == 0.0
