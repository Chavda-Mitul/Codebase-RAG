"""Unit tests for routing components (no LLM, no Neo4j calls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
from src.routing.router import RouteDecision
from src.routing.decomposer import DecomposedQuestions
from src.routing.pipeline import RoutedAnswer, _synthesize
from src.retrieval.models import RetrievedNode


# --- RouteDecision model ---

def test_route_decision_simple():
    d = RouteDecision(route="simple", reason="direct lookup")
    assert d.route == "simple"

def test_route_decision_complex():
    d = RouteDecision(route="complex", reason="multi-hop")
    assert d.route == "complex"

def test_route_decision_conceptual():
    d = RouteDecision(route="conceptual", reason="broad architecture question")
    assert d.route == "conceptual"

def test_route_decision_invalid():
    import pytest
    with pytest.raises(Exception):
        RouteDecision(route="unknown", reason="bad")


# --- DecomposedQuestions model ---

def test_decomposed_questions_model():
    d = DecomposedQuestions(
        sub_questions=["What classes exist?", "What are the imports?"],
        synthesis_hint="Combine class and import info",
    )
    assert len(d.sub_questions) == 2
    assert "Combine" in d.synthesis_hint

def test_decomposed_questions_min_2():
    import pytest
    with pytest.raises(Exception):
        DecomposedQuestions(sub_questions=["only one"], synthesis_hint="x")

def test_decomposed_questions_max_4():
    import pytest
    with pytest.raises(Exception):
        DecomposedQuestions(
            sub_questions=["q1", "q2", "q3", "q4", "q5"],
            synthesis_hint="x"
        )


# --- RoutedAnswer dataclass ---

def test_routed_answer_simple():
    r = RoutedAnswer(question="What does X do?", answer="It does Y.", route="simple")
    assert r.route == "simple"
    assert r.sub_questions == []
    assert r.step_back_question == ""

def test_routed_answer_complex_fields():
    r = RoutedAnswer(
        question="Compare A and B",
        answer="A does X, B does Y",
        route="complex",
        sub_questions=["What does A do?", "What does B do?"],
        sub_answers=["A does X", "B does Y"],
    )
    assert len(r.sub_questions) == 2
    assert len(r.sub_answers) == 2

def test_routed_answer_conceptual_fields():
    r = RoutedAnswer(
        question="What is the architecture?",
        answer="It is layered.",
        route="conceptual",
        step_back_question="What are the main components?",
    )
    assert r.step_back_question == "What are the main components?"


# --- _synthesize (mocked LLM) ---

def test_synthesize_calls_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Combined answer.")
    result = _synthesize(
        question="Compare A and B",
        hint="Combine A and B descriptions",
        sub_qs=["What is A?", "What is B?"],
        sub_answers=["A is a loader", "B is a detector"],
        llm=mock_llm,
    )
    assert result == "Combined answer."
    assert mock_llm.invoke.called
    # Verify the prompt included both sub-answers
    call_args = mock_llm.invoke.call_args[0][0]
    full_text = str(call_args)
    assert "A is a loader" in full_text
    assert "B is a detector" in full_text


# --- route_question (mocked LLM) ---

def test_route_question_mock():
    from src.routing.router import route_question
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
        route="simple", reason="direct lookup"
    )
    decision = route_question("What does AnomalyDetector do?", llm=mock_llm)
    assert decision.route == "simple"


# --- decompose_question (mocked LLM) ---

def test_decompose_question_mock():
    from src.routing.decomposer import decompose_question
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = DecomposedQuestions(
        sub_questions=["What classes handle anomaly detection?", "What data does it use?"],
        synthesis_hint="Combine class info with data info",
    )
    result = decompose_question("How does anomaly detection work end to end?", llm=mock_llm)
    assert len(result.sub_questions) == 2


# --- step_back (mocked LLM) ---

def test_step_back_mock():
    from src.routing.step_back import generate_step_back_question
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="What is the overall data processing architecture?")
    result = generate_step_back_question("Why does InvoiceLoader use pandas?", llm=mock_llm)
    assert "architecture" in result.lower() or len(result) > 0
