"""Unit tests for the agentic pipeline: memory, tools, graph."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.agents.memory import ConversationMemory, get_memory, clear_memory
from src.agents.state import AgentState
from src.agents.agent_graph import decide_after_planning


# ---------------------------------------------------------------------------
# ConversationMemory tests
# ---------------------------------------------------------------------------

def test_memory_add_and_retrieve():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("What is X?", "X is Y.")
    history = mem.get_history_text()
    assert "What is X?" in history
    assert "X is Y." in history


def test_memory_empty_history_is_empty_string():
    mem = ConversationMemory()
    assert mem.get_history_text() == ""


def test_memory_sliding_window():
    mem = ConversationMemory(max_turns=3)
    for i in range(5):
        mem.add_turn(f"Q{i}", f"A{i}")
    history = mem.get_history_text()
    # Only last 3 turns should be present
    assert "Q4" in history
    assert "Q3" in history
    assert "Q2" in history
    assert "Q1" not in history
    assert "Q0" not in history


def test_memory_len():
    mem = ConversationMemory(max_turns=10)
    assert len(mem) == 0
    mem.add_turn("q", "a")
    assert len(mem) == 1


def test_memory_clear():
    mem = ConversationMemory()
    mem.add_turn("q", "a")
    mem.clear()
    assert len(mem) == 0
    assert mem.get_history_text() == ""


def test_memory_history_format_contains_turn_numbers():
    mem = ConversationMemory()
    mem.add_turn("Q1", "A1")
    mem.add_turn("Q2", "A2")
    history = mem.get_history_text()
    assert "Turn 1" in history
    assert "Turn 2" in history


# ---------------------------------------------------------------------------
# get_memory / clear_memory tests
# ---------------------------------------------------------------------------

def test_get_memory_creates_new():
    clear_memory("test-session-new")
    mem = get_memory("test-session-new")
    assert isinstance(mem, ConversationMemory)
    assert len(mem) == 0


def test_get_memory_returns_same_instance():
    clear_memory("test-session-same")
    m1 = get_memory("test-session-same")
    m1.add_turn("q", "a")
    m2 = get_memory("test-session-same")
    assert len(m2) == 1  # same object


def test_clear_memory_removes_session():
    get_memory("test-clear-me").add_turn("q", "a")
    clear_memory("test-clear-me")
    fresh = get_memory("test-clear-me")
    assert len(fresh) == 0


# ---------------------------------------------------------------------------
# Tool: run_code_snippet
# ---------------------------------------------------------------------------

def test_run_code_snippet_captures_output():
    from src.agents.tools import run_code_snippet
    result = run_code_snippet('print("hello")')
    assert "hello" in result


def test_run_code_snippet_captures_stderr():
    from src.agents.tools import run_code_snippet
    result = run_code_snippet('import sys; sys.stderr.write("err")')
    assert "err" in result


def test_run_code_snippet_timeout():
    from src.agents.tools import run_code_snippet
    result = run_code_snippet('import time; time.sleep(10)', timeout=1)
    assert "timeout" in result.lower() or "timed out" in result.lower()


def test_run_code_snippet_no_output():
    from src.agents.tools import run_code_snippet
    result = run_code_snippet('x = 1 + 1')
    assert result == "(no output)"


# ---------------------------------------------------------------------------
# Tool: _mermaid_id helper
# ---------------------------------------------------------------------------

def test_mermaid_id_removes_special_chars():
    from src.agents.tools import _mermaid_id
    assert _mermaid_id("My Class!") == "My_Class_"


def test_mermaid_id_truncates():
    from src.agents.tools import _mermaid_id
    long_name = "a" * 50
    assert len(_mermaid_id(long_name)) <= 30


# ---------------------------------------------------------------------------
# Tool: search_codebase (mocked retriever)
# ---------------------------------------------------------------------------

@patch("src.agents.tools.hybrid_retrieve")
@patch("src.agents.tools._format_context")
def test_search_codebase_returns_text_and_nodes(mock_fmt, mock_retrieve):
    from src.agents.tools import search_codebase
    from src.retrieval.models import RetrievedNode

    node = RetrievedNode(node_id="n1", label="Function", name="foo", text="def foo(): pass")
    mock_retrieve.return_value = [node]
    mock_fmt.return_value = "formatted context"

    client = MagicMock()
    text, nodes = search_codebase("find foo", client)

    assert text == "formatted context"
    assert nodes == [node]
    mock_retrieve.assert_called_once_with("find foo", client, top_k=5)


# ---------------------------------------------------------------------------
# Agent graph: conditional edge
# ---------------------------------------------------------------------------

def test_decide_after_planning_with_tools():
    state: AgentState = {
        "question": "q", "session_id": "s", "history": "",
        "tool_plan": [{"tool_name": "search_codebase", "argument": "foo"}],
        "tool_calls": [], "documents": [], "answer": "", "iteration": 0,
    }
    assert decide_after_planning(state) == "execute_tools"


def test_decide_after_planning_no_tools():
    state: AgentState = {
        "question": "q", "session_id": "s", "history": "",
        "tool_plan": [],
        "tool_calls": [], "documents": [], "answer": "", "iteration": 0,
    }
    assert decide_after_planning(state) == "generate_with_tools"


# ---------------------------------------------------------------------------
# Agent graph: node factory (smoke test with mocks)
# ---------------------------------------------------------------------------

@patch("src.agents.agent_graph.ChatGroq")
def test_make_agent_nodes_returns_all_nodes(mock_groq):
    from src.agents.agent_graph import make_agent_nodes
    client = MagicMock()
    nodes = make_agent_nodes(client)
    assert set(nodes.keys()) == {
        "inject_memory", "plan_tools", "execute_tools",
        "generate_with_tools", "save_memory",
    }


@patch("src.agents.agent_graph.ChatGroq")
def test_inject_memory_loads_history(mock_groq):
    from src.agents.agent_graph import make_agent_nodes
    from src.agents.memory import get_memory, clear_memory

    session = "test-inject-memory"
    clear_memory(session)
    get_memory(session).add_turn("prev Q", "prev A")

    client = MagicMock()
    nodes = make_agent_nodes(client)

    state: AgentState = {
        "question": "new Q", "session_id": session, "history": "",
        "tool_plan": [], "tool_calls": [], "documents": [], "answer": "", "iteration": 0,
    }
    result = nodes["inject_memory"](state)
    assert "prev Q" in result["history"]
    assert "prev A" in result["history"]
    clear_memory(session)
