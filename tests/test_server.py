"""Unit tests for server models and graph layout logic (no Neo4j required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import (
    AskRequest, SourceNode, TraceInfo, DocGrade,
    FlowNode, FlowEdge, GraphNodeData, GraphData,
    AskResponse, GraphStatsResponse, HealthResponse,
)
from server.main import _layout_nodes


# --- Request/Response models ---

def test_ask_request_valid():
    r = AskRequest(question="What does X do?")
    assert r.question == "What does X do?"
    assert r.top_k == 8

def test_ask_request_custom_topk():
    r = AskRequest(question="test", top_k=5)
    assert r.top_k == 5

def test_ask_request_too_short():
    import pytest
    with pytest.raises(Exception):
        AskRequest(question="Hi")

def test_source_node_model():
    s = SourceNode(node_id="abc", label="Function", name="detect", score=0.85, source="vector")
    assert s.label == "Function"
    assert s.score == 0.85

def test_trace_info_defaults():
    t = TraceInfo(
        route="simple", iterations=1, query_rewrites=[],
        correction_triggered=False, hallucination_check="grounded",
        answer_check="useful", doc_grades=[], step_back_question="",
        sub_questions=[],
    )
    assert t.route == "simple"
    assert t.correction_triggered is False

def test_health_response():
    h = HealthResponse(status="ok", neo4j=True, tracing_enabled=False)
    assert h.status == "ok"
    assert h.neo4j is True

def test_graph_stats_response():
    gs = GraphStatsResponse(
        node_counts={"File": 5, "Class": 3},
        relationship_counts={"CONTAINS": 10},
        total_nodes=8,
        total_relationships=10,
    )
    assert gs.total_nodes == 8

def test_graph_data_empty():
    g = GraphData(nodes=[], edges=[])
    assert g.nodes == []
    assert g.edges == []


# --- Graph layout ---

def _make_raw_nodes(specs: list[tuple[str, str]]) -> list[dict]:
    return [{"id": f"id_{i}", "label": label, "name": name} for i, (label, name) in enumerate(specs)]

def test_layout_nodes_returns_flow_nodes():
    raw = _make_raw_nodes([("File", "main.py"), ("Class", "Detector"), ("Function", "fit")])
    source_ids = {"id_0", "id_1"}
    flow_nodes = _layout_nodes(raw, source_ids)
    assert len(flow_nodes) == 3
    assert all(isinstance(n, FlowNode) for n in flow_nodes)

def test_layout_nodes_marks_sources():
    raw = _make_raw_nodes([("File", "main.py"), ("Function", "predict")])
    source_ids = {"id_0"}
    flow_nodes = _layout_nodes(raw, source_ids)
    by_id = {n.id: n for n in flow_nodes}
    assert by_id["id_0"].data.isSource is True
    assert by_id["id_1"].data.isSource is False

def test_layout_nodes_groups_by_type():
    raw = _make_raw_nodes([
        ("Function", "fit"), ("Function", "predict"),
        ("Class", "Detector"),
        ("File", "model.py"),
    ])
    flow_nodes = _layout_nodes(raw, set())
    # File column (x=0), Class column (x=260), Function column (x=520)
    file_nodes = [n for n in flow_nodes if n.data.nodeType == "File"]
    class_nodes = [n for n in flow_nodes if n.data.nodeType == "Class"]
    func_nodes = [n for n in flow_nodes if n.data.nodeType == "Function"]
    assert len(file_nodes) == 1
    assert len(class_nodes) == 1
    assert len(func_nodes) == 2
    # Functions in same column should have different y positions
    func_ys = [n.position["y"] for n in func_nodes]
    assert len(set(func_ys)) == 2

def test_layout_nodes_positions_are_numbers():
    raw = _make_raw_nodes([("Concept", "AnomalyDetection")])
    flow_nodes = _layout_nodes(raw, set())
    pos = flow_nodes[0].position
    assert isinstance(pos["x"], (int, float))
    assert isinstance(pos["y"], (int, float))

def test_layout_nodes_empty():
    assert _layout_nodes([], set()) == []
