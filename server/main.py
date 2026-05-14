"""FastAPI server — bridges the Python RAG pipeline to the Next.js frontend."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.graph.neo4j_client import Neo4jClient
from src.routing.pipeline import answer_with_routing, answer_with_agent
from src.observability.langfuse_handler import is_tracing_enabled
from server.eval_routes import router as eval_router
from server.models import (
    AskRequest, AskResponse, SourceNode, TraceInfo, DocGrade,
    GraphData, FlowNode, FlowEdge, GraphNodeData,
    GraphStatsResponse, HealthResponse,
    AgentRequest, AgentResponse, ToolCallInfo,
)

app = FastAPI(title="Code-RAG API", version="1.0.0")
app.include_router(eval_router, prefix="/eval", tags=["eval"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_NODE_TYPE_ORDER = ["File", "Class", "Function", "Concept", "Module", "Node"]


def _layout_nodes(nodes: list[dict], source_ids: set[str]) -> list[FlowNode]:
    """Assign React Flow positions: group by label into columns."""
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        label = n.get("label", "Node")
        by_type.setdefault(label, []).append(n)

    flow_nodes: list[FlowNode] = []
    col = 0
    for label in _NODE_TYPE_ORDER:
        group = by_type.pop(label, [])
        if not group:
            continue
        for row, node in enumerate(group):
            flow_nodes.append(FlowNode(
                id=node["id"],
                type="codeNode",
                position={"x": col * 260, "y": row * 110},
                data=GraphNodeData(
                    name=node.get("name") or node.get("path") or node["id"],
                    nodeType=label,
                    isSource=node["id"] in source_ids,
                    docstring=(node.get("docstring") or "")[:120],
                    file_path=node.get("path") or node.get("file_path") or "",
                ),
            ))
        col += 1

    # Any remaining types
    for label, group in by_type.items():
        for row, node in enumerate(group):
            flow_nodes.append(FlowNode(
                id=node["id"],
                type="codeNode",
                position={"x": col * 260, "y": row * 110},
                data=GraphNodeData(
                    name=node.get("name") or node["id"],
                    nodeType=label,
                    isSource=node["id"] in source_ids,
                ),
            ))
        col += 1

    return flow_nodes


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        with Neo4jClient() as client:
            result = answer_with_routing(req.question, client, use_reranker=req.use_reranker)

            # Build graph subgraph from sources
            source_ids = {s.node_id for s in result.sources}
            graph_nodes_raw, graph_edges_raw = client.get_subgraph(list(source_ids))

            flow_nodes = _layout_nodes(graph_nodes_raw, source_ids)
            flow_edges = [
                FlowEdge(
                    id=e["id"], source=e["source"], target=e["target"],
                    label=e["label"], animated=e["source"] in source_ids,
                )
                for e in graph_edges_raw
                # Only include edges where both endpoints are in the flow graph
                if any(n.id == e["source"] for n in flow_nodes)
                and any(n.id == e["target"] for n in flow_nodes)
            ]

            # Build trace from CRAG state if available
            crag = result.crag_state or {}
            trace = TraceInfo(
                route=result.route,
                iterations=crag.get("iteration", 0),
                query_rewrites=crag.get("query_rewrites") or [],
                correction_triggered=bool(crag.get("correction_triggered")),
                hallucination_check=crag.get("hallucination_check") or "",
                answer_check=crag.get("answer_check") or "",
                doc_grades=[
                    DocGrade(**g) for g in (crag.get("doc_grades") or [])
                    if all(k in g for k in ("node_id", "name", "score", "reason"))
                ],
                step_back_question=result.step_back_question,
                sub_questions=result.sub_questions,
            )

            sources = [
                SourceNode(
                    node_id=s.node_id, label=s.label, name=s.name,
                    score=round(s.score, 3), source=s.source,
                )
                for s in result.sources[:10]
            ]

            return AskResponse(
                answer=result.answer,
                question=req.question,
                trace=trace,
                sources=sources,
                graph=GraphData(nodes=flow_nodes, edges=flow_edges),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/ask", response_model=AgentResponse)
def agent_ask(req: AgentRequest):
    try:
        with Neo4jClient() as client:
            result = answer_with_agent(req.question, client, req.session_id)

            source_ids = {s.node_id for s in result.sources}
            graph_nodes_raw, graph_edges_raw = client.get_subgraph(list(source_ids))

            flow_nodes = _layout_nodes(graph_nodes_raw, source_ids)
            flow_edges = [
                FlowEdge(
                    id=e["id"], source=e["source"], target=e["target"],
                    label=e["label"], animated=e["source"] in source_ids,
                )
                for e in graph_edges_raw
                if any(n.id == e["source"] for n in flow_nodes)
                and any(n.id == e["target"] for n in flow_nodes)
            ]

            sources = [
                SourceNode(
                    node_id=s.node_id, label=s.label, name=s.name,
                    score=round(s.score, 3), source=s.source,
                )
                for s in result.sources[:10]
            ]

            tool_calls = [
                ToolCallInfo(tool=tc["tool"], input=tc["input"], output=tc["output"])
                for tc in result.tool_calls
            ]

            return AgentResponse(
                answer=result.answer,
                question=req.question,
                tool_calls=tool_calls,
                sources=sources,
                graph=GraphData(nodes=flow_nodes, edges=flow_edges),
                session_id=result.session_id,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats", response_model=GraphStatsResponse)
def graph_stats():
    try:
        with Neo4jClient() as client:
            nc = client.node_count()
            rc = client.relationship_count()
            return GraphStatsResponse(
                node_counts=nc,
                relationship_counts=rc,
                total_nodes=sum(nc.values()),
                total_relationships=sum(rc.values()),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/communities")
def get_communities():
    try:
        with Neo4jClient() as client:
            return client.get_all_communities()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health():
    neo4j_ok = False
    try:
        with Neo4jClient() as client:
            client.verify_connectivity()
            neo4j_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="ok" if neo4j_ok else "degraded",
        neo4j=neo4j_ok,
        tracing_enabled=is_tracing_enabled(),
    )
