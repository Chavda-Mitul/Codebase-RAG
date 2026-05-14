"""Pydantic request/response models for the FastAPI server."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=8, ge=1, le=20)
    use_reranker: bool = True


class SourceNode(BaseModel):
    node_id: str
    label: str
    name: str
    score: float
    source: str


class DocGrade(BaseModel):
    node_id: str
    name: str
    score: str
    reason: str


class TraceInfo(BaseModel):
    route: str
    iterations: int
    query_rewrites: list[str]
    correction_triggered: bool
    hallucination_check: str
    answer_check: str
    doc_grades: list[DocGrade]
    step_back_question: str
    sub_questions: list[str]


class GraphNodeData(BaseModel):
    name: str
    nodeType: str
    isSource: bool
    docstring: str = ""
    file_path: str = ""


class FlowNode(BaseModel):
    id: str
    type: str = "codeNode"
    position: dict
    data: GraphNodeData


class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    animated: bool = False


class GraphData(BaseModel):
    nodes: list[FlowNode]
    edges: list[FlowEdge]


class AskResponse(BaseModel):
    answer: str
    question: str
    trace: TraceInfo
    sources: list[SourceNode]
    graph: GraphData


class GraphStatsResponse(BaseModel):
    node_counts: dict[str, int]
    relationship_counts: dict[str, int]
    total_nodes: int
    total_relationships: int


class HealthResponse(BaseModel):
    status: str
    neo4j: bool
    tracing_enabled: bool


class AgentRequest(BaseModel):
    question: str = Field(min_length=3)
    session_id: str = ""


class ToolCallInfo(BaseModel):
    tool: str
    input: str
    output: str


class AgentResponse(BaseModel):
    answer: str
    question: str
    tool_calls: list[ToolCallInfo]
    sources: list[SourceNode]
    graph: GraphData
    session_id: str
