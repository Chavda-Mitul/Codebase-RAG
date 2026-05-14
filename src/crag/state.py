"""LangGraph state definition for the CRAG pipeline."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.retrieval.models import RetrievedNode


class CRAGState(TypedDict):
    question: str
    documents: list[RetrievedNode]          # raw retrieved docs
    filtered_documents: list[RetrievedNode] # after relevance grading
    answer: str
    query_rewrites: list[str]               # history of rewritten queries
    correction_triggered: bool              # whether a correction loop fired
    doc_grades: list[dict]                  # per-doc grade results
    hallucination_check: str                # "grounded" | "hallucinated"
    answer_check: str                       # "useful" | "not useful"
    iteration: int                          # loop counter (max 3)
