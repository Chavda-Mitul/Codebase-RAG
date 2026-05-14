"""AgentState TypedDict for the LangGraph agentic pipeline."""
from __future__ import annotations
from typing import TypedDict
from src.retrieval.models import RetrievedNode


class AgentState(TypedDict):
    question: str
    session_id: str
    history: str               # formatted conversation turns injected at start
    tool_plan: list[dict]      # [{tool_name, argument}] — planned by LLM
    tool_calls: list[dict]     # [{tool, input, output}] — executed results
    documents: list[RetrievedNode]  # populated by search_codebase tool
    answer: str
    iteration: int
