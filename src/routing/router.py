"""Query router: classifies a question and decides which pipeline path to use."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import settings


RouteType = Literal["simple", "complex", "conceptual"]


class RouteDecision(BaseModel):
    route: RouteType = Field(
        description=(
            "'simple' for direct factual lookups (what does X do, where is Y defined); "
            "'complex' for multi-hop questions requiring multiple pieces of information to be combined; "
            "'conceptual' for broad architectural or design questions needing high-level context first"
        )
    )
    reason: str = Field(description="One sentence explaining the routing decision")


_SYSTEM = """You are a query router for a code knowledge base Q&A system.
Classify each incoming question into exactly one route:

- simple: direct lookup — "What does function X do?", "Where is class Y defined?", "List the imports in file Z"
- complex: multi-hop — requires combining info from multiple files/components — "Compare how auth is handled across services", "What are all the places that call method X and what do they do with the result?"
- conceptual: broad design questions — "What is the overall architecture?", "What are the main sources of technical debt?", "Explain the data flow"

When in doubt between complex and conceptual, choose complex."""

_USER = "Classify this question: {question}"


def route_question(question: str, llm: ChatGroq | None = None) -> RouteDecision:
    if llm is None:
        llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    router = llm.with_structured_output(RouteDecision)
    return router.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(question=question)},
    ])
