"""Step-back prompting: generates an abstract version of a conceptual question
to retrieve broader architectural context before answering the specific question."""
from __future__ import annotations
from langchain_groq import ChatGroq
from src.config import settings


_SYSTEM = """You are a step-back prompting specialist for a code knowledge base.
Given a specific conceptual question about a codebase, generate ONE broader, more abstract
question that captures the general principle or architecture behind it.

The step-back question should retrieve high-level architectural context that will help
answer the original specific question.

Examples:
- Original: "Why does the invoice loader use pandas instead of polars?"
  Step-back: "What are the data processing libraries and patterns used across the codebase?"

- Original: "What are the main sources of technical debt in the payment service?"
  Step-back: "What is the overall architecture and design patterns of the codebase?"

- Original: "How does anomaly detection integrate with the invoice pipeline?"
  Step-back: "What are the main data flows and component interactions in the system?"

Return only the step-back question, nothing else."""

_USER = "Original question: {question}\n\nStep-back question:"


def generate_step_back_question(question: str, llm: ChatGroq | None = None) -> str:
    if llm is None:
        llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(question=question)},
    ])
    return response.content.strip()
