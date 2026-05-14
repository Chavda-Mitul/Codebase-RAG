"""Query decomposer: splits a complex multi-hop question into independent sub-questions."""
from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import settings


class DecomposedQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description="2-4 specific, self-contained sub-questions that together answer the original question",
        min_length=2,
        max_length=4,
    )
    synthesis_hint: str = Field(
        description="One sentence describing how the sub-answers should be combined"
    )


_SYSTEM = """You are a query decomposition specialist for a code knowledge base.
Break complex multi-hop questions into 2-4 simpler, self-contained sub-questions.

Rules:
- Each sub-question must be answerable independently from the codebase
- Sub-questions should cover different aspects of the original question
- Keep sub-questions specific and technical (mention class names, file paths, patterns if implied)
- The sub-answers, when combined, must fully answer the original question"""

_USER = """Decompose this complex question into sub-questions:

{question}"""


def decompose_question(question: str, llm: ChatGroq | None = None) -> DecomposedQuestions:
    if llm is None:
        llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    decomposer = llm.with_structured_output(DecomposedQuestions)
    return decomposer.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(question=question)},
    ])
