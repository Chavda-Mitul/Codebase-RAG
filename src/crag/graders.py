"""LLM-based graders for document relevance, hallucination, and answer quality."""
from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import settings


def _llm() -> ChatGroq:
    return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)


# --- Relevance grader ---

class RelevanceGrade(BaseModel):
    score: str = Field(description="'yes' if the document is relevant to the question, 'no' otherwise")
    reason: str = Field(description="One sentence explaining the relevance decision")


_RELEVANCE_SYSTEM = """You are a relevance grader for a code knowledge base Q&A system.
Given a question and a retrieved code artifact (function, class, file, concept), decide if it contains
information useful for answering the question.
Be lenient: partial relevance counts as 'yes'. Only grade 'no' if the document is clearly unrelated."""

_RELEVANCE_USER = """Question: {question}

Retrieved document:
{document}

Is this document relevant?"""


def grade_relevance(question: str, document_text: str, llm: ChatGroq | None = None) -> RelevanceGrade:
    if llm is None:
        llm = _llm()
    grader = llm.with_structured_output(RelevanceGrade)
    return grader.invoke([
        {"role": "system", "content": _RELEVANCE_SYSTEM},
        {"role": "user", "content": _RELEVANCE_USER.format(question=question, document=document_text)},
    ])


# --- Hallucination grader ---

class HallucinationGrade(BaseModel):
    score: str = Field(description="'grounded' if the answer is supported by the context, 'hallucinated' if it contains unsupported claims")
    reason: str = Field(description="Brief explanation")


_HALLUCINATION_SYSTEM = """You are a hallucination detector for a code Q&A system.
Given retrieved context and a generated answer, check if the answer makes claims NOT supported by the context.
If the answer only uses information from the context, grade as 'grounded'.
If the answer invents details, functions, or relationships not in the context, grade as 'hallucinated'."""

_HALLUCINATION_USER = """Context:
{context}

Answer:
{answer}

Is the answer grounded in the context?"""


def grade_hallucination(context: str, answer: str, llm: ChatGroq | None = None) -> HallucinationGrade:
    if llm is None:
        llm = _llm()
    grader = llm.with_structured_output(HallucinationGrade)
    return grader.invoke([
        {"role": "system", "content": _HALLUCINATION_SYSTEM},
        {"role": "user", "content": _HALLUCINATION_USER.format(context=context, answer=answer)},
    ])


# --- Answer quality grader ---

class AnswerGrade(BaseModel):
    score: str = Field(description="'useful' if the answer addresses the question, 'not useful' if it is vague, incomplete, or off-topic")
    reason: str = Field(description="Brief explanation")


_ANSWER_SYSTEM = """You are an answer quality evaluator for a code Q&A system.
Assess whether the generated answer actually addresses the user's question with useful, specific information.
'useful' = the answer contains specific, actionable information about the codebase.
'not useful' = the answer is too vague, says 'I don't know', or doesn't address the question."""

_ANSWER_USER = """Question: {question}

Answer: {answer}

Is this answer useful?"""


def grade_answer(question: str, answer: str, llm: ChatGroq | None = None) -> AnswerGrade:
    if llm is None:
        llm = _llm()
    grader = llm.with_structured_output(AnswerGrade)
    return grader.invoke([
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": _ANSWER_USER.format(question=question, answer=answer)},
    ])


# --- Query rewriter ---

_REWRITE_SYSTEM = """You are a query optimizer for a code knowledge base search system.
Given a question that didn't retrieve useful documents, rewrite it to be more specific and use
technical terminology that would better match code artifacts (function names, class names, patterns).
Return only the rewritten question, nothing else."""

_REWRITE_USER = """Original question: {question}

Previous rewrites (avoid repeating): {rewrites}

Rewrite the question to retrieve better code context:"""


def rewrite_query(question: str, previous_rewrites: list[str], llm: ChatGroq | None = None) -> str:
    if llm is None:
        llm = _llm()
    response = llm.invoke([
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": _REWRITE_USER.format(
            question=question,
            rewrites=", ".join(previous_rewrites) if previous_rewrites else "none",
        )},
    ])
    return response.content.strip()
