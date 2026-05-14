"""LangGraph node functions for the CRAG pipeline."""
from __future__ import annotations
from langchain_groq import ChatGroq
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.models import RetrievedNode
from src.qa.chain import _format_context
from src.qa.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.crag.state import CRAGState
from src.crag.graders import (
    grade_relevance, grade_hallucination, grade_answer, rewrite_query
)

_MAX_ITERATIONS = 3


def make_nodes(client: Neo4jClient):
    """Return node functions bound to the given Neo4j client."""

    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)

    def retrieve(state: CRAGState) -> dict:
        """Hybrid retrieval using the current question. Merges with any pre-seeded docs (e.g. step-back context)."""
        question = state.get("query_rewrites", [])[-1] if state.get("query_rewrites") else state["question"]
        fresh_docs = hybrid_retrieve(question, client, top_k=8)
        # Merge with pre-seeded docs (step-back context), keeping deduped by node_id
        existing = state.get("documents") or []
        seen: dict[str, RetrievedNode] = {d.node_id: d for d in existing}
        for d in fresh_docs:
            if d.node_id not in seen:
                seen[d.node_id] = d
        merged = list(seen.values())
        return {"documents": merged, "iteration": state.get("iteration", 0) + 1}

    def grade_documents(state: CRAGState) -> dict:
        """Grade each retrieved document for relevance. Filter irrelevant ones."""
        question = state["question"]
        docs = state["documents"]
        grades: list[dict] = []
        filtered: list[RetrievedNode] = []

        for doc in docs:
            grade = grade_relevance(question, doc.text, llm=llm)
            grades.append({"node_id": doc.node_id, "name": doc.name, "score": grade.score, "reason": grade.reason})
            if grade.score == "yes":
                filtered.append(doc)

        correction_triggered = len(filtered) < len(docs) // 2  # majority irrelevant
        return {
            "filtered_documents": filtered,
            "doc_grades": grades,
            "correction_triggered": correction_triggered,
        }

    def rewrite(state: CRAGState) -> dict:
        """Rewrite the query to improve retrieval."""
        previous_rewrites = state.get("query_rewrites", [])
        new_query = rewrite_query(state["question"], previous_rewrites, llm=llm)
        return {"query_rewrites": previous_rewrites + [new_query]}

    def generate(state: CRAGState) -> dict:
        """Generate an answer from filtered (or all) documents."""
        docs = state.get("filtered_documents") or state.get("documents", [])
        context = _format_context(docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context, question=state["question"])},
        ]
        response = llm.invoke(messages)
        return {"answer": response.content}

    def check_hallucination(state: CRAGState) -> dict:
        """Check whether the answer is grounded in the retrieved context."""
        docs = state.get("filtered_documents") or state.get("documents", [])
        context = _format_context(docs)
        grade = grade_hallucination(context, state["answer"], llm=llm)
        return {"hallucination_check": grade.score}

    def check_answer_quality(state: CRAGState) -> dict:
        """Check whether the answer is actually useful for the question."""
        grade = grade_answer(state["question"], state["answer"], llm=llm)
        return {"answer_check": grade.score}

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "rewrite": rewrite,
        "generate": generate,
        "check_hallucination": check_hallucination,
        "check_answer_quality": check_answer_quality,
    }


# --- Conditional edge functions ---

def decide_after_grading(state: CRAGState) -> str:
    """After grading docs: go to rewrite if correction needed, else generate."""
    if state.get("correction_triggered") and state.get("iteration", 0) < _MAX_ITERATIONS:
        return "rewrite"
    return "generate"


def decide_after_hallucination(state: CRAGState) -> str:
    """After hallucination check: retry generation or check answer quality."""
    if state.get("hallucination_check") == "hallucinated" and state.get("iteration", 0) < _MAX_ITERATIONS:
        return "generate"  # retry with same context
    return "check_answer_quality"


def decide_after_answer_check(state: CRAGState) -> str:
    """After answer quality check: return answer or rewrite and retry."""
    if state.get("answer_check") == "not useful" and state.get("iteration", 0) < _MAX_ITERATIONS:
        return "rewrite"
    return "end"
