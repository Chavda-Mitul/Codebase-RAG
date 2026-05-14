"""Routing pipeline: classifies question → routes to simple/complex/conceptual path."""
from __future__ import annotations
from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import hybrid_retrieve, community_retrieve
from src.retrieval.models import RetrievedNode
from src.crag.graph import run_crag
from src.crag.state import CRAGState
from src.qa.chain import _format_context
from src.qa.prompts import SYSTEM_PROMPT
from src.routing.router import route_question, RouteType
from src.routing.decomposer import decompose_question
from src.routing.step_back import generate_step_back_question


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    sources: list[RetrievedNode] = field(default_factory=list)
    session_id: str = ""


@dataclass
class RoutedAnswer:
    question: str
    answer: str
    route: RouteType
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[str] = field(default_factory=list)
    step_back_question: str = ""
    crag_state: CRAGState | None = None
    sources: list[RetrievedNode] = field(default_factory=list)
    use_reranker: bool = True


_SYNTHESIS_PROMPT = """You are synthesizing answers to sub-questions into a final comprehensive answer.

Original question: {question}

Hint on how to combine: {hint}

Sub-question answers:
{sub_answers}

Write a unified, comprehensive answer to the original question based on the sub-answers above.
Use citations like [Sub-Q 1], [Sub-Q 2] etc. to reference which sub-answer supports each claim."""


def _synthesize(question: str, hint: str, sub_qs: list[str], sub_answers: list[str], llm: ChatGroq) -> str:
    formatted = "\n\n".join(
        f"Sub-Q {i+1}: {q}\nAnswer: {a}"
        for i, (q, a) in enumerate(zip(sub_qs, sub_answers))
    )
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _SYNTHESIS_PROMPT.format(
            question=question, hint=hint, sub_answers=formatted
        )},
    ])
    return response.content


def _run_simple(question: str, client: Neo4jClient, use_reranker: bool = True) -> RoutedAnswer:
    state = run_crag(question, client)
    return RoutedAnswer(
        question=question,
        answer=state["answer"],
        route="simple",
        crag_state=state,
        sources=state.get("filtered_documents") or state.get("documents", []),
        use_reranker=use_reranker,
    )


def _run_complex(question: str, client: Neo4jClient, llm: ChatGroq, use_reranker: bool = True) -> RoutedAnswer:
    decomposed = decompose_question(question, llm=llm)
    sub_answers: list[str] = []

    for sub_q in decomposed.sub_questions:
        state = run_crag(sub_q, client)
        sub_answers.append(state["answer"])

    synthesis = _synthesize(
        question, decomposed.synthesis_hint,
        decomposed.sub_questions, sub_answers, llm
    )

    # Collect all sources
    sources: list[RetrievedNode] = []
    seen_ids: set[str] = set()
    for sub_q in decomposed.sub_questions:
        docs = hybrid_retrieve(sub_q, client, top_k=3, use_reranker=use_reranker)
        for d in docs:
            if d.node_id not in seen_ids:
                sources.append(d)
                seen_ids.add(d.node_id)

    return RoutedAnswer(
        question=question,
        answer=synthesis,
        route="complex",
        sub_questions=decomposed.sub_questions,
        sub_answers=sub_answers,
        sources=sources,
        use_reranker=use_reranker,
    )


def _run_conceptual(question: str, client: Neo4jClient, use_reranker: bool = True) -> RoutedAnswer:
    # Generate step-back question and retrieve abstract context
    step_back_q = generate_step_back_question(question)
    abstract_docs = hybrid_retrieve(step_back_q, client, top_k=5, use_reranker=use_reranker)

    # Enrich with community-level summaries for architectural overview
    community_docs = community_retrieve(step_back_q, client, top_k=2)
    seen_ids = {d.node_id for d in abstract_docs}
    for doc in community_docs:
        if doc.node_id not in seen_ids:
            abstract_docs.append(doc)
            seen_ids.add(doc.node_id)

    # Seed the CRAG pipeline with the abstract context pre-loaded
    from src.crag.graph import build_crag_graph
    from src.crag.state import CRAGState

    pipeline = build_crag_graph(client)
    initial_state: CRAGState = {
        "question": question,
        "documents": abstract_docs,   # pre-seed with step-back context
        "filtered_documents": [],
        "answer": "",
        "query_rewrites": [],
        "correction_triggered": False,
        "doc_grades": [],
        "hallucination_check": "",
        "answer_check": "",
        "iteration": 0,
    }
    state = pipeline.invoke(initial_state)

    return RoutedAnswer(
        question=question,
        answer=state["answer"],
        route="conceptual",
        step_back_question=step_back_q,
        crag_state=state,
        sources=state.get("filtered_documents") or state.get("documents", []),
        use_reranker=use_reranker,
    )


def answer_with_agent(question: str, client: Neo4jClient, session_id: str = "") -> AgentResult:
    """Run the agentic tool-calling pipeline with conversation memory."""
    from src.agents.agent_graph import run_agent
    state = run_agent(question, client, session_id)
    return AgentResult(
        question=question,
        answer=state["answer"],
        tool_calls=state.get("tool_calls") or [],
        sources=state.get("documents") or [],
        session_id=session_id,
    )


def answer_with_routing(question: str, client: Neo4jClient, use_reranker: bool = True) -> RoutedAnswer:
    """Route the question and run the appropriate pipeline."""
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    decision = route_question(question, llm=llm)

    if decision.route == "simple":
        return _run_simple(question, client, use_reranker=use_reranker)
    elif decision.route == "complex":
        return _run_complex(question, client, llm, use_reranker=use_reranker)
    else:  # conceptual
        return _run_conceptual(question, client, use_reranker=use_reranker)
