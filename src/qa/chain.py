"""Q&A chain: retrieve context → format → generate answer with Groq."""
from __future__ import annotations
from dataclasses import dataclass
from langchain_groq import ChatGroq
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.models import RetrievedNode
from src.qa.prompts import SYSTEM_PROMPT, USER_PROMPT


@dataclass
class QAResult:
    question: str
    answer: str
    sources: list[RetrievedNode]
    context_used: str


def _format_context(nodes: list[RetrievedNode]) -> str:
    lines = []
    for i, node in enumerate(nodes, 1):
        lines.append(f"{i}. [{node.label}] {node.name}")
        if node.text:
            lines.append(f"   {node.text}")
        if node.source:
            lines.append(f"   (retrieved via: {node.source}, score: {node.score:.3f})")
        lines.append("")
    return "\n".join(lines)


def answer_question(
    question: str,
    client: Neo4jClient,
    top_k: int = 8,
    llm: ChatGroq | None = None,
) -> QAResult:
    """Run hybrid retrieval then generate a grounded answer."""
    if llm is None:
        llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    # Retrieve context
    nodes = hybrid_retrieve(question, client, top_k=top_k)
    context = _format_context(nodes)

    # Generate answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
    ]
    response = llm.invoke(messages)
    answer = response.content

    return QAResult(
        question=question,
        answer=answer,
        sources=nodes,
        context_used=context,
    )
