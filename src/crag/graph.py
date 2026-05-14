"""Build and compile the CRAG LangGraph pipeline."""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from src.graph.neo4j_client import Neo4jClient
from src.crag.state import CRAGState
from src.crag.nodes import (
    make_nodes,
    decide_after_grading,
    decide_after_hallucination,
    decide_after_answer_check,
)


def build_crag_graph(client: Neo4jClient):
    """Construct and compile the CRAG StateGraph."""
    nodes = make_nodes(client)

    graph = StateGraph(CRAGState)

    # Register nodes
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("grade_documents", nodes["grade_documents"])
    graph.add_node("rewrite", nodes["rewrite"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("check_hallucination", nodes["check_hallucination"])
    graph.add_node("check_answer_quality", nodes["check_answer_quality"])

    # Entry point
    graph.set_entry_point("retrieve")

    # Fixed edges
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check_hallucination")

    # Conditional edges
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination,
        {"generate": "generate", "check_answer_quality": "check_answer_quality"},
    )
    graph.add_conditional_edges(
        "check_answer_quality",
        decide_after_answer_check,
        {"rewrite": "rewrite", "end": END},
    )

    return graph.compile()


def run_crag(question: str, client: Neo4jClient) -> CRAGState:
    """Run the full CRAG pipeline and return the final state."""
    pipeline = build_crag_graph(client)
    initial_state: CRAGState = {
        "question": question,
        "documents": [],
        "filtered_documents": [],
        "answer": "",
        "query_rewrites": [],
        "correction_triggered": False,
        "doc_grades": [],
        "hallucination_check": "",
        "answer_check": "",
        "iteration": 0,
    }
    final_state = pipeline.invoke(initial_state)
    return final_state
