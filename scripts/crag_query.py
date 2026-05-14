"""CRAG pipeline CLI — uses the full corrective agentic loop.

Usage:
    python scripts/crag_query.py --question "What are the main classes and how do they relate?"
    python scripts/crag_query.py   # interactive REPL
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.graph.neo4j_client import Neo4jClient
from src.crag.graph import run_crag

app = typer.Typer(add_completion=False)


def _print_trace(state: dict) -> None:
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Q: {state['question']}")
    typer.echo("=" * 60)

    if state.get("query_rewrites"):
        typer.echo(f"\n[CRAG] Query rewritten {len(state['query_rewrites'])} time(s):")
        for i, rw in enumerate(state["query_rewrites"], 1):
            typer.echo(f"  {i}. {rw}")

    if state.get("correction_triggered"):
        typer.echo("[CRAG] Correction loop was triggered (low relevance detected)")

    typer.echo(f"[CRAG] Hallucination check: {state.get('hallucination_check', 'n/a')}")
    typer.echo(f"[CRAG] Answer quality:      {state.get('answer_check', 'n/a')}")
    typer.echo(f"[CRAG] Iterations:          {state.get('iteration', 0)}")

    grades = state.get("doc_grades", [])
    if grades:
        yes = sum(1 for g in grades if g["score"] == "yes")
        typer.echo(f"[CRAG] Docs graded: {yes}/{len(grades)} relevant")

    typer.echo("\n--- Answer ---")
    typer.echo(state.get("answer", "(no answer generated)"))

    typer.echo("\n--- Sources used ---")
    for i, doc in enumerate(state.get("filtered_documents") or state.get("documents", []), 1):
        typer.echo(f"  {i}. [{doc.label}] {doc.name}  (score: {doc.score:.3f})")

    typer.echo("")


@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Question to answer"),
):
    with Neo4jClient() as client:
        client.verify_connectivity()

        if question:
            state = run_crag(question, client)
            _print_trace(state)
        else:
            typer.echo("Code-RAG CRAG Pipeline  (type 'exit' to quit)\n")
            while True:
                try:
                    q = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() in ("exit", "quit"):
                    break
                state = run_crag(q, client)
                _print_trace(state)


if __name__ == "__main__":
    app()
