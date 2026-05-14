"""Unified Q&A CLI — uses the full Phase 4 routing pipeline.

Routes automatically to:
  simple     → direct CRAG lookup
  complex    → decompose → answer sub-questions → synthesize
  conceptual → step-back → enriched CRAG

Usage:
    python scripts/ask.py --question "What are the main sources of technical debt?"
    python scripts/ask.py   # interactive REPL
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.graph.neo4j_client import Neo4jClient
from src.routing.pipeline import answer_with_routing, RoutedAnswer

app = typer.Typer(add_completion=False)

_ROUTE_ICONS = {"simple": "→", "complex": "⋱", "conceptual": "↑"}


def _print_result(result: RoutedAnswer) -> None:
    icon = _ROUTE_ICONS.get(result.route, "?")
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Q: {result.question}")
    typer.echo(f"Route: [{result.route.upper()}]  {icon}")
    typer.echo("=" * 60)

    if result.route == "conceptual" and result.step_back_question:
        typer.echo(f"\n[Step-back] Abstract question used for context:")
        typer.echo(f"  {result.step_back_question}")

    if result.route == "complex" and result.sub_questions:
        typer.echo(f"\n[Decomposed into {len(result.sub_questions)} sub-questions]")
        for i, (sq, sa) in enumerate(zip(result.sub_questions, result.sub_answers), 1):
            typer.echo(f"\n  Sub-Q {i}: {sq}")
            typer.echo(f"  Answer:  {sa[:300]}{'...' if len(sa) > 300 else ''}")

    if result.crag_state:
        state = result.crag_state
        typer.echo(f"\n[CRAG] iterations={state.get('iteration',0)} | "
                   f"hallucination={state.get('hallucination_check','n/a')} | "
                   f"quality={state.get('answer_check','n/a')}")
        if state.get("query_rewrites"):
            typer.echo(f"[CRAG] Rewrites: {state['query_rewrites']}")

    typer.echo("\n--- Answer ---")
    typer.echo(result.answer)

    if result.sources:
        typer.echo("\n--- Sources ---")
        for i, src in enumerate(result.sources[:8], 1):
            typer.echo(f"  {i}. [{src.label}] {src.name}  (score: {src.score:.3f})")

    typer.echo("")


@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Question to answer"),
    top_k: int = typer.Option(8, "--top-k", help="Context nodes to retrieve per query"),
):
    with Neo4jClient() as client:
        client.verify_connectivity()

        if question:
            result = answer_with_routing(question, client)
            _print_result(result)
        else:
            typer.echo("Code-RAG  [smart routing: simple | complex | conceptual]\n")
            typer.echo("Type 'exit' to quit.\n")
            while True:
                try:
                    q = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() in ("exit", "quit"):
                    break
                result = answer_with_routing(q, client)
                _print_result(result)


if __name__ == "__main__":
    app()
