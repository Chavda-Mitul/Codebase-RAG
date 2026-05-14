"""Interactive Q&A CLI for the Code-RAG system.

Usage:
    # Single question
    python scripts/query.py --question "What are the main classes in this codebase?"

    # Interactive REPL
    python scripts/query.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.graph.neo4j_client import Neo4jClient
from src.qa.chain import answer_question

app = typer.Typer(add_completion=False)


def _print_result(result) -> None:
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Q: {result.question}")
    typer.echo("=" * 60)
    typer.echo(result.answer)
    typer.echo("\n--- Sources ---")
    for i, src in enumerate(result.sources, 1):
        typer.echo(f"  {i}. [{src.label}] {src.name}  (score: {src.score:.3f}, via: {src.source})")
    typer.echo("")


@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Question to ask"),
    top_k: int = typer.Option(8, "--top-k", help="Number of context nodes to retrieve"),
):
    with Neo4jClient() as client:
        client.verify_connectivity()

        if question:
            result = answer_question(question, client, top_k=top_k)
            _print_result(result)
        else:
            typer.echo("Code-RAG Q&A  (type 'exit' to quit)\n")
            while True:
                try:
                    q = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() in ("exit", "quit"):
                    break
                result = answer_question(q, client, top_k=top_k)
                _print_result(result)


if __name__ == "__main__":
    app()
