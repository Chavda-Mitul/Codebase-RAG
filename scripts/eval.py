"""Eval CLI — runs the golden QA suite and reports metrics.

Usage:
    # Eval a single pipeline
    python scripts/eval.py --pipeline crag

    # Compare all three pipelines (naive vs crag vs routed)
    python scripts/eval.py --compare

    # Limit to N questions for a quick check
    python scripts/eval.py --pipeline naive --limit 5

    # Save results to JSON
    python scripts/eval.py --pipeline routed --output results/eval_routed.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.graph.neo4j_client import Neo4jClient
from src.eval.runner import run_eval
from src.eval.golden_qa import GOLDEN_QA_PAIRS
from src.observability.langfuse_handler import is_tracing_enabled

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pipeline: str = typer.Option("crag", "--pipeline", "-p", help="Pipeline to evaluate: naive | crag | routed"),
    compare: bool = typer.Option(False, "--compare", help="Compare all three pipelines"),
    limit: int = typer.Option(0, "--limit", "-n", help="Limit to N questions (0 = all)"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON report to this path"),
    difficulty: str = typer.Option(None, "--difficulty", help="Filter by difficulty: easy | medium | hard"),
    category: str = typer.Option(None, "--category", help="Filter by category: factual | structural | architectural | comparative"),
):
    if is_tracing_enabled():
        typer.echo("Langfuse tracing: ENABLED")
    else:
        typer.echo("Langfuse tracing: disabled (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)")

    # Filter QA pairs
    pairs = GOLDEN_QA_PAIRS
    if difficulty:
        pairs = [q for q in pairs if q.difficulty == difficulty]
    if category:
        pairs = [q for q in pairs if q.category == category]
    if limit > 0:
        pairs = pairs[:limit]

    typer.echo(f"QA pairs selected: {len(pairs)}")

    with Neo4jClient() as client:
        client.verify_connectivity()

        if compare:
            reports = []
            for pl in ("naive", "crag", "routed"):
                out = f"{output.replace('.json', f'_{pl}.json')}" if output else None
                report = run_eval(pl, client, qa_pairs=pairs, output_path=out)
                reports.append(report)
                report.print_summary()

            # Print comparison table
            typer.echo(f"\n{'Pipeline':<12} {'Faithful':>10} {'Relevance':>10} {'Recall':>10} {'Precision':>10} {'Average':>10}")
            typer.echo("-" * 62)
            for r in reports:
                avg = r.avg_scores()
                typer.echo(f"{r.pipeline:<12} {avg.faithfulness:>10.3f} {avg.answer_relevance:>10.3f} {avg.context_recall:>10.3f} {avg.context_precision:>10.3f} {avg.average():>10.3f}")
        else:
            report = run_eval(pipeline, client, qa_pairs=pairs, output_path=output)
            report.print_summary()

            # Per-question breakdown
            typer.echo("\n--- Per-question scores ---")
            for r in report.results:
                typer.echo(
                    f"  {r.qa_id:<8} faith={r.scores.faithfulness:.2f} "
                    f"rel={r.scores.answer_relevance:.2f} "
                    f"recall={r.scores.context_recall:.2f} "
                    f"prec={r.scores.context_precision:.2f} "
                    f"({r.latency_ms:.0f}ms)"
                )


if __name__ == "__main__":
    app()
