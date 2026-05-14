"""CLI entrypoint for the ingestion pipeline.

Usage:
    python scripts/ingest.py --repo https://github.com/Chavda-Mitul/invoice-anomaly-detection
    python scripts/ingest.py --local /path/to/local/repo
"""
import json
import sys
from pathlib import Path

# Ensure src is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from src.ingestion.pipeline import run_ingestion

app = typer.Typer(add_completion=False)


@app.command()
def main(
    repo: str = typer.Option(None, "--repo", help="GitHub repo URL to clone and ingest"),
    local: str = typer.Option(None, "--local", help="Local directory path to ingest"),
):
    if not repo and not local:
        typer.echo("Error: provide --repo or --local", err=True)
        raise typer.Exit(1)

    typer.echo("Starting ingestion pipeline...")
    summary = run_ingestion(repo_url=repo, local_path=local)

    typer.echo("\n--- Ingestion Complete ---")
    typer.echo(f"Nodes written:         {summary['nodes_written']}")
    typer.echo(f"Relationships written: {summary['relationships_written']}")
    typer.echo(f"Elapsed:               {summary['elapsed_seconds']}s")
    typer.echo(f"\nNode distribution:")
    for label, count in summary.get("node_distribution", {}).items():
        typer.echo(f"  {label}: {count}")
    typer.echo(f"\nRelationship distribution:")
    for rel_type, count in summary.get("relationship_distribution", {}).items():
        typer.echo(f"  {rel_type}: {count}")


if __name__ == "__main__":
    app()
