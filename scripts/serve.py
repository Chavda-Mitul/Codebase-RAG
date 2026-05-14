"""Start the FastAPI server.

Usage:
    python scripts/serve.py              # default: localhost:8000
    python scripts/serve.py --port 8080  # custom port
    python scripts/serve.py --reload     # hot-reload for development
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import uvicorn

app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload on code changes"),
):
    typer.echo(f"Starting Code-RAG API server at http://{host}:{port}")
    typer.echo("Frontend should run at http://localhost:3000")
    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()
