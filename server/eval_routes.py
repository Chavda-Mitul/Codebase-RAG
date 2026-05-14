"""FastAPI router for evaluation endpoints."""
from __future__ import annotations
import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.eval import store
from src.eval.golden_qa import GOLDEN_QA_PAIRS

router = APIRouter()

# Tracks currently running background eval jobs: {pipeline: status_string}
_running: dict[str, str] = {}


def _background_eval(pipeline: str, limit: int) -> None:
    """Run eval in a background thread and store results."""
    try:
        from src.graph.neo4j_client import Neo4jClient
        from src.eval.runner import run_eval
        pairs = GOLDEN_QA_PAIRS[:limit]
        with Neo4jClient() as client:
            run_eval(pipeline, client, qa_pairs=pairs, auto_save=True)
    except Exception as e:
        print(f"[eval background] {pipeline} failed: {e}")
    finally:
        _running.pop(pipeline, None)


@router.get("/results")
def list_eval_results() -> list[dict]:
    """List all evaluation runs, newest first."""
    try:
        return store.list_runs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{run_id}")
def get_eval_result(run_id: int) -> dict:
    """Return a single run with all question-level results."""
    try:
        result = store.get_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return result


@router.post("/run")
def trigger_eval(pipeline: str = "crag", limit: int = 5) -> dict:
    """Start a background eval run. Returns immediately."""
    if pipeline not in ("naive", "crag", "routed"):
        raise HTTPException(status_code=400, detail="pipeline must be naive | crag | routed")
    if limit < 1 or limit > 18:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 18")
    if pipeline in _running:
        return {"status": "already_running", "pipeline": pipeline}

    _running[pipeline] = "running"
    thread = threading.Thread(target=_background_eval, args=(pipeline, limit), daemon=True)
    thread.start()
    return {"status": "started", "pipeline": pipeline, "limit": limit}


@router.get("/status")
def eval_status() -> dict:
    """Return which pipelines are currently evaluating."""
    return {"running": list(_running.keys())}


@router.get("/compare")
def compare_pipelines() -> dict:
    """Return {pipeline: avg_scores} for the latest run of each pipeline."""
    try:
        return store.compare_pipelines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
