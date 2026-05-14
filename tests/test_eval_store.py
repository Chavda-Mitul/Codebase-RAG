"""Unit tests for the SQLite eval store."""
from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.eval.metrics import EvalScores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temporary file for every test."""
    import src.eval.store as store_module
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test_eval.db")
    return tmp_path / "test_eval.db"


def _make_report(pipeline: str, n: int = 2):
    """Build a mock EvalReport with n results."""
    from src.eval.runner import EvalReport, EvalResult

    results = []
    for i in range(n):
        results.append(EvalResult(
            qa_id=f"q{i}",
            question=f"Question {i}?",
            pipeline=pipeline,
            answer=f"Answer {i}",
            ground_truth=f"Truth {i}",
            context="some context",
            scores=EvalScores(
                faithfulness=0.8 + i * 0.01,
                answer_relevance=0.75,
                context_recall=0.7,
                context_precision=0.65,
            ),
            latency_ms=100.0 + i * 10,
        ))
    report = EvalReport(pipeline=pipeline)
    report.results = results
    return report


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(tmp_db):
    from src.eval import store
    store.init_db()
    import sqlite3
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "eval_runs" in tables
    assert "eval_questions" in tables
    conn.close()


def test_init_db_is_idempotent(tmp_db):
    from src.eval import store
    store.init_db()
    store.init_db()  # should not raise


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_returns_run_id():
    from src.eval import store
    report = _make_report("crag")
    run_id = store.save_report(report)
    assert isinstance(run_id, int)
    assert run_id >= 1


def test_save_report_stores_avg_scores():
    from src.eval import store
    report = _make_report("crag", n=2)
    run_id = store.save_report(report)
    run = store.get_run(run_id)
    assert run is not None
    assert run["pipeline"] == "crag"
    assert run["question_count"] == 2
    assert 0 < run["avg_faithfulness"] <= 1.0
    assert 0 < run["avg_overall"] <= 1.0


def test_save_report_stores_questions():
    from src.eval import store
    report = _make_report("naive", n=3)
    run_id = store.save_report(report)
    run = store.get_run(run_id)
    assert len(run["questions"]) == 3
    assert all(q["run_id"] == run_id for q in run["questions"])


def test_save_report_multiple_pipelines():
    from src.eval import store
    id1 = store.save_report(_make_report("crag"))
    id2 = store.save_report(_make_report("naive"))
    assert id1 != id2


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_empty():
    from src.eval import store
    assert store.list_runs() == []


def test_list_runs_returns_newest_first():
    from src.eval import store
    store.save_report(_make_report("crag"))
    store.save_report(_make_report("naive"))
    runs = store.list_runs()
    assert len(runs) == 2
    # Newer run (naive) should be first
    assert runs[0]["pipeline"] == "naive"


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------

def test_get_run_missing_returns_none():
    from src.eval import store
    assert store.get_run(9999) is None


def test_get_run_contains_questions_key():
    from src.eval import store
    run_id = store.save_report(_make_report("routed", n=2))
    run = store.get_run(run_id)
    assert "questions" in run
    assert isinstance(run["questions"], list)


# ---------------------------------------------------------------------------
# compare_pipelines
# ---------------------------------------------------------------------------

def test_compare_pipelines_empty():
    from src.eval import store
    assert store.compare_pipelines() == {}


def test_compare_pipelines_returns_latest_per_pipeline():
    from src.eval import store
    store.save_report(_make_report("crag"))
    store.save_report(_make_report("crag"))  # second run (should be latest)
    store.save_report(_make_report("naive"))
    cmp = store.compare_pipelines()
    assert set(cmp.keys()) == {"crag", "naive"}


def test_compare_pipelines_has_score_fields():
    from src.eval import store
    store.save_report(_make_report("crag"))
    cmp = store.compare_pipelines()
    crag = cmp["crag"]
    for field in ("avg_faithfulness", "avg_answer_relevance", "avg_context_recall", "avg_context_precision"):
        assert field in crag
