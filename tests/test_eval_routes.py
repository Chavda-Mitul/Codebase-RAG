"""Tests for the FastAPI eval routes."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_running():
    """Ensure _running is empty before and after each test."""
    from server import eval_routes
    eval_routes._running.clear()
    yield
    eval_routes._running.clear()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with DB redirected to a temp path."""
    import src.eval.store as store_module
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test_eval_routes.db")
    from server.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /eval/results
# ---------------------------------------------------------------------------

def test_eval_results_empty(client):
    res = client.get("/eval/results")
    assert res.status_code == 200
    assert res.json() == []


def test_eval_results_after_save(client, tmp_path, monkeypatch):
    import src.eval.store as store_module
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test_eval_routes.db")

    from tests.test_eval_store import _make_report
    import src.eval.store as store
    store.save_report(_make_report("crag"))

    res = client.get("/eval/results")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["pipeline"] == "crag"


# ---------------------------------------------------------------------------
# GET /eval/results/{run_id}
# ---------------------------------------------------------------------------

def test_eval_result_not_found(client):
    res = client.get("/eval/results/9999")
    assert res.status_code == 404


def test_eval_result_found(client, tmp_path, monkeypatch):
    import src.eval.store as store_module
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test_eval_routes.db")

    from tests.test_eval_store import _make_report
    import src.eval.store as store
    run_id = store.save_report(_make_report("naive", n=2))

    res = client.get(f"/eval/results/{run_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["pipeline"] == "naive"
    assert "questions" in data
    assert len(data["questions"]) == 2


# ---------------------------------------------------------------------------
# POST /eval/run
# ---------------------------------------------------------------------------

def test_trigger_eval_invalid_pipeline(client):
    res = client.post("/eval/run?pipeline=badpipeline&limit=5")
    assert res.status_code == 400


def test_trigger_eval_invalid_limit(client):
    res = client.post("/eval/run?pipeline=crag&limit=0")
    assert res.status_code == 400


def test_trigger_eval_returns_started(client):
    # Patch background thread so it doesn't actually run
    with patch("server.eval_routes.threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        res = client.post("/eval/run?pipeline=crag&limit=3")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "started"
    assert data["pipeline"] == "crag"
    assert data["limit"] == 3


def test_trigger_eval_already_running(client):
    from server import eval_routes
    eval_routes._running["routed"] = "running"
    res = client.post("/eval/run?pipeline=routed&limit=5")
    assert res.status_code == 200
    assert res.json()["status"] == "already_running"
    eval_routes._running.pop("routed", None)


# ---------------------------------------------------------------------------
# GET /eval/compare
# ---------------------------------------------------------------------------

def test_eval_compare_empty(client):
    res = client.get("/eval/compare")
    assert res.status_code == 200
    assert res.json() == {}


def test_eval_compare_returns_pipelines(client, tmp_path, monkeypatch):
    import src.eval.store as store_module
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test_eval_routes.db")

    from tests.test_eval_store import _make_report
    import src.eval.store as store
    store.save_report(_make_report("crag"))
    store.save_report(_make_report("naive"))

    res = client.get("/eval/compare")
    assert res.status_code == 200
    data = res.json()
    assert "crag" in data or "naive" in data


# ---------------------------------------------------------------------------
# GET /eval/status
# ---------------------------------------------------------------------------

def test_eval_status_returns_running(client):
    from server import eval_routes
    eval_routes._running["test_pipe"] = "running"
    res = client.get("/eval/status")
    assert res.status_code == 200
    assert "test_pipe" in res.json()["running"]
    eval_routes._running.pop("test_pipe", None)
