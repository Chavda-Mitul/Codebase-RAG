"""SQLite-backed persistence for evaluation results."""
from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("results/eval.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline            TEXT    NOT NULL,
                timestamp           TEXT    NOT NULL,
                question_count      INTEGER NOT NULL DEFAULT 0,
                avg_faithfulness    REAL    NOT NULL DEFAULT 0,
                avg_answer_relevance REAL   NOT NULL DEFAULT 0,
                avg_context_recall  REAL    NOT NULL DEFAULT 0,
                avg_context_precision REAL  NOT NULL DEFAULT 0,
                avg_overall         REAL    NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_questions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER NOT NULL,
                qa_id               TEXT,
                question            TEXT    NOT NULL,
                answer              TEXT,
                faithfulness        REAL    NOT NULL DEFAULT 0,
                answer_relevance    REAL    NOT NULL DEFAULT 0,
                context_recall      REAL    NOT NULL DEFAULT 0,
                context_precision   REAL    NOT NULL DEFAULT 0,
                latency_ms          REAL    NOT NULL DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES eval_runs(id)
            )
        """)
        conn.commit()


def save_report(report) -> int:
    """Insert EvalReport into the DB. Returns the new run_id."""
    init_db()
    avg = report.avg_scores()
    ts = datetime.now(timezone.utc).isoformat()

    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO eval_runs
               (pipeline, timestamp, question_count,
                avg_faithfulness, avg_answer_relevance,
                avg_context_recall, avg_context_precision, avg_overall)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                report.pipeline, ts, len(report.results),
                avg.faithfulness, avg.answer_relevance,
                avg.context_recall, avg.context_precision, avg.average(),
            ),
        )
        run_id = cur.lastrowid

        for r in report.results:
            conn.execute(
                """INSERT INTO eval_questions
                   (run_id, qa_id, question, answer,
                    faithfulness, answer_relevance,
                    context_recall, context_precision, latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, r.qa_id, r.question, r.answer[:500],
                    r.scores.faithfulness, r.scores.answer_relevance,
                    r.scores.context_recall, r.scores.context_precision,
                    r.latency_ms,
                ),
            )
        conn.commit()
    return run_id


def list_runs() -> list[dict]:
    """Return all eval runs sorted by timestamp descending."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    """Return a single run with all its question rows."""
    init_db()
    with _conn() as conn:
        run_row = conn.execute(
            "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            return None
        questions = conn.execute(
            "SELECT * FROM eval_questions WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    return {**dict(run_row), "questions": [dict(q) for q in questions]}


def compare_pipelines() -> dict[str, dict]:
    """Return {pipeline: avg_scores} for the latest run of each pipeline."""
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT r.*
            FROM eval_runs r
            INNER JOIN (
                SELECT pipeline, MAX(timestamp) AS max_ts
                FROM eval_runs
                GROUP BY pipeline
            ) latest ON r.pipeline = latest.pipeline AND r.timestamp = latest.max_ts
            ORDER BY r.pipeline
        """).fetchall()
    result: dict[str, dict] = {}
    for row in rows:
        d = dict(row)
        pipeline = d.pop("pipeline")
        result[pipeline] = d
    return result
