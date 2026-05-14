"""Tracer: wraps pipeline calls with Langfuse spans for observability."""
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Any
from src.observability.langfuse_handler import get_langfuse_client, is_tracing_enabled


@contextmanager
def trace_span(name: str, input: dict | None = None, metadata: dict | None = None):
    """Context manager that creates a Langfuse span if tracing is enabled, otherwise no-op."""
    client = get_langfuse_client() if is_tracing_enabled() else None
    trace = None
    start = time.perf_counter()

    if client:
        try:
            trace = client.trace(name=name, input=input or {}, metadata=metadata or {})
        except Exception:
            trace = None

    try:
        yield trace
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if trace:
            try:
                trace.update(metadata={**(metadata or {}), "elapsed_ms": round(elapsed_ms, 1)})
                client.flush()
            except Exception:
                pass


def trace_pipeline_run(
    question: str,
    route: str,
    answer: str,
    sources: list,
    corrections: int = 0,
    hallucination_check: str = "",
    latency_ms: float = 0.0,
) -> None:
    """Log a complete pipeline run to Langfuse."""
    client = get_langfuse_client() if is_tracing_enabled() else None
    if not client:
        return
    try:
        trace = client.trace(
            name="code-rag-query",
            input={"question": question},
            output={"answer": answer},
            metadata={
                "route": route,
                "sources_count": len(sources),
                "corrections": corrections,
                "hallucination_check": hallucination_check,
                "latency_ms": round(latency_ms, 1),
            },
        )
        client.flush()
    except Exception:
        pass


def trace_eval_result(
    question: str,
    pipeline: str,
    metrics: dict[str, float],
) -> None:
    """Log eval metrics as a Langfuse score."""
    client = get_langfuse_client() if is_tracing_enabled() else None
    if not client:
        return
    try:
        trace = client.trace(
            name="code-rag-eval",
            input={"question": question, "pipeline": pipeline},
            metadata=metrics,
        )
        for metric_name, value in metrics.items():
            client.score(
                trace_id=trace.id,
                name=metric_name,
                value=value,
            )
        client.flush()
    except Exception:
        pass
