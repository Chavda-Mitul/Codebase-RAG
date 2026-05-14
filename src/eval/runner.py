"""Eval runner: runs golden QA pairs through a pipeline and collects metrics."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from langchain_groq import ChatGroq
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.models import RetrievedNode
from src.qa.chain import answer_question, _format_context
from src.crag.graph import run_crag
from src.routing.pipeline import answer_with_routing
from src.eval.golden_qa import GoldenQA, GOLDEN_QA_PAIRS
from src.eval.metrics import evaluate_all, EvalScores
from src.eval.store import save_report as _store_save_report
from src.observability.tracer import trace_eval_result


@dataclass
class EvalResult:
    qa_id: str
    question: str
    pipeline: str
    answer: str
    ground_truth: str
    context: str
    scores: EvalScores
    latency_ms: float


@dataclass
class EvalReport:
    pipeline: str
    results: list[EvalResult] = field(default_factory=list)

    def avg_scores(self) -> EvalScores:
        if not self.results:
            return EvalScores()
        n = len(self.results)
        return EvalScores(
            faithfulness=round(sum(r.scores.faithfulness for r in self.results) / n, 3),
            answer_relevance=round(sum(r.scores.answer_relevance for r in self.results) / n, 3),
            context_recall=round(sum(r.scores.context_recall for r in self.results) / n, 3),
            context_precision=round(sum(r.scores.context_precision for r in self.results) / n, 3),
        )

    def print_summary(self) -> None:
        avg = self.avg_scores()
        print(f"\n{'='*55}")
        print(f"Pipeline: {self.pipeline}  ({len(self.results)} questions evaluated)")
        print(f"{'='*55}")
        print(f"  Faithfulness:      {avg.faithfulness:.3f}")
        print(f"  Answer Relevance:  {avg.answer_relevance:.3f}")
        print(f"  Context Recall:    {avg.context_recall:.3f}")
        print(f"  Context Precision: {avg.context_precision:.3f}")
        print(f"  Overall Average:   {avg.average():.3f}")

    def to_json(self) -> str:
        data = {
            "pipeline": self.pipeline,
            "avg_scores": self.avg_scores().model_dump(),
            "results": [
                {
                    "id": r.qa_id,
                    "question": r.question,
                    "answer": r.answer[:500],
                    "latency_ms": r.latency_ms,
                    "scores": r.scores.model_dump(),
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2)


def _run_naive(question: str, client: Neo4jClient) -> tuple[str, str, list[RetrievedNode]]:
    """Naive pipeline: hybrid retrieve → format → generate (no CRAG)."""
    docs = hybrid_retrieve(question, client, top_k=8)
    result = answer_question(question, client)
    context = _format_context(docs)
    return result.answer, context, docs


def _run_crag(question: str, client: Neo4jClient) -> tuple[str, str, list[RetrievedNode]]:
    state = run_crag(question, client)
    docs = state.get("filtered_documents") or state.get("documents", [])
    context = _format_context(docs)
    return state["answer"], context, docs


def _run_routed(question: str, client: Neo4jClient) -> tuple[str, str, list[RetrievedNode]]:
    result = answer_with_routing(question, client)
    context = _format_context(result.sources)
    return result.answer, context, result.sources


_RUNNERS = {
    "naive": _run_naive,
    "crag": _run_crag,
    "routed": _run_routed,
}


def run_eval(
    pipeline: str,
    client: Neo4jClient,
    qa_pairs: list[GoldenQA] | None = None,
    output_path: str | None = None,
    auto_save: bool = True,
) -> EvalReport:
    """Run evaluation for a given pipeline. pipeline: 'naive' | 'crag' | 'routed'."""
    assert pipeline in _RUNNERS, f"Unknown pipeline '{pipeline}'. Choose from: {list(_RUNNERS)}"
    runner = _RUNNERS[pipeline]
    pairs = qa_pairs or GOLDEN_QA_PAIRS
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    report = EvalReport(pipeline=pipeline)

    print(f"\nRunning eval: pipeline={pipeline}, questions={len(pairs)}")
    for qa in tqdm(pairs, desc=f"Eval [{pipeline}]"):
        t0 = time.perf_counter()
        try:
            answer, context, docs = runner(qa.question, client)
        except Exception as e:
            print(f"  [SKIP] {qa.id}: {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        doc_texts = [d.text for d in docs]
        scores = evaluate_all(
            question=qa.question,
            answer=answer,
            context=context,
            retrieved_doc_texts=doc_texts,
            ground_truth=qa.ground_truth,
            llm=llm,
        )

        result = EvalResult(
            qa_id=qa.id,
            question=qa.question,
            pipeline=pipeline,
            answer=answer,
            ground_truth=qa.ground_truth,
            context=context,
            scores=scores,
            latency_ms=round(latency_ms, 1),
        )
        report.results.append(result)

        # Send to Langfuse if configured
        trace_eval_result(qa.question, pipeline, scores.model_dump())

    if auto_save:
        try:
            run_id = _store_save_report(report)
            print(f"Report saved to eval store (run_id={run_id})")
        except Exception as e:
            print(f"[warn] Could not save report to store: {e}")

    if output_path:
        Path(output_path).write_text(report.to_json(), encoding="utf-8")
        print(f"Report saved to {output_path}")

    return report
