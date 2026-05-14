"""RAGAS-style evaluation metrics implemented with Groq structured output.

Metrics:
  - Faithfulness:        fraction of answer claims supported by retrieved context
  - Answer Relevance:    how well the answer addresses the question
  - Context Recall:      fraction of ground-truth statements found in context
  - Context Precision:   precision-at-k weighting relevant retrieved docs higher
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import settings


def _llm() -> ChatGroq:
    return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

class ClaimList(BaseModel):
    claims: list[str] = Field(description="Atomic factual claims extracted from the text")


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool = Field(description="True if the claim is supported by the provided context")


class FaithfulnessResult(BaseModel):
    score: float          # 0.0 – 1.0
    supported: int
    total: int
    verdicts: list[ClaimVerdict]


_CLAIM_EXTRACTION_SYSTEM = """Extract every atomic factual claim from the given answer text.
Each claim should be a single, self-contained statement that can be verified independently.
Focus on claims about the codebase: function names, class names, relationships, behaviors."""

_CLAIM_VERIFY_SYSTEM = """You are a fact-checker for a code Q&A system.
Given a context and a claim about the codebase, decide if the claim is supported by the context.
A claim is supported if the context explicitly mentions or clearly implies it."""

_CLAIM_VERIFY_USER = """Context:
{context}

Claim: {claim}

Is this claim supported by the context?"""


def compute_faithfulness(answer: str, context: str, llm: ChatGroq | None = None) -> FaithfulnessResult:
    """Score how many answer claims are grounded in the retrieved context."""
    if llm is None:
        llm = _llm()

    # Step 1: extract claims
    claim_extractor = llm.with_structured_output(ClaimList)
    try:
        extracted = claim_extractor.invoke([
            {"role": "system", "content": _CLAIM_EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Answer:\n{answer}"},
        ])
        claims = extracted.claims
    except Exception:
        return FaithfulnessResult(score=0.0, supported=0, total=0, verdicts=[])

    if not claims:
        return FaithfulnessResult(score=1.0, supported=0, total=0, verdicts=[])

    class _SupportedBool(BaseModel):
        supported: bool

    # Step 2: verify each claim against context
    verifier = llm.with_structured_output(_SupportedBool)
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        try:
            result = verifier.invoke([
                {"role": "system", "content": _CLAIM_VERIFY_SYSTEM},
                {"role": "user", "content": _CLAIM_VERIFY_USER.format(context=context, claim=claim)},
            ])
            verdicts.append(ClaimVerdict(claim=claim, supported=result.supported))
        except Exception:
            verdicts.append(ClaimVerdict(claim=claim, supported=False))

    supported = sum(1 for v in verdicts if v.supported)
    score = supported / len(verdicts) if verdicts else 1.0
    return FaithfulnessResult(score=round(score, 3), supported=supported, total=len(verdicts), verdicts=verdicts)


# ---------------------------------------------------------------------------
# Answer Relevance
# ---------------------------------------------------------------------------

class AnswerRelevanceResult(BaseModel):
    score: float   # 0.0 – 1.0
    reason: str


class _RelevanceScore(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0: how well the answer addresses the question", ge=0.0, le=1.0)
    reason: str = Field(description="One sentence explanation")


_AR_SYSTEM = """You are an answer relevance evaluator for a code Q&A system.
Score how well the answer addresses the question on a scale of 0.0 to 1.0:
  1.0 = fully answers the question with specific, accurate details
  0.7 = mostly answers but misses some aspects
  0.4 = partially answers or is too vague
  0.0 = does not address the question at all"""

_AR_USER = """Question: {question}

Answer: {answer}

Score the answer relevance:"""


def compute_answer_relevance(question: str, answer: str, llm: ChatGroq | None = None) -> AnswerRelevanceResult:
    if llm is None:
        llm = _llm()
    grader = llm.with_structured_output(_RelevanceScore)
    try:
        result = grader.invoke([
            {"role": "system", "content": _AR_SYSTEM},
            {"role": "user", "content": _AR_USER.format(question=question, answer=answer)},
        ])
        return AnswerRelevanceResult(score=round(result.score, 3), reason=result.reason)
    except Exception:
        return AnswerRelevanceResult(score=0.0, reason="evaluation failed")


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------

class ContextRecallResult(BaseModel):
    score: float    # 0.0 – 1.0
    attributed: int
    total: int


class _StatementAttribution(BaseModel):
    attributed: bool = Field(description="True if this ground-truth statement can be found in the context")


_CR_SYSTEM = """You are evaluating whether a ground-truth statement about a codebase
can be attributed to (found in) the given retrieved context.
Return True only if the context clearly contains or implies the statement."""

_CR_USER = """Context:
{context}

Ground-truth statement: {statement}

Is this statement present in the context?"""

class _StatementList(BaseModel):
    statements: list[str] = Field(description="Atomic statements extracted from the ground truth answer")


def compute_context_recall(
    ground_truth: str,
    context: str,
    llm: ChatGroq | None = None,
) -> ContextRecallResult:
    """Fraction of ground-truth statements attributable to retrieved context."""
    if llm is None:
        llm = _llm()

    # Extract atomic statements from ground truth
    extractor = llm.with_structured_output(_StatementList)
    try:
        extracted = extractor.invoke([
            {"role": "system", "content": "Extract atomic factual statements from the text. Each statement should be a single verifiable fact."},
            {"role": "user", "content": f"Text:\n{ground_truth}"},
        ])
        statements = extracted.statements
    except Exception:
        return ContextRecallResult(score=0.0, attributed=0, total=0)

    if not statements:
        return ContextRecallResult(score=1.0, attributed=0, total=0)

    attributor = llm.with_structured_output(_StatementAttribution)
    attributed = 0
    for stmt in statements:
        try:
            result = attributor.invoke([
                {"role": "system", "content": _CR_SYSTEM},
                {"role": "user", "content": _CR_USER.format(context=context, statement=stmt)},
            ])
            if result.attributed:
                attributed += 1
        except Exception:
            pass

    score = attributed / len(statements) if statements else 1.0
    return ContextRecallResult(score=round(score, 3), attributed=attributed, total=len(statements))


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------

class ContextPrecisionResult(BaseModel):
    score: float    # 0.0 – 1.0 (average precision, weighted by rank)
    relevant_count: int
    total_count: int


class _DocRelevance(BaseModel):
    relevant: bool = Field(description="True if this document is useful for answering the question")


_CP_SYSTEM = """You are evaluating retrieved documents for a code Q&A system.
Decide whether a given retrieved code artifact (function, class, file, concept)
is useful for answering the question."""

_CP_USER = """Question: {question}

Retrieved document:
{document}

Is this document useful for answering the question?"""


def compute_context_precision(
    question: str,
    retrieved_docs: list[str],
    llm: ChatGroq | None = None,
) -> ContextPrecisionResult:
    """Average precision at K, weighting relevant docs appearing earlier more highly."""
    if llm is None:
        llm = _llm()
    if not retrieved_docs:
        return ContextPrecisionResult(score=0.0, relevant_count=0, total_count=0)

    grader = llm.with_structured_output(_DocRelevance)
    relevance: list[bool] = []

    for doc in retrieved_docs:
        try:
            result = grader.invoke([
                {"role": "system", "content": _CP_SYSTEM},
                {"role": "user", "content": _CP_USER.format(question=question, document=doc)},
            ])
            relevance.append(result.relevant)
        except Exception:
            relevance.append(False)

    # Average precision: sum over positions where doc is relevant of precision@k
    precision_sum = 0.0
    relevant_so_far = 0
    for k, is_rel in enumerate(relevance, 1):
        if is_rel:
            relevant_so_far += 1
            precision_sum += relevant_so_far / k

    total_relevant = sum(relevance)
    avg_precision = precision_sum / total_relevant if total_relevant > 0 else 0.0

    return ContextPrecisionResult(
        score=round(avg_precision, 3),
        relevant_count=total_relevant,
        total_count=len(retrieved_docs),
    )


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

class EvalScores(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0

    def average(self) -> float:
        vals = [self.faithfulness, self.answer_relevance, self.context_recall, self.context_precision]
        return round(sum(vals) / len(vals), 3)


def evaluate_all(
    question: str,
    answer: str,
    context: str,
    retrieved_doc_texts: list[str],
    ground_truth: str | None = None,
    llm: ChatGroq | None = None,
) -> EvalScores:
    if llm is None:
        llm = _llm()

    faith = compute_faithfulness(answer, context, llm=llm)
    ar = compute_answer_relevance(question, answer, llm=llm)
    cp = compute_context_precision(question, retrieved_doc_texts, llm=llm)
    cr = compute_context_recall(ground_truth or answer, context, llm=llm) if ground_truth else ContextRecallResult(score=0.0, attributed=0, total=0)

    return EvalScores(
        faithfulness=faith.score,
        answer_relevance=ar.score,
        context_recall=cr.score,
        context_precision=cp.score,
    )
