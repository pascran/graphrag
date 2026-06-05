"""POST /v1/evaluate — RAGAS-style scoring for a question / RAG pipeline.

Runs the full retrieve+generate flow, then scores the answer against the
retrieved chunks using a configurable subset of four metrics:

    - faithfulness          (reference-free)
    - answer_relevancy      (reference-free)
    - context_precision     (reference-free)
    - context_recall        (requires expected_answer)

Default metric set = the three reference-free metrics. When the request
includes expected_answer, context_recall is added automatically.

Heavy: each metric runs multiple LLM calls. Not a streaming endpoint.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.core.auth import AuthenticatedTenant
from app.db import neo4j as neo4j_db
from app.db import qdrant as qdrant_db
from app.deps import current_tenant
from app.evaluate.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from app.generate.llm import chat_once
from app.generate.prompt import render_rag_prompt
from app.retrieve.orchestrator import retrieve
from app.utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["evaluate"])
log = get_logger("app.api.evaluate")

MetricName = Literal[
    "faithfulness", "answer_relevancy", "context_precision", "context_recall"
]
_DEFAULT_REFERENCE_FREE: tuple[MetricName, ...] = (
    "faithfulness", "answer_relevancy", "context_precision",
)


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str | None = Field(default=None, max_length=10_000)
    mode: str = "auto"
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str | int | bool | None] | None = None
    metrics: list[MetricName] | None = Field(
        default=None,
        description="Subset of metrics to run. Defaults to the 3 reference-free "
                    "metrics; if expected_answer is supplied, context_recall is "
                    "added automatically.",
    )

    @field_validator("question")
    @classmethod
    def _non_blank_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class EvaluateResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    mode_used: str
    scores: dict[str, float | None]


def _resolve_metrics(req: EvaluateRequest) -> list[MetricName]:
    if req.metrics:
        return list(req.metrics)
    metrics = list(_DEFAULT_REFERENCE_FREE)
    if req.expected_answer and req.expected_answer.strip():
        metrics.append("context_recall")
    return metrics


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    req: EvaluateRequest,
    auth: AuthenticatedTenant = Depends(current_tenant),
) -> EvaluateResponse:
    settings = get_settings()
    qclient = qdrant_db.get_client()
    driver = neo4j_db.get_driver() if settings.graph_retrieval_enabled else None

    retrieval = await retrieve(
        qclient,
        tenant_id=auth.tenant_id,
        question=req.question,
        mode=req.mode,  # type: ignore[arg-type]
        top_k=req.top_k,
        payload_filters=req.filters,
        neo4j=driver,
    )

    messages = render_rag_prompt(req.question, retrieval.chunks)
    answer = await chat_once(messages)

    chosen = _resolve_metrics(req)
    scores: dict[str, float | None] = {}
    for name in chosen:
        if name == "faithfulness":
            scores[name] = await faithfulness(
                answer=answer, chunks=retrieval.chunks
            )
        elif name == "answer_relevancy":
            scores[name] = await answer_relevancy(
                question=req.question, answer=answer
            )
        elif name == "context_precision":
            scores[name] = await context_precision(
                question=req.question, chunks=retrieval.chunks
            )
        elif name == "context_recall":
            if not (req.expected_answer and req.expected_answer.strip()):
                scores[name] = None
            else:
                scores[name] = await context_recall(
                    expected_answer=req.expected_answer,
                    chunks=retrieval.chunks,
                )

    log.info(
        "evaluate_done",
        chunks=len(retrieval.chunks),
        mode=retrieval.mode_used,
        metrics=chosen,
    )
    return EvaluateResponse(
        question=req.question,
        answer=answer,
        sources=[{"filename": c.filename, "page": c.page} for c in retrieval.chunks],
        mode_used=retrieval.mode_used,
        scores=scores,
    )
