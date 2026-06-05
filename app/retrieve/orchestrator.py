"""Retrieval orchestrator.

Vector search via Qdrant is always primary. When a Neo4j driver is passed
in and the intent is non-casual, GraphRAG local search runs in parallel
and its hits are prepended to the result, deduped against vector hits by
(document_id, chunk_index). Graph failures degrade silently to vector-only.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal

from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from app.generate.prompt import RetrievedChunk
from app.retrieve.graph import GraphHit, graph_search
from app.retrieve.router import classify
from app.retrieve.vector import VectorHit, hybrid_search
from app.utils.logging import get_logger

log = get_logger("app.retrieve.orchestrator")

Mode = Literal["auto", "vector", "graph", "hybrid", "fact", "analysis", "casual"]


@dataclass(frozen=True)
class RetrievalResult:
    mode_used: str
    chunks: list[RetrievedChunk]
    raw: list[VectorHit]


def _merge(
    graph_hits: list[GraphHit], vector_hits: list[VectorHit]
) -> list[RetrievedChunk]:
    """Graph hits first, then vector hits with dedup on (document_id, chunk_index)."""
    seen: set[tuple[str, int]] = set()
    out: list[RetrievedChunk] = []
    for g in graph_hits:
        key = (g.document_id, g.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(RetrievedChunk(filename=g.filename, page=g.page, text=g.text))
    for v in vector_hits:
        key = (v.document_id, v.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(RetrievedChunk(filename=v.filename, page=v.page, text=v.text))
    return out


async def retrieve(
    qdrant: AsyncQdrantClient,
    *,
    tenant_id: uuid.UUID,
    question: str,
    mode: Mode = "auto",
    top_k: int = 5,
    payload_filters: dict | None = None,
    neo4j: AsyncDriver | None = None,
) -> RetrievalResult:
    if mode == "auto":
        intent = await classify(question)
    elif mode in ("casual", "fact", "analysis"):
        intent = mode
    else:
        intent = "fact"

    if intent == "casual":
        return RetrievalResult(mode_used="casual", chunks=[], raw=[])

    effective_k = top_k * 2 if intent == "analysis" else top_k

    vector_coro = hybrid_search(
        qdrant,
        tenant_id=tenant_id,
        question=question,
        top_k=effective_k,
        payload_filters=payload_filters,
    )

    if neo4j is None:
        vector_hits = await vector_coro
        chunks = _merge([], vector_hits)
        return RetrievalResult(mode_used=intent, chunks=chunks, raw=vector_hits)

    graph_coro = graph_search(
        neo4j, tenant_id=tenant_id, question=question, top_k=effective_k,
    )
    results = await asyncio.gather(vector_coro, graph_coro, return_exceptions=True)
    vector_out, graph_out = results

    if isinstance(vector_out, BaseException):
        raise vector_out
    if isinstance(graph_out, BaseException):
        log.warning("graph_search_failed", error=str(graph_out))
        graph_out = []

    chunks = _merge(graph_out, vector_out)
    return RetrievalResult(mode_used=intent, chunks=chunks, raw=vector_out)
