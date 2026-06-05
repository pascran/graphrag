"""Retrieval orchestrator — for now, vector-only.

Graph and hybrid modes will land in a follow-up alongside GraphRAG entity
extraction (Phase 3i).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from qdrant_client import AsyncQdrantClient

from app.generate.prompt import RetrievedChunk
from app.retrieve.vector import VectorHit, hybrid_search

Mode = Literal["auto", "vector", "graph", "hybrid", "fact", "analysis"]


@dataclass(frozen=True)
class RetrievalResult:
    mode_used: str
    chunks: list[RetrievedChunk]
    raw: list[VectorHit]


async def retrieve(
    qdrant: AsyncQdrantClient,
    *,
    tenant_id: uuid.UUID,
    question: str,
    mode: Mode = "auto",
    top_k: int = 5,
    payload_filters: dict | None = None,
) -> RetrievalResult:
    # Phase 4 v1: route everything to vector. Router (LLM-based) added later.
    hits = await hybrid_search(
        qdrant,
        tenant_id=tenant_id,
        question=question,
        top_k=top_k,
        payload_filters=payload_filters,
    )
    chunks = [RetrievedChunk(filename=h.filename, page=h.page, text=h.text) for h in hits]
    return RetrievalResult(mode_used="vector", chunks=chunks, raw=hits)
