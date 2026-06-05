"""Qdrant hybrid (dense + sparse) search using BGE-M3 embeddings."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.ingest.embedder import embed_texts
from app.utils.logging import get_logger

log = get_logger("app.retrieve.vector")


@dataclass(frozen=True)
class VectorHit:
    filename: str
    page: int
    text: str
    score: float
    document_id: str
    chunk_index: int


async def hybrid_search(
    client: AsyncQdrantClient,
    *,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int = 5,
    payload_filters: dict | None = None,
) -> list[VectorHit]:
    settings = get_settings()
    embeds = embed_texts([question])
    if not embeds:
        return []
    q = embeds[0]

    must = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id)))]
    if payload_filters:
        for k, v in payload_filters.items():
            if v is None or isinstance(v, (dict, list, tuple, set)):
                continue
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    flt = qm.Filter(must=must)

    response = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            qm.Prefetch(
                query=q.dense,
                using="dense",
                limit=top_k * 4,
                filter=flt,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=q.sparse_indices, values=q.sparse_values),
                using="sparse",
                limit=top_k * 4,
                filter=flt,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    out: list[VectorHit] = []
    for p in response.points:
        pl = p.payload or {}
        out.append(
            VectorHit(
                filename=pl.get("filename", "?"),
                page=int(pl.get("page", 0) or 0),
                text=pl.get("text", ""),
                score=float(p.score or 0.0),
                document_id=str(pl.get("document_id", "")),
                chunk_index=int(pl.get("chunk_index", 0) or 0),
            )
        )
    log.info("vector_search", question_chars=len(question), hits=len(out), top_k=top_k)
    return out
