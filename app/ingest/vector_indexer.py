"""Qdrant vector indexer — upserts named dense + sparse vectors with payload."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.ingest.embedder import EmbeddedChunk
from app.utils.logging import get_logger

log = get_logger("app.ingest.vector_indexer")


@dataclass(frozen=True)
class IndexableChunk:
    id: uuid.UUID
    tenant_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int
    doc_type: str | None
    chunk_index: int
    text: str
    embedding: EmbeddedChunk


async def upsert_chunks(client: AsyncQdrantClient, chunks: list[IndexableChunk]) -> int:
    if not chunks:
        return 0
    settings = get_settings()
    points: list[qm.PointStruct] = []
    for c in chunks:
        points.append(
            qm.PointStruct(
                id=str(c.id),
                vector={
                    "dense": c.embedding.dense,
                    "sparse": qm.SparseVector(
                        indices=c.embedding.sparse_indices,
                        values=c.embedding.sparse_values,
                    ),
                },
                payload={
                    "tenant_id": str(c.tenant_id),
                    "document_id": str(c.document_id),
                    "filename": c.filename,
                    "page": c.page,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                },
            )
        )
    await client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    log.info("qdrant_upsert", count=len(points))
    return len(points)


async def delete_document_chunks(
    client: AsyncQdrantClient, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> None:
    settings = get_settings()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id))),
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=str(document_id))),
            ]
        ),
    )
