"""Ingestion pipeline orchestrator: OCR -> chunk -> embed -> vector + graph index."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.ingest.chunker import chunk_pages
from app.ingest.embedder import embed_texts
from app.ingest.graph_indexer import GraphChunk, upsert_chunks_skeleton
from app.ingest.ocr import ocr_file
from app.ingest.vector_indexer import IndexableChunk, upsert_chunks
from app.utils.logging import get_logger

log = get_logger("app.ingest.pipeline")


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    chunk_count: int
    page_count: int


async def ingest_document(
    *,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    file_path: str | Path,
    filename: str,
    doc_type: str | None = None,
) -> IngestResult:
    settings = get_settings()
    log.info("ingest_start", document_id=str(document_id), filename=filename)

    pages = await ocr_file(file_path)
    page_pairs = [(p.page_number, p.markdown) for p in pages]
    chunks = chunk_pages(page_pairs, settings.chunk_size, settings.chunk_overlap)
    log.info("ocr_chunked", pages=len(pages), chunks=len(chunks))
    if not chunks:
        return IngestResult(document_id=document_id, chunk_count=0, page_count=len(pages))

    embeddings = embed_texts([c.text for c in chunks])
    indexable = [
        IndexableChunk(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            page=c.page_number,
            doc_type=doc_type,
            chunk_index=c.index,
            text=c.text,
            embedding=embeddings[i],
        )
        for i, c in enumerate(chunks)
    ]

    # Fresh, locally-scoped clients — celery tasks each create a new asyncio
    # event loop, so module-level cached clients leak "Event loop is closed".
    qclient = AsyncQdrantClient(url=settings.qdrant_url)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_url, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        await upsert_chunks(qclient, indexable)

        graph_chunks = [
            GraphChunk(
                chunk_id=ic.id,
                tenant_id=ic.tenant_id,
                document_id=ic.document_id,
                filename=ic.filename,
                page=ic.page,
                text=ic.text,
            )
            for ic in indexable
        ]
        await upsert_chunks_skeleton(driver, graph_chunks)
    finally:
        await qclient.close()
        await driver.close()

    log.info("ingest_done", document_id=str(document_id), chunks=len(indexable))
    return IngestResult(
        document_id=document_id, chunk_count=len(indexable), page_count=len(pages)
    )
