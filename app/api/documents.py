"""GET /v1/documents — list. DELETE /v1/documents/{id} — cascade across PG, Qdrant, Neo4j, uploads."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete as sqla_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthenticatedTenant
from app.db import neo4j as neo4j_db
from app.db import qdrant as qdrant_db
from app.deps import current_tenant, get_db
from app.ingest.graph_indexer import delete_document_graph
from app.ingest.vector_indexer import delete_document_chunks
from app.models.orm import Document
from app.models.schemas import DocumentOut
from app.utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["documents"])
log = get_logger("app.api.documents")

UPLOAD_DIR = Path("/data/uploads")


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    auth: AuthenticatedTenant = Depends(current_tenant),
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    doc_type: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
) -> list[Document]:
    stmt = select(Document).where(Document.tenant_id == auth.tenant_id)
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if status_filter:
        stmt = stmt.where(Document.status == status_filter)
    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    auth: AuthenticatedTenant = Depends(current_tenant),
    session: AsyncSession = Depends(get_db),
) -> None:
    doc = (
        await session.execute(
            select(Document).where(
                Document.id == document_id, Document.tenant_id == auth.tenant_id
            )
        )
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )

    # Vector
    try:
        await delete_document_chunks(qdrant_db.get_client(), auth.tenant_id, document_id)
    except Exception as e:
        log.warning("qdrant_delete_fail", document_id=str(document_id), error=str(e))

    # Graph
    try:
        await delete_document_graph(neo4j_db.get_driver(), auth.tenant_id, document_id)
    except Exception as e:
        log.warning("neo4j_delete_fail", document_id=str(document_id), error=str(e))

    # Uploaded file (best effort)
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        candidate = UPLOAD_DIR / f"{document_id}{ext}"
        if candidate.exists():
            try:
                os.unlink(candidate)
            except OSError as e:
                log.warning("file_delete_fail", path=str(candidate), error=str(e))

    # Postgres last (so we keep the row if upstream cascades fail catastrophically)
    await session.execute(
        sqla_delete(Document).where(
            Document.id == document_id, Document.tenant_id == auth.tenant_id
        )
    )
    await session.commit()
    log.info("document_deleted", document_id=str(document_id), tenant_id=str(auth.tenant_id))
