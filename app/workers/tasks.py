"""Celery tasks for ingestion."""
from __future__ import annotations

import asyncio
import uuid

from celery.utils.log import get_task_logger
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.ingest.pipeline import ingest_document
from app.models.orm import Document, Job
from app.workers.celery_app import celery_app

clog = get_task_logger(__name__)


async def _set_status(*, job_id: uuid.UUID | None, document_id: uuid.UUID | None,
                     status: str, progress: float | None = None,
                     error: str | None = None) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        if job_id is not None:
            stmt = update(Job).where(Job.id == job_id).values(
                status=status,
                **({"progress": progress} if progress is not None else {}),
                **({"error": error} if error else {}),
            )
            await session.execute(stmt)
        if document_id is not None:
            await session.execute(
                update(Document).where(Document.id == document_id).values(status=status)
            )
        await session.commit()
    await engine.dispose()


async def _run(*, tenant_id: uuid.UUID, document_id: uuid.UUID, job_id: uuid.UUID,
              file_path: str, filename: str, doc_type: str | None) -> None:
    try:
        await _set_status(job_id=job_id, document_id=document_id, status="running",
                          progress=0.0)
        result = await ingest_document(
            tenant_id=tenant_id,
            document_id=document_id,
            file_path=file_path,
            filename=filename,
            doc_type=doc_type,
        )
        clog.info("ingested document_id=%s chunks=%d", document_id, result.chunk_count)
        await _set_status(job_id=job_id, document_id=document_id, status="completed",
                          progress=1.0)
    except Exception as e:
        clog.exception("ingest_failed document_id=%s", document_id)
        await _set_status(job_id=job_id, document_id=document_id, status="failed",
                          error=f"{type(e).__name__}: {e}")
        raise


@celery_app.task(name="ingest.document", bind=True, max_retries=0)
def ingest_document_task(
    self,
    tenant_id: str,
    document_id: str,
    job_id: str,
    file_path: str,
    filename: str,
    doc_type: str | None,
) -> dict:
    asyncio.run(
        _run(
            tenant_id=uuid.UUID(tenant_id),
            document_id=uuid.UUID(document_id),
            job_id=uuid.UUID(job_id),
            file_path=file_path,
            filename=filename,
            doc_type=doc_type,
        )
    )
    return {"document_id": document_id, "status": "completed"}
