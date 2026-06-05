"""POST /v1/upload — accept files, dedupe by SHA256, enqueue Celery ingest."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import AuthenticatedTenant
from app.deps import current_tenant, get_db
from app.models.orm import Document, Job
from app.utils.hashing import sha256_bytes
from app.utils.logging import get_logger
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/v1", tags=["upload"])
log = get_logger("app.api.upload")

UPLOAD_DIR = Path("/data/uploads")
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED, response_model=None)
async def upload(
    files: Annotated[list[UploadFile], File(description="PDFs / images")],
    doc_type: Annotated[str | None, Form()] = None,
    auth: AuthenticatedTenant = Depends(current_tenant),
    session: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if len(files) > settings.upload_max_files_per_request:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"too many files (max {settings.upload_max_files_per_request})",
        )

    job = Job(tenant_id=auth.tenant_id, status="pending", progress=0.0,
              payload={"filenames": [f.filename for f in files]})
    session.add(job)
    await session.flush()

    accepted: list[dict] = []
    rejected: list[dict] = []
    enqueue: list[dict] = []
    max_bytes = settings.upload_max_file_size_mb * 1024 * 1024

    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXT:
            rejected.append({"name": f.filename, "reason": f"unsupported extension {suffix}"})
            continue
        data = await f.read()
        if len(data) == 0:
            rejected.append({"name": f.filename, "reason": "empty file"})
            continue
        if len(data) > max_bytes:
            rejected.append({"name": f.filename, "reason": f"exceeds {settings.upload_max_file_size_mb} MB"})
            continue

        digest = sha256_bytes(data)
        existing = (
            await session.execute(
                select(Document).where(
                    Document.tenant_id == auth.tenant_id, Document.file_hash == digest
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            rejected.append({"name": f.filename, "reason": "duplicate (same content already indexed)"})
            continue

        document = Document(
            tenant_id=auth.tenant_id,
            filename=f.filename or "",
            file_hash=digest,
            doc_type=doc_type,
            status="pending",
        )
        session.add(document)
        await session.flush()

        target = UPLOAD_DIR / f"{document.id}{suffix}"
        target.write_bytes(data)
        accepted.append({"name": f.filename, "document_id": str(document.id)})
        enqueue.append(
            {
                "tenant_id": str(auth.tenant_id),
                "document_id": str(document.id),
                "job_id": str(job.id),
                "file_path": str(target),
                "filename": f.filename or "",
                "doc_type": doc_type,
            }
        )

    await session.commit()

    for kwargs in enqueue:
        celery_app.send_task("ingest.document", kwargs=kwargs)

    log.info("upload_accepted", job_id=str(job.id),
             accepted=len(accepted), rejected=len(rejected))

    return {
        "job_id": str(job.id),
        "accepted_files": accepted,
        "rejected_files": rejected,
    }
