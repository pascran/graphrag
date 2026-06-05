"""GET /v1/jobs/{id} — poll status. GET /v1/jobs/{id}/stream — SSE.

The stream variant polls the job row at JOB_STREAM_POLL_SECONDS intervals
and emits `event: progress` lines until the row reaches a terminal status
(completed | failed), at which point it emits a single `event: done` and
closes. JOB_STREAM_MAX_SECONDS bounds total wall time as a safety net.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthenticatedTenant
from app.deps import current_tenant, get_db
from app.models.orm import Job
from app.models.schemas import JobOut

router = APIRouter(prefix="/v1", tags=["jobs"])


JOB_STREAM_POLL_SECONDS = 1.0
JOB_STREAM_MAX_SECONDS = 600.0  # 10 min safety cap
TERMINAL_STATUSES = {"completed", "failed"}


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(
    job_id: uuid.UUID,
    auth: AuthenticatedTenant = Depends(current_tenant),
    session: AsyncSession = Depends(get_db),
) -> Job:
    job = (
        await session.execute(
            select(Job).where(Job.id == job_id, Job.tenant_id == auth.tenant_id)
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return job


def _job_payload(job: Job) -> str:
    return json.dumps(
        {
            "id": str(job.id),
            "status": job.status,
            "progress": float(job.progress) if job.progress is not None else 0.0,
            "error": job.error,
        },
        separators=(",", ":"),
    )


@router.get("/jobs/{job_id}/stream")
async def stream_job(
    job_id: uuid.UUID,
    auth: AuthenticatedTenant = Depends(current_tenant),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    # First read happens before opening the stream so 404 returns a normal
    # JSON error response, not an SSE error event.
    first = (
        await session.execute(
            select(Job).where(Job.id == job_id, Job.tenant_id == auth.tenant_id)
        )
    ).scalar_one_or_none()
    if first is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    async def gen() -> AsyncIterator[str]:
        elapsed = 0.0
        last_payload: str | None = None
        current = first
        while True:
            payload = _job_payload(current)
            if current.status in TERMINAL_STATUSES:
                if payload != last_payload:
                    yield f"event: progress\ndata: {payload}\n\n"
                yield f"event: done\ndata: {payload}\n\n"
                return
            if payload != last_payload:
                yield f"event: progress\ndata: {payload}\n\n"
                last_payload = payload
            if elapsed >= JOB_STREAM_MAX_SECONDS:
                yield f"event: done\ndata: {payload}\n\n"
                return
            await asyncio.sleep(JOB_STREAM_POLL_SECONDS)
            elapsed += JOB_STREAM_POLL_SECONDS
            # End the current read transaction so the next query sees
            # writes committed by the celery worker since the last poll.
            await session.rollback()
            current = (
                await session.execute(
                    select(Job).where(
                        Job.id == job_id, Job.tenant_id == auth.tenant_id
                    )
                )
            ).scalar_one_or_none()
            if current is None:
                yield 'event: done\ndata: {"status":"missing"}\n\n'
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
