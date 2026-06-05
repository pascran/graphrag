"""GET /v1/jobs/{id} — poll status. SSE stream is a future Phase 3j enhancement."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthenticatedTenant
from app.deps import current_tenant, get_db
from app.models.orm import Job
from app.models.schemas import JobOut

router = APIRouter(prefix="/v1", tags=["jobs"])


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
