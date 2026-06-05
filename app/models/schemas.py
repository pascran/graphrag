"""Pydantic request/response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TenantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    created_at: datetime


class HealthCheckOut(BaseModel):
    status: str
    checks: dict[str, str]


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    file_hash: str
    doc_type: str | None
    page_count: int | None
    status: str
    created_at: datetime


class UploadAccepted(BaseModel):
    job_id: uuid.UUID
    accepted_files: list[str]
    rejected_files: list[dict]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = "auto"
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str | int | bool | None] | None = Field(
        default=None,
        examples=[None],
        description="Optional payload filters as a flat object. Values must be primitives. "
        "Leave null for no filter. Example when needed: {\"doc_type\": \"manual\"}.",
    )
    stream: bool = Field(
        default=True,
        description="If false, returns a single JSON with the full answer + sources. "
        "Use false in Swagger UI for easy manual testing.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional opaque conversation id. If supplied, prior turns "
        "(up to 30 min sliding TTL) are prepended to the prompt. Reuse the same "
        "value across calls in one chat thread.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode_used: str
    latency_ms: int
