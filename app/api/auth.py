"""Auth verification endpoint — confirms API key resolves to a tenant."""
from __future__ import annotations

import uuid
from typing import TypedDict

from fastapi import APIRouter, Depends

from app.core.auth import AuthenticatedTenant
from app.deps import current_tenant

router = APIRouter(prefix="/v1", tags=["auth"])


class MeResponse(TypedDict):
    tenant_id: uuid.UUID
    tenant_name: str
    api_key_id: uuid.UUID


@router.get("/me", response_model=None)
async def me(auth: AuthenticatedTenant = Depends(current_tenant)) -> MeResponse:
    return {
        "tenant_id": auth.tenant_id,
        "tenant_name": auth.tenant_name,
        "api_key_id": auth.api_key_id,
    }
