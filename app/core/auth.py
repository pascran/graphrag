"""API Key validation + tenant resolution.

Strategy: store key_hash = sha256(key). Constant-time compare via hashlib digest.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import ApiKey, Tenant


def hash_api_key(plain_key: str) -> str:
    return hashlib.sha256(plain_key.encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "graphrag") -> str:
    """Generate a fresh, opaque key. Uses URL-safe base64."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class AuthenticatedTenant(NamedTuple):
    tenant_id: uuid.UUID
    tenant_name: str
    api_key_id: uuid.UUID


async def authenticate(session: AsyncSession, plain_key: str) -> AuthenticatedTenant | None:
    if not plain_key:
        return None
    digest = hash_api_key(plain_key)
    stmt = (
        select(ApiKey, Tenant)
        .join(Tenant, Tenant.id == ApiKey.tenant_id)
        .where(ApiKey.key_hash == digest, ApiKey.is_active.is_(True))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if not row:
        return None
    api_key, tenant = row
    return AuthenticatedTenant(
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        api_key_id=api_key.id,
    )


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
