"""FastAPI dependencies — DB session, current tenant, etc."""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.auth import AuthenticatedTenant, authenticate, extract_bearer_token
from app.db.postgres import get_engine

_session_factory: async_sessionmaker[AsyncSession] | None = None


def _factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    async with _factory()() as session:
        yield session


async def current_tenant(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_db),
) -> AuthenticatedTenant:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    auth = await authenticate(session, token)
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth
