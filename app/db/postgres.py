"""PostgreSQL async client + healthcheck."""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.db.postgres")

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True, future=True)
    return _engine


async def ping() -> str:
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as e:
        log.warning("postgres_ping_fail", error=str(e))
        return f"error: {type(e).__name__}"
