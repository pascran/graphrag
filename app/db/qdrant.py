"""Qdrant async client + healthcheck."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.db.qdrant")

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def ping() -> str:
    try:
        await get_client().get_collections()
        return "ok"
    except Exception as e:
        log.warning("qdrant_ping_fail", error=str(e))
        return f"error: {type(e).__name__}"
