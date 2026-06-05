"""Redis async client + healthcheck."""
from __future__ import annotations

import redis.asyncio as aioredis

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.db.redis")

_client: aioredis.Redis | None = None


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def ping() -> str:
    try:
        pong = await get_client().ping()
        return "ok" if pong else "error: no pong"
    except Exception as e:
        log.warning("redis_ping_fail", error=str(e))
        return f"error: {type(e).__name__}"
