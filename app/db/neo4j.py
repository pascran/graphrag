"""Neo4j async driver + healthcheck."""
from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.db.neo4j")

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def ping() -> str:
    try:
        async with get_driver().session() as s:
            result = await s.run("RETURN 1 AS ok")
            await result.consume()
        return "ok"
    except Exception as e:
        log.warning("neo4j_ping_fail", error=str(e))
        return f"error: {type(e).__name__}"
