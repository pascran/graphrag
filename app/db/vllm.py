"""vLLM OpenAI-compatible endpoint healthcheck."""
from __future__ import annotations

import httpx

from app.utils.logging import get_logger

log = get_logger("app.db.vllm")


async def ping(base_url: str, timeout: float = 3.0) -> str:
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(health_url)
            r.raise_for_status()
        return "ok"
    except Exception as e:
        log.warning("vllm_ping_fail", url=health_url, error=str(e))
        return f"error: {type(e).__name__}"
