"""Health check endpoint that pings every backing service."""
from __future__ import annotations

import asyncio
from typing import Literal, TypedDict

from fastapi import APIRouter

from app.config import get_settings
from app.db import neo4j as neo4j_client
from app.db import postgres as postgres_client
from app.db import qdrant as qdrant_client
from app.db import redis as redis_client
from app.db import vllm as vllm_client
from app.utils.logging import get_logger

router = APIRouter(tags=["health"])
log = get_logger("app.api.health")


class HealthResponse(TypedDict):
    status: Literal["ok", "degraded"]
    checks: dict[str, str]


@router.get("/health", response_model=None)
async def health() -> HealthResponse:
    settings = get_settings()
    checks = await asyncio.gather(
        postgres_client.ping(),
        qdrant_client.ping(),
        neo4j_client.ping(),
        redis_client.ping(),
        vllm_client.ping(settings.vllm_llm_url),
        vllm_client.ping(settings.vllm_ocr_url),
        return_exceptions=False,
    )
    keys = ["postgres", "qdrant", "neo4j", "redis", "vllm_gemma", "vllm_chandra"]
    result = dict(zip(keys, checks, strict=True))
    overall: Literal["ok", "degraded"] = "ok" if all(v == "ok" for v in result.values()) else "degraded"
    return {"status": overall, "checks": result}
