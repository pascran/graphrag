"""FastAPI entry point for the graphrag llm-engine."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.api import auth as auth_api
from app.api import documents as documents_api
from app.api import health as health_api
from app.api import jobs as jobs_api
from app.api import query as query_api
from app.api import upload as upload_api
from app.config import get_settings
from app.core.limiter import limiter
from app.utils.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

settings = get_settings()
configure_logging(level=settings.app_log_level)
log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", env=settings.app_env, port=settings.app_port)
    yield
    log.info("shutdown")


app = FastAPI(
    title="graphrag LLM Engine",
    version="0.1.0",
    description="Hybrid RAG (Vector + Graph) LLM Engine for DGX Spark",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    bind_request_context(request_id=request_id, path=request.url.path, method=request.method)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
            client=request.client.host if request.client else None,
        )
        clear_request_context()


app.include_router(health_api.router)
app.include_router(auth_api.router)
app.include_router(upload_api.router)
app.include_router(jobs_api.router)
app.include_router(query_api.router)
app.include_router(documents_api.router)
