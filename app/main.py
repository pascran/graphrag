"""FastAPI entry point for the graphrag llm-engine."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth as auth_api
from app.api import documents as documents_api
from app.api import health as health_api
from app.api import jobs as jobs_api
from app.api import query as query_api
from app.api import upload as upload_api
from app.config import get_settings
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
    try:
        response = await call_next(request)
    finally:
        clear_request_context()
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health_api.router)
app.include_router(auth_api.router)
app.include_router(upload_api.router)
app.include_router(jobs_api.router)
app.include_router(query_api.router)
app.include_router(documents_api.router)
