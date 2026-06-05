"""Celery application bootstrap.

Phase 1: empty queue — only verifies that the worker can boot and connect to Redis.
Real tasks are added in Phase 3.
"""
from __future__ import annotations

from celery import Celery

from app.config import get_settings
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging(level=settings.app_log_level)

celery_app = Celery(
    "llm_engine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
    worker_prefetch_multiplier=1,
    timezone="UTC",
    enable_utc=True,
)
