"""Celery application bootstrap.

Phase 1: empty queue — only verifies that the worker can boot and connect to Redis.
Real tasks are added in Phase 3.

The orphan-:Entity cleanup task (``cleanup.orphan_entities``) is registered
via the ``include`` list below. A nightly beat schedule entry runs it in
``dry_run=True`` mode by default, gated on ``Settings.cleanup_orphans_enabled``.
"""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging(level=settings.app_log_level)

celery_app = Celery(
    "llm_engine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks", "app.workers.cleanup"],
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

# Beat schedule (env-gated). When CLEANUP_ORPHANS_ENABLED is false (default),
# nothing is registered so a stray celery-beat process cannot delete anything.
if settings.cleanup_orphans_enabled:
    celery_app.conf.beat_schedule = {
        "orphan-entities-nightly": {
            "task": "cleanup.orphan_entities",
            "schedule": crontab(hour=4, minute=15),
            "kwargs": {"dry_run": True},
        },
    }
