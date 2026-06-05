"""Structured logging setup using structlog."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_request_context(**ctx: Any) -> None:
    structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
