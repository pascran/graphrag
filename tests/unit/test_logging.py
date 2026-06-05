"""Unit tests for app.utils.logging — structlog setup + request-context binding."""
from __future__ import annotations

import json
import logging as stdlogging

import pytest
import structlog

from app.utils import logging as applog


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Force structlog and stdlib logging back to a clean state per test.

    logging.basicConfig captures sys.stdout at first call, so capsys
    won't see anything unless we drop the existing handlers first and
    let configure_logging re-attach them to the redirected stream.
    """
    applog._configured = False
    structlog.contextvars.clear_contextvars()
    root = stdlogging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    applog._configured = False
    structlog.contextvars.clear_contextvars()
    root = stdlogging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def test_configure_is_idempotent():
    applog.configure_logging(level="INFO")
    first = applog._configured
    applog.configure_logging(level="DEBUG")
    assert first is True and applog._configured is True


def test_get_logger_returns_bound_logger():
    applog.configure_logging(level="INFO")
    log = applog.get_logger("app.test")
    assert hasattr(log, "info")
    assert hasattr(log, "warning")


def test_get_logger_initial_context_is_attached(caplog):
    applog.configure_logging(level="INFO")
    log = applog.get_logger("app.test", request_id="rid-123")
    with caplog.at_level("INFO"):
        log.info("ping")
    payload = json.loads(caplog.records[-1].message)
    assert payload["request_id"] == "rid-123"
    assert payload["event"] == "ping"


def test_request_context_bind_and_clear(caplog):
    applog.configure_logging(level="INFO")
    log = applog.get_logger("app.test")

    with caplog.at_level("INFO"):
        applog.bind_request_context(request_id="abc", path="/v1/x", method="GET")
        log.info("during")
        payload_during = json.loads(caplog.records[-1].message)
        assert payload_during["request_id"] == "abc"
        assert payload_during["path"] == "/v1/x"
        assert payload_during["method"] == "GET"

        applog.clear_request_context()
        log.info("after")
        payload_after = json.loads(caplog.records[-1].message)
        assert "request_id" not in payload_after
        assert "path" not in payload_after


def test_log_output_is_json_with_iso_timestamp_and_level(caplog):
    applog.configure_logging(level="INFO")
    log = applog.get_logger("app.test")
    with caplog.at_level("INFO"):
        log.info("event_name", extra="value")
    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "event_name"
    assert payload["extra"] == "value"
    assert payload["level"] == "info"
    assert "T" in payload["timestamp"] and payload["timestamp"].endswith("Z")
