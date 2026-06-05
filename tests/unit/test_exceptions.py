"""Unit tests for app.core.exceptions — domain error class hierarchy."""
from __future__ import annotations

import pytest

from app.core.exceptions import (
    AppError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    UnauthorizedError,
    ValidationError,
)


@pytest.mark.parametrize(
    "cls,expected_status,expected_code",
    [
        (AppError, 500, "internal_error"),
        (UnauthorizedError, 401, "unauthorized"),
        (ForbiddenError, 403, "forbidden"),
        (NotFoundError, 404, "not_found"),
        (ValidationError, 422, "validation_error"),
        (RateLimitError, 429, "rate_limited"),
    ],
)
def test_each_exception_has_status_and_code(cls, expected_status, expected_code):
    assert cls.status_code == expected_status
    assert cls.code == expected_code


def test_subclasses_inherit_from_app_error():
    for sub in (UnauthorizedError, ForbiddenError, NotFoundError, ValidationError, RateLimitError):
        assert issubclass(sub, AppError)
        assert issubclass(sub, Exception)


def test_exception_message_is_preserved():
    exc = NotFoundError("missing widget xyz")
    assert str(exc) == "missing widget xyz"
    assert exc.status_code == 404


def test_status_codes_are_distinct():
    statuses = {AppError.status_code, UnauthorizedError.status_code,
                ForbiddenError.status_code, NotFoundError.status_code,
                ValidationError.status_code, RateLimitError.status_code}
    assert len(statuses) == 6
