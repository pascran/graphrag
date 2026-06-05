"""Unit tests for app.core.auth."""
from __future__ import annotations

import hashlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.core.auth import (
    AuthenticatedTenant,
    authenticate,
    extract_bearer_token,
    generate_api_key,
    hash_api_key,
)


def test_hash_is_sha256_hex():
    plain = "graphrag_abcdef"
    assert hash_api_key(plain) == hashlib.sha256(plain.encode()).hexdigest()
    assert len(hash_api_key(plain)) == 64


def test_hash_is_deterministic_and_distinct_per_input():
    a = hash_api_key("alpha")
    b = hash_api_key("alpha")
    c = hash_api_key("beta")
    assert a == b
    assert a != c


def test_generate_api_key_is_unique_and_prefixed():
    keys = {generate_api_key() for _ in range(20)}
    assert len(keys) == 20
    assert all(k.startswith("graphrag_") for k in keys)


def test_generate_api_key_custom_prefix():
    k = generate_api_key(prefix="ops")
    assert k.startswith("ops_")
    assert len(k) > len("ops_") + 16


def test_bearer_extract_happy_path():
    assert extract_bearer_token("Bearer abc123") == "abc123"
    assert extract_bearer_token("bearer abc123") == "abc123"
    assert extract_bearer_token("BEARER   spaced-token  ") == "spaced-token"


def test_bearer_extract_rejects_malformed():
    assert extract_bearer_token(None) is None
    assert extract_bearer_token("") is None
    assert extract_bearer_token("Token abc") is None
    assert extract_bearer_token("Bearer") is None
    assert extract_bearer_token("abc123") is None


# ---------- authenticate() — DB-backed lookup --------------------------------

def _result_first(value):
    res = MagicMock()
    res.first.return_value = value
    return res


async def test_authenticate_returns_none_for_empty_token():
    session = MagicMock()
    session.execute = AsyncMock()
    assert await authenticate(session, "") is None
    session.execute.assert_not_awaited()


async def test_authenticate_returns_none_when_no_row_matches():
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_first(None))
    assert await authenticate(session, "graphrag_unknown") is None
    session.execute.assert_awaited_once()


async def test_authenticate_returns_tenant_on_match():
    api_key_row = SimpleNamespace(id=uuid.uuid4(), tenant_id=uuid.uuid4())
    tenant_row = SimpleNamespace(id=api_key_row.tenant_id, name="acme")
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_first((api_key_row, tenant_row)))

    result = await authenticate(session, "graphrag_real")
    assert isinstance(result, AuthenticatedTenant)
    assert result.tenant_id == tenant_row.id
    assert result.tenant_name == "acme"
    assert result.api_key_id == api_key_row.id
