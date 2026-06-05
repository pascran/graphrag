"""Unit tests for app.core.auth."""
from __future__ import annotations

import hashlib

from app.core.auth import (
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
