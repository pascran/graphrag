"""Unit tests for app.core.limiter._bucket_key."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.auth import hash_api_key
from app.core.limiter import _bucket_key, limiter


def _request(*, headers: dict[str, str] | None = None, peer: str = "10.0.0.42"):
    """Smallest object satisfying what _bucket_key + slowapi.get_remote_address read."""
    return SimpleNamespace(
        headers=headers or {},
        client=SimpleNamespace(host=peer),
        scope={"client": (peer, 0), "type": "http"},
    )


def test_authed_request_uses_token_hash_prefix():
    token = "graphrag_abc123"
    req = _request(headers={"Authorization": f"Bearer {token}"})
    bucket = _bucket_key(req)
    assert bucket == f"key:{hash_api_key(token)[:16]}"
    assert bucket.startswith("key:")


def test_anonymous_request_falls_back_to_ip():
    req = _request(peer="192.0.2.10")
    bucket = _bucket_key(req)
    assert bucket.startswith("ip:")
    assert "192.0.2.10" in bucket


def test_malformed_authorization_falls_back_to_ip():
    req = _request(headers={"Authorization": "Token nope"}, peer="203.0.113.5")
    bucket = _bucket_key(req)
    assert bucket.startswith("ip:")


def test_distinct_tokens_get_distinct_buckets():
    a = _request(headers={"Authorization": "Bearer key-a"})
    b = _request(headers={"Authorization": "Bearer key-b"})
    assert _bucket_key(a) != _bucket_key(b)


def test_same_token_is_deterministic():
    req1 = _request(headers={"Authorization": "Bearer same-key"})
    req2 = _request(headers={"Authorization": "Bearer same-key"})
    assert _bucket_key(req1) == _bucket_key(req2)


def test_limiter_module_singleton_is_configured():
    assert limiter is not None
    assert limiter._default_limits, "default_limits should be set from settings"
