"""Per-tenant + per-IP rate limiting with slowapi, backed by Redis.

Uses tenant_id (extracted from the Bearer token's hash, never the raw key) as
the limit key when an Authorization header is present. Falls back to client IP
for unauthenticated routes such as /healthz.

The hash is what's persisted in api_keys.key_hash, so it's a stable, low-
sensitivity identifier — and using it as the bucket key avoids leaking
plaintext keys into Redis even though the bucket key is short-lived.
"""
from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.core.auth import extract_bearer_token, hash_api_key


def _bucket_key(request: Request) -> str:
    token = extract_bearer_token(request.headers.get("Authorization"))
    if token:
        return f"key:{hash_api_key(token)[:16]}"
    return f"ip:{get_remote_address(request)}"


_settings = get_settings()

limiter = Limiter(
    key_func=_bucket_key,
    storage_uri=_settings.redis_url,
    default_limits=[f"{_settings.rate_limit_per_minute}/minute"],
    headers_enabled=True,
)
