"""Unit tests for app.db.* — get_client/get_engine/get_driver caches + ping().

External libraries (aioredis, AsyncQdrantClient, sqlalchemy create_async_engine,
neo4j AsyncGraphDatabase, httpx) are monkeypatched so tests don't need a live
Postgres / Qdrant / Neo4j / Redis / vLLM.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from app.db import neo4j as neo4j_db
from app.db import postgres as pg_db
from app.db import qdrant as qdrant_db
from app.db import redis as redis_db
from app.db import vllm as vllm_db


# ---------- Redis -------------------------------------------------------------

async def test_redis_ping_ok(monkeypatch):
    fake = AsyncMock()
    fake.ping = AsyncMock(return_value=True)
    monkeypatch.setattr(redis_db, "_client", fake)
    assert await redis_db.ping() == "ok"


async def test_redis_ping_pong_falsy(monkeypatch):
    fake = AsyncMock()
    fake.ping = AsyncMock(return_value=False)
    monkeypatch.setattr(redis_db, "_client", fake)
    assert "no pong" in await redis_db.ping()


async def test_redis_ping_exception_returns_error_string(monkeypatch):
    fake = AsyncMock()
    fake.ping = AsyncMock(side_effect=ConnectionError("nope"))
    monkeypatch.setattr(redis_db, "_client", fake)
    out = await redis_db.ping()
    assert out.startswith("error:")
    assert "ConnectionError" in out


def test_redis_get_client_caches(monkeypatch):
    monkeypatch.setattr(redis_db, "_client", None)
    sentinel = object()
    monkeypatch.setattr(redis_db, "aioredis",
                        type("M", (), {"from_url": staticmethod(lambda *a, **kw: sentinel)}))
    a = redis_db.get_client()
    b = redis_db.get_client()
    assert a is b is sentinel


# ---------- Qdrant ------------------------------------------------------------

async def test_qdrant_ping_ok(monkeypatch):
    fake = AsyncMock()
    fake.get_collections = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(qdrant_db, "_client", fake)
    assert await qdrant_db.ping() == "ok"


async def test_qdrant_ping_failure(monkeypatch):
    fake = AsyncMock()
    fake.get_collections = AsyncMock(side_effect=RuntimeError("down"))
    monkeypatch.setattr(qdrant_db, "_client", fake)
    out = await qdrant_db.ping()
    assert out.startswith("error:")
    assert "RuntimeError" in out


def test_qdrant_get_client_caches(monkeypatch):
    monkeypatch.setattr(qdrant_db, "_client", None)
    sentinel = object()
    monkeypatch.setattr(qdrant_db, "AsyncQdrantClient", lambda *a, **kw: sentinel)
    assert qdrant_db.get_client() is qdrant_db.get_client() is sentinel


# ---------- Postgres ----------------------------------------------------------

async def test_postgres_ping_ok(monkeypatch):
    conn = AsyncMock()
    conn.execute = AsyncMock()

    @asynccontextmanager
    async def fake_connect():
        yield conn

    engine = MagicMock()
    engine.connect = fake_connect
    monkeypatch.setattr(pg_db, "_engine", engine)
    assert await pg_db.ping() == "ok"


async def test_postgres_ping_failure(monkeypatch):
    @asynccontextmanager
    async def fake_connect():
        raise RuntimeError("pg down")
        yield  # pragma: no cover

    engine = MagicMock()
    engine.connect = fake_connect
    monkeypatch.setattr(pg_db, "_engine", engine)
    out = await pg_db.ping()
    assert out.startswith("error:")


def test_postgres_get_engine_caches(monkeypatch):
    monkeypatch.setattr(pg_db, "_engine", None)
    sentinel = object()
    monkeypatch.setattr(pg_db, "create_async_engine", lambda *a, **kw: sentinel)
    assert pg_db.get_engine() is pg_db.get_engine() is sentinel


# ---------- Neo4j -------------------------------------------------------------

async def test_neo4j_ping_ok(monkeypatch):
    result = AsyncMock()
    result.consume = AsyncMock()
    sess = AsyncMock()
    sess.run = AsyncMock(return_value=result)

    @asynccontextmanager
    async def fake_session():
        yield sess

    driver = MagicMock()
    driver.session = fake_session
    monkeypatch.setattr(neo4j_db, "_driver", driver)
    assert await neo4j_db.ping() == "ok"


async def test_neo4j_ping_failure(monkeypatch):
    @asynccontextmanager
    async def fake_session():
        raise ConnectionError("neo4j down")
        yield  # pragma: no cover

    driver = MagicMock()
    driver.session = fake_session
    monkeypatch.setattr(neo4j_db, "_driver", driver)
    out = await neo4j_db.ping()
    assert "ConnectionError" in out


def test_neo4j_get_driver_caches(monkeypatch):
    monkeypatch.setattr(neo4j_db, "_driver", None)
    sentinel = object()
    monkeypatch.setattr(
        neo4j_db, "AsyncGraphDatabase",
        type("G", (), {"driver": staticmethod(lambda *a, **kw: sentinel)}),
    )
    assert neo4j_db.get_driver() is neo4j_db.get_driver() is sentinel


# ---------- vLLM --------------------------------------------------------------

async def test_vllm_ping_ok(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            assert url.endswith("/health")
            return _Resp()

    monkeypatch.setattr(vllm_db.httpx, "AsyncClient", _Client)
    assert await vllm_db.ping("http://vllm-x:8000/v1") == "ok"


async def test_vllm_ping_failure(monkeypatch):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            raise RuntimeError("vllm down")

    monkeypatch.setattr(vllm_db.httpx, "AsyncClient", _Client)
    out = await vllm_db.ping("http://vllm-x:8000/v1")
    assert out.startswith("error:")
    assert "RuntimeError" in out
