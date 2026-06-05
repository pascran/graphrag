"""Coverage push for app.generate.llm, app.retrieve.vector, app.deps,
and app.workers.tasks._set_status. External services are monkeypatched."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import deps
from app.core.auth import AuthenticatedTenant
from app.generate import llm as llm_mod
from app.ingest.embedder import EmbeddedChunk
from app.retrieve import vector as vector_mod
from app.workers import tasks as tasks_mod


# ============================================================================
# generate.llm.chat_once
# ============================================================================

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    """httpx.AsyncClient stand-in with class-level callbacks."""
    on_post = None
    on_stream = None

    def __init__(self, *a, **kw):
        self.kw = kw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json):
        return type(self).on_post(url, json)

    def stream(self, method, url, json):
        return type(self).on_stream(method, url, json)


async def test_chat_once_extracts_first_choice_content(monkeypatch):
    seen = {}

    def on_post(url, json):
        seen["url"] = url
        seen["json"] = json
        return _Resp({"choices": [{"message": {"content": "hello world"}}]})

    _Client.on_post = on_post
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _Client)

    out = await llm_mod.chat_once(
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=64,
    )
    assert out == "hello world"
    assert seen["url"] == "/chat/completions"
    assert seen["json"]["temperature"] == 0.0
    assert seen["json"]["max_tokens"] == 64


# ============================================================================
# generate.llm.stream_chat
# ============================================================================

class _StreamCM:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


async def test_stream_chat_yields_content_deltas_and_stops_on_done(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: [DONE]',
        'data: {"choices":[{"delta":{"content":"shouldnotappear"}}]}',
    ]

    def on_stream(method, url, json):
        return _StreamCM(lines)

    _Client.on_stream = on_stream
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _Client)

    chunks = []
    async for c in llm_mod.stream_chat([{"role": "user", "content": "x"}]):
        chunks.append(c)
    assert chunks == ["Hel", "lo"]


async def test_stream_chat_skips_non_data_and_malformed_lines(monkeypatch):
    lines = [
        ': keepalive',
        'event: foo',
        'data: not-json',
        'data: {"choices":[]}',
        'data: {"choices":[{"delta":{}}]}',
        'data: {"choices":[{"delta":{"content":"final"}}]}',
        'data: [DONE]',
    ]

    def on_stream(method, url, json):
        return _StreamCM(lines)

    _Client.on_stream = on_stream
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _Client)

    out = []
    async for c in llm_mod.stream_chat([{"role": "user", "content": "x"}]):
        out.append(c)
    assert out == ["final"]


# ============================================================================
# retrieve.vector.hybrid_search
# ============================================================================

async def test_hybrid_search_short_circuits_when_no_embeds(monkeypatch):
    monkeypatch.setattr(vector_mod, "embed_texts", lambda texts: [])
    client = MagicMock()
    client.query_points = AsyncMock()
    out = await vector_mod.hybrid_search(
        client, tenant_id=uuid.uuid4(), question="q",
    )
    assert out == []
    client.query_points.assert_not_awaited()


async def test_hybrid_search_passes_filters_and_maps_payload(monkeypatch):
    monkeypatch.setattr(
        vector_mod, "embed_texts",
        lambda texts: [EmbeddedChunk(dense=[0.1, 0.2], sparse_indices=[3], sparse_values=[0.5])],
    )

    fake_response = SimpleNamespace(
        points=[
            SimpleNamespace(
                payload={
                    "filename": "a.pdf",
                    "page": 2,
                    "text": "first hit",
                    "document_id": "doc-1",
                    "chunk_index": 5,
                },
                score=0.91,
            ),
            SimpleNamespace(payload=None, score=None),
        ]
    )
    client = MagicMock()
    client.query_points = AsyncMock(return_value=fake_response)

    tenant = uuid.uuid4()
    out = await vector_mod.hybrid_search(
        client,
        tenant_id=tenant,
        question="q",
        top_k=3,
        payload_filters={"doc_type": "policy", "skip_dict": {"a": 1}, "skip_none": None},
    )
    assert len(out) == 2
    assert out[0].filename == "a.pdf"
    assert out[0].page == 2
    assert out[0].score == pytest.approx(0.91)
    assert out[0].document_id == "doc-1"
    assert out[1].filename == "?"
    assert out[1].score == 0.0

    kwargs = client.query_points.await_args.kwargs
    must = kwargs["prefetch"][0].filter.must
    keys = sorted(c.key for c in must)
    assert "tenant_id" in keys
    assert "doc_type" in keys
    assert "skip_dict" not in keys
    assert "skip_none" not in keys


# ============================================================================
# deps.current_tenant
# ============================================================================

async def test_current_tenant_raises_401_when_no_credentials():
    session = MagicMock()
    with pytest.raises(Exception) as exc:
        await deps.current_tenant(creds=None, session=session)
    assert exc.value.status_code == 401


async def test_current_tenant_raises_401_when_authenticate_returns_none(monkeypatch):
    async def fake_auth(session, token):
        return None

    monkeypatch.setattr(deps, "authenticate", fake_auth)
    creds = SimpleNamespace(credentials="graphrag_invalid")
    session = MagicMock()
    with pytest.raises(Exception) as exc:
        await deps.current_tenant(creds=creds, session=session)
    assert exc.value.status_code == 401
    assert "invalid" in exc.value.detail


async def test_current_tenant_returns_tenant_on_success(monkeypatch):
    tenant = AuthenticatedTenant(
        tenant_id=uuid.uuid4(), tenant_name="acme", api_key_id=uuid.uuid4()
    )

    async def fake_auth(session, token):
        return tenant

    monkeypatch.setattr(deps, "authenticate", fake_auth)
    creds = SimpleNamespace(credentials="  graphrag_valid  ")
    session = MagicMock()
    out = await deps.current_tenant(creds=creds, session=session)
    assert out is tenant


def test_factory_caches_session_factory(monkeypatch):
    monkeypatch.setattr(deps, "_session_factory", None)
    fake_engine = object()
    monkeypatch.setattr(deps, "get_engine", lambda: fake_engine)

    captured = {}

    def fake_async_sessionmaker(engine, **kw):
        captured["engine"] = engine
        captured["kw"] = kw
        return "factory-instance"

    monkeypatch.setattr(deps, "async_sessionmaker", fake_async_sessionmaker)
    a = deps._factory()
    b = deps._factory()
    assert a is b == "factory-instance"
    assert captured["engine"] is fake_engine
    assert captured["kw"] == {"expire_on_commit": False}


# ============================================================================
# workers.tasks._set_status
# ============================================================================

async def test_set_status_updates_job_and_document(monkeypatch):
    executed: list = []
    committed = {"hit": False}
    disposed = {"hit": False}

    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=lambda stmt: executed.append(stmt))
    fake_session.commit = AsyncMock(side_effect=lambda: committed.__setitem__("hit", True))

    @asynccontextmanager
    async def fake_session_cm():
        yield fake_session

    fake_factory = MagicMock(side_effect=lambda: fake_session_cm())

    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock(side_effect=lambda: disposed.__setitem__("hit", True))

    monkeypatch.setattr(tasks_mod, "create_async_engine", lambda *a, **kw: fake_engine)
    monkeypatch.setattr(tasks_mod, "async_sessionmaker", lambda *a, **kw: fake_factory)

    await tasks_mod._set_status(
        job_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        status="completed",
        progress=1.0,
    )

    assert len(executed) == 2
    assert committed["hit"] is True
    assert disposed["hit"] is True


async def test_set_status_only_job_when_document_id_none(monkeypatch):
    executed: list = []
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=lambda stmt: executed.append(stmt))
    fake_session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_session_cm():
        yield fake_session

    fake_factory = MagicMock(side_effect=lambda: fake_session_cm())
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()

    monkeypatch.setattr(tasks_mod, "create_async_engine", lambda *a, **kw: fake_engine)
    monkeypatch.setattr(tasks_mod, "async_sessionmaker", lambda *a, **kw: fake_factory)

    await tasks_mod._set_status(
        job_id=uuid.uuid4(),
        document_id=None,
        status="failed",
        error="boom",
    )
    assert len(executed) == 1


async def test_set_status_only_document_when_job_id_none(monkeypatch):
    executed: list = []
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(side_effect=lambda stmt: executed.append(stmt))
    fake_session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_session_cm():
        yield fake_session

    fake_factory = MagicMock(side_effect=lambda: fake_session_cm())
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()

    monkeypatch.setattr(tasks_mod, "create_async_engine", lambda *a, **kw: fake_engine)
    monkeypatch.setattr(tasks_mod, "async_sessionmaker", lambda *a, **kw: fake_factory)

    await tasks_mod._set_status(
        job_id=None,
        document_id=uuid.uuid4(),
        status="running",
    )
    assert len(executed) == 1
