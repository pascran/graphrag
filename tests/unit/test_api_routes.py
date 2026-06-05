"""In-process unit tests for the FastAPI routers using TestClient.

Drives every route through the real ASGI app so coverage tracks app/api/*
and app/deps.py without standing up Postgres / Qdrant / Neo4j / Redis /
vLLM. DB sessions and the auth dep are replaced via app.dependency_overrides;
external clients are monkeypatched.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.api import health as health_api
from app.api import documents as documents_api
from app.api import query as query_api
from app.api import upload as upload_api
from app.core.auth import AuthenticatedTenant
from app.main import app


@pytest.fixture()
def fake_tenant() -> AuthenticatedTenant:
    return AuthenticatedTenant(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        tenant_name="unit-tenant",
        api_key_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    )


@pytest.fixture()
def stub_session():
    session = MagicMock(name="AsyncSession")
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture()
def client(fake_tenant, stub_session):
    async def _override_session():
        yield stub_session

    app.dependency_overrides[deps.current_tenant] = lambda: fake_tenant
    app.dependency_overrides[deps.get_db] = _override_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _result_with_scalars(rows: list):
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    return res


def _result_with_one(value):
    res = MagicMock()
    res.scalar_one_or_none.return_value = value
    return res


# ---------- /health -----------------------------------------------------------

@pytest.fixture()
def all_ok(monkeypatch):
    async def ok():
        return "ok"

    async def ok_url(url):
        return "ok"

    monkeypatch.setattr(health_api.postgres_client, "ping", ok)
    monkeypatch.setattr(health_api.qdrant_client, "ping", ok)
    monkeypatch.setattr(health_api.neo4j_client, "ping", ok)
    monkeypatch.setattr(health_api.redis_client, "ping", ok)
    monkeypatch.setattr(health_api.vllm_client, "ping", ok_url)


def test_health_all_ok(client, all_ok):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["checks"].keys()) == {
        "postgres", "qdrant", "neo4j", "redis", "vllm_gemma", "vllm_chandra"
    }
    assert all(v == "ok" for v in body["checks"].values())


def test_health_one_failure_returns_degraded(client, all_ok, monkeypatch):
    async def bad():
        return "error: boom"

    monkeypatch.setattr(health_api.qdrant_client, "ping", bad)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"].startswith("error:")


# ---------- /v1/me ------------------------------------------------------------

def test_me_returns_tenant_info(client, fake_tenant):
    r = client.get("/v1/me")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(fake_tenant.tenant_id)
    assert body["tenant_name"] == "unit-tenant"


# ---------- /v1/jobs/{id} -----------------------------------------------------

def test_jobs_get_returns_job_when_present(client, stub_session, fake_tenant):
    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        tenant_id=fake_tenant.tenant_id,
        status="completed",
        progress=1.0,
        error=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    stub_session.execute.return_value = _result_with_one(job)

    r = client.get(f"/v1/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["id"] == str(job_id)
    assert r.json()["status"] == "completed"


def test_jobs_get_returns_404_when_missing(client, stub_session):
    stub_session.execute.return_value = _result_with_one(None)
    r = client.get(f"/v1/jobs/{uuid.uuid4()}")
    assert r.status_code == 404
    assert "job not found" in r.json()["detail"]


# ---------- /v1/documents (GET) ----------------------------------------------

def test_documents_list_empty(client, stub_session):
    stub_session.execute.return_value = _result_with_scalars([])
    r = client.get("/v1/documents")
    assert r.status_code == 200
    assert r.json() == []


def test_documents_list_serializes_rows(client, stub_session, fake_tenant):
    row = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=fake_tenant.tenant_id,
        filename="x.pdf",
        file_hash="0" * 64,
        doc_type="policy",
        page_count=1,
        status="completed",
        created_at=datetime.now(tz=timezone.utc),
    )
    stub_session.execute.return_value = _result_with_scalars([row])
    r = client.get("/v1/documents")
    body = r.json()
    assert len(body) == 1
    assert body[0]["filename"] == "x.pdf"
    assert body[0]["doc_type"] == "policy"


def test_documents_list_accepts_query_filters(client, stub_session):
    stub_session.execute.return_value = _result_with_scalars([])
    r = client.get("/v1/documents?limit=10&offset=5&doc_type=policy&status=completed")
    assert r.status_code == 200
    stub_session.execute.assert_awaited_once()


# ---------- /v1/documents/{id} (DELETE) --------------------------------------

def test_documents_delete_404_when_missing(client, stub_session):
    stub_session.execute.return_value = _result_with_one(None)
    r = client.delete(f"/v1/documents/{uuid.uuid4()}")
    assert r.status_code == 404
    assert "document not found" in r.json()["detail"]


def test_documents_delete_cascades_and_returns_204(
    client, stub_session, fake_tenant, monkeypatch, tmp_path
):
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(id=doc_id, tenant_id=fake_tenant.tenant_id)
    stub_session.execute.side_effect = [
        _result_with_one(doc),
        MagicMock(),
    ]

    qdrant_mock = AsyncMock()
    neo4j_mock = AsyncMock()
    monkeypatch.setattr(documents_api, "delete_document_chunks", qdrant_mock)
    monkeypatch.setattr(documents_api, "delete_document_graph", neo4j_mock)
    monkeypatch.setattr(documents_api.qdrant_db, "get_client", lambda: object())
    monkeypatch.setattr(documents_api.neo4j_db, "get_driver", lambda: object())
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)

    fake_pdf = tmp_path / f"{doc_id}.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")

    r = client.delete(f"/v1/documents/{doc_id}")
    assert r.status_code == 204
    qdrant_mock.assert_awaited_once()
    neo4j_mock.assert_awaited_once()
    assert not fake_pdf.exists()
    stub_session.commit.assert_awaited()


def test_documents_delete_swallows_qdrant_failure(
    client, stub_session, fake_tenant, monkeypatch, tmp_path
):
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(id=doc_id, tenant_id=fake_tenant.tenant_id)
    stub_session.execute.side_effect = [
        _result_with_one(doc),
        MagicMock(),
    ]
    monkeypatch.setattr(
        documents_api, "delete_document_chunks",
        AsyncMock(side_effect=RuntimeError("qdrant down")),
    )
    monkeypatch.setattr(documents_api, "delete_document_graph", AsyncMock())
    monkeypatch.setattr(documents_api.qdrant_db, "get_client", lambda: object())
    monkeypatch.setattr(documents_api.neo4j_db, "get_driver", lambda: object())
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)

    r = client.delete(f"/v1/documents/{doc_id}")
    assert r.status_code == 204
    stub_session.commit.assert_awaited()


# ---------- /v1/upload --------------------------------------------------------

def test_upload_rejects_disallowed_extension(client, stub_session, monkeypatch, tmp_path):
    monkeypatch.setattr(upload_api, "UPLOAD_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(upload_api.celery_app, "send_task",
                        lambda name, kwargs: sent.append((name, kwargs)))
    stub_session.execute.return_value = _result_with_one(None)

    r = client.post(
        "/v1/upload",
        files={"files": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["accepted_files"] == []
    assert len(body["rejected_files"]) == 1
    assert ".exe" in body["rejected_files"][0]["reason"]
    assert sent == []


def test_upload_rejects_empty_file(client, stub_session, monkeypatch, tmp_path):
    monkeypatch.setattr(upload_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(upload_api.celery_app, "send_task", lambda *a, **kw: None)
    stub_session.execute.return_value = _result_with_one(None)

    r = client.post(
        "/v1/upload",
        files={"files": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["accepted_files"] == []
    assert "empty file" in body["rejected_files"][0]["reason"]


def test_upload_rejects_duplicate_content(client, stub_session, fake_tenant, monkeypatch, tmp_path):
    monkeypatch.setattr(upload_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(upload_api.celery_app, "send_task", lambda *a, **kw: None)
    existing = SimpleNamespace(id=uuid.uuid4(), tenant_id=fake_tenant.tenant_id)
    stub_session.execute.return_value = _result_with_one(existing)

    r = client.post(
        "/v1/upload",
        files={"files": ("x.pdf", b"%PDF-1.4 hello", "application/pdf")},
    )
    body = r.json()
    assert body["accepted_files"] == []
    assert "duplicate" in body["rejected_files"][0]["reason"].lower()


def test_upload_accepts_pdf_and_enqueues_celery(
    client, stub_session, fake_tenant, monkeypatch, tmp_path
):
    monkeypatch.setattr(upload_api, "UPLOAD_DIR", tmp_path)
    sent: list[tuple] = []

    def fake_send(name, kwargs):
        sent.append((name, kwargs))

    monkeypatch.setattr(upload_api.celery_app, "send_task", fake_send)
    stub_session.execute.return_value = _result_with_one(None)

    r = client.post(
        "/v1/upload",
        files={"files": ("good.pdf", b"%PDF-1.4 valid bytes\n", "application/pdf")},
        data={"doc_type": "policy"},
    )
    assert r.status_code == 202
    body = r.json()
    assert len(body["accepted_files"]) == 1
    assert body["accepted_files"][0]["name"] == "good.pdf"
    assert body["rejected_files"] == []
    assert sent and sent[0][0] == "ingest.document"
    assert sent[0][1]["filename"] == "good.pdf"
    assert sent[0][1]["doc_type"] == "policy"


def test_upload_rejects_when_too_many_files(client, stub_session, monkeypatch, tmp_path):
    monkeypatch.setattr(upload_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(upload_api.celery_app, "send_task", lambda *a, **kw: None)
    fake_settings = upload_api.get_settings().model_copy(
        update={"upload_max_files_per_request": 1}
    )
    monkeypatch.setattr(upload_api, "get_settings", lambda: fake_settings)

    r = client.post(
        "/v1/upload",
        files=[
            ("files", ("a.pdf", b"%PDF a", "application/pdf")),
            ("files", ("b.pdf", b"%PDF b", "application/pdf")),
        ],
    )
    assert r.status_code == 413
    assert "too many files" in r.json()["detail"]


# ---------- /v1/query ---------------------------------------------------------

def _retrieval(chunks=()):
    return SimpleNamespace(
        mode_used="fact",
        chunks=[SimpleNamespace(filename=f, page=p, text=t) for f, p, t in chunks],
        raw=[],
    )


def test_query_json_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(query_api, "retrieve",
                        AsyncMock(return_value=_retrieval([("a.pdf", 1, "alpha")])))
    monkeypatch.setattr(query_api, "chat_once", AsyncMock(return_value="answer body"))
    monkeypatch.setattr(query_api.qdrant_db, "get_client", lambda: object())

    r = client.post(
        "/v1/query",
        json={"question": "hello", "mode": "auto", "top_k": 3, "stream": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "answer body"
    assert body["sources"] == [{"filename": "a.pdf", "page": 1}]
    assert body["mode_used"] == "fact"


def test_query_json_loads_session_when_session_id_provided(client, monkeypatch):
    monkeypatch.setattr(query_api, "retrieve",
                        AsyncMock(return_value=_retrieval()))
    monkeypatch.setattr(query_api, "chat_once", AsyncMock(return_value="ack"))
    monkeypatch.setattr(query_api.qdrant_db, "get_client", lambda: object())
    load_mock = AsyncMock(return_value=SimpleNamespace(summary="prior", turns=[]))
    append_mock = AsyncMock()
    monkeypatch.setattr(query_api, "load_session", load_mock)
    monkeypatch.setattr(query_api, "append_turn", append_mock)

    r = client.post(
        "/v1/query",
        json={"question": "hi", "stream": False, "session_id": "s-1"},
    )
    assert r.status_code == 200
    load_mock.assert_awaited_once()
    append_mock.assert_awaited_once()


def test_query_sse_emits_citation_token_done(client, monkeypatch):
    async def fake_stream(messages):
        yield "hel"
        yield "lo"

    monkeypatch.setattr(query_api, "retrieve",
                        AsyncMock(return_value=_retrieval([("x.pdf", 7, "hi")])))
    monkeypatch.setattr(query_api, "stream_chat", fake_stream)
    monkeypatch.setattr(query_api.qdrant_db, "get_client", lambda: object())

    with client.stream(
        "POST", "/v1/query",
        json={"question": "stream me", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        events = [line for line in resp.iter_lines() if line.startswith("event:")]
    names = [e.removeprefix("event:").strip() for e in events]
    assert names[0] == "citation"
    assert "token" in names
    assert names[-1] == "done"


def test_query_sse_yields_error_event_on_retrieval_failure(client, monkeypatch):
    monkeypatch.setattr(query_api, "retrieve",
                        AsyncMock(side_effect=RuntimeError("qdrant boom")))
    monkeypatch.setattr(query_api.qdrant_db, "get_client", lambda: object())

    with client.stream(
        "POST", "/v1/query",
        json={"question": "x", "stream": True},
    ) as resp:
        events = [line for line in resp.iter_lines() if line.startswith("event:")]
    names = [e.removeprefix("event:").strip() for e in events]
    assert "error" in names
