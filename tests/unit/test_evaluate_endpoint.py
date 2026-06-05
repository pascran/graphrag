"""In-process tests for POST /v1/evaluate."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.api import evaluate as evaluate_api
from app.core.auth import AuthenticatedTenant
from app.generate.prompt import RetrievedChunk
from app.main import app


@pytest.fixture()
def fake_tenant():
    return AuthenticatedTenant(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        tenant_name="eval-tenant",
        api_key_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    )


@pytest.fixture()
def client(fake_tenant):
    stub_session = MagicMock()
    stub_session.execute = AsyncMock()
    stub_session.commit = AsyncMock()
    stub_session.rollback = AsyncMock()

    async def _override_session():
        yield stub_session

    app.dependency_overrides[deps.current_tenant] = lambda: fake_tenant
    app.dependency_overrides[deps.get_db] = _override_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def patched(monkeypatch):
    """Mock retrieve(), chat_once(), and the 4 metric functions."""
    state = {"metrics_called": [], "expected_answer_seen": None}

    fake_chunks = [
        RetrievedChunk(filename="a.pdf", page=1, text="ctx-a"),
        RetrievedChunk(filename="b.pdf", page=2, text="ctx-b"),
    ]

    async def fake_retrieve(*a, **kw):
        return SimpleNamespace(
            mode_used="fact",
            chunks=fake_chunks,
            raw=[],
        )

    async def fake_chat_once(*a, **kw):
        return "stubbed answer"

    async def fake_faith(*, answer, chunks):
        state["metrics_called"].append("faithfulness")
        return 0.87

    async def fake_rel(*, question, answer):
        state["metrics_called"].append("answer_relevancy")
        return 0.91

    async def fake_prec(*, question, chunks):
        state["metrics_called"].append("context_precision")
        return 0.75

    async def fake_recall(*, expected_answer, chunks):
        state["metrics_called"].append("context_recall")
        state["expected_answer_seen"] = expected_answer
        return 0.66

    monkeypatch.setattr(evaluate_api, "retrieve", fake_retrieve)
    monkeypatch.setattr(evaluate_api, "chat_once", fake_chat_once)
    monkeypatch.setattr(evaluate_api, "faithfulness", fake_faith)
    monkeypatch.setattr(evaluate_api, "answer_relevancy", fake_rel)
    monkeypatch.setattr(evaluate_api, "context_precision", fake_prec)
    monkeypatch.setattr(evaluate_api, "context_recall", fake_recall)
    return state


def test_evaluate_default_runs_reference_free_three(client, patched):
    r = client.post("/v1/evaluate", json={"question": "Q?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "Q?"
    assert body["answer"] == "stubbed answer"
    assert body["mode_used"] == "fact"
    assert body["sources"] == [
        {"filename": "a.pdf", "page": 1},
        {"filename": "b.pdf", "page": 2},
    ]
    assert body["scores"]["faithfulness"] == 0.87
    assert body["scores"]["answer_relevancy"] == 0.91
    assert body["scores"]["context_precision"] == 0.75
    assert body["scores"].get("context_recall") is None
    assert "context_recall" not in patched["metrics_called"]


def test_evaluate_includes_recall_when_expected_answer_present(client, patched):
    r = client.post(
        "/v1/evaluate",
        json={
            "question": "Q?",
            "expected_answer": "The known correct answer.",
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scores"]["context_recall"] == 0.66
    assert "context_recall" in patched["metrics_called"]
    assert patched["expected_answer_seen"] == "The known correct answer."


def test_evaluate_respects_metric_subset(client, patched):
    r = client.post(
        "/v1/evaluate",
        json={"question": "Q?", "metrics": ["faithfulness"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scores"] == {"faithfulness": 0.87}
    assert patched["metrics_called"] == ["faithfulness"]


def test_evaluate_rejects_unknown_metric(client, patched):
    r = client.post(
        "/v1/evaluate",
        json={"question": "Q?", "metrics": ["bogus"]},
    )
    assert r.status_code == 422


def test_evaluate_rejects_empty_question(client, patched):
    r = client.post("/v1/evaluate", json={"question": "  "})
    assert r.status_code == 422


def test_evaluate_returns_null_score_when_metric_undefined(client, patched, monkeypatch):
    async def faith_none(*, answer, chunks):
        return None

    monkeypatch.setattr(evaluate_api, "faithfulness", faith_none)
    r = client.post(
        "/v1/evaluate",
        json={"question": "Q?", "metrics": ["faithfulness"]},
    )
    assert r.status_code == 200
    assert r.json()["scores"]["faithfulness"] is None
