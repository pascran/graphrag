"""End-to-end smoke against the live API.

Requires `docker compose up` and an active vLLM Gemma. Skipped automatically
when the API is unreachable (see conftest.py).
"""
from __future__ import annotations

import httpx


def test_health_is_ok():
    r = httpx.get("http://localhost:8000/health", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "checks" in body


def test_query_requires_auth():
    r = httpx.post(
        "http://localhost:8000/v1/query",
        json={"question": "hi", "stream": False},
        timeout=10.0,
    )
    assert r.status_code == 401


def test_documents_list_for_fresh_tenant_is_empty(client: httpx.Client):
    r = client.get("/v1/documents")
    assert r.status_code == 200
    assert r.json() == []


def test_query_casual_path_returns_no_sources(client: httpx.Client):
    r = client.post(
        "/v1/query",
        json={"question": "안녕!", "mode": "auto", "top_k": 3, "stream": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode_used"] == "casual"
    assert body["sources"] == []
    assert isinstance(body["answer"], str) and body["answer"]


def test_query_session_memory_recalls_prior_turn(client: httpx.Client):
    sid = "itest-session"
    r1 = client.post(
        "/v1/query",
        json={
            "question": "내 이름은 박씨야. 기억해.",
            "mode": "auto",
            "top_k": 3,
            "stream": False,
            "session_id": sid,
        },
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/v1/query",
        json={
            "question": "내 이름이 뭐였지?",
            "mode": "auto",
            "top_k": 3,
            "stream": False,
            "session_id": sid,
        },
    )
    assert r2.status_code == 200
    answer = r2.json()["answer"]
    assert "박" in answer or "박씨" in answer


def test_query_sse_emits_done_event(client: httpx.Client):
    seen: list[str] = []
    with client.stream(
        "POST",
        "/v1/query",
        json={"question": "안녕!", "mode": "auto", "top_k": 3, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        for raw_line in resp.iter_lines():
            if raw_line.startswith("event:"):
                seen.append(raw_line.removeprefix("event:").strip())
            if "done" in seen:
                break
    assert "citation" in seen
    assert "done" in seen


def test_rate_limit_headers_present(client: httpx.Client):
    r = client.get("/v1/documents")
    assert r.status_code == 200
    lower = {k.lower() for k in r.headers.keys()}
    assert "x-ratelimit-limit" in lower
    assert "x-ratelimit-remaining" in lower


def test_request_id_round_trip(client: httpx.Client):
    rid = "test-req-id-abcdef"
    r = client.get("/v1/documents", headers={"X-Request-ID": rid})
    assert r.headers.get("X-Request-ID") == rid


def test_delete_unknown_document_returns_404(client: httpx.Client):
    r = client.delete("/v1/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_query_invalid_top_k_rejected(client: httpx.Client):
    r = client.post(
        "/v1/query",
        json={"question": "ok", "top_k": 999, "stream": False},
    )
    assert r.status_code == 422


def test_health_returns_check_map_keys():
    r = httpx.get("http://localhost:8000/health", timeout=5.0)
    body = r.json()
    expected_some = {"postgres", "qdrant", "neo4j", "redis"}
    assert expected_some & set(body["checks"].keys())
