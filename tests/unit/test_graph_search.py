"""Unit tests for app.retrieve.graph.graph_search."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from app.retrieve.graph import GraphHit, graph_search


def _make_driver(rows: list[dict] | None = None, captured: list | None = None):
    """Mock Neo4j AsyncDriver. session.run(cypher, **params) returns an async
    iterator over the given rows. Each row supports dict-style item access."""
    rows = rows or []

    async def aiter_records():
        for r in rows:
            yield r

    class _Result:
        def __aiter__(self):
            return aiter_records()

        async def consume(self):
            return None

    async def fake_run(cypher, **params):
        if captured is not None:
            captured.append((cypher, params))
        return _Result()

    sess = MagicMock()
    sess.run = fake_run

    @asynccontextmanager
    async def fake_session():
        yield sess

    driver = MagicMock()
    driver.session = fake_session
    return driver


# ---------- happy path -----------------------------------------------------

async def test_graph_search_empty_question_short_circuits():
    captured: list = []
    driver = _make_driver(captured=captured)
    out = await graph_search(
        driver, tenant_id=uuid.uuid4(), question="   ", top_k=5,
    )
    assert out == []
    assert captured == []


async def test_graph_search_returns_empty_when_no_matches():
    driver = _make_driver(rows=[])
    out = await graph_search(
        driver, tenant_id=uuid.uuid4(), question="Who is Bob Park?", top_k=5,
    )
    assert out == []


async def test_graph_search_maps_rows_to_graph_hits():
    rows = [
        {
            "chunk_id": "c1",
            "filename": "policy.pdf",
            "page": 1,
            "text": "Bob Park reports to Alice Han.",
            "document_id": "d1",
            "chunk_index": 0,
            "score": 3,
        },
        {
            "chunk_id": "c2",
            "filename": "policy.pdf",
            "page": 2,
            "text": "Alice Han is CEO.",
            "document_id": "d1",
            "chunk_index": 1,
            "score": 1,
        },
    ]
    driver = _make_driver(rows=rows)
    out = await graph_search(
        driver, tenant_id=uuid.uuid4(),
        question="Who does Bob Park report to?", top_k=5,
    )
    assert len(out) == 2
    assert all(isinstance(h, GraphHit) for h in out)
    assert out[0].filename == "policy.pdf"
    assert out[0].page == 1
    assert out[0].score == 3
    assert out[0].text.startswith("Bob Park")
    assert out[0].document_id == "d1"
    assert out[0].chunk_index == 0


async def test_graph_search_passes_tenant_question_top_k_to_cypher():
    captured: list = []
    driver = _make_driver(rows=[], captured=captured)
    tenant = uuid.uuid4()
    await graph_search(
        driver, tenant_id=tenant, question="Acme acquired Northstar", top_k=7,
    )
    assert len(captured) == 1
    cypher, params = captured[0]
    assert "Entity" in cypher
    assert "MENTIONS" in cypher
    assert "RELATES_TO" in cypher
    assert params["tenant_id"] == str(tenant)
    assert params["question"] == "acme acquired northstar"
    assert params["top_k"] == 7


async def test_graph_search_caps_top_k_for_safety():
    """top_k <= 0 should normalize to >=1 (cannot pass 0 to Neo4j LIMIT)."""
    captured: list = []
    driver = _make_driver(rows=[], captured=captured)
    await graph_search(
        driver, tenant_id=uuid.uuid4(), question="q", top_k=0,
    )
    assert captured[0][1]["top_k"] >= 1


async def test_graph_search_handles_missing_fields_gracefully():
    rows = [
        {
            "chunk_id": "c1",
            "filename": None,
            "page": None,
            "text": None,
            "document_id": None,
            "chunk_index": None,
            "score": None,
        },
    ]
    driver = _make_driver(rows=rows)
    out = await graph_search(
        driver, tenant_id=uuid.uuid4(), question="q", top_k=5,
    )
    assert len(out) == 1
    h = out[0]
    assert h.filename == "?"
    assert h.page == 0
    assert h.text == ""
    assert h.score == 0.0
