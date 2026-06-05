"""Unit tests for app.retrieve.orchestrator.retrieve."""
from __future__ import annotations

import uuid

import pytest

from app.retrieve import orchestrator
from app.retrieve.vector import VectorHit


def _hits(n: int) -> list[VectorHit]:
    return [
        VectorHit(
            filename=f"f{i}.pdf",
            page=i,
            text=f"chunk-{i}",
            score=1.0 - i * 0.01,
            document_id=str(uuid.uuid4()),
            chunk_index=i,
        )
        for i in range(n)
    ]


@pytest.fixture()
def patched(monkeypatch):
    state = {"classify_called_with": None, "search_top_k": None, "intent": "fact"}

    async def fake_classify(question: str) -> str:
        state["classify_called_with"] = question
        return state["intent"]

    async def fake_search(qdrant, *, tenant_id, question, top_k, payload_filters):
        state["search_top_k"] = top_k
        return _hits(min(top_k, 5))

    monkeypatch.setattr(orchestrator, "classify", fake_classify)
    monkeypatch.setattr(orchestrator, "hybrid_search", fake_search)
    return state


async def test_auto_casual_skips_qdrant(patched):
    patched["intent"] = "casual"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="hi", mode="auto", top_k=5
    )
    assert result.mode_used == "casual"
    assert result.chunks == []
    assert result.raw == []
    assert patched["classify_called_with"] == "hi"
    assert patched["search_top_k"] is None


async def test_auto_fact_uses_top_k_as_is(patched):
    patched["intent"] = "fact"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=5
    )
    assert result.mode_used == "fact"
    assert patched["search_top_k"] == 5
    assert len(result.chunks) == 5


async def test_auto_analysis_doubles_top_k(patched):
    patched["intent"] = "analysis"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="compare", mode="auto", top_k=5
    )
    assert result.mode_used == "analysis"
    assert patched["search_top_k"] == 10


async def test_explicit_casual_bypasses_router(patched):
    patched["intent"] = "fact"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="anything", mode="casual", top_k=3
    )
    assert result.mode_used == "casual"
    assert patched["classify_called_with"] is None


async def test_unknown_mode_falls_back_to_fact(patched):
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="vector", top_k=4
    )
    assert result.mode_used == "fact"
    assert patched["search_top_k"] == 4
    assert patched["classify_called_with"] is None


async def test_chunks_carry_filename_and_page_from_hits(patched):
    patched["intent"] = "fact"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=2
    )
    assert [c.filename for c in result.chunks] == ["f0.pdf", "f1.pdf"]
    assert [c.page for c in result.chunks] == [0, 1]
    assert [c.text for c in result.chunks] == ["chunk-0", "chunk-1"]
