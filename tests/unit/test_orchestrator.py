"""Unit tests for app.retrieve.orchestrator.retrieve."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.retrieve import orchestrator
from app.retrieve.graph import GraphHit
from app.retrieve.vector import VectorHit


def _hits(n: int) -> list[VectorHit]:
    return [
        VectorHit(
            filename=f"f{i}.pdf",
            page=i,
            text=f"chunk-{i}",
            score=1.0 - i * 0.01,
            document_id=f"doc-{i}",
            chunk_index=i,
        )
        for i in range(n)
    ]


def _graph_hits(items: list[tuple[str, int, str, int]]) -> list[GraphHit]:
    """items = [(filename, page, document_id, chunk_index), ...]"""
    return [
        GraphHit(
            filename=fn, page=p, text=f"g-{ci}", score=float(3 - i),
            document_id=did, chunk_index=ci,
        )
        for i, (fn, p, did, ci) in enumerate(items)
    ]


@pytest.fixture()
def patched(monkeypatch):
    state = {
        "classify_called_with": None,
        "search_top_k": None,
        "intent": "fact",
        "graph_called": False,
        "graph_top_k": None,
        "graph_return": [],
        "rerank_called": False,
        "rerank_input_n": None,
        "rerank_top_k": None,
        "reranker_enabled": False,
        "reranker_oversample": 4,
    }

    async def fake_classify(question: str) -> str:
        state["classify_called_with"] = question
        return state["intent"]

    async def fake_search(qdrant, *, tenant_id, question, top_k, payload_filters):
        state["search_top_k"] = top_k
        return _hits(min(top_k, 20))

    async def fake_graph(driver, *, tenant_id, question, top_k):
        state["graph_called"] = True
        state["graph_top_k"] = top_k
        return state["graph_return"]

    def fake_rerank(*, question, hits, top_k):
        state["rerank_called"] = True
        state["rerank_input_n"] = len(hits)
        state["rerank_top_k"] = top_k
        # Reverse the list so we can detect that reranker actually ran.
        return list(reversed(hits))[:top_k]

    def fake_get_settings():
        return SimpleNamespace(
            reranker_enabled=state["reranker_enabled"],
            reranker_oversample=state["reranker_oversample"],
        )

    monkeypatch.setattr(orchestrator, "classify", fake_classify)
    monkeypatch.setattr(orchestrator, "hybrid_search", fake_search)
    monkeypatch.setattr(orchestrator, "graph_search", fake_graph)
    monkeypatch.setattr(orchestrator, "rerank", fake_rerank)
    monkeypatch.setattr(orchestrator, "get_settings", fake_get_settings)
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


# ---------- graph-search merge -----------------------------------------------

async def test_graph_search_skipped_when_driver_is_none(patched):
    patched["intent"] = "fact"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=3,
        neo4j=None,
    )
    assert result.mode_used == "fact"
    assert patched["graph_called"] is False


async def test_graph_search_skipped_for_casual_intent(patched):
    patched["intent"] = "casual"
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="hi", mode="auto", top_k=3,
        neo4j=object(),  # truthy stub
    )
    assert result.mode_used == "casual"
    assert patched["graph_called"] is False


async def test_graph_hits_prepended_then_vector(patched):
    patched["intent"] = "fact"
    patched["graph_return"] = _graph_hits([
        ("policy.pdf", 9, "doc-graph", 99),
    ])
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=3,
        neo4j=object(),
    )
    assert patched["graph_called"] is True
    assert [c.filename for c in result.chunks][:1] == ["policy.pdf"]
    # Plus the 3 vector hits after, deduped against (doc-graph, 99).
    assert [c.filename for c in result.chunks] == [
        "policy.pdf", "f0.pdf", "f1.pdf", "f2.pdf"
    ]


async def test_graph_hits_dedup_against_vector_by_doc_and_chunk(patched):
    """Same (document_id, chunk_index) → graph version wins, vector dropped."""
    patched["intent"] = "fact"
    # Vector returns chunk_index 0 for doc-0. Graph hit collides with that.
    patched["graph_return"] = _graph_hits([
        ("f0.pdf", 0, "doc-0", 0),  # collides with vector hit #0
        ("policy.pdf", 5, "doc-9", 5),
    ])
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=3,
        neo4j=object(),
    )
    # First two = graph hits; vector "f0.pdf chunk 0" must be dropped, but
    # f1/f2 still appear.
    filenames = [c.filename for c in result.chunks]
    assert filenames == ["f0.pdf", "policy.pdf", "f1.pdf", "f2.pdf"]
    # Text comes from the graph hit (g-0), not the vector hit (chunk-0).
    assert result.chunks[0].text == "g-0"


async def test_graph_search_uses_same_effective_top_k_as_vector(patched):
    patched["intent"] = "analysis"
    patched["graph_return"] = []
    await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="compare", mode="auto",
        top_k=5, neo4j=object(),
    )
    assert patched["search_top_k"] == 10
    assert patched["graph_top_k"] == 10


async def test_graph_search_failure_falls_back_to_vector_only(patched, monkeypatch):
    patched["intent"] = "fact"

    async def boom(driver, *, tenant_id, question, top_k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(orchestrator, "graph_search", boom)
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=2,
        neo4j=object(),
    )
    assert result.mode_used == "fact"
    assert [c.filename for c in result.chunks] == ["f0.pdf", "f1.pdf"]


# ---------- reranker integration ---------------------------------------------

async def test_rerank_skipped_when_disabled(patched):
    patched["intent"] = "fact"
    patched["reranker_enabled"] = False
    await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=3,
    )
    # Without reranker, vector search asks for exactly top_k.
    assert patched["search_top_k"] == 3
    assert patched["rerank_called"] is False


async def test_rerank_oversamples_vector_then_trims_to_top_k(patched):
    patched["intent"] = "fact"
    patched["reranker_enabled"] = True
    patched["reranker_oversample"] = 4
    result = await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="q", mode="auto", top_k=3,
    )
    # Vector pulls top_k * oversample = 12
    assert patched["search_top_k"] == 12
    # Reranker gets the 12 hits and is asked to return top_k=3
    assert patched["rerank_called"] is True
    assert patched["rerank_input_n"] == 12
    assert patched["rerank_top_k"] == 3
    # fake_rerank reverses input order, so final ranking is reversed too.
    # _hits(min(12, 20)) -> 12 hits f0..f11; reversed top 3 = f11,f10,f9
    filenames = [c.filename for c in result.chunks]
    assert filenames == ["f11.pdf", "f10.pdf", "f9.pdf"]


async def test_rerank_combines_with_analysis_double_top_k(patched):
    patched["intent"] = "analysis"
    patched["reranker_enabled"] = True
    patched["reranker_oversample"] = 4
    await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="compare", mode="auto", top_k=5,
    )
    # effective_k = top_k * 2 = 10 (analysis), oversample = 4, so search = 40
    # But _hits caps at 20.
    assert patched["search_top_k"] == 40
    assert patched["rerank_top_k"] == 10  # post-rerank top_n = effective_k


async def test_rerank_skipped_for_casual_intent(patched):
    patched["intent"] = "casual"
    patched["reranker_enabled"] = True
    await orchestrator.retrieve(
        qdrant=None, tenant_id=uuid.uuid4(), question="hi", mode="auto", top_k=3,
    )
    assert patched["rerank_called"] is False
    assert patched["search_top_k"] is None  # casual short-circuits vector too
