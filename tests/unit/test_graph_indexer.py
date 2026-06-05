"""Unit tests for app.ingest.graph_indexer — :Document/:Chunk skeleton plus
GraphRAG :Entity / :MENTIONS / :RELATES_TO upsert."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from app.ingest.graph_extract import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelation,
)
from app.ingest.graph_indexer import (
    ChunkWithGraph,
    GraphChunk,
    delete_document_graph,
    upsert_chunks_skeleton,
    upsert_chunks_with_graph,
)


def _make_driver():
    """Mock AsyncDriver whose session() is an async context manager.

    Records every Cypher statement + params for assertion.
    """
    statements: list[tuple[str, dict]] = []

    result = AsyncMock()
    result.consume = AsyncMock()

    sess = MagicMock()

    async def fake_run(cypher, **kwargs):
        params = dict(kwargs)
        statements.append((cypher, params))
        return result

    sess.run = fake_run

    @asynccontextmanager
    async def fake_session():
        yield sess

    driver = MagicMock()
    driver.session = fake_session
    driver._statements = statements
    return driver


# ---------- skeleton upsert (regression coverage for existing helper) ---------

async def test_upsert_chunks_skeleton_no_chunks_is_noop():
    driver = _make_driver()
    await upsert_chunks_skeleton(driver, [])
    assert driver._statements == []


async def test_upsert_chunks_skeleton_writes_doc_and_chunk_rows():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    chunks = [
        GraphChunk(chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
                   filename="a.pdf", page=1, text="alpha"),
        GraphChunk(chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
                   filename="a.pdf", page=2, text="beta"),
    ]
    await upsert_chunks_skeleton(driver, chunks)
    assert len(driver._statements) == 2
    doc_cypher, doc_params = driver._statements[0]
    assert "MERGE (d:Document" in doc_cypher
    assert len(doc_params["rows"]) == 1
    chunk_cypher, chunk_params = driver._statements[1]
    assert "MERGE (c:Chunk" in chunk_cypher
    assert len(chunk_params["rows"]) == 2


# ---------- GraphRAG entity/relation upsert ----------------------------------

async def test_upsert_chunks_with_graph_no_chunks_is_noop():
    driver = _make_driver()
    await upsert_chunks_with_graph(driver, [])
    assert driver._statements == []


async def test_upsert_chunks_with_graph_creates_entities_and_mentions():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    chunk_id = uuid.uuid4()
    base = GraphChunk(
        chunk_id=chunk_id, tenant_id=tenant, document_id=doc,
        filename="policy.pdf", page=1, text="Acme employs Alice in Seoul.",
    )
    extraction = ChunkExtraction(
        entities=[
            ExtractedEntity(name="Acme", type="ORG", description="company"),
            ExtractedEntity(name="Alice", type="PERSON", description="employee"),
            ExtractedEntity(name="Seoul", type="LOCATION", description=""),
        ],
        relations=[
            ExtractedRelation(source="Acme", target="Alice", kind="employs"),
            ExtractedRelation(
                source="Acme", target="Seoul", kind="headquartered_in"
            ),
        ],
    )
    await upsert_chunks_with_graph(
        driver, [ChunkWithGraph(chunk=base, extraction=extraction)]
    )

    cyphers = [c for c, _ in driver._statements]
    joined = "\n".join(cyphers)
    assert "Document" in joined
    assert "Chunk" in joined
    assert "Entity" in joined
    assert "MENTIONS" in joined
    assert "RELATES_TO" in joined

    entity_stmt = next(
        (cy, p) for cy, p in driver._statements if "MERGE (e:Entity" in cy
    )
    names = {row["name"] for row in entity_stmt[1]["rows"]}
    assert names == {"Acme", "Alice", "Seoul"}

    mentions_stmt = next(
        (cy, p) for cy, p in driver._statements if "MENTIONS" in cy
    )
    assert len(mentions_stmt[1]["rows"]) == 3
    assert all(row["chunk_id"] == str(chunk_id) for row in mentions_stmt[1]["rows"])

    rel_stmt = next(
        (cy, p) for cy, p in driver._statements if "RELATES_TO" in cy
    )
    assert len(rel_stmt[1]["rows"]) == 2
    kinds = {row["kind"] for row in rel_stmt[1]["rows"]}
    assert kinds == {"employs", "headquartered_in"}


async def test_upsert_chunks_with_graph_skips_relations_when_no_entities():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    base = GraphChunk(
        chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
        filename="x.pdf", page=1, text="empty extraction",
    )
    await upsert_chunks_with_graph(
        driver,
        [ChunkWithGraph(chunk=base, extraction=ChunkExtraction(entities=[], relations=[]))],
    )
    cyphers = [c for c, _ in driver._statements]
    joined = "\n".join(cyphers)
    assert "Document" in joined
    assert "Chunk" in joined
    assert "Entity" not in joined
    assert "MENTIONS" not in joined
    assert "RELATES_TO" not in joined


async def test_upsert_chunks_with_graph_dedupes_entities_across_chunks():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    e_acme = ExtractedEntity(name="Acme", type="ORG")
    items = [
        ChunkWithGraph(
            chunk=GraphChunk(
                chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
                filename="a.pdf", page=1, text="t1",
            ),
            extraction=ChunkExtraction(entities=[e_acme], relations=[]),
        ),
        ChunkWithGraph(
            chunk=GraphChunk(
                chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
                filename="a.pdf", page=2, text="t2",
            ),
            extraction=ChunkExtraction(entities=[e_acme], relations=[]),
        ),
    ]
    await upsert_chunks_with_graph(driver, items)
    entity_stmt = next(
        (cy, p) for cy, p in driver._statements if "MERGE (e:Entity" in cy
    )
    assert len({row["name"] for row in entity_stmt[1]["rows"]}) == 1


async def test_upsert_chunks_with_graph_drops_relations_with_unknown_endpoints():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    item = ChunkWithGraph(
        chunk=GraphChunk(
            chunk_id=uuid.uuid4(), tenant_id=tenant, document_id=doc,
            filename="a.pdf", page=1, text="t",
        ),
        extraction=ChunkExtraction(
            entities=[ExtractedEntity(name="A", type="ORG")],
            relations=[
                ExtractedRelation(source="A", target="GHOST", kind="x"),
                ExtractedRelation(source="GHOST", target="A", kind="y"),
            ],
        ),
    )
    await upsert_chunks_with_graph(driver, [item])
    rel_stmts = [
        (cy, p) for cy, p in driver._statements if "RELATES_TO" in cy
    ]
    if rel_stmts:
        for _, params in rel_stmts:
            assert params["rows"] == []


# ---------- delete cascade ---------------------------------------------------

async def test_delete_document_graph_removes_chunks_and_doc():
    driver = _make_driver()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    await delete_document_graph(driver, tenant, doc)
    assert len(driver._statements) >= 1
    cy, params = driver._statements[-1]
    assert "DETACH DELETE" in cy
    assert params["doc_id"] == str(doc)
    assert params["tenant_id"] == str(tenant)
