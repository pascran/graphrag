"""Unit tests for app.ingest.{vector_indexer, ocr, embedder, pipeline}.

External services (Qdrant, vLLM Chandra, FlagEmbedding, Neo4j, pdf2image)
are monkeypatched so the suite stays in-process.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingest import embedder as embedder_mod
from app.ingest import ocr as ocr_mod
from app.ingest import pipeline as pipeline_mod
from app.ingest import vector_indexer as vi_mod
from app.ingest.embedder import EmbeddedChunk
from app.ingest.graph_extract import ChunkExtraction, ExtractedEntity
from app.ingest.vector_indexer import IndexableChunk


# ============================================================================
# vector_indexer
# ============================================================================

async def test_vector_upsert_empty_is_noop():
    client = MagicMock()
    client.upsert = AsyncMock()
    out = await vi_mod.upsert_chunks(client, [])
    assert out == 0
    client.upsert.assert_not_awaited()


def _make_indexable(n: int = 1) -> list[IndexableChunk]:
    return [
        IndexableChunk(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename=f"f{i}.pdf",
            page=i + 1,
            doc_type="brief",
            chunk_index=i,
            text=f"chunk {i} text",
            embedding=EmbeddedChunk(
                dense=[0.1] * 4,
                sparse_indices=[0, 7],
                sparse_values=[0.4, 0.6],
            ),
        )
        for i in range(n)
    ]


async def test_vector_upsert_writes_dense_sparse_payload():
    client = MagicMock()
    client.upsert = AsyncMock()
    chunks = _make_indexable(2)
    out = await vi_mod.upsert_chunks(client, chunks)
    assert out == 2
    client.upsert.assert_awaited_once()
    kwargs = client.upsert.await_args.kwargs
    assert kwargs["wait"] is True
    points = kwargs["points"]
    assert len(points) == 2
    p0 = points[0]
    assert p0.id == str(chunks[0].id)
    assert p0.vector["dense"] == chunks[0].embedding.dense
    assert p0.vector["sparse"].indices == [0, 7]
    assert p0.payload["filename"] == "f0.pdf"
    assert p0.payload["chunk_index"] == 0


async def test_vector_delete_filters_by_tenant_and_document():
    client = MagicMock()
    client.delete = AsyncMock()
    tenant = uuid.uuid4()
    doc = uuid.uuid4()
    await vi_mod.delete_document_chunks(client, tenant, doc)
    client.delete.assert_awaited_once()
    sel = client.delete.await_args.kwargs["points_selector"]
    keys = sorted(c.key for c in sel.must)
    assert keys == ["document_id", "tenant_id"]


# ============================================================================
# ocr — pure helpers + dispatcher
# ============================================================================

def test_pil_to_data_uri_emits_png_base64():
    from PIL import Image
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    uri = ocr_mod._pil_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > len("data:image/png;base64,")


def test_file_to_data_uri_uses_path_mime(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    uri = ocr_mod._file_to_data_uri(p)
    assert uri.startswith("data:image/png;base64,")


def test_file_to_data_uri_falls_back_to_png_when_unknown_mime(tmp_path):
    p = tmp_path / "x.unknownext"
    p.write_bytes(b"\x00\x01")
    uri = ocr_mod._file_to_data_uri(p)
    assert uri.startswith("data:image/png;base64,")


async def test_ocr_file_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "weird.docx"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        await ocr_mod.ocr_file(p)


async def test_ocr_file_dispatches_pdf_to_ocr_pdf(monkeypatch, tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    called = {}

    async def fake_pdf(path):
        called["path"] = str(path)
        return [ocr_mod.OcrPage(page_number=1, markdown="hi")]

    monkeypatch.setattr(ocr_mod, "ocr_pdf", fake_pdf)
    out = await ocr_mod.ocr_file(p)
    assert out[0].markdown == "hi"
    assert called["path"].endswith("x.pdf")


async def test_ocr_file_dispatches_image_to_ocr_image(monkeypatch, tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    called = {}

    async def fake_image(path):
        called["path"] = str(path)
        return [ocr_mod.OcrPage(page_number=1, markdown="img-md")]

    monkeypatch.setattr(ocr_mod, "ocr_image", fake_image)
    out = await ocr_mod.ocr_file(p)
    assert out[0].markdown == "img-md"


async def test_ocr_image_calls_chandra_with_data_uri(monkeypatch, tmp_path):
    p = tmp_path / "scan.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "## Heading\nbody"}}]}

    class _Client:
        def __init__(self, *a, **kw):
            captured["base_url"] = kw.get("base_url")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, timeout):
            captured["url"] = url
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", _Client)
    out = await ocr_mod.ocr_image(p)
    assert out == [ocr_mod.OcrPage(page_number=1, markdown="## Heading\nbody")]
    assert captured["url"] == "/chat/completions"
    content = captured["payload"]["messages"][0]["content"]
    image_url = next(c for c in content if c["type"] == "image_url")
    assert image_url["image_url"]["url"].startswith("data:image/png;base64,")


async def test_ocr_pdf_iterates_pages_with_chandra(monkeypatch, tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    from PIL import Image

    monkeypatch.setattr(
        ocr_mod, "convert_from_path",
        lambda path, dpi: [Image.new("RGB", (4, 4), (i * 50, 0, 0)) for i in range(2)],
    )

    async def fake_call(client, model, data_uri):
        return f"page-md-{data_uri[-3:]}"

    monkeypatch.setattr(ocr_mod, "_ocr_image_data_uri", fake_call)

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", _Client)
    out = await ocr_mod.ocr_pdf(p)
    assert len(out) == 2
    assert out[0].page_number == 1
    assert out[1].page_number == 2
    assert out[0].markdown.startswith("page-md-")


# ============================================================================
# embedder
# ============================================================================

def test_embed_texts_empty_returns_empty(monkeypatch):
    def boom():
        raise AssertionError("model loader hit on empty input")

    monkeypatch.setattr(embedder_mod, "_load", boom)
    assert embedder_mod.embed_texts([]) == []


def test_embed_texts_packs_dense_and_sorted_sparse(monkeypatch):
    class _Vec(list):
        def tolist(self):
            return list(self)

    fake_model = MagicMock()
    fake_model.encode.return_value = {
        "dense_vecs": [_Vec([0.1, 0.2, 0.3])],
        "lexical_weights": [{"5": 0.7, "1": 0.3, "9": 0.5}],
    }
    monkeypatch.setattr(embedder_mod, "_load", lambda: fake_model)

    out = embedder_mod.embed_texts(["hello"])
    assert len(out) == 1
    assert out[0].dense == [0.1, 0.2, 0.3]
    assert out[0].sparse_indices == [1, 5, 9]
    assert out[0].sparse_values == [0.3, 0.7, 0.5]


def test_embed_texts_handles_dense_without_tolist(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = {
        "dense_vecs": [[0.4, 0.5]],
        "lexical_weights": [{}],
    }
    monkeypatch.setattr(embedder_mod, "_load", lambda: fake_model)
    out = embedder_mod.embed_texts(["x"])
    assert out[0].dense == [0.4, 0.5]
    assert out[0].sparse_indices == []


# ============================================================================
# pipeline.ingest_document — full path with every collaborator mocked
# ============================================================================

@pytest.fixture()
def mock_pipeline_collaborators(monkeypatch):
    pages = [SimpleNamespace(page_number=1, markdown="alpha bravo")]

    async def fake_ocr_file(path):
        return pages

    monkeypatch.setattr(pipeline_mod, "ocr_file", fake_ocr_file)

    chunks = [
        SimpleNamespace(page_number=1, index=0, text="alpha"),
        SimpleNamespace(page_number=1, index=1, text="bravo"),
    ]
    monkeypatch.setattr(pipeline_mod, "chunk_pages", lambda pairs, size, ov: chunks)

    monkeypatch.setattr(
        pipeline_mod, "embed_texts",
        lambda texts: [
            EmbeddedChunk(dense=[0.0] * 4, sparse_indices=[], sparse_values=[])
            for _ in texts
        ],
    )

    upsert_calls = {"vector": 0, "graph": 0, "skeleton": 0}

    async def fake_upsert_vector(client, items):
        upsert_calls["vector"] += 1
        return len(items)

    async def fake_upsert_graph(driver, items):
        upsert_calls["graph"] += 1
        upsert_calls["graph_items"] = items

    async def fake_upsert_skeleton(driver, items):
        upsert_calls["skeleton"] += 1

    monkeypatch.setattr(pipeline_mod, "upsert_chunks", fake_upsert_vector)
    monkeypatch.setattr(pipeline_mod, "upsert_chunks_with_graph", fake_upsert_graph)
    monkeypatch.setattr(pipeline_mod, "upsert_chunks_skeleton", fake_upsert_skeleton)

    qclient = MagicMock()
    qclient.close = AsyncMock()
    monkeypatch.setattr(pipeline_mod, "AsyncQdrantClient", lambda **kw: qclient)

    driver = MagicMock()
    driver.close = AsyncMock()
    monkeypatch.setattr(
        pipeline_mod, "AsyncGraphDatabase",
        SimpleNamespace(driver=lambda *a, **kw: driver),
    )

    return upsert_calls


async def test_pipeline_returns_zero_when_no_chunks(monkeypatch):
    async def fake_ocr_file(path):
        return [SimpleNamespace(page_number=1, markdown="")]

    monkeypatch.setattr(pipeline_mod, "ocr_file", fake_ocr_file)
    monkeypatch.setattr(pipeline_mod, "chunk_pages", lambda *a, **kw: [])

    out = await pipeline_mod.ingest_document(
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_path="/tmp/never.pdf",
        filename="never.pdf",
    )
    assert out.chunk_count == 0
    assert out.page_count == 1


async def test_pipeline_runs_extraction_when_graphrag_enabled(
    monkeypatch, mock_pipeline_collaborators
):
    monkeypatch.setattr(
        pipeline_mod.get_settings(), "graphrag_enabled", True, raising=False
    )

    async def fake_extract(text):
        return ChunkExtraction(
            entities=[ExtractedEntity(name=f"E-{text}", type="ORG")],
            relations=[],
        )

    monkeypatch.setattr(
        pipeline_mod, "extract_entities_and_relations", fake_extract
    )

    out = await pipeline_mod.ingest_document(
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_path="/tmp/x.pdf",
        filename="x.pdf",
    )
    assert out.chunk_count == 2
    assert mock_pipeline_collaborators["vector"] == 1
    assert mock_pipeline_collaborators["graph"] == 1
    assert mock_pipeline_collaborators["skeleton"] == 0
    items = mock_pipeline_collaborators["graph_items"]
    assert len(items) == 2
    assert items[0].extraction.entities[0].name == "E-alpha"


async def test_pipeline_falls_back_to_skeleton_when_extraction_raises(
    monkeypatch, mock_pipeline_collaborators
):
    monkeypatch.setattr(
        pipeline_mod.get_settings(), "graphrag_enabled", True, raising=False
    )

    async def boom_extract_all(texts, concurrency):
        raise RuntimeError("vllm gone")

    monkeypatch.setattr(pipeline_mod, "_extract_all", boom_extract_all)

    out = await pipeline_mod.ingest_document(
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_path="/tmp/x.pdf",
        filename="x.pdf",
    )
    assert out.chunk_count == 2
    assert mock_pipeline_collaborators["vector"] == 1
    assert mock_pipeline_collaborators["skeleton"] == 1
    assert mock_pipeline_collaborators["graph"] == 0


async def test_pipeline_uses_skeleton_when_graphrag_disabled(
    monkeypatch, mock_pipeline_collaborators
):
    monkeypatch.setattr(
        pipeline_mod.get_settings(), "graphrag_enabled", False, raising=False
    )

    out = await pipeline_mod.ingest_document(
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_path="/tmp/x.pdf",
        filename="x.pdf",
    )
    assert out.chunk_count == 2
    assert mock_pipeline_collaborators["skeleton"] == 1
    assert mock_pipeline_collaborators["graph"] == 0


async def test_pipeline_extract_all_bounds_concurrency(monkeypatch):
    in_flight = {"now": 0, "max": 0}

    async def fake_extract(text):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        import asyncio
        await asyncio.sleep(0)
        in_flight["now"] -= 1
        return ChunkExtraction(entities=[], relations=[])

    monkeypatch.setattr(
        pipeline_mod, "extract_entities_and_relations", fake_extract
    )
    out = await pipeline_mod._extract_all([f"t{i}" for i in range(10)], concurrency=2)
    assert len(out) == 10
    assert in_flight["max"] <= 2
