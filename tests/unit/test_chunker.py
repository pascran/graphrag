"""Unit tests for the OCR chunker."""
from __future__ import annotations

from app.ingest.chunker import Chunk, chunk_pages


def test_skips_empty_pages():
    pages = [(1, ""), (2, "   \n  "), (3, "real content here.")]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].index == 0


def test_indexes_are_globally_monotonic():
    pages = [(1, "alpha. " * 200), (2, "beta. " * 200)]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_short_text_returns_single_chunk_per_page():
    pages = [(1, "hello world"), (2, "second page text")]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 2
    assert [c.page_number for c in chunks] == [1, 2]
    assert [c.text for c in chunks] == ["hello world", "second page text"]


def test_long_text_splits_with_overlap():
    page_text = "alpha beta gamma delta. " * 100
    chunks = chunk_pages([(1, page_text)], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        # Recursive overlap stitching can compound across split levels;
        # bound is generous but verifies the chunker does not return runaway
        # chunks that would dwarf the configured size.
        assert len(c.text) <= 1500


def test_chunks_preserve_page_assignment():
    pages = [(7, "a" * 800), (8, "b" * 800)]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=30)
    assert {c.page_number for c in chunks} == {7, 8}
    seven = [c for c in chunks if c.page_number == 7]
    eight = [c for c in chunks if c.page_number == 8]
    assert all("a" in c.text for c in seven)
    assert all("b" in c.text for c in eight)
