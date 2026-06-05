"""Recursive character chunker for OCR markdown output."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    index: int
    page_number: int
    text: str


def _recursive_split(text: str, size: int, overlap: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    sep, rest = (seps[0], seps[1:]) if seps else ("", [])
    if not sep:
        return [text[i : i + size] for i in range(0, len(text), max(1, size - overlap))]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > size:
            out.extend(_recursive_split(part, size, overlap, rest))
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)

    # apply overlap by stitching the tail of each chunk onto the next one
    if overlap > 0 and len(out) > 1:
        with_overlap: list[str] = [out[0]]
        for prev, cur in zip(out[:-1], out[1:], strict=True):
            tail = prev[-overlap:] if len(prev) > overlap else prev
            with_overlap.append(tail + sep + cur if sep else tail + cur)
        out = with_overlap
    return out


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Pages = [(page_number, markdown_text), ...]."""
    chunks: list[Chunk] = []
    idx = 0
    for page_number, text in pages:
        if not text or not text.strip():
            continue
        for piece in _recursive_split(text, chunk_size, chunk_overlap, DEFAULT_SEPARATORS):
            chunks.append(Chunk(index=idx, page_number=page_number, text=piece.strip()))
            idx += 1
    return chunks
