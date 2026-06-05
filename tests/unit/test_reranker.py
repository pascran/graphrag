"""Unit tests for app.retrieve.reranker.rerank."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieve import reranker as r
from app.retrieve.vector import VectorHit


def _hit(i: int, score: float = 0.5) -> VectorHit:
    return VectorHit(
        filename=f"f{i}.pdf",
        page=i,
        text=f"chunk-{i}",
        score=score,
        document_id=f"doc-{i}",
        chunk_index=i,
    )


# ---------- short-circuits --------------------------------------------------

def test_rerank_empty_hits_returns_empty(monkeypatch):
    def boom():
        raise AssertionError("model loader hit on empty input")

    monkeypatch.setattr(r, "_load", boom)
    assert r.rerank(question="q?", hits=[], top_k=5) == []


def test_rerank_single_hit_skips_model(monkeypatch):
    def boom():
        raise AssertionError("model loader hit with single hit")

    monkeypatch.setattr(r, "_load", boom)
    h = _hit(0, score=0.42)
    out = r.rerank(question="q?", hits=[h], top_k=5)
    assert len(out) == 1
    assert out[0] is h


def test_rerank_blank_question_returns_input_unchanged(monkeypatch):
    def boom():
        raise AssertionError("must not call model on blank question")

    monkeypatch.setattr(r, "_load", boom)
    hits = [_hit(0), _hit(1)]
    out = r.rerank(question="  ", hits=hits, top_k=5)
    assert out == hits


# ---------- happy path -----------------------------------------------------

def test_rerank_reorders_by_descending_cross_encoder_score(monkeypatch):
    fake_model = MagicMock()
    fake_model.compute_score.return_value = [0.2, 0.7, 0.9]
    monkeypatch.setattr(r, "_load", lambda: fake_model)

    in_hits = [_hit(0), _hit(1), _hit(2)]
    out = r.rerank(question="q?", hits=in_hits, top_k=3)

    assert [h.filename for h in out] == ["f2.pdf", "f1.pdf", "f0.pdf"]
    assert out[0].score == pytest.approx(0.9)
    assert out[1].score == pytest.approx(0.7)
    assert out[2].score == pytest.approx(0.2)


def test_rerank_truncates_to_top_k(monkeypatch):
    fake_model = MagicMock()
    fake_model.compute_score.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]
    monkeypatch.setattr(r, "_load", lambda: fake_model)

    in_hits = [_hit(i) for i in range(5)]
    out = r.rerank(question="q?", hits=in_hits, top_k=2)
    assert len(out) == 2
    assert [h.chunk_index for h in out] == [1, 4]


def test_rerank_passes_question_paired_with_each_chunk(monkeypatch):
    captured = {}

    fake_model = MagicMock()

    def fake_score(pairs, normalize=True):
        captured["pairs"] = pairs
        captured["normalize"] = normalize
        return [0.1] * len(pairs)

    fake_model.compute_score.side_effect = fake_score
    monkeypatch.setattr(r, "_load", lambda: fake_model)

    in_hits = [_hit(0), _hit(1)]
    r.rerank(question="my question", hits=in_hits, top_k=5)

    assert len(captured["pairs"]) == 2
    assert captured["pairs"][0] == ["my question", "chunk-0"]
    assert captured["pairs"][1] == ["my question", "chunk-1"]
    assert captured["normalize"] is True


# ---------- failure handling ----------------------------------------------

def test_rerank_model_load_failure_falls_back_to_input_order(monkeypatch):
    def boom():
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(r, "_load", boom)
    hits = [_hit(0, score=0.3), _hit(1, score=0.9), _hit(2, score=0.5)]
    out = r.rerank(question="q?", hits=hits, top_k=2)
    assert [h.chunk_index for h in out] == [0, 1]


def test_rerank_score_call_failure_falls_back_to_input_order(monkeypatch):
    fake_model = MagicMock()
    fake_model.compute_score.side_effect = RuntimeError("inference error")
    monkeypatch.setattr(r, "_load", lambda: fake_model)

    hits = [_hit(0), _hit(1), _hit(2)]
    out = r.rerank(question="q?", hits=hits, top_k=2)
    assert [h.chunk_index for h in out] == [0, 1]


def test_rerank_caps_top_k_to_input_length(monkeypatch):
    fake_model = MagicMock()
    fake_model.compute_score.return_value = [0.5, 0.6]
    monkeypatch.setattr(r, "_load", lambda: fake_model)
    out = r.rerank(question="q?", hits=[_hit(0), _hit(1)], top_k=99)
    assert len(out) == 2
