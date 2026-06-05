"""BGE Reranker v2 cross-encoder re-ranking.

The orchestrator oversamples from Qdrant (top_k * reranker_oversample),
then this module scores every (question, chunk) pair via FlagReranker and
returns the top_k by descending score. Falls back to the input ordering on
any failure (model load, inference error) so a flaky reranker never blocks
a query.

Lazy-loaded singleton — only the FastAPI workers that actually call
/v1/query pay the model load cost.
"""
from __future__ import annotations

from dataclasses import replace
from threading import Lock
from typing import Any

from app.config import get_settings
from app.retrieve.vector import VectorHit
from app.utils.logging import get_logger

log = get_logger("app.retrieve.reranker")

_lock = Lock()
_model: Any = None


class _Reranker:
    """Thin wrapper exposing compute_score(pairs, normalize=True) over
    sentence_transformers.CrossEncoder. We use sentence-transformers
    instead of FlagEmbedding.FlagReranker because the latter's tokenizer
    path is incompatible with transformers >= 5.x.
    """

    def __init__(self, model_name: str, use_fp16: bool):
        from sentence_transformers import CrossEncoder  # heavy import — keep local

        kwargs: dict[str, Any] = {"max_length": 512}
        if use_fp16:
            try:
                import torch
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            except Exception:
                pass
        self._ce = CrossEncoder(model_name, **kwargs)

    def compute_score(self, pairs: list[list[str]], normalize: bool = True):
        # CrossEncoder.predict returns numpy array of raw logits.
        raw = self._ce.predict(pairs)
        scores = [float(s) for s in raw]
        if not normalize:
            return scores
        # bge-reranker logits → [0, 1] via sigmoid for comparability.
        import math
        return [1.0 / (1.0 + math.exp(-s)) for s in scores]


def _load() -> Any:
    global _model
    with _lock:
        if _model is not None:
            return _model
        settings = get_settings()
        log.info(
            "loading_reranker",
            model=settings.reranker_model,
            use_fp16=settings.reranker_use_fp16,
        )
        _model = _Reranker(settings.reranker_model, settings.reranker_use_fp16)
        log.info("reranker_loaded")
        return _model


def rerank(*, question: str, hits: list[VectorHit], top_k: int) -> list[VectorHit]:
    if not hits:
        return []
    limit = max(1, min(top_k, len(hits)))
    if len(hits) == 1:
        return [hits[0]]
    q = (question or "").strip()
    if not q:
        return list(hits[:limit])

    try:
        model = _load()
    except Exception as e:
        log.warning("reranker_load_failed", error=str(e))
        return list(hits[:limit])

    pairs = [[q, h.text] for h in hits]
    try:
        scores = model.compute_score(pairs, normalize=True)
    except Exception as e:
        log.warning("reranker_score_failed", error=str(e))
        return list(hits[:limit])

    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = [float(s) for s in scores]
    if len(scores) != len(hits):
        log.warning(
            "reranker_score_length_mismatch",
            expected=len(hits), got=len(scores),
        )
        return list(hits[:limit])

    paired = [(s, h) for s, h in zip(scores, hits, strict=True)]
    paired.sort(key=lambda sh: sh[0], reverse=True)
    out = [replace(h, score=s) for s, h in paired[:limit]]
    log.info("rerank_done", input=len(hits), kept=len(out))
    return out
