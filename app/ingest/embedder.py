"""BGE-M3 embedder — produces dense (1024-d) and sparse vectors.

Lazy-loaded singleton to avoid loading the model in API workers that never
embed (only celery worker uses it).
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.ingest.embedder")

_lock = Lock()
_model: Any = None


def _load() -> Any:
    global _model
    with _lock:
        if _model is not None:
            return _model
        from FlagEmbedding import BGEM3FlagModel  # heavy import — keep local

        settings = get_settings()
        log.info("loading_bge_m3", model=settings.embedding_model, device=settings.embedding_device)
        _model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=settings.embedding_device == "cuda",
        )
        log.info("bge_m3_loaded")
        return _model


@dataclass(frozen=True)
class EmbeddedChunk:
    dense: list[float]               # 1024-dim
    sparse_indices: list[int]
    sparse_values: list[float]


def embed_texts(texts: list[str]) -> list[EmbeddedChunk]:
    if not texts:
        return []
    model = _load()
    settings = get_settings()
    out_raw = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vecs = out_raw["dense_vecs"]
    sparse_lex = out_raw["lexical_weights"]
    results: list[EmbeddedChunk] = []
    for i in range(len(texts)):
        dense = dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else list(dense_vecs[i])
        lex = sparse_lex[i]  # dict[token_id_str, weight]
        items = sorted(((int(k), float(v)) for k, v in lex.items()), key=lambda kv: kv[0])
        idxs = [k for k, _ in items]
        vals = [v for _, v in items]
        results.append(EmbeddedChunk(dense=dense, sparse_indices=idxs, sparse_values=vals))
    return results
