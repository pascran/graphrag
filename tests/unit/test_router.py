"""Unit tests for app.retrieve.router.classify with LLM mocked."""
from __future__ import annotations

import pytest

from app.retrieve import router


def _patch_chat(monkeypatch, response: str | Exception):
    async def fake_chat_once(*args, **kwargs):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(router, "chat_once", fake_chat_once)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("casual", "casual"),
        ("CASUAL", "casual"),
        ("  fact  \n", "fact"),
        ("analysis.", "analysis"),
        ('"fact"', "fact"),
        ("fact something extra", "fact"),
    ],
)
async def test_classify_normalizes_label_variants(monkeypatch, raw, expected):
    _patch_chat(monkeypatch, raw)
    assert await router.classify("anything") == expected


async def test_unknown_label_falls_back_to_fact(monkeypatch):
    _patch_chat(monkeypatch, "uncategorized")
    assert await router.classify("ambiguous") == "fact"


async def test_empty_response_falls_back_to_fact(monkeypatch):
    _patch_chat(monkeypatch, "")
    assert await router.classify("anything") == "fact"


async def test_llm_failure_falls_back_to_fact(monkeypatch):
    _patch_chat(monkeypatch, RuntimeError("vllm down"))
    assert await router.classify("query") == "fact"
