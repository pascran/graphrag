"""Unit tests for app.evaluate.metrics — RAGAS-style scoring primitives.

All external dependencies (Gemma chat_once, BGE-M3 embed_texts) are
monkeypatched. Each metric returns a float in [0.0, 1.0] or None when
the inputs make the metric undefined.
"""
from __future__ import annotations

from app.evaluate import metrics as m
from app.generate.prompt import RetrievedChunk
from app.ingest.embedder import EmbeddedChunk


def _chunk(filename: str, page: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(filename=filename, page=page, text=text)


# ============================================================================
# faithfulness — answer claims vs retrieved passages
# ============================================================================

async def test_faithfulness_returns_none_when_no_chunks():
    out = await m.faithfulness(answer="Anything.", chunks=[])
    assert out is None


async def test_faithfulness_perfect_when_every_claim_supported(monkeypatch):
    calls = []

    async def fake_chat(messages, **kw):
        calls.append(messages)
        body = messages[-1]["content"]
        if "decompose" in body.lower() or "atomic" in body.lower():
            return '["Anthropic was founded in 2021.", "Claude is its model."]'
        return "yes"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.faithfulness(
        answer="Anthropic was founded in 2021. Claude is its model.",
        chunks=[_chunk("a.pdf", 1, "Anthropic founded 2021. Claude is its flagship.")],
    )
    assert out == 1.0
    assert len(calls) == 3  # 1 decompose + 2 judge


async def test_faithfulness_half_when_half_unsupported(monkeypatch):
    judge_idx = {"i": 0}
    judgments = ["yes", "no", "yes", "no"]

    async def fake_chat(messages, **kw):
        body = messages[-1]["content"]
        if "atomic" in body.lower() or "decompose" in body.lower():
            return '["a", "b", "c", "d"]'
        out = judgments[judge_idx["i"]]
        judge_idx["i"] += 1
        return out

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.faithfulness(answer="a b c d.", chunks=[_chunk("x.pdf", 1, "ctx")])
    assert out == 0.5


async def test_faithfulness_handles_empty_claim_list(monkeypatch):
    async def fake_chat(messages, **kw):
        return "[]"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.faithfulness(answer="", chunks=[_chunk("x.pdf", 1, "ctx")])
    assert out is None


async def test_faithfulness_tolerates_malformed_decompose_json(monkeypatch):
    async def fake_chat(messages, **kw):
        return "not json"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.faithfulness(
        answer="Some answer.", chunks=[_chunk("x.pdf", 1, "ctx")]
    )
    assert out is None


# ============================================================================
# answer_relevancy — generate Qs from answer, embed-compare with original Q
# ============================================================================

async def test_answer_relevancy_returns_none_for_empty_answer(monkeypatch):
    async def boom(*a, **kw):
        raise AssertionError("must not call LLM on empty answer")

    monkeypatch.setattr(m, "chat_once", boom)
    out = await m.answer_relevancy(question="Q?", answer="")
    assert out is None


async def test_answer_relevancy_high_score_when_generated_questions_match(monkeypatch):
    async def fake_chat(messages, **kw):
        return '["When was Anthropic founded?", "What year was Anthropic founded?"]'

    def fake_embed(texts):
        return [EmbeddedChunk(dense=[1.0, 0.0], sparse_indices=[], sparse_values=[])
                for _ in texts]

    monkeypatch.setattr(m, "chat_once", fake_chat)
    monkeypatch.setattr(m, "embed_texts", fake_embed)
    out = await m.answer_relevancy(
        question="When was Anthropic founded?",
        answer="Anthropic was founded in 2021.",
    )
    assert out is not None
    assert out > 0.99


async def test_answer_relevancy_low_score_when_questions_unrelated(monkeypatch):
    async def fake_chat(messages, **kw):
        return '["What is the weather?"]'

    def fake_embed(texts):
        return [
            EmbeddedChunk(
                dense=[1.0, 0.0] if "weather" not in t else [0.0, 1.0],
                sparse_indices=[], sparse_values=[],
            )
            for t in texts
        ]

    monkeypatch.setattr(m, "chat_once", fake_chat)
    monkeypatch.setattr(m, "embed_texts", fake_embed)
    out = await m.answer_relevancy(
        question="When was Anthropic founded?",
        answer="The sky is blue.",
    )
    assert out == 0.0


async def test_answer_relevancy_returns_none_when_decompose_fails(monkeypatch):
    async def fake_chat(messages, **kw):
        return "bogus output"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.answer_relevancy(question="q?", answer="a.")
    assert out is None


# ============================================================================
# context_precision — was each retrieved chunk useful?
# ============================================================================

async def test_context_precision_returns_none_for_no_chunks():
    out = await m.context_precision(question="q?", chunks=[])
    assert out is None


async def test_context_precision_all_useful_perfect(monkeypatch):
    async def fake_chat(messages, **kw):
        return "yes"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.context_precision(
        question="q?",
        chunks=[_chunk("a.pdf", 1, "ctx-a"), _chunk("b.pdf", 1, "ctx-b")],
    )
    assert out == 1.0


async def test_context_precision_first_chunk_useless_drops_score(monkeypatch):
    seq = ["no", "yes", "yes"]
    i = {"k": 0}

    async def fake_chat(messages, **kw):
        out = seq[i["k"]]
        i["k"] += 1
        return out

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.context_precision(
        question="q?",
        chunks=[
            _chunk("a.pdf", 1, "off-topic"),
            _chunk("b.pdf", 1, "relevant"),
            _chunk("c.pdf", 1, "relevant"),
        ],
    )
    # Rank-aware MAP:
    # rank1=0, rank2=1 (precision_at_2=1/2), rank3=1 (precision_at_3=2/3)
    # MAP = (1/2 + 2/3) / 2 ~= 0.583
    assert 0.55 < out < 0.62


# ============================================================================
# context_recall — given expected_answer, are its claims supported by chunks?
# ============================================================================

async def test_context_recall_returns_none_without_expected_answer():
    out = await m.context_recall(
        expected_answer="",
        chunks=[_chunk("a.pdf", 1, "ctx")],
    )
    assert out is None


async def test_context_recall_returns_none_for_no_chunks():
    out = await m.context_recall(expected_answer="something", chunks=[])
    assert out is None


async def test_context_recall_perfect_when_every_claim_supported(monkeypatch):
    async def fake_chat(messages, **kw):
        body = messages[-1]["content"]
        if "atomic" in body.lower() or "decompose" in body.lower():
            return '["c1", "c2"]'
        return "yes"

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.context_recall(
        expected_answer="c1. c2.",
        chunks=[_chunk("a.pdf", 1, "ctx that supports c1 and c2")],
    )
    assert out == 1.0


async def test_context_recall_partial(monkeypatch):
    judgments = ["yes", "no", "yes"]
    i = {"k": 0}

    async def fake_chat(messages, **kw):
        body = messages[-1]["content"]
        if "atomic" in body.lower() or "decompose" in body.lower():
            return '["a", "b", "c"]'
        out = judgments[i["k"]]
        i["k"] += 1
        return out

    monkeypatch.setattr(m, "chat_once", fake_chat)
    out = await m.context_recall(
        expected_answer="a b c",
        chunks=[_chunk("x.pdf", 1, "partial ctx")],
    )
    assert abs(out - 2 / 3) < 1e-6


# ============================================================================
# helpers
# ============================================================================

def test_cosine_similarity_handles_zero_vectors():
    assert m._cosine([0, 0, 0], [1, 2, 3]) == 0.0
    assert m._cosine([1, 0, 0], [0, 0, 0]) == 0.0


def test_cosine_similarity_value_for_known_vectors():
    assert m._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert m._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    val = m._cosine([1.0, 1.0], [1.0, 0.0])
    assert abs(val - (1 / (2 ** 0.5))) < 1e-6


def test_parse_yes_no_robust_to_punctuation_and_case():
    assert m._parse_yes_no("Yes.") is True
    assert m._parse_yes_no("  YES  ") is True
    assert m._parse_yes_no("yes, it is supported") is True
    assert m._parse_yes_no("No") is False
    assert m._parse_yes_no("nope") is False
    assert m._parse_yes_no("maybe") is False
    assert m._parse_yes_no("") is False
