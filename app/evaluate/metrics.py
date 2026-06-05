"""RAGAS-style evaluation metrics — native implementation.

Four reference-(mostly-)free metrics for RAG quality:

- faithfulness          : do claims in the answer follow from retrieved chunks?
- answer_relevancy      : does the answer actually address the question?
- context_precision     : were the retrieved chunks useful (rank-aware)?
- context_recall        : (needs ground truth) are expected_answer's claims
                          present in the retrieved chunks?

Each function returns a float in [0.0, 1.0] or None when the inputs make
the metric undefined. All LLM judging uses temperature=0 for determinism.
"""
from __future__ import annotations

import json
import math
import re

from app.generate.llm import chat_once
from app.generate.prompt import RetrievedChunk
from app.ingest.embedder import embed_texts
from app.utils.logging import get_logger

log = get_logger("app.evaluate.metrics")


# ============================================================================
# helpers
# ============================================================================

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _parse_yes_no(raw: str) -> bool:
    s = (raw or "").strip().lower()
    if not s:
        return False
    first = re.split(r"[\s,.;:!?\"'`]+", s, maxsplit=1)[0]
    return first == "yes"


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[#{i} {c.filename} p.{c.page}]\n{c.text}")
    return "\n\n".join(parts)


async def _decompose_claims(text: str) -> list[str] | None:
    """Ask the LLM to split text into atomic factual claims as a JSON array."""
    if not text or not text.strip():
        return None
    system = (
        "Decompose the given text into a JSON array of atomic factual claims. "
        "Each claim must be a self-contained sentence. Output the JSON array only, "
        "no commentary, no markdown fence."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Decompose into atomic claims:\n\n{text.strip()}"},
    ]
    try:
        raw = await chat_once(messages, temperature=0.0, max_tokens=512, timeout=30.0)
    except Exception as e:
        log.warning("decompose_llm_failed", error=str(e))
        return None
    body = raw.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.IGNORECASE | re.DOTALL)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        log.warning("decompose_parse_failed", preview=body[:200])
        return None
    if not isinstance(parsed, list):
        return None
    return [str(x).strip() for x in parsed if str(x).strip()]


async def _judge_entailment(claim: str, passages: str) -> bool:
    """Ask the LLM whether `claim` is entailed by `passages`. Returns yes/no."""
    system = (
        "You are an entailment judge. Reply with exactly one word: yes OR no. "
        "Reply 'yes' iff the claim is directly supported by the passages."
    )
    user = f"Passages:\n\n{passages}\n\n---\n\nClaim: {claim}"
    try:
        raw = await chat_once(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=4, timeout=20.0,
        )
    except Exception as e:
        log.warning("judge_llm_failed", error=str(e))
        return False
    return _parse_yes_no(raw)


# ============================================================================
# faithfulness
# ============================================================================

async def faithfulness(
    *, answer: str, chunks: list[RetrievedChunk]
) -> float | None:
    if not chunks:
        return None
    claims = await _decompose_claims(answer)
    if claims is None or not claims:
        return None
    passages = _format_passages(chunks)
    supported = 0
    for c in claims:
        if await _judge_entailment(c, passages):
            supported += 1
    return supported / len(claims)


# ============================================================================
# answer_relevancy
# ============================================================================

async def answer_relevancy(*, question: str, answer: str) -> float | None:
    if not answer or not answer.strip():
        return None
    system = (
        "Generate a JSON array of 3 questions that the given answer fully answers. "
        "Each question must be a complete sentence ending in '?'. "
        "Output the JSON array only, no commentary, no markdown fence."
    )
    try:
        raw = await chat_once(
            [{"role": "system", "content": system},
             {"role": "user", "content": answer.strip()}],
            temperature=0.0, max_tokens=256, timeout=30.0,
        )
    except Exception as e:
        log.warning("relevancy_llm_failed", error=str(e))
        return None
    body = raw.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.IGNORECASE | re.DOTALL)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        log.warning("relevancy_parse_failed", preview=body[:200])
        return None
    if not isinstance(parsed, list):
        return None
    generated = [str(x).strip() for x in parsed if str(x).strip()]
    if not generated:
        return None

    embeds = embed_texts([question.strip()] + generated)
    if not embeds or len(embeds) < 2:
        return None
    q_vec = embeds[0].dense
    sims = [_cosine(q_vec, e.dense) for e in embeds[1:]]
    if not sims:
        return None
    return sum(sims) / len(sims)


# ============================================================================
# context_precision (rank-aware Mean Average Precision over usefulness)
# ============================================================================

async def context_precision(
    *, question: str, chunks: list[RetrievedChunk]
) -> float | None:
    if not chunks:
        return None
    system = (
        "You judge whether a passage is useful for answering a question. "
        "Reply with exactly one word: yes OR no."
    )
    relevance: list[int] = []
    for c in chunks:
        user = f"Question: {question}\n\nPassage ({c.filename} p.{c.page}):\n{c.text}"
        try:
            raw = await chat_once(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.0, max_tokens=4, timeout=20.0,
            )
        except Exception as e:
            log.warning("precision_llm_failed", error=str(e))
            relevance.append(0)
            continue
        relevance.append(1 if _parse_yes_no(raw) else 0)

    relevant_total = sum(relevance)
    if relevant_total == 0:
        return 0.0
    seen = 0
    ap = 0.0
    for k, r in enumerate(relevance, start=1):
        if r == 1:
            seen += 1
            ap += seen / k
    return ap / relevant_total


# ============================================================================
# context_recall — ground-truth-driven
# ============================================================================

async def context_recall(
    *, expected_answer: str, chunks: list[RetrievedChunk]
) -> float | None:
    if not expected_answer or not expected_answer.strip():
        return None
    if not chunks:
        return None
    claims = await _decompose_claims(expected_answer)
    if claims is None or not claims:
        return None
    passages = _format_passages(chunks)
    supported = 0
    for c in claims:
        if await _judge_entailment(c, passages):
            supported += 1
    return supported / len(claims)
