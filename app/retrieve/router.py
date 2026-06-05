"""LLM-based query router.

Classifies the user question into one of three intents so the orchestrator can
pick the right retrieval strategy:

  casual    → no retrieval (general knowledge, greetings, math, coding help)
  fact      → narrow vector search (top_k as-is)  — answer pinpointed in 1-3 chunks
  analysis  → broader vector search (top_k * 2)   — needs synthesis across passages

Uses Gemma at temperature=0 with a 4-token cap. Falls back to 'fact' on any
parse / network failure so RAG still serves the question rather than blocking.
"""
from __future__ import annotations

from typing import Literal

from app.generate.llm import chat_once
from app.utils.logging import get_logger

Intent = Literal["casual", "fact", "analysis"]

log = get_logger("app.retrieve.router")

_VALID: set[Intent] = {"casual", "fact", "analysis"}

_SYSTEM = (
    "Classify the user's question into exactly ONE label and reply with ONLY "
    "that label, no punctuation, no explanation:\n"
    "- casual    : greetings, small talk, general knowledge, math, coding help, "
    "definitions answerable without a private document\n"
    "- fact      : asks for a specific value, name, date, number, definition, or "
    "policy that should be looked up in indexed enterprise documents\n"
    "- analysis  : asks to compare, summarize, explain trade-offs, or synthesize "
    "across multiple sections of indexed documents\n"
    "Reply with one word: casual OR fact OR analysis."
)


async def classify(question: str) -> Intent:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question.strip()},
    ]
    try:
        raw = await chat_once(messages, temperature=0.0, max_tokens=4, timeout=15.0)
    except Exception as e:
        log.warning("router_llm_failed", error=str(e))
        return "fact"

    label = raw.strip().lower().split()[0] if raw.strip() else ""
    label = label.strip(".,;:!?\"'`*")
    if label in _VALID:
        return label  # type: ignore[return-value]
    log.info("router_unknown_label", raw=raw[:80])
    return "fact"
