"""RAG prompt templates."""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_RAG = (
    "You are a helpful assistant for an enterprise system. The user may ask "
    "document-related questions OR general / casual questions.\n\n"
    "Source passages from indexed documents may be attached below. Use them "
    "ONLY when they are actually relevant to the user's question. If the "
    "passages are clearly off-topic (e.g. casual greetings, general "
    "knowledge, definitions, math, coding help), ignore them and answer "
    "normally from your own knowledge.\n\n"
    "Citation rules:\n"
    "- If you used one or more passages, end your answer with "
    "`[Source: filename1.pdf, filename2.pdf]` listing ONLY filenames you "
    "actually relied on.\n"
    "- If you did not use the passages, do NOT include a Source line.\n"
    "- Never invent filenames.\n\n"
    "Always answer in the same language as the user's question."
)


@dataclass(frozen=True)
class RetrievedChunk:
    filename: str
    page: int
    text: str


def render_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    prior_turns: list[dict[str, str]] | None = None,
    prior_summary: str | None = None,
) -> list[dict[str, str]]:
    if not chunks:
        passages_block = "(no passages retrieved)"
    else:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(f"[#{i} {c.filename} p.{c.page}]\n{c.text}")
        passages_block = "\n\n".join(parts)

    user_content = (
        f"Source passages:\n\n{passages_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Follow the system rules: cite passages only if you used them."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_RAG}]
    if prior_summary:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation summary so far:\n{prior_summary}",
            }
        )
    if prior_turns:
        messages.extend(prior_turns)
    messages.append({"role": "user", "content": user_content})
    return messages
