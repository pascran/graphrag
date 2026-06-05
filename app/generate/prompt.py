"""RAG prompt templates."""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_RAG = (
    "You are a helpful assistant for an enterprise document Q&A system. "
    "Answer the user's question using ONLY the provided source passages. "
    "If the passages do not contain the answer, say so plainly. "
    "Always answer in the same language as the user's question. "
    "Cite sources by listing the filenames you used at the end as `[Source: filename1.pdf, filename2.pdf]`. "
    "Do not invent filenames."
)


@dataclass(frozen=True)
class RetrievedChunk:
    filename: str
    page: int
    text: str


def render_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
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
        f"Answer in the same language as the question, then cite filenames."
    )
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": user_content},
    ]
