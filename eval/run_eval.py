"""GraphRAG vs vector-only evaluation harness.

Runs each question in `eval/golden_qa.json` twice — once with the Neo4j graph
retriever enabled (graph+vector) and once without (vector-only) — using the
project's own RAGAS-style metrics (`app.evaluate.metrics`). Writes raw per-
question results to `eval/raw_results.json` and a summary table to
`eval/results.md`.

Designed to be executed *inside* the app container (so it can import `app.*`):

    docker cp eval                     llm-engine-app-1:/app/
    docker exec llm-engine-app-1 python -m eval.run_eval
    docker cp llm-engine-app-1:/app/eval/raw_results.json eval/
    docker cp llm-engine-app-1:/app/eval/results.md       eval/
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

from app.db.neo4j import get_driver
from app.db.qdrant import get_client as get_qdrant
from app.evaluate.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from app.generate.llm import chat_once
from app.generate.prompt import render_rag_prompt
from app.retrieve.orchestrator import retrieve

EVAL_DIR = Path("/app/eval")
TENANT_ID = uuid.UUID("78c24aa0-b596-4b8d-8ca7-a3025fdc3348")
TOP_K = 5

MODES = ("graph+vector", "vector-only")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


async def run_one(q: dict, mode: str) -> dict:
    qdrant = get_qdrant()
    neo4j = get_driver() if mode == "graph+vector" else None

    t0 = time.time()
    retrieval = await retrieve(
        qdrant,
        tenant_id=TENANT_ID,
        question=q["question"],
        mode="fact",
        top_k=TOP_K,
        neo4j=neo4j,
    )
    messages = render_rag_prompt(q["question"], retrieval.chunks)
    answer = await chat_once(messages)
    latency_s = round(time.time() - t0, 3)

    scores: dict[str, float | None] = {}
    scores["faithfulness"] = await faithfulness(answer=answer, chunks=retrieval.chunks)
    scores["answer_relevancy"] = await answer_relevancy(question=q["question"], answer=answer)
    scores["context_precision"] = await context_precision(
        question=q["question"], chunks=retrieval.chunks
    )
    gt = q.get("ground_truth") or ""
    if gt.strip():
        scores["context_recall"] = await context_recall(
            expected_answer=gt, chunks=retrieval.chunks
        )
    else:
        scores["context_recall"] = None

    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "ground_truth": q.get("ground_truth", ""),
        "answer": answer,
        "sources": [{"filename": c.filename, "page": c.page} for c in retrieval.chunks],
        "n_chunks": len(retrieval.chunks),
        "mode_used": retrieval.mode_used,
        "latency_s": latency_s,
        "scores": scores,
    }


def _safe_mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if isinstance(x, float) else "N/A"


def build_results_md(raw: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("# GraphRAG vs Vector-only — synthetic evaluation\n")
    lines.append(
        "Synthetic Korean public-institute travel-policy corpus (4 PNG documents, "
        f"{len(raw[MODES[0]])} golden questions). Both modes use the same retrieval "
        "orchestrator and the same RAG prompt; the only difference is whether the "
        "Neo4j GraphRAG local-search retriever is consulted alongside Qdrant "
        "dense+sparse RRF.\n"
    )
    lines.append("## Aggregate metrics (mean across all questions)\n")
    lines.append("| Mode | faithfulness | answer_relevancy | context_precision | context_recall | latency (s) |")
    lines.append("|---|---|---|---|---|---|")
    for mode in MODES:
        rows = raw[mode]
        means = {
            name: _safe_mean([r["scores"][name] for r in rows]) for name in METRIC_NAMES
        }
        latency_vals = [r["latency_s"] for r in rows]
        latency_mean = round(statistics.mean(latency_vals), 2) if latency_vals else None
        lines.append(
            "| {mode} | {f} | {r} | {p} | {cr} | {lat} |".format(
                mode=mode,
                f=_fmt(means["faithfulness"]),
                r=_fmt(means["answer_relevancy"]),
                p=_fmt(means["context_precision"]),
                cr=_fmt(means["context_recall"]),
                lat=f"{latency_mean:.2f}" if latency_mean is not None else "N/A",
            )
        )

    lines.append("\n## Mean by question category\n")
    cats = ("single-fact", "multi-hop", "trap-no-evidence")
    for cat in cats:
        lines.append(f"### {cat}\n")
        lines.append("| Mode | faithfulness | answer_relevancy | context_precision | context_recall |")
        lines.append("|---|---|---|---|---|")
        for mode in MODES:
            rows = [r for r in raw[mode] if r["category"] == cat]
            means = {
                name: _safe_mean([r["scores"][name] for r in rows]) for name in METRIC_NAMES
            }
            lines.append(
                "| {mode} | {f} | {r} | {p} | {cr} |".format(
                    mode=mode,
                    f=_fmt(means["faithfulness"]),
                    r=_fmt(means["answer_relevancy"]),
                    p=_fmt(means["context_precision"]),
                    cr=_fmt(means["context_recall"]),
                )
            )
        lines.append("")

    lines.append("## Per-question result\n")
    lines.append("| id | category | mode | faithfulness | answer_relevancy | context_precision | context_recall |")
    lines.append("|---|---|---|---|---|---|---|")
    by_id: dict[str, dict[str, dict]] = {}
    for mode in MODES:
        for r in raw[mode]:
            by_id.setdefault(r["id"], {})[mode] = r
    for qid in sorted(by_id):
        for mode in MODES:
            r = by_id[qid].get(mode)
            if not r:
                continue
            s = r["scores"]
            lines.append(
                "| {qid} | {cat} | {mode} | {f} | {r} | {p} | {cr} |".format(
                    qid=qid,
                    cat=r["category"],
                    mode=mode,
                    f=_fmt(s["faithfulness"]),
                    r=_fmt(s["answer_relevancy"]),
                    p=_fmt(s["context_precision"]),
                    cr=_fmt(s["context_recall"]),
                )
            )

    lines.append("\n## Honest limitations\n")
    lines.append(
        "- Single labeller (the author) wrote both the synthetic documents and "
        "the golden answers, so the ground truth is biased by what felt natural "
        "to ask. This is a *directional sanity check*, not a precision benchmark.\n"
        "- N=12 questions is too small to make strong claims about absolute "
        "metric levels; only the relative deltas between modes are informative.\n"
        "- `faithfulness`, `context_precision`, and `answer_relevancy` use the "
        "same local vLLM (Gemma 4 26B AWQ) as both generator *and* judge — judge "
        "and generator share failure modes, so absolute scores are optimistic.\n"
        "- `context_recall` requires a ground-truth reference; trap questions "
        "intentionally have no reference and report N/A for that metric.\n"
        "- Latency is wall-clock from request to answer (excluding metric "
        "scoring), measured once per question, no warm-up.\n"
    )

    return "\n".join(lines) + "\n"


async def main() -> None:
    qa_path = EVAL_DIR / "golden_qa.json"
    with qa_path.open(encoding="utf-8") as f:
        qa = json.load(f)
    questions = qa["questions"]

    raw: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for mode in MODES:
        print(f"\n=== {mode} ({len(questions)} questions) ===", flush=True)
        for q in questions:
            print(f"  {q['id']:12s} ...", flush=True, end=" ")
            try:
                rec = await run_one(q, mode)
                raw[mode].append(rec)
                s = rec["scores"]

                def _f(x: float | None) -> str:
                    return f"{x:.2f}" if isinstance(x, float) else "N/A"

                print(
                    f"f={_f(s['faithfulness'])} "
                    f"r={_f(s['answer_relevancy'])} "
                    f"p={_f(s['context_precision'])} "
                    f"cr={_f(s['context_recall'])} "
                    f"t={rec['latency_s']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                raw[mode].append(
                    {
                        "id": q["id"],
                        "category": q["category"],
                        "question": q["question"],
                        "error": str(exc),
                        "scores": {n: None for n in METRIC_NAMES},
                        "latency_s": 0.0,
                    }
                )

    raw_path = EVAL_DIR / "raw_results.json"
    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {raw_path}")

    md = build_results_md(raw)
    md_path = EVAL_DIR / "results.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
