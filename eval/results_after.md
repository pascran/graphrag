# GraphRAG vs Vector-only — synthetic evaluation

Synthetic Korean public-institute travel-policy corpus (4 PNG documents, 12 golden questions). Both modes use the same retrieval orchestrator and the same RAG prompt; the only difference is whether the Neo4j GraphRAG local-search retriever is consulted alongside Qdrant dense+sparse RRF.

## Aggregate metrics (mean across all questions)

| Mode | faithfulness | answer_relevancy | context_precision | context_recall | latency (s) |
|---|---|---|---|---|---|
| graph+vector | 0.903 | 0.861 | 0.815 | 0.778 | 13.54 |
| vector-only | 0.944 | 0.871 | 0.815 | 0.778 | 12.23 |

## Mean by question category

### single-fact

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 1.000 | 0.897 | 0.983 | 1.000 |
| vector-only | 0.933 | 0.888 | 0.983 | 1.000 |

### multi-hop

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 0.867 | 0.821 | 0.973 | 0.867 |
| vector-only | 0.933 | 0.852 | 0.973 | 0.867 |

### trap-no-evidence

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 0.750 | 0.875 | 0.000 | 0.000 |
| vector-only | 1.000 | 0.876 | 0.000 | 0.000 |

## Per-question result

| id | category | mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|
| multihop-01 | multi-hop | graph+vector | 0.667 | 0.771 | 1.000 | 0.333 |
| multihop-01 | multi-hop | vector-only | 0.667 | 0.808 | 1.000 | 0.333 |
| multihop-02 | multi-hop | graph+vector | 1.000 | 0.891 | 1.000 | 1.000 |
| multihop-02 | multi-hop | vector-only | 1.000 | 0.891 | 1.000 | 1.000 |
| multihop-03 | multi-hop | graph+vector | 0.667 | 0.754 | 1.000 | 1.000 |
| multihop-03 | multi-hop | vector-only | 1.000 | 0.906 | 1.000 | 1.000 |
| multihop-04 | multi-hop | graph+vector | 1.000 | 0.900 | 0.867 | 1.000 |
| multihop-04 | multi-hop | vector-only | 1.000 | 0.900 | 0.867 | 1.000 |
| multihop-05 | multi-hop | graph+vector | 1.000 | 0.786 | 1.000 | 1.000 |
| multihop-05 | multi-hop | vector-only | 1.000 | 0.755 | 1.000 | 1.000 |
| single-01 | single-fact | graph+vector | 1.000 | 0.969 | 1.000 | 1.000 |
| single-01 | single-fact | vector-only | 1.000 | 0.963 | 1.000 | 1.000 |
| single-02 | single-fact | graph+vector | 1.000 | 0.877 | 1.000 | 1.000 |
| single-02 | single-fact | vector-only | 1.000 | 0.875 | 1.000 | 1.000 |
| single-03 | single-fact | graph+vector | 1.000 | 0.719 | 1.000 | 1.000 |
| single-03 | single-fact | vector-only | 1.000 | 0.693 | 1.000 | 1.000 |
| single-04 | single-fact | graph+vector | 1.000 | 0.932 | 0.917 | 1.000 |
| single-04 | single-fact | vector-only | 0.667 | 0.925 | 0.917 | 1.000 |
| single-05 | single-fact | graph+vector | 1.000 | 0.986 | 1.000 | 1.000 |
| single-05 | single-fact | vector-only | 1.000 | 0.986 | 1.000 | 1.000 |
| trap-01 | trap-no-evidence | graph+vector | 0.500 | 0.805 | 0.000 | 0.000 |
| trap-01 | trap-no-evidence | vector-only | 1.000 | 0.809 | 0.000 | 0.000 |
| trap-02 | trap-no-evidence | graph+vector | 1.000 | 0.944 | 0.000 | 0.000 |
| trap-02 | trap-no-evidence | vector-only | 1.000 | 0.944 | 0.000 | 0.000 |

## Honest limitations

- Single labeller (the author) wrote both the synthetic documents and the golden answers, so the ground truth is biased by what felt natural to ask. This is a *directional sanity check*, not a precision benchmark.
- N=12 questions is too small to make strong claims about absolute metric levels; only the relative deltas between modes are informative.
- `faithfulness`, `context_precision`, and `answer_relevancy` use the same local vLLM (Gemma 4 26B AWQ) as both generator *and* judge — judge and generator share failure modes, so absolute scores are optimistic.
- `context_recall` requires a ground-truth reference; trap questions intentionally have no reference and report N/A for that metric.
- Latency is wall-clock from request to answer (excluding metric scoring), measured once per question, no warm-up.

