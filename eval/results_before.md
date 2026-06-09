# GraphRAG vs Vector-only — synthetic evaluation

Synthetic Korean public-institute travel-policy corpus (4 PNG documents, 12 golden questions). Both modes use the same retrieval orchestrator and the same RAG prompt; the only difference is whether the Neo4j GraphRAG local-search retriever is consulted alongside Qdrant dense+sparse RRF.

## Aggregate metrics (mean across all questions)

| Mode | faithfulness | answer_relevancy | context_precision | context_recall | latency (s) |
|---|---|---|---|---|---|
| graph+vector | 0.881 | 0.865 | 0.350 | 0.778 | 13.18 |
| vector-only | 0.956 | 0.865 | 0.815 | 0.778 | 12.12 |

## Mean by question category

### single-fact

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 0.933 | 0.898 | 0.407 | 1.000 |
| vector-only | 0.933 | 0.890 | 0.983 | 1.000 |

### multi-hop

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 0.780 | 0.825 | 0.433 | 0.867 |
| vector-only | 0.960 | 0.835 | 0.973 | 0.867 |

### trap-no-evidence

| Mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graph+vector | 1.000 | 0.882 | 0.000 | 0.000 |
| vector-only | 1.000 | 0.876 | 0.000 | 0.000 |

## Per-question result

| id | category | mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|
| multihop-01 | multi-hop | graph+vector | 0.400 | 0.778 | 0.639 | 0.333 |
| multihop-01 | multi-hop | vector-only | 0.800 | 0.756 | 1.000 | 0.333 |
| multihop-02 | multi-hop | graph+vector | 1.000 | 0.891 | 0.333 | 1.000 |
| multihop-02 | multi-hop | vector-only | 1.000 | 0.852 | 1.000 | 1.000 |
| multihop-03 | multi-hop | graph+vector | 0.500 | 0.799 | 0.333 | 1.000 |
| multihop-03 | multi-hop | vector-only | 1.000 | 0.825 | 1.000 | 1.000 |
| multihop-04 | multi-hop | graph+vector | 1.000 | 0.900 | 0.444 | 1.000 |
| multihop-04 | multi-hop | vector-only | 1.000 | 0.900 | 0.867 | 1.000 |
| multihop-05 | multi-hop | graph+vector | 1.000 | 0.754 | 0.417 | 1.000 |
| multihop-05 | multi-hop | vector-only | 1.000 | 0.840 | 1.000 | 1.000 |
| single-01 | single-fact | graph+vector | 1.000 | 0.969 | 0.583 | 1.000 |
| single-01 | single-fact | vector-only | 1.000 | 0.969 | 1.000 | 1.000 |
| single-02 | single-fact | graph+vector | 0.667 | 0.874 | 0.333 | 1.000 |
| single-02 | single-fact | vector-only | 1.000 | 0.878 | 1.000 | 1.000 |
| single-03 | single-fact | graph+vector | 1.000 | 0.738 | 0.333 | 1.000 |
| single-03 | single-fact | vector-only | 1.000 | 0.693 | 1.000 | 1.000 |
| single-04 | single-fact | graph+vector | 1.000 | 0.925 | 0.367 | 1.000 |
| single-04 | single-fact | vector-only | 0.667 | 0.925 | 0.917 | 1.000 |
| single-05 | single-fact | graph+vector | 1.000 | 0.986 | 0.417 | 1.000 |
| single-05 | single-fact | vector-only | 1.000 | 0.986 | 1.000 | 1.000 |
| trap-01 | trap-no-evidence | graph+vector | 1.000 | 0.819 | 0.000 | 0.000 |
| trap-01 | trap-no-evidence | vector-only | 1.000 | 0.809 | 0.000 | 0.000 |
| trap-02 | trap-no-evidence | graph+vector | 1.000 | 0.944 | 0.000 | 0.000 |
| trap-02 | trap-no-evidence | vector-only | 1.000 | 0.944 | 0.000 | 0.000 |

## Honest limitations

- Single labeller (the author) wrote both the synthetic documents and the golden answers, so the ground truth is biased by what felt natural to ask. This is a *directional sanity check*, not a precision benchmark.
- N=12 questions is too small to make strong claims about absolute metric levels; only the relative deltas between modes are informative.
- `faithfulness`, `context_precision`, and `answer_relevancy` use the same local vLLM (Gemma 4 26B AWQ) as both generator *and* judge — judge and generator share failure modes, so absolute scores are optimistic.
- `context_recall` requires a ground-truth reference; trap questions intentionally have no reference and report N/A for that metric.
- Latency is wall-clock from request to answer (excluding metric scoring), measured once per question, no warm-up.

