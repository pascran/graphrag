# graphrag — LLM Engine

Hybrid RAG (Vector + Graph) LLM Engine for DGX Spark.

Self-contained, runs entirely on the local box via `docker compose up`.
No external LLM API. Per-tenant API key. Streaming SSE answers.

## Stack
- **API**: FastAPI + SSE
- **LLM**: Gemma 4 26B AWQ-4bit (vLLM)
- **OCR**: Chandra OCR 2 (vLLM)
- **Embedding**: BGE-M3 (dense + sparse)
- **Vector DB**: Qdrant (hybrid)
- **Graph DB**: Neo4j (community-edition, populated from Microsoft GraphRAG)
- **Metadata**: PostgreSQL 16
- **Queue/Cache**: Redis + Celery

See [`graphrag-PLAN.md`](../graphrag-PLAN.md) for the full architecture and decisions.

## Quick Start (5 minutes — once models are cached)

```bash
# 0. One-time: HuggingFace token for gated/private models (export, don't commit)
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# 1. Clone, then enter
cd llm-engine

# 2. Configure
cp .env.example .env
# (optional) edit .env: change passwords, choose ports, etc.

# 3. Pre-download models into a shared docker volume (saves vLLM cold start)
./scripts/download_models.sh

# 4. Bring up the whole stack
make up
make ps        # all services should reach `healthy` within ~5 min

# 5. Initialise DB schema + issue first API key (Phase 2 — coming next)
docker compose exec app python -m scripts.init_db

# 6. Sanity check
make health
```

## Phase Status (per PLAN v1.1)

| Phase | Status |
|---|---|
| 0. Scaffolding | DONE |
| 1. Infra containers | DONE — runtime verification pending first `docker compose up` |
| 2. DB schema + auth | TODO |
| 3. Ingestion pipeline | TODO |
| 4. Retrieval + generation | TODO |
| 5. Document mgmt + ops | TODO |
| 6. Tests + 80% coverage | TODO |

## Layout

```
llm-engine/
├── app/
│   ├── api/              # routers (health, upload, query, jobs, documents)
│   ├── core/             # auth, session, exceptions
│   ├── ingest/           # OCR, chunker, embedder, vector/graph indexer, pipeline
│   ├── retrieve/         # router, vector, graph, reranker, orchestrator
│   ├── generate/         # llm client, prompt, sse streamer
│   ├── workers/          # celery tasks
│   ├── db/               # postgres, qdrant, neo4j, redis, vllm clients
│   ├── models/           # ORM + Pydantic schemas
│   ├── utils/            # logging, hashing
│   ├── config.py         # pydantic-settings
│   └── main.py           # FastAPI entrypoint
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── scripts/
│   ├── init_db.py        # Phase 2
│   ├── create_api_key.py # Phase 5
│   └── download_models.sh
└── tests/
```

## Environment

Set values in `.env`. See `.env.example` for the full set.

Locked decisions (PLAN v1.1):

| Setting | Value |
|---|---|
| vLLM image | `vllm/vllm-openai:gemma4-cu130` |
| `max_model_len` | 16384 |
| `max_num_seqs` | 8 |
| GPU mem util | Chandra 0.25, Gemma 0.55 |
| GraphRAG | parquet → Neo4j import → Cypher search |
| Inputs (1차) | PDF, JPG, PNG (PPTX 2차) |
| Citation | filename only |
| API key delivery | stdout once + `.env` auto-injection |

## License & external services

- Chandra OCR 2: **OpenRAIL-M** — commercial threshold check required before production deploy
- Gemma 4: Apache 2.0
- BGE-M3: MIT
- Qdrant / Neo4j Community / PostgreSQL / Redis: open-source
