# graphrag — LLM Engine

Hybrid RAG (Vector + Graph) LLM Engine for DGX Spark.

Self-contained, runs entirely on the local box via `docker compose up`.
No external LLM API. Per-tenant API key. Streaming SSE answers.

## Stack
- **API**: FastAPI + SSE (token / citation / done / error events)
- **LLM**: Gemma 4 26B AWQ-4bit (vLLM, OpenAI-compatible)
- **OCR**: Chandra OCR 2 (vLLM, OpenAI-compatible chat-completions)
- **Embedding**: BGE-M3 (1024-d dense + sparse lexical weights)
- **Vector DB**: Qdrant (named vectors `dense` + `sparse`, RRF fusion)
- **Graph DB**: Neo4j (community-edition, GraphRAG entity/relation)
- **Metadata**: PostgreSQL 16 + SQLAlchemy 2.0 async + Alembic
- **Queue/Cache**: Redis + Celery
- **Rate limit**: slowapi (60 req/min/tenant, bucket = sha256(token)[:16])

See [`graphrag-PLAN.md`](../graphrag-PLAN.md) for the full architecture and decisions.

## Phase Status

| Phase | Status |
|---|---|
| 0. Scaffolding | DONE |
| 1. Infra containers | DONE |
| 2. DB schema + auth | DONE |
| 3. Ingestion pipeline (incl. 3i GraphRAG, 3j SSE) | DONE |
| 4. Retrieval + generation | DONE |
| 5. Document mgmt + ops | DONE |
| 6. Tests + 80% coverage | DONE — **95.63% / 172 unit tests** |
| 7. Optional (graph retriever, RAGAS, reranker, admin UI) | TODO |

## Quick Start (5 minutes — once models are cached)

```bash
# 0. One-time: HuggingFace token for gated/private models
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# 1. Configure
cd llm-engine
cp .env.example .env        # edit passwords/ports if you care

# 2. Pre-download models into a shared docker volume
./scripts/download_models.sh

# 3. Bring up the whole stack
make up
make ps                     # wait until app + workers + vLLM are healthy

# 4. Initialise DB schema (idempotent)
docker compose exec app python -m scripts.init_db

# 5. Sanity check
make health
# {"status":"ok","checks":{"postgres":"ok","qdrant":"ok","neo4j":"ok","redis":"ok",
#                          "vllm_gemma":"ok","vllm_chandra":"ok"}}
```

## End-to-End Demo

A full happy-path walk-through against a running stack. All commands use only
`curl` + `jq` + the `scripts/create_api_key.py` admin CLI.

### 1. Issue an API key

```bash
docker compose exec app python scripts/create_api_key.py tenants
# (creates default tenant on first run)

docker compose exec app python scripts/create_api_key.py issue \
  --tenant default --name demo
# ================================================================
# tenant   : default (2039714d-...-0b3a)
# key id   : 051edbed-...-de78
# key name : demo
# PLAIN KEY (shown ONCE — store it now):
# graphrag_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ================================================================

export SK=graphrag_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Upload a PDF

```bash
curl -s -X POST http://localhost:8000/v1/upload \
  -H "Authorization: Bearer $SK" \
  -F "files=@./tests/fixtures/sample_pdfs/korean_form.pdf" \
  -F "doc_type=policy" | jq
# {
#   "job_id": "3d5bcfcb-df51-4671-819c-2d140f37d6cb",
#   "accepted_files": [{"name":"korean_form.pdf","document_id":"2e881c1d-..."}],
#   "rejected_files": []
# }
```

Rejected reasons you may hit on real input:
- `unsupported extension .docx` — accept only `.pdf .png .jpg .jpeg .webp .bmp .tiff`
- `duplicate (same content already indexed)` — SHA-256 dedup per tenant
- `too many files` — `upload_max_files_per_request` (default 100)

### 3. Watch the job — two options

**A. Poll**
```bash
JOB=3d5bcfcb-df51-4671-819c-2d140f37d6cb
curl -s "http://localhost:8000/v1/jobs/$JOB" -H "Authorization: Bearer $SK" | jq
# {"id":"...","status":"completed","progress":1.0,"error":null,
#  "created_at":"2026-04-27T09:35:20Z","updated_at":"2026-04-27T09:35:37Z"}
```

**B. Stream (SSE, Phase 3j)**
```bash
curl -sN "http://localhost:8000/v1/jobs/$JOB/stream" -H "Authorization: Bearer $SK"
# event: progress
# data: {"id":"...","status":"pending","progress":0.0,"error":null}
#
# event: progress
# data: {"id":"...","status":"running","progress":0.0,"error":null}
#
# event: progress
# data: {"id":"...","status":"completed","progress":1.0,"error":null}
#
# event: done
# data: {"id":"...","status":"completed","progress":1.0,"error":null}
```

What happens under the hood during ingest:

1. **OCR**: PDF → per-page PNG → Chandra returns Markdown per page
2. **Chunk**: recursive splitter (`chunk_size=1000`, `overlap=200`)
3. **Embed**: BGE-M3 dense (1024-d) + sparse lexical weights per chunk
4. **Vector upsert**: Qdrant `chunks` collection, named vectors `dense` + `sparse`,
   payload carries `tenant_id`, `document_id`, `filename`, `page`, `doc_type`,
   `chunk_index`, `text`
5. **GraphRAG extract** (Phase 3i): Gemma per chunk →
   `{entities:[{name,type,description}], relations:[{source,target,kind,description}]}`,
   bounded-parallel via `asyncio.Semaphore(graphrag_extract_concurrency)`
6. **Graph upsert**: Neo4j — `(:Document)<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(:Entity)`
   and `(:Entity)-[:RELATES_TO {kind}]->(:Entity)`. Tenant-scoped entity dedup
   on `(tenant_id, name, type)`.

### 4. Query — JSON

```bash
curl -s -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d '{"question":"국내 출장 일일 식비 한도가 얼마야?","top_k":5,"stream":false}' | jq
# {
#   "answer": "국내 출장 일일 식비 한도는 50,000원입니다 [korean_form.pdf].",
#   "sources": [
#     {"filename":"korean_form.pdf","page":1,"chunk_index":0,"score":0.91}
#   ],
#   "mode_used": "fact",
#   "latency_ms": 842
# }
```

`stream` defaults to `true` (SSE). Set `stream:false` for a single JSON payload —
this is the path used by Swagger UI testing.

### 5. Query — streaming (SSE)

```bash
curl -sN -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question":"출장 신청 절차 알려줘","top_k":5,"stream":true}'
# event: citation
# data: [{"filename":"korean_form.pdf","page":1,"chunk_index":0,"score":0.91}]
#
# event: token
# data: 출장
# event: token
# data:  신청은
# ...
# event: done
# data: {}
```

Errors during retrieval/generation arrive as:
```
event: error
data: {"type":"RetrievalError","message":"qdrant timeout"}
```

### 6. Multi-turn session

Pass a stable `session_id` to thread context across requests. Last 10 turns kept
in Redis (`sess:{tenant}:{sid}:turns` LIST), sliding 30-minute TTL, older turns
collapsed into a Gemma-written summary.

```bash
SID=$(uuidgen)
curl -s -X POST http://localhost:8000/v1/query -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"내 이름은 박씨야\",\"session_id\":\"$SID\"}" | jq .answer
# "박씨님, 반갑습니다…"

curl -s -X POST http://localhost:8000/v1/query -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"내 이름이 뭐였지?\",\"session_id\":\"$SID\"}" | jq .answer
# "박씨라고 하셨습니다."
```

### 7. List / delete

```bash
curl -s "http://localhost:8000/v1/documents?doc_type=policy" \
  -H "Authorization: Bearer $SK" | jq

curl -s -X DELETE "http://localhost:8000/v1/documents/$DOC_ID" \
  -H "Authorization: Bearer $SK" -w "%{http_code}\n"
# 204
```

Delete cascades across **Postgres** (document row + jobs), **Qdrant** (all
chunks filtered by `tenant_id` + `document_id`), **Neo4j** (`:Chunk` +
`:Document` with `DETACH DELETE`; orphan `:Entity` cleanup is a separate
maintenance job), and the on-disk `/data/uploads/<doc_id>.<ext>` file.

### 8. Inspect the knowledge graph (optional)

```bash
docker compose exec neo4j cypher-shell -u neo4j -p change_me_neo4j \
  "MATCH (d:Document {id:'$DOC_ID'})<-[:PART_OF]-(c:Chunk)-[:MENTIONS]->(e:Entity)
   RETURN e.type, count(*) AS n ORDER BY n DESC;"
# +----------------+----+
# | e.type         | n  |
# +----------------+----+
# | ORG            | 7  |
# | PERSON         | 4  |
# | LOCATION       | 3  |
# | ...                 |
# +----------------+----+
```

### 9. Revoke the demo key

```bash
docker compose exec app python scripts/create_api_key.py revoke \
  --key-id 051edbed-330b-4181-b893-49c6875bde78
# revoked key 051edbed-...
```

## Layout

```
llm-engine/
├── app/
│   ├── api/              # health, upload, query, jobs (incl. SSE), documents, auth, me
│   ├── core/             # auth, limiter, exceptions
│   ├── ingest/           # ocr, chunker, embedder, vector_indexer,
│   │                     # graph_extract, graph_indexer, pipeline
│   ├── retrieve/         # router (mode classifier), vector (hybrid+RRF), orchestrator
│   ├── generate/         # llm (vLLM client), prompt, streamer, session
│   ├── workers/          # celery_app + tasks
│   ├── db/               # postgres, qdrant, neo4j, redis, vllm clients
│   ├── models/           # ORM + Pydantic schemas
│   ├── utils/            # logging, hashing
│   ├── config.py         # pydantic-settings
│   └── main.py           # FastAPI entrypoint
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── scripts/
│   ├── init_db.py            # idempotent schema bootstrap + initial key
│   ├── init_qdrant.py        # collection + named vectors
│   ├── init_neo4j.py         # constraints + indexes
│   ├── create_api_key.py     # tenants / issue / list / revoke
│   ├── build_fixture_pdf.py  # synthetic Korean travel policy PDF for E2E
│   └── download_models.sh
└── tests/
    ├── unit/                  # 172 cases, 95.63% coverage
    ├── integration/           # PDF upload → jobs poll → query round-trip
    ├── load/                  # locust profile, 40 users / 2 min baseline
    └── fixtures/sample_pdfs/  # korean_form.pdf
```

## Testing

```bash
# Inside the app container (so deps + Postgres mocks are wired up)
docker compose exec app pytest tests/unit                       # 172 unit tests
docker compose exec app pytest tests/unit --cov=app             # coverage report
docker compose exec app pytest tests/integration                # live PG + Qdrant + Neo4j
```

Locust load profile (4 weighted tasks: query_casual_json, query_factual_json,
list_documents, query_streaming):

```bash
docker compose exec app sh -c "cd /app && locust -f tests/load/locustfile.py \
  --headless -u 40 -r 5 -t 2m --host http://localhost:8000 \
  --csv /tmp/locust --html /tmp/locust.html"
```

Baseline result with one tenant against the default `rate_limit_per_minute=60`:
~2893 requests @ ~24 req/s, ~91% returning `429 Too Many Requests` with proper
`X-RateLimit-*` headers (expected: hitting the rate-limit ceiling).

## Environment

Set values in `.env`. See `.env.example` for the full set.

Locked decisions (PLAN v1.1):

| Setting | Value |
|---|---|
| vLLM image | `vllm/vllm-openai:gemma4-cu130` |
| `max_model_len` | 16384 |
| `max_num_seqs` | 8 |
| GPU mem util | Chandra 0.25, Gemma 0.55 |
| GraphRAG | per-chunk LLM extraction → Neo4j |
| Inputs (1차) | PDF, JPG, PNG (PPTX 2차) |
| Citation | filename only |
| API key delivery | stdout once + `.env` auto-injection |
| Rate limit | 60 req/min/tenant (slowapi + Redis) |
| Session | Redis, 10 turns, 30-min sliding TTL, Gemma summary on overflow |
| `graphrag_extract_concurrency` | 4 |

## License & external services

- Chandra OCR 2: **OpenRAIL-M** — commercial threshold check required before production deploy
- Gemma 4: Apache 2.0
- BGE-M3: MIT
- Qdrant / Neo4j Community / PostgreSQL / Redis: open-source
