# llm-engine 운영 매뉴얼

> 본 문서는 graphrag/llm-engine 의 일상 운영(데이/온콜)용 런북이다.
> 모든 명령은 `docker compose` 가 동작하는 호스트(보통 DGX Spark) 에서 `/home/graphrag/llm-engine` 작업 디렉터리 기준으로 실행한다.
> 단일 진실 소스는 **이 파일과 docker-compose.yml, alembic migrations, scripts/**\* 이며, README 의 Phase Status 표는 마케팅용 요약일 뿐 운영 기준이 아니다.

---

## 빠른 헬스체크 (각 서비스 ping 명령)

`/v1/health` 가 통합 답을 주지만, 장애 시점에는 서비스별 ping 으로 누가 누락됐는지 잘라봐야 한다.

| 서비스 | 1차 체크 (호스트에서) | 2차 체크 (컨테이너 내부 / 인증 필요) |
|---|---|---|
| app (FastAPI) | `curl -fsS http://localhost:8000/v1/health` | `curl -fsS -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/health/deep` |
| postgres | `docker compose exec postgres pg_isready -U llm -d llm_engine` | `docker compose exec postgres psql -U llm -d llm_engine -c 'select 1;'` |
| qdrant | `curl -fsS http://localhost:6333/readyz` | `curl -fsS http://localhost:6333/collections/chunks` |
| neo4j | `docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" 'RETURN 1;'` | `docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" 'CALL dbms.components();'` |
| redis | `docker compose exec redis redis-cli ping` | `docker compose exec redis redis-cli -n 1 LLEN celery` |
| vllm-gemma | `curl -fsS http://localhost:8001/v1/models` | `curl -fsS http://localhost:8001/health` |
| vllm-chandra | `curl -fsS http://localhost:8002/v1/models` | `curl -fsS http://localhost:8002/health` |
| celery-worker | `docker compose exec celery-worker celery -A app.workers.celery_app inspect ping` | `docker compose exec celery-worker celery -A app.workers.celery_app inspect active` |
| celery-beat (옵션) | `docker compose ps celery-beat` | `docker compose logs --tail=20 celery-beat` |

**원라이너 — 전체 일괄 확인:**

```bash
docker compose ps --format 'table {{.Service}}\t{{.Status}}\t{{.Health}}'
```

`Status` 가 `Up (healthy)` 이 아닌 서비스부터 위 표의 1차 체크로 좁힌다.

---

## 시작 / 정지 / 재시작 (docker compose)

```bash
# 최초 부팅 (모델 캐시가 있는 상태에서 ~3~5분 소요, vLLM 워밍업 별도)
docker compose up -d

# 정상 정지 (볼륨 유지)
docker compose stop

# 정지 + 컨테이너 제거 (볼륨은 유지, 다음 up 으로 동일 상태 복귀)
docker compose down

# 특정 서비스만 재시작 (앱 코드 hot reload 안 됨, 이미지 재기동)
docker compose restart app celery-worker

# 코드 변경 후 앱만 다시 빌드해서 띄움
docker compose up -d --build app celery-worker

# 비상 — 모든 데이터 삭제 (POSTGRES/QDRANT/NEO4J/REDIS 볼륨 전부 날아감)
docker compose down -v   # 위험. 백업 확인 후에만.

# 로그 팔로우
docker compose logs -f app
docker compose logs -f celery-worker
docker compose logs -f vllm-gemma   # 모델 로드는 stdout 으로 진행 표시

# 단일 명령 실행
docker compose exec app python -m scripts.create_api_key tenants
docker compose exec app alembic current
```

**기동 순서 의존성**: `app` 은 postgres / qdrant / neo4j / redis 의 `service_healthy` 와 vLLM 두 개의 `service_started` 를 기다린다. vLLM 은 모델 로드 종료 전에 healthcheck 통과를 안 하므로 `app` 컨테이너가 먼저 떠도 첫 요청에서 5xx 가 날 수 있다. `curl localhost:8001/v1/models` 가 200 을 줘야 라우터 LLM 호출이 가능하다.

---

## 환경 변수 (.env 키 + 의미 + 기본값 + 안전성 노트)

전부 `app/config.py:Settings` 에서 검증되며, `.env` → 환경변수 → 기본값 순으로 우선한다. 굵게 표시된 항목은 운영 배포 전 반드시 교체해야 한다.

| 키 | 기본값 | 의미 / 안전성 노트 |
|---|---|---|
| APP_ENV | `development` | `production` 으로 바꾸면 디버그 응답이 닫힌다. 라우트별 limiter 강제는 별건 (보안 체크리스트 참조). |
| APP_HOST | `0.0.0.0` | 내부망 한정이면 `127.0.0.1` 권장. |
| APP_PORT | `8000` | reverse proxy 뒤에 두는 게 정석. |
| APP_LOG_LEVEL | `INFO` | structlog 레벨. 디버그 시 `DEBUG`, 운영은 `INFO`. |
| INITIAL_API_KEY | `""` | `scripts/init_db.py` 가 처음 부팅 시 발급한 키를 여기에 주입. 운영에선 비워두고 admin CLI 로 발급한다. |
| API_KEY_HEADER | `Authorization` | Bearer 스키마 기준. 변경 시 클라이언트 전부 변경. |
| **POSTGRES_PASSWORD** | `change_me_postgres` | **프로덕션 차단 사유.** 32+ 문자 랜덤. alembic.ini 의 plaintext DSN 도 같이 봐야 함 — 추후 작업. |
| POSTGRES_HOST/PORT/DB/USER | `postgres / 5432 / llm_engine / llm` | 외부 RDS 로 빼면 DSN 만 갈아끼우면 됨. |
| QDRANT_URL | `http://qdrant:6333` | 내부 도커 네트워크. 외부 노출 금지 (API key 없음). |
| QDRANT_COLLECTION | `chunks` | 한 클러스터에 환경 분리 시 `chunks_prod` 식으로 prefix. |
| **NEO4J_PASSWORD** | `change_me_neo4j` | **프로덕션 차단 사유.** community 2026.04, 기본 계정은 `neo4j`. |
| NEO4J_URL/USER | `bolt://neo4j:7687 / neo4j` | bolt 프로토콜. 외부 노출 시 TLS 필수. |
| REDIS_URL | `redis://redis:6379/0` | DB0 = 캐시 + rate limit + 세션 메모리. |
| CELERY_BROKER_URL | `redis://redis:6379/1` | DB1 = Celery 큐. |
| CELERY_RESULT_BACKEND | `redis://redis:6379/2` | DB2 = 작업 결과. |
| VLLM_LLM_URL | `http://vllm-gemma:8000/v1` | 사설 vLLM. OpenAI 호환. |
| VLLM_LLM_MODEL | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | AWQ 4bit, 16k 컨텍스트. |
| VLLM_LLM_MAX_TOKENS | `2048` | 응답 상한. 길어지면 SSE 타임아웃 위험. |
| VLLM_LLM_TEMPERATURE | `0.3` | RAG 답변은 0.2~0.4 권장. |
| VLLM_LLM_MAX_MODEL_LEN | `16384` | 컨텍스트 + 응답. 초과 시 vLLM 이 잘라냄. |
| VLLM_LLM_MAX_NUM_SEQS | `8` | 동시 디코딩. GPU mem 과 trade-off. |
| VLLM_LLM_GPU_MEM_UTIL | `0.55` | Chandra(0.25)와 합쳐 0.8 미만 유지. |
| VLLM_OCR_URL | `http://vllm-chandra:8000/v1` | Chandra OCR 2. data URI 로 이미지 전송. |
| VLLM_OCR_GPU_MEM_UTIL | `0.25` | 합산 GPU 점유율 관리 포인트. |
| EMBEDDING_MODEL | `BAAI/bge-m3` | dense 1024d + sparse. |
| EMBEDDING_DEVICE | `cuda` | CPU 폴백 시 `cpu` (성능 급락). |
| EMBEDDING_BATCH_SIZE | `32` | OOM 시 16 로. |
| SESSION_TTL_SECONDS | `1800` | Redis 세션 슬라이딩 TTL. |
| SESSION_MAX_TURNS | `10` | 초과 시 요약 fold. |
| SESSION_MAX_TOKENS | `4000` | 동상. |
| CHUNK_SIZE / CHUNK_OVERLAP | `1000 / 200` | recursive splitter. 변경 시 기존 인덱스 재구축 권장. |
| GRAPHRAG_ENABLED | `true` | false 면 :Entity 추출/그래프 retriever 양쪽 우회. |
| GRAPHRAG_EXTRACT_CONCURRENCY | `4` | LLM 추출 동시성. vLLM 큐와 맞춰서. |
| GRAPH_RETRIEVAL_ENABLED | `true` | false 면 벡터 only. |
| RERANKER_ENABLED | `true` | false 면 RRF 순서 그대로. 콜드스타트 회피 시 일시 off. |
| RERANKER_MODEL | `BAAI/bge-reranker-v2-m3` | CrossEncoder, fp16. |
| RERANKER_OVERSAMPLE | `4` | top_k * 4 만큼 가져와 재랭킹. |
| UPLOAD_MAX_FILE_SIZE_MB | `50` | 단일 파일. |
| UPLOAD_MAX_FILES_PER_REQUEST | `100` | 멀티 업로드. |
| RATE_LIMIT_PER_MINUTE | `60` | slowapi 기본. 라우트별 강제는 미적용 — 보안 체크리스트 참조. |
| CLEANUP_ORPHANS_ENABLED | `false` | **신규.** Celery Beat 의 orphan :Entity 야간 청소 작업 활성 게이트. 운영 도입 1주는 false 로 두고 수동 dry-run 으로 충분히 검증한 뒤 true 로 전환한다. |

---

## API 키 운영 (admin CLI: 발급, 목록, 철회 — 정확한 명령)

`scripts/create_api_key.py` 가 단일 진입점이며, **`app` 컨테이너 안에서** 실행해야 `postgres_dsn` 이 해석된다. 평문 키는 발급 시점 **한 번만** 출력되고 저장되지 않는다(sha256 hash 만 DB 에 남음). 잃어버리면 새로 발급한다.

```bash
# 1. 테넌트 목록
docker compose exec app python -m scripts.create_api_key tenants

# 2. 신규 테넌트 + 첫 키 동시 발급
docker compose exec app python -m scripts.create_api_key issue \
    --new-tenant "acme-corp" --name "first-key"

# 3. 기존 테넌트에 키 추가 (이름으로 지정)
docker compose exec app python -m scripts.create_api_key issue \
    --tenant "acme-corp" --name "ops-laptop"

# 4. 기존 테넌트에 키 추가 (UUID 로 지정)
docker compose exec app python -m scripts.create_api_key issue \
    --tenant-id 8cf7XXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX --name "ci-runner"

# 5. 특정 테넌트의 키 목록 (hash 앞 12자, 활성 여부, 생성시각)
docker compose exec app python -m scripts.create_api_key list --tenant "acme-corp"

# 6. 키 철회 (soft delete — is_active=false, 감사 추적 유지)
docker compose exec app python -m scripts.create_api_key revoke --key-id 4d2eXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
```

**테스트 호출**:

```bash
curl -fsS -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/health/deep
```

**키 노출 사고 대응**: `revoke` → 새 키 발급 → 클라이언트 교체 → access log 에서 해당 hash prefix 로 조회한 IP/UA 확인 → 침해 범위 평가.

---

## 백업

전제: 운영 호스트의 `/var/backups/llm-engine/` 같은 별도 디렉터리에 저장하고, 일정 주기로 외부(S3/오브젝트 스토리지)로 푸시한다. 본 절은 호스트 로컬 백업 명령만 제공한다.

### Postgres 백업 / 복원

```bash
# 백업 (논리 덤프, custom format — alembic/스키마 같이 잡힘)
docker compose exec -T postgres pg_dump -U llm -d llm_engine -Fc \
    > /var/backups/llm-engine/postgres-$(date +%Y%m%d-%H%M).dump

# 복원 (대상 DB 비어 있어야 함)
cat /var/backups/llm-engine/postgres-YYYYMMDD-HHMM.dump | \
    docker compose exec -T postgres pg_restore -U llm -d llm_engine --clean --if-exists

# 스키마만 체크
docker compose exec -T postgres pg_dump -U llm -d llm_engine --schema-only | head -50
```

### Qdrant 스냅샷

Qdrant 는 컬렉션 단위 snapshot API 를 제공한다. 볼륨 자체를 tar 로 떠도 되지만 hot-copy 는 권장하지 않는다.

```bash
# 스냅샷 생성 → JSON 응답에 snapshot 이름이 나옴
curl -fsS -X POST http://localhost:6333/collections/chunks/snapshots

# 스냅샷 목록
curl -fsS http://localhost:6333/collections/chunks/snapshots

# 다운로드 (NAME 은 위에서 받은 파일명)
curl -fsS http://localhost:6333/collections/chunks/snapshots/NAME \
    -o /var/backups/llm-engine/qdrant-chunks-$(date +%Y%m%d-%H%M).snapshot

# 복원 — 컬렉션을 한 번 비우고 (또는 새 이름으로) 업로드
curl -fsS -X PUT \
    "http://localhost:6333/collections/chunks/snapshots/upload?priority=snapshot" \
    -F snapshot=@/var/backups/llm-engine/qdrant-chunks-YYYYMMDD-HHMM.snapshot
```

### Neo4j 백업 (community 제약 명시)

**중요 제약**: `neo4j-admin database backup` (온라인 백업) 은 **Enterprise 전용**이다. 본 환경은 `neo4j:2026.04.0-community` 이므로 **오프라인 dump 만 가능**하다. 즉 백업 동안에는 컨테이너를 멈춰야 한다.

```bash
# 1. Neo4j 컨테이너 정지 (앱은 잠시 그래프 unavailable 상태가 됨)
docker compose stop neo4j

# 2. 같은 볼륨을 일회용 컨테이너로 마운트해 dump 생성
docker run --rm \
    -v llm-engine_neo4j_data:/data \
    -v /var/backups/llm-engine:/backups \
    neo4j:2026.04.0-community \
    neo4j-admin database dump neo4j --to-path=/backups

# 3. 다시 기동
docker compose start neo4j

# 복원 (대상 DB 가 stopped 여야 함)
docker compose stop neo4j
docker run --rm \
    -v llm-engine_neo4j_data:/data \
    -v /var/backups/llm-engine:/backups \
    neo4j:2026.04.0-community \
    neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true
docker compose start neo4j
```

대안: `cypher-shell` 로 `MATCH (n) RETURN n` 을 CSV 로 빼는 논리 백업은 인덱스/제약/APOC 설정을 따로 들고 다녀야 해서 권장하지 않는다.

### Redis 백업 (RDB/AOF)

Redis 는 캐시 + 세션 + Celery 큐 겸용이라 손실 허용도가 데이터 종류마다 다르다. DB0(세션) 손실은 사용자 재로그인 수준, DB1(Celery 큐) 손실은 진행 중 작업 유실 가능, DB2(결과) 손실은 작업 결과 조회 실패.

```bash
# 1. BGSAVE 트리거 → background 로 RDB 덤프 생성
docker compose exec redis redis-cli BGSAVE

# 2. 마지막 저장 시각 확인 (UNIX timestamp)
docker compose exec redis redis-cli LASTSAVE

# 3. dump.rdb 를 호스트로 복사
docker cp $(docker compose ps -q redis):/data/dump.rdb \
    /var/backups/llm-engine/redis-$(date +%Y%m%d-%H%M).rdb

# AOF 활성화하려면 redis.conf 에 appendonly yes (현재 기본 미사용).
# 복원: 컨테이너 정지 → 볼륨에 dump.rdb 덮어쓰기 → 기동
```

---

## 데이터 정합성 유지

### Documents 캐스케이드 삭제 동작 (Postgres + Qdrant + Neo4j 흐름, 부분 실패 시 잔여 노드 가능성)

`DELETE /v1/documents/{document_id}` 의 실제 동작은 다음 3단계 cascade 다.

1. **Postgres**: `documents` 행 삭제. FK 로 묶인 `jobs` 도 같이 정리.
2. **Qdrant**: 해당 `document_id` 페이로드를 가진 모든 포인트 삭제 (`filter: payload.document_id == <id>`).
3. **Neo4j**: `MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id}) OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d) DETACH DELETE c, d`
   - **`:Document` + 그 아래 `:Chunk` 만** 제거된다.
   - `:MENTIONS` (Chunk→Entity) 와 `:PART_OF` (Chunk→Document) 엣지는 DETACH 로 자동 삭제.
   - **`:Entity` 노드와 `:RELATES_TO` 엣지는 보존**된다. 다른 테넌트의 다른 청크가 같은 entity 를 가리킬 수 있기 때문 (entity dedup key 는 `(tenant_id, name, type)`).

**부분 실패 시나리오**:

| 실패 지점 | 잔여물 | 복구 |
|---|---|---|
| Postgres 성공, Qdrant 실패 | Qdrant 에 고아 포인트 다수 | `document_id` 로 직접 `POST /collections/chunks/points/delete` |
| Postgres + Qdrant 성공, Neo4j 실패 | Neo4j 에 `:Document` + `:Chunk` 잔존 | 위 cypher 수동 실행 (`app/ingest/graph_indexer.py:217`) |
| 어디든 부분 성공 | 어떤 layer 에 무엇이 남았는지 직접 조회 필요 | 아래 진단 cypher / SQL 참고 |

진단 쿼리:

```sql
-- Postgres: 특정 doc 잔존 확인
SELECT id, filename, tenant_id, created_at FROM documents WHERE id = '<doc_id>';
```

```cypher
// Neo4j: 특정 doc 잔존 확인
MATCH (d:Document {id: $doc_id}) OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
RETURN d, count(c) AS chunks;
```

```bash
# Qdrant: 특정 doc 의 잔존 포인트 수
curl -fsS -X POST http://localhost:6333/collections/chunks/points/count \
    -H 'content-type: application/json' \
    -d '{"filter":{"must":[{"key":"document_id","match":{"value":"<doc_id>"}}]},"exact":true}'
```

### orphan :Entity 정리 작업

**무엇이 orphan 인가**

`delete_document_graph` 가 `:Entity` 를 보존하기 때문에, 어떤 `:Chunk` 도 더 이상 가리키지 않는 `:Entity` 가 그래프에 부유한다. 운영적으로 *orphan* 의 정의는:

> 임의의 `:Chunk` 로부터 `:MENTIONS` 엣지가 들어오지 않는 `:Entity` 노드. `:RELATES_TO` 엣지만으로 다른 entity 와 연결된 island component 도 동일하게 orphan 으로 본다 (현재 추출기는 chunk 단위로만 relation 을 만들기 때문에 MENTIONS 가 없는 entity 가 RELATES_TO 만으로 의미 있게 살아 있을 가능성은 사실상 없다).

**수동 실행** (`scripts/cleanup_orphans.py`)

```bash
# Dry-run (기본) — 삭제 후보만 스캔, 샘플 20개 출력
docker compose exec app python scripts/cleanup_orphans.py

# 특정 테넌트로 범위 좁히기
docker compose exec app python scripts/cleanup_orphans.py --tenant <TENANT_UUID>

# 배치 상한 변경 (기본 1000)
docker compose exec app python scripts/cleanup_orphans.py --limit 500

# 실제 삭제 — DETACH DELETE 수행
docker compose exec app python scripts/cleanup_orphans.py --apply
docker compose exec app python scripts/cleanup_orphans.py --apply --tenant <TENANT_UUID> --limit 500
```

출력에는 `scanned`, `would_delete` 또는 `deleted`, 첫 20개 샘플(`name`, `tenant_id`, `elementId`)이 포함된다. 모든 실행은 structlog 에 `action="orphan_cleanup"` 으로 남는다.

**자동 실행 (Celery Beat)**

`CLEANUP_ORPHANS_ENABLED=true` 로 켜면 Celery Beat 가 매일 **04:15 KST** 에 `cleanup.orphan_entities` 작업을 **dry-run** 모드로 실행한다. 결과는 worker 로그에 동일한 JSON 구조로 남는다. 실제 삭제를 자동화하려면 별도 결정과 PR 이 필요하다 — 현재는 dry-run 이 기본이다.

`docker-compose.yml` 에 `celery-beat` 서비스가 같이 떠 있어야 schedule 이 동작한다. Beat 스케줄은 `app/workers/celery_app.py` 의 `beat_schedule` 항목에서 조정한다.

**검증 가이드라인**

첫 운영 도입 시에는 다음 절차를 권장한다.

1. **Week 0–1**: `CLEANUP_ORPHANS_ENABLED=false` 유지. 수동 `--dry-run` 만 매일 1회.
2. **Week 1**: 매일 dry-run 로그에서 `would_delete` 추이 관찰. 폭증 구간이 있으면 entity 추출 프롬프트 / 문서 삭제 패턴을 의심.
3. **Week 2**: `CLEANUP_ORPHANS_ENABLED=true` 로 전환. 여전히 야간 Beat 는 dry-run 모드.
4. **Week 3+**: 안정화 확인 후, 별도 운영 결정으로 야간 작업을 `--apply` 로 옮길지 결정.

**복구 시나리오**: 잘못 삭제된 :Entity 는 원본 청크가 그대로 있다면 재추출 가능하다. 해당 문서를 다시 ingest(또는 강제 reindex) 하면 동일 dedup key 로 :Entity 가 재생성된다.

---

## 부하 / 용량 계획 (locust 실행법, 알려진 한계)

```bash
# tests/load/locustfile.py 가 있다고 가정. 없다면 작성 필요.
docker compose exec app locust -f tests/load/locustfile.py \
    --host http://localhost:8000 --headless -u 50 -r 5 -t 5m \
    --csv=/tmp/locust-$(date +%Y%m%d-%H%M)
```

**알려진 한계 (병목 후보)**

| 영역 | 한계 | 대응 |
|---|---|---|
| Reranker 콜드스타트 | BGE-Reranker v2-m3 lazy singleton, 첫 호출에 모델 로드. 멀티 워커일수록 워커별 중복 로드. | 워밍업 요청을 헬스체크에 포함하거나, 단일 worker + concurrency 모델 사용. |
| OCR 페이지 직렬 | `app/ingest/pipeline.py` 가 페이지를 순차로 vLLM-Chandra 에 보냄. | 대용량 PDF 는 분할 업로드. concurrency 도입은 별도 PR. |
| BGE-M3 embed | asyncio 안에서 동기 호출을 `run_in_executor` 없이 수행. 큰 배치는 이벤트 루프 블로킹. | EMBEDDING_BATCH_SIZE 줄이거나 별도 thread executor 도입. |
| Job SSE | `app/api/jobs.py` 가 Postgres 1초 폴링, 최대 600s. | Celery PubSub 기반으로 교체는 추후 작업. |
| Celery max_retries=0 | `ingest_document_task` 는 실패 시 즉시 죽음. 부분 실패 ingest 가 그대로 남음. | Job 상태가 `failed` 가 된 문서는 수동으로 삭제 후 재업로드. |
| Rate limit | slowapi `default_limits` 만 적용, 라우트별 데코레이터 미적용. | 보안 체크리스트 참조. |
| GPU 메모리 | Gemma(0.55) + Chandra(0.25) = 0.80. 다른 모델 추가 시 합산 > 0.9 금지. | `nvidia-smi` 로 실측. |

Baseline 참고치는 환경마다 다르므로 본 문서에서는 명시하지 않는다. 측정 시 `RATE_LIMIT_PER_MINUTE` 값과 429 응답 비율을 같이 기록한다.

---

## 모니터링 & 로그 (structlog JSON, 어디서 보나)

- **app / celery-worker / celery-beat** 모두 stdout 으로 structlog JSON 라인을 뿌린다. `docker compose logs` 가 1차 진입점.
- 주요 event key:
  - `event="request_complete"` — FastAPI 요청 (method, path, status, latency_ms, tenant_id, key_hash_prefix).
  - `event="ingest_pipeline"` — phase, document_id, page_count, chunk_count, duration_ms.
  - `event="qdrant_upsert"`, `event="neo4j_graph_upsert"` — 인덱싱 결과.
  - `event="rag_retrieve"` — router_label, vector_hits, graph_hits, final_passages, rerank_used.
  - `event="orphan_cleanup"` — dry_run, scanned, would_delete/deleted, tenant_id, duration_ms.
- 권장: 호스트에서 `docker compose logs --tail=0 -f app celery-worker | jq -c` 로 실시간 JSON 파싱.
- 영속 수집: 운영 배포 시 Loki / Promtail 또는 Fluent Bit 로 stdout 을 흡수. 현재는 도커 기본 로깅 드라이버에 의존 (logrotate 설정은 호스트 docker 데몬 설정 따라감).

```bash
# 최근 1000줄에서 에러만
docker compose logs --tail=1000 app | grep -E '"level":"(error|critical)"'

# 특정 tenant 의 요청만
docker compose logs --tail=5000 app | jq -c 'select(.tenant_id == "<TENANT_UUID>")'

# orphan cleanup 결과만 추적
docker compose logs celery-worker | jq -c 'select(.event == "orphan_cleanup")'
```

---

## 흔한 장애 시나리오 & 해결

### Job stuck in running

증상: `GET /v1/jobs/{id}` 가 `status=running, progress=0.6` 에서 멈춤. SSE 는 600s 후 끊김.

원인 후보:
- Celery worker 가 죽었다 (`max_retries=0` → 재시도 없음).
- vLLM-Chandra 가 OOM / 행 상태.
- Neo4j 가 데드락.

대응:
```bash
docker compose exec celery-worker celery -A app.workers.celery_app inspect active
docker compose exec celery-worker celery -A app.workers.celery_app inspect reserved
docker compose logs --tail=200 celery-worker | jq -c 'select(.level=="error")'
```

해결: `app/api/jobs.py` 에서 Postgres 상태 확인 후 `jobs.status = 'failed'` 로 수동 업데이트, `documents` 도 같이 정리 후 재업로드.

### SSE 600s 타임아웃

증상: 긴 ingest 가 client 측에서 `EventSource` 끊김.

원인: `app/api/jobs.py` 의 폴링 루프 상한이 600초.

대응: 클라이언트가 끊긴 뒤 같은 endpoint 를 다시 호출하면 마지막 상태부터 이어 받는다. 백엔드 작업 자체는 SSE 와 독립적으로 진행 중이므로 재접속하면 됨.

### vLLM OOM

증상: `vllm-gemma` 또는 `vllm-chandra` 가 healthcheck 실패, 로그에 `CUDA out of memory`.

원인: 두 모델의 `gpu_mem_util` 합산이 GPU 메모리 한계 초과. 또는 다른 프로세스가 GPU 점유.

대응:
```bash
nvidia-smi
docker compose stop vllm-chandra   # 한 쪽만 살려 우선순위 회복
# .env 의 VLLM_*_GPU_MEM_UTIL 을 조정 후 재기동
docker compose up -d vllm-gemma vllm-chandra
```

`VLLM_LLM_MAX_NUM_SEQS` 를 줄이면 concurrency 가 떨어지는 대신 더 적은 KV cache 로 운영 가능.

### Neo4j APOC 로드 실패

증상: Neo4j 기동 후 `cypher-shell` 에서 `CALL apoc.help('apoc')` 가 `Unknown procedure` 로 실패.

원인: `docker-compose.yml` 에서 `NEO4J_PLUGINS=["apoc"]` (또는 동등 환경 변수) 가 빠졌거나, 플러그인 디렉터리 마운트 누락.

대응:
```bash
docker compose exec neo4j ls /var/lib/neo4j/plugins
docker compose logs neo4j | grep -i apoc
# 누락 시 docker-compose.yml 의 neo4j 서비스 env 에:
#   NEO4J_PLUGINS: '["apoc"]'
#   NEO4J_dbms_security_procedures_unrestricted: 'apoc.*'
# 추가 후 docker compose up -d neo4j
```

### Celery worker 죽음 (max_retries=0 영향)

증상: ingest 가 절반쯤에서 실패. 같은 문서 재업로드도 SHA256 dedupe 로 거부.

원인: `ingest_document_task` 가 `max_retries=0` 이라 예외 한 번에 종결. 이미 Postgres 에 `Document` 행이 생성되어 있어 SHA256 충돌.

대응:
```bash
# 실패한 document 식별
docker compose exec postgres psql -U llm -d llm_engine -c \
  "select d.id, d.filename, j.status, j.error from documents d join jobs j on j.document_id = d.id where j.status='failed' order by d.created_at desc limit 20;"

# 해당 document_id 로 DELETE /v1/documents/{id} 호출 (cascade)
curl -X DELETE -H "Authorization: Bearer $API_KEY" \
    http://localhost:8000/v1/documents/<doc_id>

# 원본 파일 재업로드
```

근본 해결은 `max_retries=3` + idempotent 청크 upsert 로의 전환 — 추후 작업.

---

## 보안 체크리스트

운영 배포 전 점검. 굵게 처리된 항목은 현재 **갭(미해결)** 으로 식별된 것이다.

- [ ] **`POSTGRES_PASSWORD` 가 `change_me_postgres` 가 아닌 강력한 비밀번호** — 추후 작업: `alembic.ini` 의 sqlalchemy.url 도 env 로 옮겨야 plaintext 가 사라짐.
- [ ] **`NEO4J_PASSWORD` 가 `change_me_neo4j` 가 아닌 강력한 비밀번호.**
- [ ] `INITIAL_API_KEY` 가 비어 있고, admin CLI 로 발급한 키만 사용.
- [ ] **prod CORS 화이트리스트가 채워져 있음** — 현재 `APP_ENV=production` 일 때 화이트리스트가 빈 상태로 출발 → 추후 작업: `app/main.py` 에서 `CORS_ALLOWED_ORIGINS` 환경 변수로 강제.
- [ ] **라우트별 rate limiter 적용** — 현재 slowapi `default_limits` 만 활성. `/v1/upload`, `/v1/query`, `/v1/evaluate` 처럼 비용 차이 큰 라우트는 개별 `@limiter.limit(...)` 데코레이터 필요 → 추후 작업.
- [ ] API 키는 sha256 hash 만 저장 (현재 OK).
- [ ] Qdrant / Neo4j / Redis 의 외부 포트가 호스트에서 닫혀 있음 (내부 도커 네트워크 only).
- [ ] vLLM endpoint 가 reverse proxy 없이 외부에 노출되지 않음 (인증 없음).
- [ ] `.env` 가 git ignore 에 있고, 호스트 권한 600.
- [ ] `docker compose logs` 에 평문 API key / secret 이 찍히지 않음 (현재 access log 는 hash prefix 만 남김 — OK).
- [ ] 업로드 파일에 대한 매직바이트 / MIME 검증이 활성 (현재 SHA256 + size 만 검증, 컨텐츠 sniff 는 OCR 단에서 간접). 악성 PDF 대비는 별도 sandbox 필요 → 추후 작업.
- [ ] **`CLEANUP_ORPHANS_ENABLED` 가 운영 검증 절차를 거친 뒤 켜짐** — 갓 도입 시 false 유지.

---

## 마이그레이션 (alembic upgrade/downgrade, 새 revision 생성 절차)

```bash
# 현재 head 확인
docker compose exec app alembic current

# 모든 마이그레이션 적용
docker compose exec app alembic upgrade head

# 한 단계만
docker compose exec app alembic upgrade +1

# 롤백
docker compose exec app alembic downgrade -1
docker compose exec app alembic downgrade base   # 위험. DB 초기화에 가까움

# 신규 revision 생성 (autogenerate)
docker compose exec app alembic revision --autogenerate -m "add foo column"

# autogenerate 없이 빈 템플릿
docker compose exec app alembic revision -m "manual data backfill"
```

**주의**:
- `migrations/versions/0001_initial.py` 만 존재. 새 revision 추가 시 dependency chain (`down_revision`) 정합성 확인 필수.
- `alembic.ini` 의 `sqlalchemy.url` 이 plaintext 로 `change_me_postgres` 를 포함 → env 기반으로 옮기는 작업 PR 필요.
- 운영 DB 에 마이그레이션 적용 전 Postgres 백업을 반드시 먼저 받는다.

---

## 부록: 비상 연락 / 로그 위치 / 메트릭 대시보드 자리

| 항목 | 위치 |
|---|---|
| 1차 온콜 | TODO — 운영 팀 채널 / 페이저듀티 등 정보 입력 |
| 백업 보관 | `/var/backups/llm-engine/` (호스트), 외부 오프사이트 TODO |
| 컨테이너 로그 | `docker compose logs <service>` — 도커 데몬 기본 로깅 드라이버 따름 |
| 호스트 docker 로그 디렉터리 | `/var/lib/docker/containers/<container_id>/<container_id>-json.log` |
| 메트릭 대시보드 | TODO — Prometheus / Grafana 도입 시 URL 등록 |
| GPU 모니터링 | `nvidia-smi -l 1` 또는 DCGM exporter (도입 TODO) |
| 감사 로그 | Postgres `api_keys` 테이블 `created_at`/`is_active`, app stdout `event="request_complete"` |
| 외부 의존 모델 | HuggingFace 캐시 `models` 볼륨 (네트워크 단절 시 새 모델 다운로드 불가) |
| 디스크 사용량 모니터링 | `docker system df`, `du -sh /var/lib/docker/volumes/llm-engine_*` |

**비상 시 우선순위**: 데이터 손실 방지 > 서비스 복구 > 근본 원인 분석. `docker compose down -v` 는 마지막 수단이며, 직전에 위 백업 절차를 반드시 한 번 더 돌린다.
