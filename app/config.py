"""Application settings loaded from environment via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Server
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # Auth
    initial_api_key: str = ""
    api_key_header: str = "Authorization"

    # PostgreSQL
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "llm_engine"
    postgres_user: str = "llm"
    postgres_password: str = Field(default="change_me_postgres")

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "chunks"

    # Neo4j
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="change_me_neo4j")

    # Redis
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # vLLM - Gemma
    vllm_llm_url: str = "http://vllm-gemma:8000/v1"
    vllm_llm_model: str = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
    vllm_llm_max_tokens: int = 2048
    vllm_llm_temperature: float = 0.3
    vllm_llm_max_model_len: int = 16384
    vllm_llm_max_num_seqs: int = 8
    vllm_llm_gpu_mem_util: float = 0.55

    # vLLM - Chandra
    vllm_ocr_url: str = "http://vllm-chandra:8000/v1"
    vllm_ocr_model: str = "datalab-to/chandra-ocr-2"
    vllm_ocr_gpu_mem_util: float = 0.25

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # Session
    session_ttl_seconds: int = 1800
    session_max_turns: int = 10
    session_max_tokens: int = 4000

    # Indexing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    graphrag_enabled: bool = True
    graphrag_workdir: str = "/data/graphrag"
    graphrag_extract_concurrency: int = 4
    graph_retrieval_enabled: bool = True
    # Fix A: when True, graph seed matching uses token-set IN matching instead
    # of CONTAINS substring (kills short-Korean-entity false positives).
    graph_seed_use_token_boundary: bool = False
    # Minimum entity-name length for token-boundary seed matching.
    graph_seed_min_entity_len: int = 2

    # Reranker (cross-encoder over vector hits)
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_oversample: int = 4  # fetch top_k * oversample from Qdrant
    reranker_use_fp16: bool = True
    # Fix B: when True (and reranker_enabled), graph+vector candidates are
    # unified into one pool and ordered purely by cross-encoder score instead
    # of prepending graph hits. When False, the legacy prepend merge is used.
    graph_rerank_fusion: bool = False

    # Limits
    upload_max_file_size_mb: int = 50
    upload_max_files_per_request: int = 100
    rate_limit_per_minute: int = 60

    # Maintenance jobs
    # When True, the orphan :Entity cleanup beat schedule is registered.
    # Default is False so a stray celery-beat process is a no-op.
    cleanup_orphans_enabled: bool = False

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
