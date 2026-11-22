"""Typed application settings loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for TerraceMem."""

    model_config = SettingsConfigDict(
        env_prefix="TERRACEMEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database & Search Backend
    storage_backend: Literal["auto", "postgres", "memory"] = Field(
        default="auto",
        description=(
            "Storage backend: 'postgres' (strict PG), 'memory' (in-process), "
            "or 'auto' (connect to PG if reachable, else memory fallback)."
        ),
    )
    database_url: str = Field(
        default="postgresql://terracemem:terracemem@localhost:5432/terracemem",
        description="PostgreSQL connection DSN.",
    )
    db_pool_min_size: int = Field(default=2, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)
    lexical_backend: Literal["paradedb", "native"] = Field(
        default="paradedb",
        description="BM25/lexical backend: ParadeDB pg_search or native tsvector fallback.",
    )
    embedding_dim: int = Field(default=1024, ge=64, le=4096)
    enable_background_worker: bool = Field(
        default=False,
        description="Run PG Outbox worker loop inside API lifespan when connected to PostgreSQL.",
    )
    outbox_notify_channel: str = Field(
        default="terracemem_jobs",
        description="PostgreSQL LISTEN/NOTIFY channel for low-latency outbox wakeups.",
    )
    eval_reports_dir: str = Field(
        default="evals/reports",
        description="Directory containing offline evaluation JSON reports for /ui dashboard.",
    )

    # LLM / Embedding / Reranker (OpenAI-compatible API-first)
    llm_api_key: str = Field(default="", description="Primary LLM API key.")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-4o-mini")
    fast_llm_model: str = Field(default="gpt-4o-mini")

    embedding_api_key: str = Field(default="")
    embedding_base_url: str = Field(default="https://api.openai.com/v1")
    embedding_model: str = Field(default="text-embedding-3-small")

    reranker_enabled: bool = Field(default=False)
    reranker_api_key: str = Field(default="")
    reranker_base_url: str = Field(default="")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    # Feature Flags
    enable_write_time_enrichment: bool = Field(
        default=False,
        description="Optional write-time LLM summarization & keyword extraction during ingestion.",
    )
    enable_graph_expansion: bool = Field(
        default=True,
        description="Enable bounded 2-hop recursive CTE entity relation expansion.",
    )
    strict_citation_mode: bool = Field(
        default=True,
        description="Enforce strict citation whitelist validation and abstention on failure.",
    )
    enable_inspector_ui: bool = Field(
        default=True,
        description="Mount embedded lightweight developer Inspector UI at /ui.",
    )

    # Agentic RAG Budgets & Thresholds
    max_retrieval_retries: int = Field(default=2, ge=0, le=5)
    max_state_transitions: int = Field(default=10, ge=2, le=30)
    max_llm_calls: int = Field(default=5, ge=1, le=20)
    request_timeout_seconds: float = Field(default=30.0, gt=1.0)
    rrf_k: int = Field(default=60, ge=1)
    parent_expansion_max_tokens: int = Field(default=1200, ge=200, le=8192)
    rerank_hard_reject_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    rerank_high_confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)

    # Memory Guardrail Thresholds
    memory_contradiction_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    memory_decay_lambda: float = Field(default=0.015, ge=0.0)
    default_session_token_budget: int = Field(default=8000, ge=512)

    # HTTP Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings instance."""
    return Settings()
