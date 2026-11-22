"""Pydantic v2 request and response schemas for TerraceMem FastAPI REST & SSE endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from terracemem.db.repositories import RetrievalMode


class ParsePreviewRequest(BaseModel):
    """Request payload for previewing structure-aware Markdown tree & chunk splitting."""

    title: str = Field(default="Untitled Document")
    markdown: str = Field(..., min_length=1)
    max_chunk_tokens: int = Field(default=450, ge=64, le=4096)


class DocumentIngestRequest(BaseModel):
    """Request payload for transactional versioned document ingestion."""

    title: str = Field(..., min_length=1)
    external_key: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default")
    namespace: str = Field(default="default")
    source_uri: str | None = None
    effective_at: datetime | None = None
    max_chunk_tokens: int = Field(default=450, ge=64, le=4096)
    enable_enrichment: bool | None = None
    enable_graph: bool | None = None
    skip_embeddings: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Request payload for hybrid retrieval."""

    query: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default")
    namespace: str = Field(default="default")
    document_ids: list[UUID] | None = None
    section_ids: list[UUID] | None = None
    effective_after: datetime | None = None
    effective_before: datetime | None = None
    only_latest: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    mode: RetrievalMode = "hybrid"
    enable_graph: bool | None = None
    enable_rerank: bool = True
    enable_parent_expand: bool = True


class AskRequest(BaseModel):
    """Request payload for synchronous or SSE Agentic RAG question answering."""

    question: str
    tenant_id: str = Field(default="default")
    namespace: str = Field(default="default")
    user_id: str | None = None
    session_id: UUID | None = None
    document_ids: list[UUID] | None = None
    section_ids: list[UUID] | None = None
    effective_after: datetime | None = None
    effective_before: datetime | None = None
    only_latest: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    retrieval_mode: RetrievalMode = "hybrid"
    enable_rerank: bool = True
    enable_parent_expand: bool = True
    enable_graph_expansion: bool | None = None
    enable_outline_navigation: bool | None = None
    enable_crag: bool = True
    strict_citation_mode: bool | None = None


class CreateSessionRequest(BaseModel):
    """Request payload for creating or retrieving an L1 working-memory session."""

    session_id: UUID | None = None
    tenant_id: str = Field(default="default")
    user_id: str = Field(default="default")
    namespace: str = Field(default="default")
    token_budget: int | None = Field(default=None, ge=256, le=128000)
    pinned_constraints: list[str] = Field(default_factory=list)


class MemoryCommitRequest(BaseModel):
    """Request payload for committing a turn into L1/L2/L3 memory."""

    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default")
    namespace: str = Field(default="default")
    session_id: UUID | None = None
    role: Literal["system", "user", "assistant", "tool"] = "user"
    event_time: datetime | None = None
    is_pinned: bool = False
    pinned_constraints: list[str] | None = None
    sync_extract: bool = True


class MemoryRecallRequest(BaseModel):
    """Request payload for L1/L2/L3 memory recall."""

    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default")
    namespace: str = Field(default="default")
    session_id: UUID | None = None
    limit: int = Field(default=8, ge=1, le=25)


class PinMemoryRequest(BaseModel):
    """Request payload for toggling `is_pinned` protection on an L3 semantic memory."""

    user_id: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default")
    is_pinned: bool = True
