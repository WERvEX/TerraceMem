"""Core Pydantic v2 domain models shared across storage, RAG, memory, and APIs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from terracemem.utils.ids import new_uuid7


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


# ------------------------------------------------------------------------------
# Document Tree & Retrieval Models
# ------------------------------------------------------------------------------


class SectionNode(BaseModel):
    """A node in a parsed document's hierarchical heading tree."""

    id: UUID = Field(default_factory=new_uuid7)
    parent_section_id: UUID | None = None
    section_path: str
    heading_path: list[str]
    heading: str
    level: int = Field(ge=0, le=10)
    ordinal: int = Field(ge=0)
    section_text: str
    token_count: int = Field(ge=0)
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """A single indexable chunk linked back to its parent document and section."""

    id: UUID = Field(default_factory=new_uuid7)
    document_id: UUID
    section_id: UUID | None = None
    chunk_index: int = Field(ge=0)
    raw_text: str
    contextual_text: str
    enriched_keywords: str = ""
    token_count: int = Field(ge=0)
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Structured result emitted by any DocumentParser implementation."""

    id: UUID = Field(default_factory=new_uuid7)
    external_key: str
    title: str
    content_hash: str
    source_uri: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    sections: list[SectionNode] = Field(default_factory=list)
    chunks: list[ChunkRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """Lightweight document summary returned by listing and outline queries."""

    id: UUID
    tenant_id: str = "default"
    namespace: str = "default"
    external_key: str
    title: str
    version: int = 1
    is_latest: bool = True
    status: str = "ready"
    effective_at: datetime | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    section_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class DocumentOutline(BaseModel):
    """Document metadata coupled with its ordered hierarchical section tree."""

    document: DocumentSummary
    sections: list[SectionNode] = Field(default_factory=list)


class IngestDocumentResult(BaseModel):
    """Transactional result of document versioned ingestion."""

    document_id: UUID
    tenant_id: str
    namespace: str
    external_key: str
    title: str
    version: int
    action: Literal["created", "updated", "unchanged"]
    section_count: int
    chunk_count: int
    enriched: bool = False
    embedded: bool = False
    entities_indexed: int = 0
    edges_indexed: int = 0


class EntityRecord(BaseModel):
    """Canonical entity node in PostgreSQL lightweight relation graph."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    namespace: str = "default"
    canonical_name: str
    entity_type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class RelationEdgeRecord(BaseModel):
    """Directed typed edge between two entities backed by a chunk citation."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    namespace: str = "default"
    src_entity_id: UUID
    rel_type: str
    dst_entity_id: UUID
    evidence_chunk_id: UUID | None = None
    weight: float = Field(default=1.0, gt=0.0)
    valid_from: datetime = Field(default_factory=utc_now)
    valid_to: datetime | None = None


class RetrievalFilter(BaseModel):
    """Deterministic SQL filter boundaries applied before/during retrieval."""

    tenant_id: str = "default"
    namespace: str = "default"
    document_ids: list[UUID] | None = None
    section_ids: list[UUID] | None = None
    effective_after: datetime | None = None
    effective_before: datetime | None = None
    only_latest: bool = True


class RetrievedEvidence(BaseModel):
    """Evidence unit returned by retrieval/rerank/parent-expansion stages."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    section_id: UUID | None = None
    heading_path: list[str] = Field(default_factory=list)
    raw_text: str
    expanded_text: str | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    graph_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


# ------------------------------------------------------------------------------
# Agentic RAG State Machine & Citation Models
# ------------------------------------------------------------------------------


class RagDecisionStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ERROR = "error"


class SufficiencyVerdict(StrEnum):
    SUFFICIENT = "sufficient"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


class Citation(BaseModel):
    """Structured citation linking a generated claim to a concrete chunk."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    heading_path: list[str] = Field(default_factory=list)
    quote: str | None = None


class ClaimWithCitations(BaseModel):
    """An atomic statement in the generated answer backed by chunk IDs."""

    statement: str
    cited_chunk_ids: list[UUID] = Field(default_factory=list)


class RagResponse(BaseModel):
    """Final trustworthy RAG response with strict citations or explicit abstention."""

    trace_id: UUID
    status: RagDecisionStatus
    answer: str
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    abstention_reason: str | None = None
    retrieval_retries: int = 0
    searched_scope: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------------------
# Three-Tier Bi-temporal Memory Models
# ------------------------------------------------------------------------------


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    PROCEDURE = "procedure"


class EpistemicModality(StrEnum):
    """Distinguishes asserted facts from hypothetical/question statements."""

    ASSERTED = "asserted"
    HYPOTHETICAL = "hypothetical"
    QUESTION = "question"
    REPORTED = "reported"


class MemoryResolutionAction(StrEnum):
    NOOP = "noop"
    MERGE = "merge"
    CONTRADICT = "contradict"
    UNRELATED = "unrelated"


class SemanticMemoryRecord(BaseModel):
    """An L3 atomic semantic memory entry with bi-temporal validity axes."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    user_id: str
    namespace: str = "default"
    memory_type: MemoryType = MemoryType.FACT
    slot_key: str | None = None
    fact_text: str
    evidence_quote: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    is_pinned: bool = False
    valid_from: datetime = Field(default_factory=utc_now)
    valid_to: datetime | None = None
    recorded_at: datetime = Field(default_factory=utc_now)
    invalidated_at: datetime | None = None
    superseded_by: UUID | None = None
    invalidation_reason: str | None = None
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: datetime = Field(default_factory=utc_now)
    importance_score: float = Field(default=0.6, ge=0.0, le=1.0)


class CandidateMemoryFact(BaseModel):
    """Newly extracted fact awaiting deduplication and conflict resolution."""

    memory_type: MemoryType
    slot_key: str | None = None
    fact_text: str
    evidence_quote: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    modality: EpistemicModality = EpistemicModality.ASSERTED
    event_time: datetime = Field(default_factory=utc_now)


class ConflictJudgment(BaseModel):
    """Proposed LLM relation judgment between a candidate fact and an existing fact."""

    target_memory_id: UUID
    action: MemoryResolutionAction
    reason: str = ""
    merged_fact_text: str | None = None


class SessionRecord(BaseModel):
    """L1 working-memory session metadata and pinned hard constraints."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    user_id: str
    namespace: str = "default"
    token_budget: int = Field(default=8000, ge=256)
    pinned_constraints: list[str] = Field(default_factory=list)
    last_active_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)


class SessionMessageRecord(BaseModel):
    """Single message turn inside an L1 session."""

    id: UUID = Field(default_factory=new_uuid7)
    session_id: UUID
    seq_no: int = Field(ge=1)
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    token_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SessionCompactionRecord(BaseModel):
    """Incremental rolling L1 summary compacting older messages."""

    id: UUID = Field(default_factory=new_uuid7)
    session_id: UUID
    compacted_up_to_seq: int = Field(ge=1)
    summary_text: str
    preserved_facts: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class WorkingMemoryContext(BaseModel):
    """Token-budgeted L1 context window ready to inject into a prompt."""

    session_id: UUID
    token_budget: int
    used_tokens: int
    pinned_constraints: list[str] = Field(default_factory=list)
    compaction_summary: str | None = None
    preserved_facts: list[str] = Field(default_factory=list)
    recent_messages: list[SessionMessageRecord] = Field(default_factory=list)
    dropped_message_count: int = 0


class EpisodicMemoryRecord(BaseModel):
    """L2 timestamped session episode archive with key decisions and mentioned entities."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    user_id: str
    namespace: str = "default"
    session_id: UUID | None = None
    event_start_at: datetime = Field(default_factory=utc_now)
    event_end_at: datetime = Field(default_factory=utc_now)
    summary: str
    key_decisions: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_seq_from: int | None = None
    source_seq_to: int | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ScoredSemanticMemory(BaseModel):
    """L3 semantic memory paired with its relevance & decay score."""

    memory: SemanticMemoryRecord
    similarity_score: float = 0.0
    final_score: float = 0.0


class ScoredEpisodicMemory(BaseModel):
    """L2 episodic memory paired with its combined time/semantic score."""

    episode: EpisodicMemoryRecord
    score: float = 0.0


class MemoryConflictAudit(BaseModel):
    """Record of how a candidate fact interacted with existing L3 memory & guardrails."""

    candidate_fact: str
    slot_key: str | None = None
    target_memory_id: UUID | None = None
    proposed_action: MemoryResolutionAction
    effective_action: MemoryResolutionAction
    guardrail_decision: str
    reason: str
    new_memory_id: UUID | None = None


class MemoryRecallResult(BaseModel):
    """Unified L1/L2/L3 recall payload for RAG injection, REST API, and MCP."""

    working_context: WorkingMemoryContext | None = None
    episodic_memories: list[ScoredEpisodicMemory] = Field(default_factory=list)
    semantic_memories: list[ScoredSemanticMemory] = Field(default_factory=list)


class MemoryTimelineResult(BaseModel):
    """Bi-temporal L3 history & supersession chain audit view."""

    user_id: str
    tenant_id: str
    namespace: str
    slot_key: str | None = None
    as_of: datetime | None = None
    active_at_as_of: list[SemanticMemoryRecord] = Field(default_factory=list)
    history: list[SemanticMemoryRecord] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Observability Trace Models
# ------------------------------------------------------------------------------


class TraceSpanRecord(BaseModel):
    """One deterministic or LLM step inside a single request execution trace."""

    id: UUID = Field(default_factory=new_uuid7)
    step_order: int
    node_name: str
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    input_meta: dict[str, Any] = Field(default_factory=dict)
    output_meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ExecutionTraceRecord(BaseModel):
    """End-to-end execution trace persisted to PostgreSQL and shown in /ui."""

    id: UUID = Field(default_factory=new_uuid7)
    tenant_id: str = "default"
    session_id: UUID | None = None
    channel: Literal["rest", "sse", "mcp", "eval"] = "rest"
    operation: str
    query_text: str | None = None
    final_status: str
    retrieval_retries: int = 0
    total_latency_ms: int = Field(default=0, ge=0)
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    spans: list[TraceSpanRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
