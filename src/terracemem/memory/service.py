"""Unified Three-Tier Memory Service coordinating L1/L2/L3 and Outbox extraction."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from terracemem.config import Settings
from terracemem.db.memory_repository import MemoryRepository
from terracemem.domain.models import (
    EpisodicMemoryRecord,
    MemoryConflictAudit,
    MemoryRecallResult,
    MemoryTimelineResult,
    SemanticMemoryRecord,
    SessionMessageRecord,
    SessionRecord,
    WorkingMemoryContext,
    utc_now,
)
from terracemem.llm.providers import EmbeddingProvider, LLMProvider
from terracemem.memory.episodic import EpisodicMemoryManager
from terracemem.memory.semantic import SemanticMemoryManager
from terracemem.memory.working import WorkingMemoryManager
from terracemem.utils.tokens import estimate_tokens
from terracemem.workers.outbox import OutboxWorkerProtocol


class MemoryCommitResult(BaseModel):
    """Result of committing a conversation turn into L1 (+ optional sync L2/L3)."""

    session_id: UUID
    message_id: UUID
    seq_no: int
    compacted: bool = False
    queued_async_job: bool = False
    episode: EpisodicMemoryRecord | None = None
    audits: list[MemoryConflictAudit] = Field(default_factory=list)


class MemoryService:
    """Facade coordinating L1 working memory, L2 episodic, L3 bi-temporal & Outbox jobs."""

    JOB_TYPE_EXTRACT = "memory.extract"

    def __init__(
        self,
        *,
        settings: Settings,
        repo: MemoryRepository,
        embedder: EmbeddingProvider,
        llm: LLMProvider | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self.working = WorkingMemoryManager(settings=settings, repo=repo, llm=llm)
        self.episodic = EpisodicMemoryManager(repo=repo, embedder=embedder, llm=llm)
        self.semantic = SemanticMemoryManager(
            settings=settings, repo=repo, embedder=embedder, llm=llm
        )

    def register_worker_handlers(self, worker: OutboxWorkerProtocol) -> None:
        """Bind async outbox job handlers on either PostgresOutboxWorker or InMemoryOutboxQueue."""
        worker.register(self.JOB_TYPE_EXTRACT, self.handle_extract_job)

    async def create_session(
        self,
        *,
        session_id: UUID | None = None,
        tenant_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        token_budget: int | None = None,
        pinned_constraints: list[str] | None = None,
    ) -> SessionRecord:
        return await self._repo.create_or_get_session(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            token_budget=token_budget or self._settings.default_session_token_budget,
            pinned_constraints=pinned_constraints,
        )

    async def get_working_context(self, session_id: UUID) -> WorkingMemoryContext | None:
        return await self.working.build_context(session_id)

    async def commit_turn(
        self,
        *,
        user_id: str,
        content: str,
        tenant_id: str = "default",
        namespace: str = "default",
        session_id: UUID | None = None,
        role: Literal["system", "user", "assistant", "tool"] = "user",
        event_time: datetime | None = None,
        is_pinned: bool = False,
        pinned_constraints: list[str] | None = None,
        sync_extract: bool = True,
    ) -> MemoryCommitResult:
        """Persist turn to L1 and either extract L2/L3 synchronously or enqueue Outbox job."""
        session = await self._repo.create_or_get_session(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            token_budget=self._settings.default_session_token_budget,
            pinned_constraints=pinned_constraints,
        )
        if pinned_constraints is not None and session.pinned_constraints != pinned_constraints:
            merged = list(dict.fromkeys([*session.pinned_constraints, *pinned_constraints]))
            await self._repo.update_pinned_constraints(session.id, merged)

        resolved_time = event_time or utc_now()
        digest = hashlib.sha256(
            f"{session.id}:{role}:{content}:{resolved_time.isoformat()}".encode()
        ).hexdigest()[:20]
        idem_key = f"mem_extract:{session.id}:{digest}"
        payload: dict[str, Any] = {
            "session_id": str(session.id),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "namespace": namespace,
            "content": content,
            "role": role,
            "event_time": resolved_time.isoformat(),
            "is_pinned": is_pinned,
        }

        msg = await self._repo.append_message(
            session.id,
            role=role,
            content=content,
            token_count=estimate_tokens(content),
            metadata={"event_time": resolved_time.isoformat()},
            outbox_job_type=None if sync_extract else self.JOB_TYPE_EXTRACT,
            outbox_idempotency_key=None if sync_extract else idem_key,
            outbox_payload=None if sync_extract else payload,
            notify_channel=self._settings.outbox_notify_channel,
        )

        compaction = await self.working.maybe_compact(session.id)

        episode: EpisodicMemoryRecord | None = None
        audits: list[MemoryConflictAudit] = []
        if sync_extract:
            episode, audits = await self._extract_from_message(
                session_id=session.id,
                message=msg,
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                event_time=resolved_time,
                is_pinned=is_pinned,
            )

        return MemoryCommitResult(
            session_id=session.id,
            message_id=msg.id,
            seq_no=msg.seq_no,
            compacted=compaction is not None,
            queued_async_job=not sync_extract,
            episode=episode,
            audits=audits,
        )

    async def handle_extract_job(self, payload: dict[str, Any]) -> None:
        """Async Outbox handler archiving L2 episode and resolving L3 bi-temporal facts."""
        session_id = UUID(str(payload["session_id"]))
        tenant_id = str(payload.get("tenant_id", "default"))
        user_id = str(payload.get("user_id", "default"))
        namespace = str(payload.get("namespace", "default"))
        content = str(payload.get("content", ""))
        raw_role = str(payload.get("role", "user"))
        role: Literal["system", "user", "assistant", "tool"]
        if raw_role == "system":
            role = "system"
        elif raw_role == "assistant":
            role = "assistant"
        elif raw_role == "tool":
            role = "tool"
        else:
            role = "user"
        raw_time = payload.get("event_time")
        event_time = datetime.fromisoformat(str(raw_time)) if raw_time else utc_now()
        is_pinned = bool(payload.get("is_pinned", False))

        msg = SessionMessageRecord(
            session_id=session_id,
            seq_no=1,
            role=role,
            content=content,
            token_count=estimate_tokens(content),
            created_at=event_time,
        )
        await self._extract_from_message(
            session_id=session_id,
            message=msg,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            event_time=event_time,
            is_pinned=is_pinned,
        )

    async def _extract_from_message(
        self,
        *,
        session_id: UUID,
        message: SessionMessageRecord,
        tenant_id: str,
        user_id: str,
        namespace: str,
        event_time: datetime,
        is_pinned: bool,
    ) -> tuple[EpisodicMemoryRecord, list[MemoryConflictAudit]]:
        episode = await self.episodic.archive_messages(
            [message],
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            session_id=session_id,
        )
        candidates = await self.semantic.extract_candidate_facts(
            message.content,
            event_time=event_time,
        )
        audits = await self.semantic.commit_candidates(
            candidates,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            source_episode_id=episode.id,
            is_pinned=is_pinned,
        )
        return (episode, audits)

    async def recall(
        self,
        *,
        user_id: str,
        query: str,
        tenant_id: str = "default",
        namespace: str = "default",
        session_id: UUID | None = None,
        limit: int = 8,
    ) -> MemoryRecallResult:
        working_ctx = (
            await self.working.build_context(session_id) if session_id is not None else None
        )
        episodes = await self.episodic.recall(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            query=query,
            limit=min(limit, 5),
        )
        semantics = await self.semantic.recall(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            query=query,
            limit=limit,
        )
        return MemoryRecallResult(
            working_context=working_ctx,
            episodic_memories=episodes,
            semantic_memories=semantics,
        )

    async def timeline(
        self,
        *,
        user_id: str,
        tenant_id: str = "default",
        namespace: str = "default",
        slot_key: str | None = None,
        as_of: datetime | None = None,
    ) -> MemoryTimelineResult:
        return await self.semantic.timeline(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            slot_key=slot_key,
            as_of=as_of,
        )

    async def set_pinned(
        self,
        memory_id: UUID,
        *,
        tenant_id: str = "default",
        user_id: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord | None:
        return await self.semantic.set_pinned(
            memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            is_pinned=is_pinned,
        )
