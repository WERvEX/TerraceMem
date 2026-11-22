"""Bi-temporal L1/L2/L3 memory repository (PostgreSQL & InMemory implementations)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from psycopg.types.json import Jsonb

from terracemem.db.pool import DatabasePool
from terracemem.db.row_mappers import (
    as_datetime,
    as_dict,
    as_float,
    as_int,
    as_opt_datetime,
    as_opt_int,
    as_opt_str,
    as_opt_uuid,
    as_str,
    as_str_list,
    as_uuid,
)
from terracemem.domain.models import (
    EpisodicMemoryRecord,
    MemoryTimelineResult,
    MemoryType,
    SemanticMemoryRecord,
    SessionCompactionRecord,
    SessionMessageRecord,
    SessionRecord,
    utc_now,
)
from terracemem.llm.providers import cosine_similarity, format_pg_halfvec
from terracemem.utils.ids import new_uuid7
from terracemem.workers.outbox import InMemoryOutboxQueue, enqueue_outbox_job


def _ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _as_role(val: object) -> Literal["system", "user", "assistant", "tool"]:
    raw = as_str(val, default="user")
    if raw == "system":
        return "system"
    if raw == "assistant":
        return "assistant"
    if raw == "tool":
        return "tool"
    return "user"


def _as_memory_type(val: object) -> MemoryType:
    raw = as_str(val, default="fact")
    for member in MemoryType:
        if member.value == raw:
            return member
    return MemoryType.FACT


def _row_to_semantic(row: tuple[object, ...]) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        id=as_uuid(row[0]),
        tenant_id=as_str(row[1]),
        user_id=as_str(row[2]),
        namespace=as_str(row[3]),
        memory_type=_as_memory_type(row[4]),
        slot_key=as_opt_str(row[5]),
        fact_text=as_str(row[6]),
        evidence_quote=as_opt_str(row[7]),
        confidence=as_float(row[8], default=0.8),
        is_pinned=bool(row[9]),
        valid_from=as_datetime(row[10]),
        valid_to=as_opt_datetime(row[11]),
        recorded_at=as_datetime(row[12]),
        invalidated_at=as_opt_datetime(row[13]),
        superseded_by=as_opt_uuid(row[14]),
        invalidation_reason=as_opt_str(row[15]),
        access_count=as_int(row[16]),
        last_accessed_at=as_datetime(row[17]),
        importance_score=as_float(row[18], default=0.6),
    )


class MemoryRepository(Protocol):
    """Repository protocol managing L1 working sessions, L2 episodes & L3 bi-temporal facts."""

    async def create_or_get_session(
        self,
        *,
        session_id: UUID | None = None,
        tenant_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        token_budget: int = 8000,
        pinned_constraints: list[str] | None = None,
    ) -> SessionRecord: ...

    async def get_session(self, session_id: UUID) -> SessionRecord | None: ...

    async def update_pinned_constraints(
        self,
        session_id: UUID,
        pinned_constraints: list[str],
    ) -> SessionRecord | None: ...

    async def append_message(
        self,
        session_id: UUID,
        *,
        role: Literal["system", "user", "assistant", "tool"],
        content: str,
        token_count: int,
        metadata: dict[str, Any] | None = None,
        outbox_job_type: str | None = None,
        outbox_idempotency_key: str | None = None,
        outbox_payload: dict[str, Any] | None = None,
        notify_channel: str = "terracemem_jobs",
    ) -> SessionMessageRecord: ...

    async def list_messages(
        self,
        session_id: UUID,
        *,
        after_seq: int = 0,
    ) -> list[SessionMessageRecord]: ...

    async def save_compaction(
        self,
        compaction: SessionCompactionRecord,
    ) -> SessionCompactionRecord: ...

    async def get_latest_compaction(
        self,
        session_id: UUID,
    ) -> SessionCompactionRecord | None: ...

    async def insert_episode(
        self,
        episode: EpisodicMemoryRecord,
        embedding: list[float] | None = None,
    ) -> EpisodicMemoryRecord: ...

    async def search_episodes(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 5,
    ) -> list[tuple[EpisodicMemoryRecord, float]]: ...

    async def find_conflict_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None,
        query_embedding: list[float] | None,
        top_k: int = 5,
    ) -> list[SemanticMemoryRecord]: ...

    async def insert_semantic_memory(
        self,
        record: SemanticMemoryRecord,
        embedding: list[float] | None = None,
        *,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord: ...

    async def supersede_semantic_memory(
        self,
        *,
        old_id: UUID,
        new_record: SemanticMemoryRecord,
        new_embedding: list[float] | None = None,
        invalidation_reason: str,
        valid_to: datetime,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord: ...

    async def search_active_semantic_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 12,
    ) -> list[tuple[SemanticMemoryRecord, float]]: ...

    async def set_memory_pinned(
        self,
        memory_id: UUID,
        *,
        tenant_id: str,
        user_id: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord | None: ...

    async def get_timeline(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None = None,
        as_of: datetime | None = None,
    ) -> MemoryTimelineResult: ...


# ------------------------------------------------------------------------------
# PostgreSQL Memory Repository
# ------------------------------------------------------------------------------


_SEMANTIC_COLS = """
    id, tenant_id, user_id, namespace, memory_type, slot_key, fact_text,
    evidence_quote, confidence, is_pinned, valid_from, valid_to, recorded_at,
    invalidated_at, superseded_by, invalidation_reason, access_count,
    last_accessed_at, importance_score
"""


class PostgresMemoryRepository:
    """PostgreSQL implementation of L1/L2/L3 bi-temporal memory tables."""

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    async def create_or_get_session(
        self,
        *,
        session_id: UUID | None = None,
        tenant_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        token_budget: int = 8000,
        pinned_constraints: list[str] | None = None,
    ) -> SessionRecord:
        sid = session_id or new_uuid7()
        if session_id is not None:
            existing = await self.get_session(session_id)
            if existing is not None:
                return existing

        now = utc_now()
        constraints = list(pinned_constraints or [])
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO sessions (
                    id, tenant_id, user_id, namespace, token_budget,
                    pinned_constraints, last_active_at, created_at
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(user_id)s, %(namespace)s,
                    %(token_budget)s, %(pinned)s, %(now)s, %(now)s
                )
                ON CONFLICT (id) DO NOTHING;
                """,
                {
                    "id": sid,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "namespace": namespace,
                    "token_budget": token_budget,
                    "pinned": Jsonb(constraints),
                    "now": now,
                },
            )
        fetched = await self.get_session(sid)
        if fetched is not None:
            return fetched
        return SessionRecord(
            id=sid,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            token_budget=token_budget,
            pinned_constraints=constraints,
            last_active_at=now,
            created_at=now,
        )

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id, tenant_id, user_id, namespace, token_budget,
                    pinned_constraints, last_active_at, created_at
                FROM sessions
                WHERE id = %(session_id)s;
                """,
                {"session_id": session_id},
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return SessionRecord(
            id=as_uuid(row[0]),
            tenant_id=as_str(row[1]),
            user_id=as_str(row[2]),
            namespace=as_str(row[3]),
            token_budget=as_int(row[4], default=8000),
            pinned_constraints=as_str_list(row[5]),
            last_active_at=as_datetime(row[6]),
            created_at=as_datetime(row[7]),
        )

    async def update_pinned_constraints(
        self,
        session_id: UUID,
        pinned_constraints: list[str],
    ) -> SessionRecord | None:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE sessions
                SET pinned_constraints = %(pinned)s, last_active_at = NOW()
                WHERE id = %(session_id)s;
                """,
                {"session_id": session_id, "pinned": Jsonb(pinned_constraints)},
            )
        return await self.get_session(session_id)

    async def append_message(
        self,
        session_id: UUID,
        *,
        role: Literal["system", "user", "assistant", "tool"],
        content: str,
        token_count: int,
        metadata: dict[str, Any] | None = None,
        outbox_job_type: str | None = None,
        outbox_idempotency_key: str | None = None,
        outbox_payload: dict[str, Any] | None = None,
        notify_channel: str = "terracemem_jobs",
    ) -> SessionMessageRecord:
        msg_id = new_uuid7()
        now = utc_now()
        meta = dict(metadata or {})
        async with self._pool.connection() as conn, conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id FROM sessions WHERE id = %(session_id)s FOR UPDATE;",
                    {"session_id": session_id},
                )
                await cur.execute(
                    """
                    SELECT COALESCE(MAX(seq_no), 0)
                    FROM session_messages
                    WHERE session_id = %(session_id)s;
                    """,
                    {"session_id": session_id},
                )
                max_row = await cur.fetchone()
                next_seq = (as_int(max_row[0]) if max_row else 0) + 1
                await cur.execute(
                    """
                    INSERT INTO session_messages (
                        id, session_id, seq_no, role, content, token_count, metadata, created_at
                    ) VALUES (
                        %(id)s, %(session_id)s, %(seq_no)s, %(role)s, %(content)s,
                        %(token_count)s, %(metadata)s, %(created_at)s
                    );
                    """,
                    {
                        "id": msg_id,
                        "session_id": session_id,
                        "seq_no": next_seq,
                        "role": role,
                        "content": content,
                        "token_count": token_count,
                        "metadata": Jsonb(meta),
                        "created_at": now,
                    },
                )
                await cur.execute(
                    "UPDATE sessions SET last_active_at = %(now)s WHERE id = %(session_id)s;",
                    {"now": now, "session_id": session_id},
                )

            if outbox_job_type and outbox_idempotency_key:
                await enqueue_outbox_job(
                    conn,
                    job_type=outbox_job_type,
                    idempotency_key=outbox_idempotency_key,
                    payload=outbox_payload or {},
                    notify_channel=notify_channel,
                )

        return SessionMessageRecord(
            id=msg_id,
            session_id=session_id,
            seq_no=next_seq,
            role=role,
            content=content,
            token_count=token_count,
            metadata=meta,
            created_at=now,
        )

    async def list_messages(
        self,
        session_id: UUID,
        *,
        after_seq: int = 0,
    ) -> list[SessionMessageRecord]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, session_id, seq_no, role, content, token_count, metadata, created_at
                FROM session_messages
                WHERE session_id = %(session_id)s AND seq_no > %(after_seq)s
                ORDER BY seq_no ASC;
                """,
                {"session_id": session_id, "after_seq": after_seq},
            )
            rows = await cur.fetchall()
        return [
            SessionMessageRecord(
                id=as_uuid(row[0]),
                session_id=as_uuid(row[1]),
                seq_no=as_int(row[2]),
                role=_as_role(row[3]),
                content=as_str(row[4]),
                token_count=as_int(row[5]),
                metadata=as_dict(row[6]),
                created_at=as_datetime(row[7]),
            )
            for row in rows
        ]

    async def save_compaction(
        self,
        compaction: SessionCompactionRecord,
    ) -> SessionCompactionRecord:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO session_compactions (
                    id, session_id, compacted_up_to_seq, summary_text,
                    preserved_facts, token_count, created_at
                ) VALUES (
                    %(id)s, %(session_id)s, %(seq)s, %(summary)s,
                    %(facts)s, %(tokens)s, %(created_at)s
                );
                """,
                {
                    "id": compaction.id,
                    "session_id": compaction.session_id,
                    "seq": compaction.compacted_up_to_seq,
                    "summary": compaction.summary_text,
                    "facts": Jsonb(compaction.preserved_facts),
                    "tokens": compaction.token_count,
                    "created_at": compaction.created_at,
                },
            )
        return compaction

    async def get_latest_compaction(
        self,
        session_id: UUID,
    ) -> SessionCompactionRecord | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id, session_id, compacted_up_to_seq, summary_text,
                    preserved_facts, token_count, created_at
                FROM session_compactions
                WHERE session_id = %(session_id)s
                ORDER BY compacted_up_to_seq DESC
                LIMIT 1;
                """,
                {"session_id": session_id},
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return SessionCompactionRecord(
            id=as_uuid(row[0]),
            session_id=as_uuid(row[1]),
            compacted_up_to_seq=as_int(row[2]),
            summary_text=as_str(row[3]),
            preserved_facts=as_str_list(row[4]),
            token_count=as_int(row[5]),
            created_at=as_datetime(row[6]),
        )

    async def insert_episode(
        self,
        episode: EpisodicMemoryRecord,
        embedding: list[float] | None = None,
    ) -> EpisodicMemoryRecord:
        emb_lit = format_pg_halfvec(embedding) if embedding is not None else None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO episodic_memories (
                    id, tenant_id, user_id, namespace, session_id,
                    event_start_at, event_end_at, summary, key_decisions,
                    entities_mentioned, importance_score, embedding,
                    source_seq_from, source_seq_to, created_at
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(user_id)s, %(namespace)s, %(session_id)s,
                    %(start)s, %(end)s, %(summary)s, %(decisions)s,
                    %(entities)s, %(importance)s, %(embedding)s::halfvec,
                    %(seq_from)s, %(seq_to)s, %(created_at)s
                );
                """,
                {
                    "id": episode.id,
                    "tenant_id": episode.tenant_id,
                    "user_id": episode.user_id,
                    "namespace": episode.namespace,
                    "session_id": episode.session_id,
                    "start": episode.event_start_at,
                    "end": episode.event_end_at,
                    "summary": episode.summary,
                    "decisions": Jsonb(episode.key_decisions),
                    "entities": episode.entities_mentioned,
                    "importance": episode.importance_score,
                    "embedding": emb_lit,
                    "seq_from": episode.source_seq_from,
                    "seq_to": episode.source_seq_to,
                    "created_at": episode.created_at,
                },
            )
        return episode

    async def search_episodes(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 5,
    ) -> list[tuple[EpisodicMemoryRecord, float]]:
        del query_text
        query_vec = format_pg_halfvec(query_embedding) if query_embedding is not None else None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if query_vec is not None:
                await cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
                await cur.execute(
                    """
                    SELECT
                        id, tenant_id, user_id, namespace, session_id,
                        event_start_at, event_end_at, summary, key_decisions,
                        entities_mentioned, importance_score,
                        source_seq_from, source_seq_to, created_at,
                        CASE
                            WHEN embedding IS NOT NULL
                            THEN GREATEST(0.0, 1.0 - (embedding <=> %(query_vec)s::halfvec))
                            ELSE 0.3
                        END AS sim
                    FROM episodic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                    ORDER BY sim DESC, event_end_at DESC
                    LIMIT %(limit)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "query_vec": query_vec,
                        "limit": limit,
                    },
                )
            else:
                await cur.execute(
                    """
                    SELECT
                        id, tenant_id, user_id, namespace, session_id,
                        event_start_at, event_end_at, summary, key_decisions,
                        entities_mentioned, importance_score,
                        source_seq_from, source_seq_to, created_at,
                        0.5::double precision AS sim
                    FROM episodic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                    ORDER BY event_end_at DESC
                    LIMIT %(limit)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "limit": limit,
                    },
                )
            rows = await cur.fetchall()
        results: list[tuple[EpisodicMemoryRecord, float]] = []
        for row in rows:
            ep = EpisodicMemoryRecord(
                id=as_uuid(row[0]),
                tenant_id=as_str(row[1]),
                user_id=as_str(row[2]),
                namespace=as_str(row[3]),
                session_id=as_opt_uuid(row[4]),
                event_start_at=as_datetime(row[5]),
                event_end_at=as_datetime(row[6]),
                summary=as_str(row[7]),
                key_decisions=as_str_list(row[8]),
                entities_mentioned=as_str_list(row[9]),
                importance_score=as_float(row[10], default=0.5),
                source_seq_from=as_opt_int(row[11]),
                source_seq_to=as_opt_int(row[12]),
                created_at=as_datetime(row[13]),
            )
            results.append((ep, as_float(row[14], default=0.5)))
        return results

    async def find_conflict_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None,
        query_embedding: list[float] | None,
        top_k: int = 5,
    ) -> list[SemanticMemoryRecord]:
        by_id: dict[UUID, SemanticMemoryRecord] = {}
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if slot_key:
                await cur.execute(
                    f"""
                    SELECT {_SEMANTIC_COLS}
                    FROM semantic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                      AND valid_to IS NULL
                      AND invalidated_at IS NULL
                      AND slot_key = %(slot_key)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "slot_key": slot_key,
                    },
                )
                for row in await cur.fetchall():
                    rec = _row_to_semantic(row)
                    by_id[rec.id] = rec

            if query_embedding is not None:
                await cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
                await cur.execute(
                    f"""
                    SELECT {_SEMANTIC_COLS}
                    FROM semantic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                      AND valid_to IS NULL
                      AND invalidated_at IS NULL
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %(query_vec)s::halfvec ASC
                    LIMIT %(top_k)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "query_vec": format_pg_halfvec(query_embedding),
                        "top_k": top_k,
                    },
                )
                for row in await cur.fetchall():
                    rec = _row_to_semantic(row)
                    by_id.setdefault(rec.id, rec)
        return list(by_id.values())

    async def insert_semantic_memory(
        self,
        record: SemanticMemoryRecord,
        embedding: list[float] | None = None,
        *,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord:
        emb_lit = format_pg_halfvec(embedding) if embedding is not None else None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO semantic_memories (
                    id, tenant_id, user_id, namespace, memory_type, slot_key,
                    fact_text, evidence_quote, confidence, is_pinned,
                    valid_from, valid_to, recorded_at, invalidated_at,
                    superseded_by, invalidation_reason, access_count,
                    last_accessed_at, importance_score, source_episode_id, embedding
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(user_id)s, %(namespace)s, %(memory_type)s,
                    %(slot_key)s, %(fact_text)s, %(evidence_quote)s, %(confidence)s,
                    %(is_pinned)s, %(valid_from)s, %(valid_to)s, %(recorded_at)s,
                    %(invalidated_at)s, %(superseded_by)s, %(invalidation_reason)s,
                    %(access_count)s, %(last_accessed_at)s, %(importance_score)s,
                    %(source_episode_id)s, %(embedding)s::halfvec
                );
                """,
                {
                    "id": record.id,
                    "tenant_id": record.tenant_id,
                    "user_id": record.user_id,
                    "namespace": record.namespace,
                    "memory_type": record.memory_type.value,
                    "slot_key": record.slot_key,
                    "fact_text": record.fact_text,
                    "evidence_quote": record.evidence_quote,
                    "confidence": record.confidence,
                    "is_pinned": record.is_pinned,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                    "recorded_at": record.recorded_at,
                    "invalidated_at": record.invalidated_at,
                    "superseded_by": record.superseded_by,
                    "invalidation_reason": record.invalidation_reason,
                    "access_count": record.access_count,
                    "last_accessed_at": record.last_accessed_at,
                    "importance_score": record.importance_score,
                    "source_episode_id": source_episode_id,
                    "embedding": emb_lit,
                },
            )
        return record

    async def supersede_semantic_memory(
        self,
        *,
        old_id: UUID,
        new_record: SemanticMemoryRecord,
        new_embedding: list[float] | None = None,
        invalidation_reason: str,
        valid_to: datetime,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord:
        emb_lit = format_pg_halfvec(new_embedding) if new_embedding is not None else None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO semantic_memories (
                    id, tenant_id, user_id, namespace, memory_type, slot_key,
                    fact_text, evidence_quote, confidence, is_pinned,
                    valid_from, valid_to, recorded_at, invalidated_at,
                    superseded_by, invalidation_reason, access_count,
                    last_accessed_at, importance_score, source_episode_id, embedding
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(user_id)s, %(namespace)s, %(memory_type)s,
                    %(slot_key)s, %(fact_text)s, %(evidence_quote)s, %(confidence)s,
                    %(is_pinned)s, %(valid_from)s, %(valid_to)s, %(recorded_at)s,
                    %(invalidated_at)s, %(superseded_by)s, %(invalidation_reason)s,
                    %(access_count)s, %(last_accessed_at)s, %(importance_score)s,
                    %(source_episode_id)s, %(embedding)s::halfvec
                );
                """,
                {
                    "id": new_record.id,
                    "tenant_id": new_record.tenant_id,
                    "user_id": new_record.user_id,
                    "namespace": new_record.namespace,
                    "memory_type": new_record.memory_type.value,
                    "slot_key": new_record.slot_key,
                    "fact_text": new_record.fact_text,
                    "evidence_quote": new_record.evidence_quote,
                    "confidence": new_record.confidence,
                    "is_pinned": new_record.is_pinned,
                    "valid_from": new_record.valid_from,
                    "valid_to": new_record.valid_to,
                    "recorded_at": new_record.recorded_at,
                    "invalidated_at": new_record.invalidated_at,
                    "superseded_by": new_record.superseded_by,
                    "invalidation_reason": new_record.invalidation_reason,
                    "access_count": new_record.access_count,
                    "last_accessed_at": new_record.last_accessed_at,
                    "importance_score": new_record.importance_score,
                    "source_episode_id": source_episode_id,
                    "embedding": emb_lit,
                },
            )
            await cur.execute(
                """
                UPDATE semantic_memories
                SET valid_to = %(valid_to)s,
                    invalidated_at = NOW(),
                    superseded_by = %(new_id)s,
                    invalidation_reason = %(reason)s
                WHERE id = %(old_id)s
                  AND valid_to IS NULL
                  AND invalidated_at IS NULL;
                """,
                {
                    "old_id": old_id,
                    "new_id": new_record.id,
                    "valid_to": valid_to,
                    "reason": invalidation_reason,
                },
            )
        return new_record

    async def search_active_semantic_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 12,
    ) -> list[tuple[SemanticMemoryRecord, float]]:
        del query_text
        query_vec = format_pg_halfvec(query_embedding) if query_embedding is not None else None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if query_vec is not None:
                await cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
                await cur.execute(
                    f"""
                    SELECT {_SEMANTIC_COLS},
                        CASE
                            WHEN embedding IS NOT NULL
                            THEN GREATEST(0.0, 1.0 - (embedding <=> %(query_vec)s::halfvec))
                            ELSE 0.4
                        END AS sim
                    FROM semantic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                      AND valid_to IS NULL
                      AND invalidated_at IS NULL
                    ORDER BY sim DESC, is_pinned DESC
                    LIMIT %(limit)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "query_vec": query_vec,
                        "limit": limit,
                    },
                )
            else:
                await cur.execute(
                    f"""
                    SELECT {_SEMANTIC_COLS}, 0.5::double precision AS sim
                    FROM semantic_memories
                    WHERE tenant_id = %(tenant_id)s
                      AND user_id = %(user_id)s
                      AND namespace = %(namespace)s
                      AND valid_to IS NULL
                      AND invalidated_at IS NULL
                    ORDER BY is_pinned DESC, last_accessed_at DESC
                    LIMIT %(limit)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "namespace": namespace,
                        "limit": limit,
                    },
                )
            rows = await cur.fetchall()
            results = [(_row_to_semantic(row[:19]), as_float(row[19], default=0.5)) for row in rows]
            if results:
                ids = [item[0].id for item in results]
                await cur.execute(
                    """
                    UPDATE semantic_memories
                    SET access_count = access_count + 1,
                        last_accessed_at = NOW()
                    WHERE id = ANY(%(ids)s::uuid[]);
                    """,
                    {"ids": ids},
                )
        return results

    async def set_memory_pinned(
        self,
        memory_id: UUID,
        *,
        tenant_id: str,
        user_id: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord | None:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                f"""
                UPDATE semantic_memories
                SET is_pinned = %(is_pinned)s
                WHERE id = %(memory_id)s
                  AND tenant_id = %(tenant_id)s
                  AND user_id = %(user_id)s
                RETURNING {_SEMANTIC_COLS};
                """,
                {
                    "memory_id": memory_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "is_pinned": is_pinned,
                },
            )
            row = await cur.fetchone()
        return _row_to_semantic(row) if row is not None else None

    async def get_timeline(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None = None,
        as_of: datetime | None = None,
    ) -> MemoryTimelineResult:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT {_SEMANTIC_COLS}
                FROM semantic_memories
                WHERE tenant_id = %(tenant_id)s
                  AND user_id = %(user_id)s
                  AND namespace = %(namespace)s
                  AND (%(slot_key)s::text IS NULL OR slot_key = %(slot_key)s)
                ORDER BY valid_from ASC, recorded_at ASC;
                """,
                {
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "namespace": namespace,
                    "slot_key": slot_key,
                },
            )
            rows = await cur.fetchall()
        history = [_row_to_semantic(row) for row in rows]
        active = _filter_active_at(history, as_of=as_of)
        return MemoryTimelineResult(
            user_id=user_id,
            tenant_id=tenant_id,
            namespace=namespace,
            slot_key=slot_key,
            as_of=as_of,
            active_at_as_of=active,
            history=history,
        )


def _filter_active_at(
    records: list[SemanticMemoryRecord],
    *,
    as_of: datetime | None,
) -> list[SemanticMemoryRecord]:
    if as_of is None:
        return [r for r in records if r.valid_to is None and r.invalidated_at is None]
    target = _ensure_utc(as_of)
    active: list[SemanticMemoryRecord] = []
    for rec in records:
        vf = _ensure_utc(rec.valid_from)
        vt = _ensure_utc(rec.valid_to) if rec.valid_to is not None else None
        if vf <= target and (vt is None or vt > target):
            active.append(rec)
    return active


# ------------------------------------------------------------------------------
# In-Memory Memory Repository
# ------------------------------------------------------------------------------


class InMemoryMemoryRepository:
    """In-memory L1/L2/L3 memory store mirroring PG bi-temporal & outbox semantics."""

    def __init__(self, *, outbox: InMemoryOutboxQueue | None = None) -> None:
        self._outbox = outbox
        self._sessions: dict[UUID, SessionRecord] = {}
        self._messages: dict[UUID, list[SessionMessageRecord]] = {}
        self._compactions: dict[UUID, list[SessionCompactionRecord]] = {}
        self._episodes: dict[UUID, tuple[EpisodicMemoryRecord, list[float] | None]] = {}
        self._semantic: dict[UUID, tuple[SemanticMemoryRecord, list[float] | None]] = {}

    async def create_or_get_session(
        self,
        *,
        session_id: UUID | None = None,
        tenant_id: str = "default",
        user_id: str = "default",
        namespace: str = "default",
        token_budget: int = 8000,
        pinned_constraints: list[str] | None = None,
    ) -> SessionRecord:
        if session_id is not None and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id or new_uuid7()
        now = utc_now()
        record = SessionRecord(
            id=sid,
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            token_budget=token_budget,
            pinned_constraints=list(pinned_constraints or []),
            last_active_at=now,
            created_at=now,
        )
        self._sessions[sid] = record
        return record

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        return self._sessions.get(session_id)

    async def update_pinned_constraints(
        self,
        session_id: UUID,
        pinned_constraints: list[str],
    ) -> SessionRecord | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        updated = session.model_copy(
            update={"pinned_constraints": list(pinned_constraints), "last_active_at": utc_now()}
        )
        self._sessions[session_id] = updated
        return updated

    async def append_message(
        self,
        session_id: UUID,
        *,
        role: Literal["system", "user", "assistant", "tool"],
        content: str,
        token_count: int,
        metadata: dict[str, Any] | None = None,
        outbox_job_type: str | None = None,
        outbox_idempotency_key: str | None = None,
        outbox_payload: dict[str, Any] | None = None,
        notify_channel: str = "terracemem_jobs",
    ) -> SessionMessageRecord:
        del notify_channel
        existing = self._messages.setdefault(session_id, [])
        next_seq = (existing[-1].seq_no if existing else 0) + 1
        msg = SessionMessageRecord(
            session_id=session_id,
            seq_no=next_seq,
            role=role,
            content=content,
            token_count=token_count,
            metadata=dict(metadata or {}),
        )
        existing.append(msg)
        if session_id in self._sessions:
            self._sessions[session_id] = self._sessions[session_id].model_copy(
                update={"last_active_at": utc_now()}
            )
        if self._outbox is not None and outbox_job_type and outbox_idempotency_key:
            self._outbox.enqueue(
                job_type=outbox_job_type,
                idempotency_key=outbox_idempotency_key,
                payload=outbox_payload or {},
            )
        return msg

    async def list_messages(
        self,
        session_id: UUID,
        *,
        after_seq: int = 0,
    ) -> list[SessionMessageRecord]:
        return [m for m in self._messages.get(session_id, []) if m.seq_no > after_seq]

    async def save_compaction(
        self,
        compaction: SessionCompactionRecord,
    ) -> SessionCompactionRecord:
        self._compactions.setdefault(compaction.session_id, []).append(compaction)
        return compaction

    async def get_latest_compaction(
        self,
        session_id: UUID,
    ) -> SessionCompactionRecord | None:
        items = self._compactions.get(session_id, [])
        if not items:
            return None
        return max(items, key=lambda c: c.compacted_up_to_seq)

    async def insert_episode(
        self,
        episode: EpisodicMemoryRecord,
        embedding: list[float] | None = None,
    ) -> EpisodicMemoryRecord:
        self._episodes[episode.id] = (episode, embedding)
        return episode

    async def search_episodes(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 5,
    ) -> list[tuple[EpisodicMemoryRecord, float]]:
        q_lower = query_text.strip().lower()
        scored: list[tuple[EpisodicMemoryRecord, float]] = []
        for ep, emb in self._episodes.values():
            if ep.tenant_id != tenant_id or ep.user_id != user_id or ep.namespace != namespace:
                continue
            sim = 0.4
            if query_embedding is not None and emb is not None:
                sim = cosine_similarity(query_embedding, emb)
            elif q_lower and q_lower in ep.summary.lower():
                sim = 0.85
            scored.append((ep, sim))
        scored.sort(key=lambda pair: (-pair[1], -pair[0].event_end_at.timestamp()))
        return scored[:limit]

    async def find_conflict_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None,
        query_embedding: list[float] | None,
        top_k: int = 5,
    ) -> list[SemanticMemoryRecord]:
        active = [
            (rec, emb)
            for rec, emb in self._semantic.values()
            if rec.tenant_id == tenant_id
            and rec.user_id == user_id
            and rec.namespace == namespace
            and rec.valid_to is None
            and rec.invalidated_at is None
        ]
        by_id: dict[UUID, SemanticMemoryRecord] = {}
        if slot_key:
            for rec, _ in active:
                if rec.slot_key == slot_key:
                    by_id[rec.id] = rec
        if query_embedding is not None:
            sim_pairs: list[tuple[SemanticMemoryRecord, float]] = []
            for rec, emb in active:
                if emb is not None:
                    sim_pairs.append((rec, cosine_similarity(query_embedding, emb)))
            sim_pairs.sort(key=lambda pair: -pair[1])
            for rec, _ in sim_pairs[:top_k]:
                by_id.setdefault(rec.id, rec)
        return list(by_id.values())

    async def insert_semantic_memory(
        self,
        record: SemanticMemoryRecord,
        embedding: list[float] | None = None,
        *,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord:
        del source_episode_id
        self._semantic[record.id] = (record, embedding)
        return record

    async def supersede_semantic_memory(
        self,
        *,
        old_id: UUID,
        new_record: SemanticMemoryRecord,
        new_embedding: list[float] | None = None,
        invalidation_reason: str,
        valid_to: datetime,
        source_episode_id: UUID | None = None,
    ) -> SemanticMemoryRecord:
        del source_episode_id
        self._semantic[new_record.id] = (new_record, new_embedding)
        old_pair = self._semantic.get(old_id)
        if old_pair is not None:
            old_rec, old_emb = old_pair
            if old_rec.valid_to is None and old_rec.invalidated_at is None:
                updated_old = old_rec.model_copy(
                    update={
                        "valid_to": valid_to,
                        "invalidated_at": utc_now(),
                        "superseded_by": new_record.id,
                        "invalidation_reason": invalidation_reason,
                    }
                )
                self._semantic[old_id] = (updated_old, old_emb)
        return new_record

    async def search_active_semantic_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query_embedding: list[float] | None,
        query_text: str,
        limit: int = 12,
    ) -> list[tuple[SemanticMemoryRecord, float]]:
        q_lower = query_text.strip().lower()
        scored: list[tuple[SemanticMemoryRecord, float]] = []
        for rec, emb in self._semantic.values():
            if (
                rec.tenant_id != tenant_id
                or rec.user_id != user_id
                or rec.namespace != namespace
                or rec.valid_to is not None
                or rec.invalidated_at is not None
            ):
                continue
            sim = 0.5
            if query_embedding is not None and emb is not None:
                sim = cosine_similarity(query_embedding, emb)
            elif q_lower and q_lower in rec.fact_text.lower():
                sim = 0.85
            scored.append((rec, sim))
        scored.sort(key=lambda pair: (-pair[1], not pair[0].is_pinned))
        top = scored[:limit]
        now = utc_now()
        updated_top: list[tuple[SemanticMemoryRecord, float]] = []
        for rec, sim in top:
            _, emb = self._semantic[rec.id]
            bumped = rec.model_copy(
                update={"access_count": rec.access_count + 1, "last_accessed_at": now}
            )
            self._semantic[rec.id] = (bumped, emb)
            updated_top.append((bumped, sim))
        return updated_top

    async def set_memory_pinned(
        self,
        memory_id: UUID,
        *,
        tenant_id: str,
        user_id: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord | None:
        pair = self._semantic.get(memory_id)
        if pair is None:
            return None
        rec, emb = pair
        if rec.tenant_id != tenant_id or rec.user_id != user_id:
            return None
        updated = rec.model_copy(update={"is_pinned": is_pinned})
        self._semantic[memory_id] = (updated, emb)
        return updated

    async def get_timeline(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None = None,
        as_of: datetime | None = None,
    ) -> MemoryTimelineResult:
        history = [
            rec
            for rec, _ in self._semantic.values()
            if rec.tenant_id == tenant_id
            and rec.user_id == user_id
            and rec.namespace == namespace
            and (slot_key is None or rec.slot_key == slot_key)
        ]
        history.sort(key=lambda r: (_ensure_utc(r.valid_from), _ensure_utc(r.recorded_at)))
        active = _filter_active_at(history, as_of=as_of)
        return MemoryTimelineResult(
            user_id=user_id,
            tenant_id=tenant_id,
            namespace=namespace,
            slot_key=slot_key,
            as_of=as_of,
            active_at_as_of=active,
            history=history,
        )
