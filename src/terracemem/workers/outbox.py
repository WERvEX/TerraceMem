"""Zero-Redis PostgreSQL transactional Outbox worker (SKIP LOCKED + LISTEN/NOTIFY)."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

import structlog
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from terracemem.db.pool import DatabasePool
from terracemem.db.row_mappers import as_int, as_uuid
from terracemem.utils.ids import new_uuid7

logger = structlog.get_logger(__name__)

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]

_CHANNEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

CLAIM_JOB_SQL = """
UPDATE async_jobs
SET status = 'running',
    attempts = attempts + 1,
    updated_at = NOW()
WHERE id = (
    SELECT id
    FROM async_jobs
    WHERE status = 'pending'
      AND run_at <= NOW()
    ORDER BY created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING id, job_type, payload, attempts, max_attempts;
"""

MARK_SUCCEEDED_SQL = """
UPDATE async_jobs
SET status = 'succeeded', updated_at = NOW(), last_error = NULL
WHERE id = %(job_id)s;
"""

MARK_FAILED_SQL = """
UPDATE async_jobs
SET status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
    run_at = NOW() + (INTERVAL '5 seconds' * POWER(2, attempts)),
    last_error = %(error)s,
    updated_at = NOW()
WHERE id = %(job_id)s;
"""


class OutboxWorkerProtocol(Protocol):
    """Common interface for PostgreSQL and in-memory outbox queues."""

    def register(self, job_type: str, handler: JobHandler) -> None: ...

    async def process_one(self) -> bool: ...


async def enqueue_outbox_job(
    conn: AsyncConnection[tuple[object, ...]],
    *,
    job_type: str,
    idempotency_key: str,
    payload: Mapping[str, Any],
    notify_channel: str = "terracemem_jobs",
    max_attempts: int = 3,
) -> UUID | None:
    """Insert an idempotent outbox job inside the caller's active transaction and notify."""
    job_id = new_uuid7()
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO async_jobs (
                id, job_type, idempotency_key, payload, status, attempts, max_attempts
            ) VALUES (
                %(id)s, %(job_type)s, %(idempotency_key)s, %(payload)s, 'pending', 0,
                %(max_attempts)s
            )
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id;
            """,
            {
                "id": job_id,
                "job_type": job_type,
                "idempotency_key": idempotency_key,
                "payload": Jsonb(dict(payload)),
                "max_attempts": max_attempts,
            },
        )
        row = await cur.fetchone()
        if row is None:
            return None
        inserted_id = as_uuid(row[0])
        if _CHANNEL_RE.match(notify_channel):
            await cur.execute(
                "SELECT pg_notify(%(channel)s, %(job_id)s);",
                {"channel": notify_channel, "job_id": str(inserted_id)},
            )
        return inserted_id


class PostgresOutboxWorker:
    """Claims and executes durable background jobs directly from PostgreSQL."""

    def __init__(
        self,
        pool: DatabasePool,
        *,
        poll_interval_seconds: float = 1.0,
        notify_channel: str | None = "terracemem_jobs",
        database_url: str | None = None,
    ) -> None:
        self._pool = pool
        self._poll_interval_seconds = poll_interval_seconds
        self._notify_channel = (
            notify_channel if notify_channel and _CHANNEL_RE.match(notify_channel) else None
        )
        self._database_url = database_url
        self._handlers: dict[str, JobHandler] = {}
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()

    def register(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    def request_stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    async def process_one(self) -> bool:
        """Attempt to claim and execute a single pending job; return True if a job ran."""
        job_row: tuple[UUID, str, dict[str, Any], int, int] | None = None
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(CLAIM_JOB_SQL)
            row = await cur.fetchone()
            if row is not None:
                raw_id, raw_type, raw_payload, raw_attempts, raw_max = row
                job_id_val = as_uuid(raw_id)
                if not isinstance(raw_payload, Mapping):
                    raise TypeError(
                        f"Expected mapping job payload, got {type(raw_payload).__name__}"
                    )
                job_row = (
                    job_id_val,
                    str(raw_type),
                    {str(k): v for k, v in raw_payload.items()},
                    as_int(raw_attempts),
                    as_int(raw_max),
                )

        if job_row is None:
            return False

        job_id, job_type, payload, _, _ = job_row
        handler = self._handlers.get(job_type)
        if handler is None:
            await self._mark_failed(job_id, f"Unregistered job_type: {job_type}")
            return True

        try:
            await handler(payload)
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(MARK_SUCCEEDED_SQL, {"job_id": job_id})
                await conn.commit()
        except Exception as exc:
            logger.exception("outbox_job_failed", job_id=str(job_id), job_type=job_type)
            await self._mark_failed(job_id, str(exc))
        return True

    async def _mark_failed(self, job_id: UUID, error: str) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(MARK_FAILED_SQL, {"job_id": job_id, "error": error[:1000]})
            await conn.commit()

    async def _listen_loop(self) -> None:
        """Listen on PostgreSQL NOTIFY channel to wake worker immediately when jobs arrive."""
        if not self._notify_channel or not self._database_url:
            return
        while not self._stop_event.is_set():
            try:
                conn = await AsyncConnection.connect(self._database_url, autocommit=True)
                async with conn:
                    await conn.execute(f"LISTEN {self._notify_channel};")
                    async for _notify in conn.notifies():
                        self._wake_event.set()
                        if self._stop_event.is_set():
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("outbox_listen_fallback_polling", error=str(exc))
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=max(self._poll_interval_seconds, 2.0),
                    )
                except TimeoutError:
                    continue

    async def run_forever(self) -> None:
        """Continuously drain jobs until graceful shutdown is requested."""
        listener_task: asyncio.Task[None] | None = None
        if self._notify_channel and self._database_url:
            listener_task = asyncio.create_task(self._listen_loop())

        try:
            while not self._stop_event.is_set():
                did_work = await self.process_one()
                if not did_work:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(),
                            timeout=self._poll_interval_seconds,
                        )
                    except TimeoutError:
                        continue
        finally:
            if listener_task is not None:
                listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await listener_task


@dataclass
class InMemoryOutboxJob:
    """In-memory representation of an `async_jobs` record."""

    id: UUID
    job_type: str
    idempotency_key: str
    payload: dict[str, Any]
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 3
    run_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryOutboxQueue:
    """Deterministic in-memory outbox queue mirroring PG SKIP LOCKED & idempotency semantics."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, InMemoryOutboxJob] = {}
        self._idempotency_keys: set[str] = set()
        self._handlers: dict[str, JobHandler] = {}

    def register(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    def enqueue(
        self,
        *,
        job_type: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
        max_attempts: int = 3,
    ) -> UUID | None:
        if idempotency_key in self._idempotency_keys:
            return None
        self._idempotency_keys.add(idempotency_key)
        job_id = new_uuid7()
        self._jobs[job_id] = InMemoryOutboxJob(
            id=job_id,
            job_type=job_type,
            idempotency_key=idempotency_key,
            payload={str(k): v for k, v in payload.items()},
            max_attempts=max_attempts,
        )
        return job_id

    def get_job(self, job_id: UUID) -> InMemoryOutboxJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[InMemoryOutboxJob]:
        return list(self._jobs.values())

    async def process_one(self, *, ignore_backoff: bool = False) -> bool:
        now = datetime.now(UTC)
        claimable = [
            job
            for job in self._jobs.values()
            if job.status == "pending" and (ignore_backoff or job.run_at <= now)
        ]
        if not claimable:
            return False
        claimable.sort(key=lambda j: j.created_at)
        job = claimable[0]
        job.status = "running"
        job.attempts += 1

        handler = self._handlers.get(job.job_type)
        if handler is None:
            self._fail(job, f"Unregistered job_type: {job.job_type}")
            return True

        try:
            await handler(dict(job.payload))
            job.status = "succeeded"
            job.last_error = None
        except Exception as exc:
            self._fail(job, str(exc))
        return True

    def _fail(self, job: InMemoryOutboxJob, error: str) -> None:
        job.last_error = error[:1000]
        if job.attempts >= job.max_attempts:
            job.status = "dead"
        else:
            job.status = "pending"
            job.run_at = datetime.now(UTC) + timedelta(seconds=5 * (2**job.attempts))

    async def drain(self, *, max_jobs: int = 100) -> int:
        processed = 0
        while processed < max_jobs and await self.process_one():
            processed += 1
        return processed
