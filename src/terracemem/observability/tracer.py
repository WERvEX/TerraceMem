"""Structured execution trace recorder persisting spans to PostgreSQL or InMemory store."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from terracemem.db.repositories import TraceRepository
from terracemem.domain.models import ExecutionTraceRecord, TraceSpanRecord
from terracemem.utils.ids import new_uuid7


@dataclass
class SpanCapture:
    """Mutable capture bag populated inside an active trace span."""

    input_meta: dict[str, Any] = field(default_factory=dict)
    output_meta: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TraceRecorder:
    """Records per-step spans and flushes an ExecutionTraceRecord at request end."""

    def __init__(
        self,
        *,
        trace_repo: TraceRepository,
        operation: str,
        tenant_id: str = "default",
        session_id: UUID | None = None,
        channel: Literal["rest", "sse", "mcp", "eval"] = "rest",
        query_text: str | None = None,
        trace_id: UUID | None = None,
    ) -> None:
        self._trace_repo = trace_repo
        self.trace_id = trace_id or new_uuid7()
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._channel: Literal["rest", "sse", "mcp", "eval"] = channel
        self._operation = operation
        self._query_text = query_text
        self._started_monotonic = time.monotonic()
        self._spans: list[TraceSpanRecord] = []
        self._retrieval_retries = 0

    def set_retrieval_retries(self, retries: int) -> None:
        self._retrieval_retries = max(0, retries)

    def record_span(
        self,
        node_name: str,
        latency_ms: int,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        input_meta: dict[str, Any] | None = None,
        output_meta: dict[str, Any] | None = None,
    ) -> TraceSpanRecord:
        span = TraceSpanRecord(
            step_order=len(self._spans) + 1,
            node_name=node_name,
            latency_ms=max(0, latency_ms),
            prompt_tokens=max(0, prompt_tokens),
            completion_tokens=max(0, completion_tokens),
            input_meta=input_meta or {},
            output_meta=output_meta or {},
        )
        self._spans.append(span)
        return span

    @asynccontextmanager
    async def span(
        self,
        node_name: str,
        *,
        input_meta: dict[str, Any] | None = None,
    ) -> AsyncIterator[SpanCapture]:
        capture = SpanCapture(input_meta=input_meta or {})
        t0 = time.monotonic()
        try:
            yield capture
        finally:
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            self.record_span(
                node_name=node_name,
                latency_ms=elapsed_ms,
                prompt_tokens=capture.prompt_tokens,
                completion_tokens=capture.completion_tokens,
                input_meta=capture.input_meta,
                output_meta=capture.output_meta,
            )

    async def finalize(self, final_status: str) -> ExecutionTraceRecord:
        total_ms = max(0, round((time.monotonic() - self._started_monotonic) * 1000))
        total_prompt = sum(s.prompt_tokens for s in self._spans)
        total_completion = sum(s.completion_tokens for s in self._spans)
        record = ExecutionTraceRecord(
            id=self.trace_id,
            tenant_id=self._tenant_id,
            session_id=self._session_id,
            channel=self._channel,
            operation=self._operation,
            query_text=self._query_text,
            final_status=final_status,
            retrieval_retries=self._retrieval_retries,
            total_latency_ms=total_ms,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            spans=list(self._spans),
        )
        await self._trace_repo.save_trace(record)
        return record
