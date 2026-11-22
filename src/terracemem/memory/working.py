"""L1 Working Memory manager: token-budgeted context packing & incremental compaction."""

from __future__ import annotations

from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from terracemem.config import Settings
from terracemem.db.memory_repository import MemoryRepository
from terracemem.domain.models import (
    SessionCompactionRecord,
    SessionMessageRecord,
    WorkingMemoryContext,
)
from terracemem.llm.providers import LLMProvider
from terracemem.utils.tokens import estimate_tokens

logger = structlog.get_logger(__name__)


class _CompactionPayload(BaseModel):
    summary_text: str
    preserved_facts: list[str] = Field(default_factory=list)


class WorkingMemoryManager:
    """Manages L1 working context within strict token budgets without losing hard constraints."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: MemoryRepository,
        llm: LLMProvider | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._llm = llm

    async def build_context(self, session_id: UUID) -> WorkingMemoryContext | None:
        """Assemble L1 working memory: pinned constraints + latest compaction + recent turns."""
        session = await self._repo.get_session(session_id)
        if session is None:
            return None

        compaction = await self._repo.get_latest_compaction(session_id)
        after_seq = compaction.compacted_up_to_seq if compaction is not None else 0
        messages = await self._repo.list_messages(session_id, after_seq=after_seq)

        budget = session.token_budget
        # Reserve pinned constraints first: hard rules are never evicted
        pinned_tokens = sum(estimate_tokens(c) for c in session.pinned_constraints)
        used = pinned_tokens
        remaining = max(0, budget - used)

        compaction_summary: str | None = None
        preserved_facts: list[str] = []
        if compaction is not None:
            comp_tokens = estimate_tokens(compaction.summary_text)
            if comp_tokens <= remaining:
                compaction_summary = compaction.summary_text
                used += comp_tokens
                remaining -= comp_tokens
            elif remaining > 24:
                compaction_summary = compaction.summary_text[: max(40, remaining * 2)]
                used += estimate_tokens(compaction_summary)
                remaining = max(0, budget - used)

            for fact in compaction.preserved_facts:
                ftok = estimate_tokens(fact)
                if ftok <= remaining:
                    preserved_facts.append(fact)
                    used += ftok
                    remaining -= ftok

        kept_reversed: list[SessionMessageRecord] = []
        dropped = 0
        for msg in reversed(messages):
            mtok = msg.token_count or estimate_tokens(msg.content)
            if mtok <= remaining:
                kept_reversed.append(msg)
                used += mtok
                remaining -= mtok
            else:
                dropped += 1

        kept_reversed.reverse()
        return WorkingMemoryContext(
            session_id=session.id,
            token_budget=budget,
            used_tokens=used,
            pinned_constraints=list(session.pinned_constraints),
            compaction_summary=compaction_summary,
            preserved_facts=preserved_facts,
            recent_messages=kept_reversed,
            dropped_message_count=dropped,
        )

    async def maybe_compact(
        self,
        session_id: UUID,
        *,
        keep_recent_turns: int = 4,
        trigger_ratio: float = 0.75,
    ) -> SessionCompactionRecord | None:
        """Incrementally compact older messages once uncompacted tokens cross trigger_ratio."""
        session = await self._repo.get_session(session_id)
        if session is None:
            return None

        prev = await self._repo.get_latest_compaction(session_id)
        after_seq = prev.compacted_up_to_seq if prev is not None else 0
        messages = await self._repo.list_messages(session_id, after_seq=after_seq)
        if len(messages) <= keep_recent_turns:
            return None

        pinned_tokens = sum(estimate_tokens(c) for c in session.pinned_constraints)
        prev_tokens = prev.token_count if prev is not None else 0
        uncompacted_tokens = sum(m.token_count or estimate_tokens(m.content) for m in messages)
        if pinned_tokens + prev_tokens + uncompacted_tokens < int(
            session.token_budget * trigger_ratio
        ):
            return None

        to_compact = messages[:-keep_recent_turns] if keep_recent_turns > 0 else messages
        if not to_compact:
            return None

        summary_text, preserved = await self._summarize_messages(prev, to_compact)
        record = SessionCompactionRecord(
            session_id=session_id,
            compacted_up_to_seq=to_compact[-1].seq_no,
            summary_text=summary_text,
            preserved_facts=preserved,
            token_count=estimate_tokens(summary_text)
            + sum(estimate_tokens(f) for f in preserved),
        )
        return await self._repo.save_compaction(record)

    async def _summarize_messages(
        self,
        prev: SessionCompactionRecord | None,
        messages: list[SessionMessageRecord],
    ) -> tuple[str, list[str]]:
        if self._llm is not None and self._llm.is_configured:
            try:
                transcript = "\n".join(f"[{m.role}] {m.content}" for m in messages)
                prev_block = f"Previous summary: {prev.summary_text}\n" if prev else ""
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Compact conversation history into a concise rolling summary "
                                "and extract concrete decisions/facts to preserve."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"{prev_block}New turns:\n{transcript[:4000]}",
                        },
                    ],
                    _CompactionPayload,
                    use_fast_model=True,
                    max_tokens=500,
                )
                return (
                    res.parsed.summary_text.strip(),
                    [f.strip() for f in res.parsed.preserved_facts if f.strip()][:8],
                )
            except Exception as exc:
                logger.warning("l1_compaction_llm_fallback", error=str(exc))

        snippets = [
            f"{m.role}: {m.content.strip()[:120]}"
            for m in messages
            if m.content.strip()
        ]
        prev_prefix = f"{prev.summary_text} | " if prev and prev.summary_text else ""
        summary_text = f"{prev_prefix}" + " ; ".join(snippets[:6])
        if len(summary_text) > 600:
            summary_text = summary_text[:597] + "..."
        preserved = list(prev.preserved_facts if prev else [])
        for msg in messages:
            text = msg.content.strip()
            if text and len(preserved) < 8 and text not in preserved:
                preserved.append(text[:140])
        return (summary_text, preserved[:8])
