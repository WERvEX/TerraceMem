"""L2 Episodic Memory manager: session episode archiving and time-blended recall."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from terracemem.db.memory_repository import MemoryRepository
from terracemem.domain.models import (
    EpisodicMemoryRecord,
    ScoredEpisodicMemory,
    SessionMessageRecord,
)
from terracemem.llm.providers import EmbeddingProvider, LLMProvider

logger = structlog.get_logger(__name__)

_DECISION_CUES = ("decided", "switched", "chose", "migrated", "agreed", "决定", "切换", "采用")
_ENTITY_RE = re.compile(r"`([A-Za-z0-9_.-]{2,40})`|\b([A-Z][a-zA-Z0-9]{2,24})\b")


class _EpisodeExtractionPayload(BaseModel):
    summary: str
    key_decisions: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class EpisodicMemoryManager:
    """Archives timestamped session episodes and recalls them by semantic + recency score."""

    def __init__(
        self,
        *,
        repo: MemoryRepository,
        embedder: EmbeddingProvider,
        llm: LLMProvider | None = None,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._llm = llm

    async def archive_messages(
        self,
        messages: list[SessionMessageRecord],
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        session_id: UUID | None = None,
    ) -> EpisodicMemoryRecord:
        """Summarize and persist a batch of session messages as an L2 episodic memory."""
        if not messages:
            raise ValueError("Cannot archive empty message list into episodic memory")

        summary, decisions, entities, importance = await self._extract_episode(messages)
        vecs = await self._embedder.embed_texts([summary])
        embedding = vecs[0] if vecs else None

        episode = EpisodicMemoryRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            session_id=session_id,
            event_start_at=messages[0].created_at,
            event_end_at=messages[-1].created_at,
            summary=summary,
            key_decisions=decisions,
            entities_mentioned=entities,
            importance_score=importance,
            source_seq_from=messages[0].seq_no,
            source_seq_to=messages[-1].seq_no,
        )
        return await self._repo.insert_episode(episode, embedding)

    async def recall(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[ScoredEpisodicMemory]:
        vecs = await self._embedder.embed_texts([query])
        query_emb = vecs[0] if vecs else None
        raw = await self._repo.search_episodes(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            query_embedding=query_emb,
            query_text=query,
            limit=max(limit * 2, 10),
        )
        ref = now or datetime.now(UTC)
        scored: list[ScoredEpisodicMemory] = []
        for ep, sim in raw:
            end_ts = (
                ep.event_end_at
                if ep.event_end_at.tzinfo is not None
                else ep.event_end_at.replace(tzinfo=UTC)
            )
            age_days = max(0.0, (ref - end_ts).total_seconds() / 86400.0)
            recency = math.exp(-0.02 * age_days)
            combined = 0.65 * sim + 0.20 * recency + 0.15 * ep.importance_score
            scored.append(ScoredEpisodicMemory(episode=ep, score=round(combined, 4)))

        scored.sort(key=lambda item: -item.score)
        return scored[:limit]

    async def _extract_episode(
        self,
        messages: list[SessionMessageRecord],
    ) -> tuple[str, list[str], list[str], float]:
        if self._llm is not None and self._llm.is_configured:
            try:
                transcript = "\n".join(f"[{m.role}] {m.content}" for m in messages)
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Summarize this interaction window into a concise episode summary, "
                                "key decisions, mentioned entities, and importance_score [0,1]."
                            ),
                        },
                        {"role": "user", "content": transcript[:3500]},
                    ],
                    _EpisodeExtractionPayload,
                    use_fast_model=True,
                    max_tokens=450,
                )
                p = res.parsed
                return (
                    p.summary.strip(),
                    [d.strip() for d in p.key_decisions if d.strip()][:6],
                    [e.strip() for e in p.entities_mentioned if e.strip()][:8],
                    p.importance_score,
                )
            except Exception as exc:
                logger.warning("l2_episode_llm_fallback", error=str(exc))

        joined = " ".join(m.content.strip() for m in messages if m.content.strip())
        summary = joined[:280] + ("..." if len(joined) > 280 else "")
        decisions: list[str] = []
        for msg in messages:
            for sent in re.split(r"[。！？.\n]+", msg.content):
                s = sent.strip()
                if s and any(cue in s.lower() for cue in _DECISION_CUES):
                    decisions.append(s[:160])

        entities: list[str] = []
        for g1, g2 in _ENTITY_RE.findall(joined):
            ent = (g1 or g2).strip()
            if ent and ent not in {"The", "This", "That", "User", "Assistant"}:
                entities.append(ent)

        importance = 0.75 if decisions else 0.55
        return (
            summary,
            list(dict.fromkeys(decisions))[:6],
            list(dict.fromkeys(entities))[:8],
            importance,
        )
