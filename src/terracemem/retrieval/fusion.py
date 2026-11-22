"""Deterministic Reciprocal Rank Fusion (RRF) and L3 memory decay scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from terracemem.domain.models import MemoryType, SemanticMemoryRecord


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[UUID]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[UUID, float]]:
    """Fuse multiple ranked candidate ID lists using weighted RRF.

    Scale-invariant across heterogeneous retrieval engines (cosine similarity vs BM25 vs graph).
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    effective_weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    scores: dict[UUID, float] = {}

    for weight, candidates in zip(effective_weights, ranked_lists, strict=True):
        for rank_1based, item_id in enumerate(candidates, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank_1based)

    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def score_semantic_memory(
    memory: SemanticMemoryRecord,
    *,
    similarity_score: float,
    now: datetime | None = None,
    decay_lambda: float = 0.015,
) -> float:
    """Compute final L3 semantic memory relevance combining similarity, confidence, recency & hits.

    Pinned memories and explicit hard constraints bypass temporal decay so critical rules
    are never evicted merely because they are old.
    """
    ref_time = now or datetime.now(UTC)
    last_touched = memory.last_accessed_at
    if last_touched.tzinfo is None:
        last_touched = last_touched.replace(tzinfo=UTC)
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=UTC)

    age_days = max(0.0, (ref_time - last_touched).total_seconds() / 86400.0)
    effective_lambda = (
        0.0 if (memory.is_pinned or memory.memory_type == MemoryType.CONSTRAINT) else decay_lambda
    )
    recency = math.exp(-effective_lambda * age_days)
    access_boost = min(1.0, math.log1p(memory.access_count) / math.log(20.0))

    clamped_sim = max(0.0, min(1.0, similarity_score))
    return (
        0.55 * clamped_sim
        + 0.20 * memory.confidence
        + 0.15 * recency
        + 0.10 * access_boost
    )
