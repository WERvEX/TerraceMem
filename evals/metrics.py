"""Deterministic RAG and Memory evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _dedup_ordered(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of relevant items found in the top-k unique retrieved items."""
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    unique_top = _dedup_ordered(retrieved_ids)[:k]
    hits = len(rel_set & set(unique_top))
    return hits / len(rel_set)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Mean Reciprocal Rank at cutoff k."""
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    for rank, item_id in enumerate(_dedup_ordered(retrieved_ids)[:k], start=1):
        if item_id in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Binary Normalized Discounted Cumulative Gain at cutoff k."""
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    dcg = 0.0
    for idx, item_id in enumerate(_dedup_ordered(retrieved_ids)[:k]):
        if item_id in rel_set:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def citation_validity(cited_ids: Sequence[str], allowed_evidence_ids: Sequence[str]) -> float:
    """Return fraction of cited IDs present in allowed evidence (0.0 if none cited)."""
    if not cited_ids:
        return 0.0
    allowed = set(allowed_evidence_ids)
    valid_count = sum(1 for cid in cited_ids if cid in allowed)
    return valid_count / len(cited_ids)


def abstention_f1(predictions: Sequence[tuple[bool, bool]]) -> tuple[float, float]:
    """Compute (abstention_f1, false_answer_rate) from (is_unanswerable, did_abstain) pairs.

    false_answer_rate (FAR): fraction of unanswerable questions that were answered anyway.
    """
    tp = fp = fn = 0
    unanswerable_total = 0
    unanswerable_answered = 0

    for is_unanswerable, did_abstain in predictions:
        if is_unanswerable:
            unanswerable_total += 1
            if did_abstain:
                tp += 1
            else:
                fn += 1
                unanswerable_answered += 1
        elif did_abstain:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = (unanswerable_answered / unanswerable_total) if unanswerable_total > 0 else 0.0
    return (f1, far)


def percentile_ms(samples: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile (e.g. 50 or 95) over latency samples."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def prompt_token_reduction_pct(baseline_tokens: float, method_tokens: float) -> float:
    """Percentage prompt tokens saved compared to M1 Full Context baseline."""
    if baseline_tokens <= 0:
        return 0.0
    return ((baseline_tokens - method_tokens) / baseline_tokens) * 100.0


def false_invalidation_rate(protected_total: int, wrongly_invalidated: int) -> float:
    """Fraction of protected/hypothetical/backfill memories wrongly overwritten."""
    if protected_total <= 0:
        return 0.0
    return wrongly_invalidated / protected_total
