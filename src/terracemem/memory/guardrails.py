"""Four-layer deterministic anti-misfire guardrails for L3 bi-temporal conflict resolution.

Prevents noisy LLM judgments from silently corrupting or invalidating stable user memories:
1. Modality & confidence gate: hypothetical/question statements or low-confidence extractions
   cannot trigger CONTRADICT or MERGE overwrites.
2. Evidence requirement: CONTRADICT must include both a non-empty reason and an evidence quote.
3. Temporal ordering guard: historical backfills (event_time < existing.valid_from) cannot
   supersede newer established facts.
4. Pinned constraint protection: user-pinned memories cannot be silently invalidated by
   background extraction; they are routed to `pending_confirmation` instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from terracemem.domain.models import (
    CandidateMemoryFact,
    ConflictJudgment,
    EpistemicModality,
    MemoryResolutionAction,
    SemanticMemoryRecord,
)


class GuardrailDecision(StrEnum):
    ALLOW_COMMIT = "allow_commit"
    DOWNGRADE_TO_NOOP = "downgrade_to_noop"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECT = "reject"


class GuardrailOutcome(BaseModel):
    """Deterministic decision returned before mutating bi-temporal memory tables."""

    decision: GuardrailDecision
    effective_action: MemoryResolutionAction
    reason: str


def _ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def evaluate_conflict_guardrails(
    candidate: CandidateMemoryFact,
    existing: SemanticMemoryRecord,
    judgment: ConflictJudgment,
    *,
    min_contradiction_confidence: float = 0.75,
) -> GuardrailOutcome:
    """Evaluate deterministic safety guardrails before executing a memory state transition."""
    if judgment.action in (MemoryResolutionAction.NOOP, MemoryResolutionAction.UNRELATED):
        return GuardrailOutcome(
            decision=GuardrailDecision.ALLOW_COMMIT,
            effective_action=judgment.action,
            reason="Non-destructive action permitted.",
        )

    # Guardrail 1: Only explicitly asserted, sufficiently confident facts may modify existing facts
    if candidate.modality != EpistemicModality.ASSERTED:
        return GuardrailOutcome(
            decision=GuardrailDecision.REJECT,
            effective_action=MemoryResolutionAction.NOOP,
            reason=(
                f"Candidate modality '{candidate.modality}' is not asserted; "
                "refusing to overwrite existing memory."
            ),
        )

    if candidate.confidence < min_contradiction_confidence:
        return GuardrailOutcome(
            decision=GuardrailDecision.DOWNGRADE_TO_NOOP,
            effective_action=MemoryResolutionAction.NOOP,
            reason=(
                f"Candidate confidence {candidate.confidence:.2f} below threshold "
                f"{min_contradiction_confidence:.2f}; refusing silent overwrite."
            ),
        )

    # Guardrail 2: CONTRADICT requires explicit rationale and verbatim quote evidence
    if judgment.action == MemoryResolutionAction.CONTRADICT and (
        not judgment.reason.strip()
        or not (candidate.evidence_quote and candidate.evidence_quote.strip())
    ):
        return GuardrailOutcome(
            decision=GuardrailDecision.REJECT,
            effective_action=MemoryResolutionAction.NOOP,
            reason="CONTRADICT requires non-empty invalidation reason and evidence_quote.",
        )

    # Guardrail 3: Chronological ordering guard prevents old backfills overriding newer truths
    if _ensure_utc(candidate.event_time) < _ensure_utc(existing.valid_from):
        return GuardrailOutcome(
            decision=GuardrailDecision.REJECT,
            effective_action=MemoryResolutionAction.NOOP,
            reason="Candidate event_time precedes existing memory valid_from.",
        )

    # Guardrail 4: Pinned hard constraints must never be silently invalidated in background jobs
    if existing.is_pinned:
        return GuardrailOutcome(
            decision=GuardrailDecision.PENDING_CONFIRMATION,
            effective_action=MemoryResolutionAction.NOOP,
            reason="Target memory is pinned; conflict flagged for explicit user confirmation.",
        )

    return GuardrailOutcome(
        decision=GuardrailDecision.ALLOW_COMMIT,
        effective_action=judgment.action,
        reason="All four bi-temporal anti-misfire guardrails passed.",
    )
