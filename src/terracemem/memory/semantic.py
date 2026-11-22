"""L3 Bi-temporal Semantic Memory manager with conflict resolution & 4-layer guardrails."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from terracemem.config import Settings
from terracemem.db.memory_repository import MemoryRepository
from terracemem.domain.models import (
    CandidateMemoryFact,
    ConflictJudgment,
    EpistemicModality,
    MemoryConflictAudit,
    MemoryResolutionAction,
    MemoryTimelineResult,
    MemoryType,
    ScoredSemanticMemory,
    SemanticMemoryRecord,
    utc_now,
)
from terracemem.llm.providers import EmbeddingProvider, LLMProvider, cosine_similarity
from terracemem.memory.guardrails import GuardrailDecision, evaluate_conflict_guardrails
from terracemem.retrieval.fusion import score_semantic_memory

logger = structlog.get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_SLOT_TAG_RE = re.compile(r"\[slot:([a-zA-Z0-9_.-]+)\]")

_HYPOTHETICAL_CUES = (
    "maybe ",
    "what if",
    "suppose ",
    "hypothetically",
    "might ",
    "could consider",
    "perhaps ",
    "假设",
    "如果",
    "假如",
    "或许",
    "说不定",
    "考虑以后",
)
_REPORTED_CUES = ("someone said", "rumor", "heard that", "allegedly", "听说", "据说", "传闻")
_QUESTION_PREFIXES = ("what ", "why ", "how ", "when ", "where ", "could ", "should ", "would ")
_CONSTRAINT_CUES = (
    "must ",
    "never ",
    "always ",
    "do not ",
    "don't ",
    "forbid",
    "prohibit",
    "必须",
    "禁止",
    "绝不",
    "严禁",
    "不得",
)
_PREFERENCE_CUES = (
    "prefer ",
    "favorite",
    "like using",
    "love using",
    "dislike",
    "偏好",
    "喜欢用",
    "习惯用",
)
_PROCEDURE_CUES = ("workflow", "procedure", "how to ", "deploy by", "步骤", "流程")

_SLOT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "project.db",
        ("postgresql", "postgres", "mysql", "sqlite", "duckdb", "mongodb", "database", "数据库"),
    ),
    ("project.cache", ("redis", "valkey", "memcached", "缓存")),
    (
        "project.framework",
        ("fastapi", "django", "flask", "react", "nextjs", "framework", "框架"),
    ),
    ("user.language", ("python", "typescript", "rust", "golang", "java", "language", "语言")),
    ("user.editor", ("vscode", "neovim", "vim", "cursor", "emacs", "intellij", "编辑器")),
    (
        "user.location",
        ("live in", "lives in", "moved to", "relocated to", "based in", "住在", "搬到", "现居"),
    ),
    ("user.timezone", ("timezone", "utc+", "utc-", "时区")),
    ("user.role", ("engineer", "architect", "manager", "职位", "担任")),
)


class _ExtractedFactItem(BaseModel):
    memory_type: MemoryType = MemoryType.FACT
    slot_key: str | None = None
    fact_text: str
    evidence_quote: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    modality: EpistemicModality = EpistemicModality.ASSERTED


class _ExtractedFactsPayload(BaseModel):
    facts: list[_ExtractedFactItem] = Field(default_factory=list)


class _ConflictJudgePayload(BaseModel):
    action: MemoryResolutionAction
    reason: str = ""
    merged_fact_text: str | None = None


def _infer_slot_key(sentence: str) -> str | None:
    tag_match = _SLOT_TAG_RE.search(sentence)
    if tag_match:
        return tag_match.group(1).lower()
    lower = sentence.lower()
    for slot_key, cues in _SLOT_RULES:
        if any(cue in lower for cue in cues):
            return slot_key
    return None


def _classify_modality(sentence: str) -> EpistemicModality:
    s = sentence.strip()
    lower = s.lower()
    if s.endswith(("?", "？")) or lower.startswith(_QUESTION_PREFIXES) or "吗" in s:
        return EpistemicModality.QUESTION
    if any(cue in lower for cue in _HYPOTHETICAL_CUES):
        return EpistemicModality.HYPOTHETICAL
    if any(cue in lower for cue in _REPORTED_CUES):
        return EpistemicModality.REPORTED
    return EpistemicModality.ASSERTED


def _classify_memory_type(sentence: str) -> MemoryType:
    lower = sentence.lower()
    if any(cue in lower for cue in _CONSTRAINT_CUES):
        return MemoryType.CONSTRAINT
    if any(cue in lower for cue in _PREFERENCE_CUES):
        return MemoryType.PREFERENCE
    if any(cue in lower for cue in _PROCEDURE_CUES):
        return MemoryType.PROCEDURE
    return MemoryType.FACT


class SemanticMemoryManager:
    """Manages L3 atomic facts, bi-temporal conflict resolution, guardrails & time-travel."""

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
        self._embedder = embedder
        self._llm = llm

    async def extract_candidate_facts(
        self,
        text: str,
        *,
        event_time: datetime | None = None,
    ) -> list[CandidateMemoryFact]:
        """Extract structured atomic facts with epistemic modality & slot_key."""
        resolved_time = event_time or utc_now()
        if self._llm is not None and self._llm.is_configured:
            try:
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Extract atomic user/project facts, preferences, constraints, or "
                                "procedures. Classify modality accurately (asserted vs "
                                "hypothetical/question/reported) and infer dot slot_key like "
                                "'project.db' or 'user.location' when applicable."
                            ),
                        },
                        {"role": "user", "content": text[:3500]},
                    ],
                    _ExtractedFactsPayload,
                    use_fast_model=True,
                    max_tokens=700,
                )
                candidates = [
                    CandidateMemoryFact(
                        memory_type=item.memory_type,
                        slot_key=item.slot_key,
                        fact_text=item.fact_text.strip(),
                        evidence_quote=(item.evidence_quote or item.fact_text).strip(),
                        confidence=item.confidence,
                        modality=item.modality,
                        event_time=resolved_time,
                    )
                    for item in res.parsed.facts
                    if item.fact_text.strip()
                ]
                if candidates:
                    return candidates
            except Exception as exc:
                logger.warning("l3_extract_llm_fallback", error=str(exc))

        return self._extract_deterministic(text, event_time=resolved_time)

    def _extract_deterministic(
        self,
        text: str,
        *,
        event_time: datetime,
    ) -> list[CandidateMemoryFact]:
        raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
        candidates: list[CandidateMemoryFact] = []
        for sent in raw_parts:
            if len(sent) < 4:
                continue
            cleaned = _SLOT_TAG_RE.sub("", sent).strip()
            modality = _classify_modality(cleaned)
            mem_type = _classify_memory_type(cleaned)
            slot_key = _infer_slot_key(sent)
            conf = 0.88 if modality == EpistemicModality.ASSERTED else 0.72
            if any(w in cleaned.lower() for w in ("not sure", "uncertain", "大概", "不太确定")):
                conf = 0.55
            candidates.append(
                CandidateMemoryFact(
                    memory_type=mem_type,
                    slot_key=slot_key,
                    fact_text=cleaned,
                    evidence_quote=cleaned,
                    confidence=conf,
                    modality=modality,
                    event_time=event_time,
                )
            )
        return candidates

    async def commit_candidates(
        self,
        candidates: list[CandidateMemoryFact],
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        source_episode_id: UUID | None = None,
        is_pinned: bool = False,
    ) -> list[MemoryConflictAudit]:
        """Resolve conflicts via dual recall + LLM judgment + 4-layer guardrails."""
        audits: list[MemoryConflictAudit] = []
        if not candidates:
            return audits

        embeddings = await self._embedder.embed_texts([c.fact_text for c in candidates])

        for candidate, cand_emb in zip(candidates, embeddings, strict=True):
            existing_list = await self._repo.find_conflict_candidates(
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                slot_key=candidate.slot_key,
                query_embedding=cand_emb,
                top_k=5,
            )

            # Never store unasserted/hypothetical/question statements as active semantic truths
            if candidate.modality != EpistemicModality.ASSERTED:
                target_id = existing_list[0].id if existing_list else None
                audits.append(
                    MemoryConflictAudit(
                        candidate_fact=candidate.fact_text,
                        slot_key=candidate.slot_key,
                        target_memory_id=target_id,
                        proposed_action=MemoryResolutionAction.NOOP,
                        effective_action=MemoryResolutionAction.NOOP,
                        guardrail_decision=GuardrailDecision.REJECT.value,
                        reason=f"Non-asserted modality '{candidate.modality}' blocked.",
                    )
                )
                continue

            if not existing_list:
                if candidate.confidence < self._settings.memory_contradiction_min_confidence * 0.7:
                    audits.append(
                        MemoryConflictAudit(
                            candidate_fact=candidate.fact_text,
                            slot_key=candidate.slot_key,
                            proposed_action=MemoryResolutionAction.NOOP,
                            effective_action=MemoryResolutionAction.NOOP,
                            guardrail_decision=GuardrailDecision.DOWNGRADE_TO_NOOP.value,
                            reason="Low confidence new fact ignored.",
                        )
                    )
                    continue
                new_rec = self._build_record(
                    candidate,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    namespace=namespace,
                    is_pinned=is_pinned,
                )
                await self._repo.insert_semantic_memory(
                    new_rec,
                    cand_emb,
                    source_episode_id=source_episode_id,
                )
                audits.append(
                    MemoryConflictAudit(
                        candidate_fact=candidate.fact_text,
                        slot_key=candidate.slot_key,
                        proposed_action=MemoryResolutionAction.UNRELATED,
                        effective_action=MemoryResolutionAction.UNRELATED,
                        guardrail_decision=GuardrailDecision.ALLOW_COMMIT.value,
                        reason="Inserted new L3 semantic memory.",
                        new_memory_id=new_rec.id,
                    )
                )
                continue

            handled = False
            for target in existing_list:
                judgment = await self._judge_conflict(candidate, cand_emb, target)
                if judgment.action == MemoryResolutionAction.UNRELATED:
                    continue

                outcome = evaluate_conflict_guardrails(
                    candidate,
                    target,
                    judgment,
                    min_contradiction_confidence=self._settings.memory_contradiction_min_confidence,
                )
                if outcome.decision != GuardrailDecision.ALLOW_COMMIT:
                    audits.append(
                        MemoryConflictAudit(
                            candidate_fact=candidate.fact_text,
                            slot_key=candidate.slot_key,
                            target_memory_id=target.id,
                            proposed_action=judgment.action,
                            effective_action=outcome.effective_action,
                            guardrail_decision=outcome.decision.value,
                            reason=outcome.reason,
                        )
                    )
                    handled = True
                    break

                if outcome.effective_action == MemoryResolutionAction.NOOP:
                    audits.append(
                        MemoryConflictAudit(
                            candidate_fact=candidate.fact_text,
                            slot_key=candidate.slot_key,
                            target_memory_id=target.id,
                            proposed_action=judgment.action,
                            effective_action=MemoryResolutionAction.NOOP,
                            guardrail_decision=GuardrailDecision.ALLOW_COMMIT.value,
                            reason=judgment.reason or "Duplicate fact already active.",
                        )
                    )
                    handled = True
                    break

                new_fact_text = (
                    judgment.merged_fact_text.strip()
                    if outcome.effective_action == MemoryResolutionAction.MERGE
                    and judgment.merged_fact_text
                    else candidate.fact_text
                )
                new_cand = candidate.model_copy(update={"fact_text": new_fact_text})
                new_rec = self._build_record(
                    new_cand,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    namespace=namespace,
                    is_pinned=is_pinned,
                )
                new_emb = (
                    (await self._embedder.embed_texts([new_fact_text]))[0]
                    if new_fact_text != candidate.fact_text
                    else cand_emb
                )
                inv_reason = judgment.reason or "Superseded by newer semantic fact"
                await self._repo.supersede_semantic_memory(
                    old_id=target.id,
                    new_record=new_rec,
                    new_embedding=new_emb,
                    invalidation_reason=inv_reason,
                    valid_to=candidate.event_time,
                    source_episode_id=source_episode_id,
                )
                audits.append(
                    MemoryConflictAudit(
                        candidate_fact=candidate.fact_text,
                        slot_key=candidate.slot_key,
                        target_memory_id=target.id,
                        proposed_action=judgment.action,
                        effective_action=outcome.effective_action,
                        guardrail_decision=GuardrailDecision.ALLOW_COMMIT.value,
                        reason=inv_reason,
                        new_memory_id=new_rec.id,
                    )
                )
                handled = True
                break

            if not handled:
                new_rec = self._build_record(
                    candidate,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    namespace=namespace,
                    is_pinned=is_pinned,
                )
                await self._repo.insert_semantic_memory(
                    new_rec,
                    cand_emb,
                    source_episode_id=source_episode_id,
                )
                audits.append(
                    MemoryConflictAudit(
                        candidate_fact=candidate.fact_text,
                        slot_key=candidate.slot_key,
                        proposed_action=MemoryResolutionAction.UNRELATED,
                        effective_action=MemoryResolutionAction.UNRELATED,
                        guardrail_decision=GuardrailDecision.ALLOW_COMMIT.value,
                        reason="Inserted distinct L3 semantic memory.",
                        new_memory_id=new_rec.id,
                    )
                )

        return audits

    async def recall(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        query: str,
        limit: int = 8,
        now: datetime | None = None,
    ) -> list[ScoredSemanticMemory]:
        vecs = await self._embedder.embed_texts([query])
        query_emb = vecs[0] if vecs else None
        raw = await self._repo.search_active_semantic_memories(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            query_embedding=query_emb,
            query_text=query,
            limit=max(limit * 2, 16),
        )
        scored = [
            ScoredSemanticMemory(
                memory=rec,
                similarity_score=round(sim, 4),
                final_score=round(
                    score_semantic_memory(
                        rec,
                        similarity_score=sim,
                        now=now,
                        decay_lambda=self._settings.memory_decay_lambda,
                    ),
                    4,
                ),
            )
            for rec, sim in raw
        ]
        scored.sort(key=lambda item: -item.final_score)
        return scored[:limit]

    async def timeline(
        self,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        slot_key: str | None = None,
        as_of: datetime | None = None,
    ) -> MemoryTimelineResult:
        return await self._repo.get_timeline(
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
        tenant_id: str,
        user_id: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord | None:
        return await self._repo.set_memory_pinned(
            memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            is_pinned=is_pinned,
        )

    def _build_record(
        self,
        candidate: CandidateMemoryFact,
        *,
        tenant_id: str,
        user_id: str,
        namespace: str,
        is_pinned: bool,
    ) -> SemanticMemoryRecord:
        return SemanticMemoryRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace=namespace,
            memory_type=candidate.memory_type,
            slot_key=candidate.slot_key,
            fact_text=candidate.fact_text,
            evidence_quote=candidate.evidence_quote,
            confidence=candidate.confidence,
            is_pinned=is_pinned or candidate.memory_type == MemoryType.CONSTRAINT,
            valid_from=candidate.event_time,
            recorded_at=utc_now(),
            last_accessed_at=utc_now(),
            importance_score=(
                0.9 if (is_pinned or candidate.memory_type == MemoryType.CONSTRAINT) else 0.7
            ),
        )

    async def _judge_conflict(
        self,
        candidate: CandidateMemoryFact,
        cand_emb: list[float],
        existing: SemanticMemoryRecord,
    ) -> ConflictJudgment:
        if self._llm is not None and self._llm.is_configured:
            try:
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Compare a new candidate memory fact against an existing active "
                                "memory. Return action in {noop, merge, contradict, unrelated}. "
                                "Use 'contradict' when the new fact supersedes/replaces the old "
                                "fact, 'merge' when both combine cleanly, 'noop' if identical, "
                                "and 'unrelated' if about different topics."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Existing [{existing.slot_key or 'none'}]: {existing.fact_text}\n"
                                f"Candidate [{candidate.slot_key or 'none'}]: {candidate.fact_text}"
                            ),
                        },
                    ],
                    _ConflictJudgePayload,
                    use_fast_model=True,
                    max_tokens=250,
                )
                return ConflictJudgment(
                    target_memory_id=existing.id,
                    action=res.parsed.action,
                    reason=res.parsed.reason.strip(),
                    merged_fact_text=res.parsed.merged_fact_text,
                )
            except Exception as exc:
                logger.warning("l3_conflict_judge_llm_fallback", error=str(exc))

        # Deterministic relation judgment fallback
        norm_new = candidate.fact_text.strip().lower()
        norm_old = existing.fact_text.strip().lower()
        if norm_new == norm_old:
            return ConflictJudgment(
                target_memory_id=existing.id,
                action=MemoryResolutionAction.NOOP,
                reason="Exact fact already recorded.",
            )

        if candidate.slot_key and existing.slot_key and candidate.slot_key == existing.slot_key:
            return ConflictJudgment(
                target_memory_id=existing.id,
                action=MemoryResolutionAction.CONTRADICT,
                reason=f"New asserted value supersedes slot '{existing.slot_key}'.",
            )

        existing_emb = (await self._embedder.embed_texts([existing.fact_text]))[0]
        sim = cosine_similarity(cand_emb, existing_emb)
        if sim >= 0.92:
            return ConflictJudgment(
                target_memory_id=existing.id,
                action=MemoryResolutionAction.NOOP,
                reason="Semantically equivalent fact already stored.",
            )
        return ConflictJudgment(
            target_memory_id=existing.id,
            action=MemoryResolutionAction.UNRELATED,
            reason="Distinct topic.",
        )
