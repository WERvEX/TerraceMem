"""Typed asynchronous Agentic RAG state machine with budgets, CRAG & citation verify."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

import jieba
import structlog
from pydantic import BaseModel, Field

from terracemem.config import Settings
from terracemem.db.repositories import DocumentRepository, RetrievalMode, TraceRepository
from terracemem.domain.models import (
    ClaimWithCitations,
    MemoryRecallResult,
    RagDecisionStatus,
    RagResponse,
    RetrievalFilter,
    RetrievedEvidence,
    SufficiencyVerdict,
)
from terracemem.llm.providers import LLMProvider
from terracemem.memory.service import MemoryService
from terracemem.observability.tracer import TraceRecorder
from terracemem.rag.budget import ExecutionBudget
from terracemem.rag.verifier import CitationVerifier
from terracemem.retrieval.service import RetrievalService, RetrievalStageResult

logger = structlog.get_logger(__name__)

EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

_QUERY_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "about",
        "tell",
        "me",
        "please",
        "explain",
        "什么",
        "是什么",
        "哪些",
        "如何",
        "怎么",
        "为什么",
        "请问",
        "吗",
        "呢",
        "的",
        "了",
        "在",
        "是",
        "和",
    }
)
_PURE_MEMORY_CUES = (
    "my preference",
    "what did i",
    "do you remember",
    "my favorite",
    "what database did we choose",
    "我的偏好",
    "我之前",
    "还记得",
    "我喜欢",
    "我住在哪里",
)
_OUTLINE_CUES = (
    "outline",
    "table of contents",
    "structure",
    "sections",
    "chapters",
    "目录",
    "章节",
    "结构",
    "大纲",
)


class RagState(StrEnum):
    """Explicit finite states for the Agentic RAG orchestrator."""

    INTAKE_AND_FILTER = "intake_and_filter"
    QUERY_ROUTER = "query_router"
    MEMORY_RECALL = "memory_recall"
    OUTLINE_NAVIGATE = "outline_navigate"
    HYBRID_RETRIEVE = "hybrid_retrieve"
    RERANK_AND_EXPAND = "rerank_and_expand"
    SUFFICIENCY_GATE = "sufficiency_gate"
    CORRECTIVE_REWRITE = "corrective_rewrite"
    GROUNDED_GENERATE = "grounded_generate"
    CITATION_VERIFY = "citation_verify"
    STRICT_ABSTAIN = "strict_abstain"
    DONE = "done"


class RagRequest(BaseModel):
    """Input specification for an Agentic RAG execution."""

    question: str
    tenant_id: str = "default"
    namespace: str = "default"
    user_id: str | None = None
    session_id: UUID | None = None
    document_ids: list[UUID] | None = None
    section_ids: list[UUID] | None = None
    effective_after: datetime | None = None
    effective_before: datetime | None = None
    only_latest: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    retrieval_mode: RetrievalMode = "hybrid"
    enable_rerank: bool = True
    enable_parent_expand: bool = True
    enable_graph_expansion: bool | None = None
    enable_outline_navigation: bool | None = None
    enable_crag: bool = True
    strict_citation_mode: bool | None = None
    channel: Literal["rest", "sse", "mcp", "eval"] = "rest"


class _SufficiencyDraft(BaseModel):
    verdict: SufficiencyVerdict
    missing_aspects: str = ""
    suggested_rewrite: str = ""


class _RewriteDraft(BaseModel):
    rewritten_query: str


class _DraftClaim(BaseModel):
    statement: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


class _GroundedAnswerDraft(BaseModel):
    cannot_answer: bool = False
    abstention_reason: str | None = None
    answer: str = ""
    claims: list[_DraftClaim] = Field(default_factory=list)


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in jieba.cut_for_search(text):
        tok = raw.strip().lower()
        if len(tok) <= 1 or tok in _QUERY_STOPWORDS:
            continue
        if all(ch in "`~!@#$%^&*()_+-=[]{}|;':\",./<>?，。！？；：（）【】 " for ch in tok):
            continue
        tokens.append(tok)
    return list(dict.fromkeys(tokens))


def _token_coverage(query: str, evidence: list[RetrievedEvidence]) -> float:
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 0.0
    combined = " ".join((ev.expanded_text or ev.raw_text).lower() for ev in evidence[:3])
    matched = sum(1 for tok in q_tokens if tok in combined)
    return matched / len(q_tokens)


def _parse_uuid_safe(raw_id: str) -> UUID:
    try:
        return UUID(raw_id.strip())
    except ValueError:
        # Sentinel non-matching UUID so CitationVerifier catches malformed/fabricated citations
        return UUID(int=0)


class AgenticRagEngine:
    """Explicit Typed Async FSM executing trustworthy retrieval, CRAG & citation checks."""

    def __init__(
        self,
        *,
        settings: Settings,
        doc_repo: DocumentRepository,
        retrieval_service: RetrievalService,
        trace_repo: TraceRepository,
        llm: LLMProvider | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._settings = settings
        self._doc_repo = doc_repo
        self._retrieval = retrieval_service
        self._trace_repo = trace_repo
        self._llm = llm
        self._memory = memory_service
        self._verifier = CitationVerifier()

    async def run(
        self,
        request: RagRequest,
        *,
        event_callback: EventCallback | None = None,
    ) -> RagResponse:
        """Execute the Agentic RAG state machine from Intake through CitationVerify/Abstain."""
        recorder = TraceRecorder(
            trace_repo=self._trace_repo,
            operation="rag.ask",
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            channel=request.channel,
            query_text=request.question,
        )
        budget = ExecutionBudget(
            max_retrieval_retries=self._settings.max_retrieval_retries,
            max_state_transitions=self._settings.max_state_transitions,
            max_llm_calls=self._settings.max_llm_calls,
            timeout_seconds=self._settings.request_timeout_seconds,
        )
        strict_mode = (
            request.strict_citation_mode
            if request.strict_citation_mode is not None
            else self._settings.strict_citation_mode
        )

        async def emit(event_type: str, payload: dict[str, Any]) -> None:
            if event_callback is not None:
                await event_callback(event_type, payload)

        state = RagState.INTAKE_AND_FILTER
        current_query = request.question.strip()
        current_filter = RetrievalFilter(
            tenant_id=request.tenant_id,
            namespace=request.namespace,
            document_ids=request.document_ids,
            section_ids=request.section_ids,
            effective_after=request.effective_after,
            effective_before=request.effective_before,
            only_latest=request.only_latest,
        )

        memory_ctx: MemoryRecallResult | None = None
        memory_text = ""
        pure_memory_only = False
        outline_narrowed = False
        evidence: list[RetrievedEvidence] = []
        last_retrieval: RetrievalStageResult | None = None
        suggested_rewrite = ""
        draft: _GroundedAnswerDraft | None = None
        verified_claims: list[ClaimWithCitations] = []
        abstention_reason: str | None = None
        final_response: RagResponse | None = None

        while state != RagState.DONE:
            await emit("state", {"state": state.value, "retries": budget.retrieval_retries})

            if state != RagState.STRICT_ABSTAIN and not budget.record_transition():
                abstention_reason = (
                    "Execution timeout reached."
                    if budget.is_timed_out
                    else "Maximum FSM state transitions exceeded."
                )
                state = RagState.STRICT_ABSTAIN
                continue

            if state == RagState.INTAKE_AND_FILTER:
                async with recorder.span(
                    RagState.INTAKE_AND_FILTER.value,
                    input_meta={"tenant_id": request.tenant_id, "namespace": request.namespace},
                ) as cap:
                    if not current_query:
                        abstention_reason = "Empty question cannot be answered."
                        cap.output_meta = {"valid": False, "reason": abstention_reason}
                        state = RagState.STRICT_ABSTAIN
                        continue
                    budget.register_query(current_query)
                    cap.output_meta = {"valid": True, "mode": request.retrieval_mode}
                state = RagState.QUERY_ROUTER

            elif state == RagState.QUERY_ROUTER:
                async with recorder.span(
                    RagState.QUERY_ROUTER.value,
                    input_meta={"query": current_query},
                ) as cap:
                    has_memory_scope = self._memory is not None and bool(
                        request.user_id or request.session_id
                    )
                    q_lower = current_query.lower()
                    is_pure_memory_cue = has_memory_scope and any(
                        cue in q_lower for cue in _PURE_MEMORY_CUES
                    )
                    wants_outline = (
                        request.enable_outline_navigation is True
                        or (
                            request.enable_outline_navigation is None
                            and self._settings.enable_write_time_enrichment
                            and request.section_ids is None
                            and any(cue in q_lower for cue in _OUTLINE_CUES)
                        )
                    )
                    pure_memory_only = is_pure_memory_cue
                    cap.output_meta = {
                        "needs_memory": has_memory_scope,
                        "pure_memory_only": pure_memory_only,
                        "use_outline": wants_outline,
                    }
                    if has_memory_scope:
                        state = RagState.MEMORY_RECALL
                    elif wants_outline:
                        state = RagState.OUTLINE_NAVIGATE
                    else:
                        state = RagState.HYBRID_RETRIEVE

            elif state == RagState.MEMORY_RECALL:
                async with recorder.span(
                    RagState.MEMORY_RECALL.value,
                    input_meta={"user_id": request.user_id, "session_id": str(request.session_id)},
                ) as cap:
                    if self._memory is not None and request.user_id:
                        memory_ctx = await self._memory.recall(
                            user_id=request.user_id,
                            query=current_query,
                            tenant_id=request.tenant_id,
                            namespace=request.namespace,
                            session_id=request.session_id,
                            limit=6,
                        )
                        memory_text = self._format_memory_context(memory_ctx)
                        has_mem_hits = bool(
                            memory_ctx.semantic_memories
                            or memory_ctx.episodic_memories
                            or (
                                memory_ctx.working_context
                                and memory_ctx.working_context.pinned_constraints
                            )
                        )
                        cap.output_meta = {
                            "semantic_hits": len(memory_ctx.semantic_memories),
                            "episodic_hits": len(memory_ctx.episodic_memories),
                            "has_hits": has_mem_hits,
                        }
                    else:
                        has_mem_hits = False
                        cap.output_meta = {"skipped": True}

                    if pure_memory_only and has_mem_hits:
                        state = RagState.GROUNDED_GENERATE
                    elif (
                        request.enable_outline_navigation is True
                        and request.section_ids is None
                    ):
                        state = RagState.OUTLINE_NAVIGATE
                    else:
                        pure_memory_only = False
                        state = RagState.HYBRID_RETRIEVE

            elif state == RagState.OUTLINE_NAVIGATE:
                async with recorder.span(
                    RagState.OUTLINE_NAVIGATE.value,
                    input_meta={"query": current_query},
                ) as cap:
                    outlines = await self._doc_repo.browse_outlines(
                        tenant_id=request.tenant_id,
                        namespace=request.namespace,
                        query=current_query,
                        limit=6,
                    )
                    q_toks = set(_content_tokens(current_query))
                    scored_secs: list[tuple[UUID, int]] = []
                    for outline in outlines:
                        for sec in outline.sections:
                            sec_str = " ".join(
                                [sec.heading, sec.summary or "", " ".join(sec.keywords)]
                            ).lower()
                            hits = sum(1 for tok in q_toks if tok in sec_str)
                            if hits > 0:
                                scored_secs.append((sec.id, hits))
                    scored_secs.sort(key=lambda pair: -pair[1])
                    if scored_secs and current_filter.section_ids is None:
                        selected_sec_ids = [sid for sid, _ in scored_secs[:4]]
                        current_filter = current_filter.model_copy(
                            update={"section_ids": selected_sec_ids}
                        )
                        outline_narrowed = True
                    cap.output_meta = {
                        "outlines_checked": len(outlines),
                        "selected_sections": len(current_filter.section_ids or []),
                    }
                state = RagState.HYBRID_RETRIEVE

            elif state == RagState.HYBRID_RETRIEVE:
                last_retrieval = await self._retrieval.search(
                    current_query,
                    current_filter,
                    mode=request.retrieval_mode,
                    top_k=request.top_k,
                    enable_graph=request.enable_graph_expansion,
                    enable_rerank=request.enable_rerank,
                    enable_parent_expand=request.enable_parent_expand,
                )
                evidence = last_retrieval.evidence
                recorder.record_span(
                    RagState.HYBRID_RETRIEVE.value,
                    last_retrieval.retrieve_latency_ms,
                    input_meta={"query": current_query, "mode": request.retrieval_mode},
                    output_meta={
                        "base_hits": last_retrieval.base_hit_count,
                        "graph_hits": last_retrieval.graph_hit_count,
                        "seed_entities": last_retrieval.seed_entities_matched,
                    },
                )
                recorder.record_span(
                    RagState.RERANK_AND_EXPAND.value,
                    last_retrieval.rerank_expand_latency_ms,
                    input_meta={
                        "enable_rerank": request.enable_rerank,
                        "enable_parent_expand": request.enable_parent_expand,
                    },
                    output_meta={
                        "evidence_count": len(evidence),
                        "top_score": last_retrieval.top_score,
                        "chunk_ids": [str(ev.chunk_id) for ev in evidence],
                    },
                )
                await emit(
                    "retrieval",
                    {
                        "count": len(evidence),
                        "top_score": last_retrieval.top_score,
                        "chunk_ids": [str(ev.chunk_id) for ev in evidence],
                    },
                )
                if request.enable_crag:
                    state = RagState.SUFFICIENCY_GATE
                elif evidence:
                    state = RagState.GROUNDED_GENERATE
                else:
                    abstention_reason = "No matching documents found in the requested scope."
                    state = RagState.STRICT_ABSTAIN

            elif state == RagState.SUFFICIENCY_GATE:
                async with recorder.span(
                    RagState.SUFFICIENCY_GATE.value,
                    input_meta={
                        "evidence_count": len(evidence),
                        "top_score": last_retrieval.top_score if last_retrieval else 0.0,
                    },
                ) as cap:
                    verdict, gate_reason, gate_rewrite, pt, ct = await self._evaluate_sufficiency(
                        current_query,
                        evidence,
                        top_score=last_retrieval.top_score if last_retrieval else 0.0,
                        rerank_used=request.enable_rerank,
                        budget=budget,
                    )
                    cap.prompt_tokens = pt
                    cap.completion_tokens = ct
                    cap.output_meta = {
                        "verdict": verdict.value,
                        "reason": gate_reason,
                        "suggested_rewrite": gate_rewrite,
                    }
                    suggested_rewrite = gate_rewrite

                    if verdict == SufficiencyVerdict.SUFFICIENT:
                        state = RagState.GROUNDED_GENERATE
                    elif (
                        verdict == SufficiencyVerdict.AMBIGUOUS
                        and budget.can_retry_retrieval()
                    ):
                        state = RagState.CORRECTIVE_REWRITE
                    else:
                        abstention_reason = gate_reason or (
                            "Retrieved evidence is insufficient to answer reliably."
                        )
                        state = RagState.STRICT_ABSTAIN

            elif state == RagState.CORRECTIVE_REWRITE:
                async with recorder.span(
                    RagState.CORRECTIVE_REWRITE.value,
                    input_meta={"previous_query": current_query},
                ) as cap:
                    if not budget.record_retrieval_retry():
                        abstention_reason = "Retrieval retry budget exhausted."
                        cap.output_meta = {"aborted": True, "reason": abstention_reason}
                        state = RagState.STRICT_ABSTAIN
                        continue

                    relaxed_sections = False
                    if outline_narrowed and current_filter.section_ids is not None:
                        current_filter = current_filter.model_copy(update={"section_ids": None})
                        outline_narrowed = False
                        relaxed_sections = True

                    rewritten, pt, ct = await self._rewrite_query(
                        current_query,
                        suggested_rewrite=suggested_rewrite,
                        budget=budget,
                    )
                    cap.prompt_tokens = pt
                    cap.completion_tokens = ct

                    is_new_query = budget.register_query(rewritten)
                    if not is_new_query and not relaxed_sections:
                        abstention_reason = (
                            "Corrective rewrite produced duplicate query without new evidence."
                        )
                        cap.output_meta = {"duplicate_blocked": True, "rewritten": rewritten}
                        state = RagState.STRICT_ABSTAIN
                        continue

                    current_query = rewritten
                    cap.output_meta = {
                        "rewritten_query": current_query,
                        "relaxed_sections": relaxed_sections,
                        "retry_no": budget.retrieval_retries,
                    }
                state = RagState.HYBRID_RETRIEVE

            elif state == RagState.GROUNDED_GENERATE:
                async with recorder.span(
                    RagState.GROUNDED_GENERATE.value,
                    input_meta={
                        "evidence_count": len(evidence),
                        "has_memory": bool(memory_text),
                    },
                ) as cap:
                    draft, pt, ct = await self._generate_grounded_draft(
                        request.question,
                        evidence,
                        memory_text=memory_text,
                        pure_memory_only=pure_memory_only,
                        strict_crag=request.enable_crag,
                        budget=budget,
                    )
                    cap.prompt_tokens = pt
                    cap.completion_tokens = ct
                    cap.output_meta = {
                        "cannot_answer": draft.cannot_answer,
                        "claim_count": len(draft.claims),
                    }
                    if draft.cannot_answer:
                        abstention_reason = (
                            draft.abstention_reason
                            or "Evidence does not support answering the question."
                        )
                        state = RagState.STRICT_ABSTAIN
                        continue

                    verified_claims = [
                        ClaimWithCitations(
                            statement=c.statement,
                            cited_chunk_ids=[_parse_uuid_safe(cid) for cid in c.cited_chunk_ids],
                        )
                        for c in draft.claims
                        if c.statement.strip()
                    ]
                state = RagState.CITATION_VERIFY

            elif state == RagState.CITATION_VERIFY:
                async with recorder.span(
                    RagState.CITATION_VERIFY.value,
                    input_meta={"claim_count": len(verified_claims), "strict": strict_mode},
                ) as cap:
                    verify_res = self._verifier.verify(
                        verified_claims,
                        evidence,
                        require_all_claims_cited=strict_mode and not pure_memory_only,
                    )
                    cap.output_meta = {
                        "passed": verify_res.passed,
                        "valid_citations": len(verify_res.valid_citations),
                        "invalid_chunk_ids": [str(i) for i in verify_res.invalid_chunk_ids],
                        "uncited_claims": len(verify_res.uncited_claims),
                    }
                    await emit(
                        "citation",
                        {
                            "passed": verify_res.passed,
                            "citations": [
                                c.model_dump(mode="json") for c in verify_res.valid_citations
                            ],
                        },
                    )
                    if not verify_res.passed and strict_mode:
                        abstention_reason = verify_res.reason or "Citation verification failed."
                        state = RagState.STRICT_ABSTAIN
                        continue

                    assert draft is not None
                    final_response = RagResponse(
                        trace_id=recorder.trace_id,
                        status=RagDecisionStatus.ANSWERED,
                        answer=draft.answer.strip(),
                        claims=verified_claims,
                        citations=verify_res.valid_citations,
                        retrieval_retries=budget.retrieval_retries,
                        searched_scope={
                            "tenant_id": request.tenant_id,
                            "namespace": request.namespace,
                            "retrieval_mode": request.retrieval_mode,
                            "evidence_count": len(evidence),
                        },
                    )
                state = RagState.DONE

            elif state == RagState.STRICT_ABSTAIN:
                reason = abstention_reason or "Insufficient verifiable evidence in scope."
                async with recorder.span(
                    RagState.STRICT_ABSTAIN.value,
                    input_meta={"reason": reason},
                ) as cap:
                    abstain_msg = (
                        f"无法基于当前授权检索范围内的已验证证据回答该问题（{reason}）。"
                    )
                    cap.output_meta = {"abstention_reason": reason}
                    final_response = RagResponse(
                        trace_id=recorder.trace_id,
                        status=RagDecisionStatus.ABSTAINED,
                        answer=abstain_msg,
                        claims=[],
                        citations=[],
                        abstention_reason=reason,
                        retrieval_retries=budget.retrieval_retries,
                        searched_scope={
                            "tenant_id": request.tenant_id,
                            "namespace": request.namespace,
                            "retrieval_mode": request.retrieval_mode,
                            "evidence_count": len(evidence),
                        },
                    )
                state = RagState.DONE

        assert final_response is not None
        recorder.set_retrieval_retries(budget.retrieval_retries)
        await recorder.finalize(final_response.status.value)
        await emit("final", final_response.model_dump(mode="json"))
        return final_response

    def _format_memory_context(self, mem: MemoryRecallResult) -> str:
        lines: list[str] = []
        if mem.working_context and mem.working_context.pinned_constraints:
            lines.append("Pinned Constraints:")
            for c in mem.working_context.pinned_constraints:
                lines.append(f"- {c}")
        if mem.semantic_memories:
            lines.append("Active Semantic Memories:")
            for item in mem.semantic_memories:
                tag = item.memory.slot_key or item.memory.memory_type
                lines.append(f"- [{tag}] {item.memory.fact_text}")
        if mem.episodic_memories:
            lines.append("Relevant Episodes:")
            for ep in mem.episodic_memories[:3]:
                lines.append(f"- {ep.episode.summary}")
        return "\n".join(lines)

    async def _evaluate_sufficiency(
        self,
        query: str,
        evidence: list[RetrievedEvidence],
        *,
        top_score: float,
        rerank_used: bool,
        budget: ExecutionBudget,
    ) -> tuple[SufficiencyVerdict, str, str, int, int]:
        if not evidence:
            return (
                SufficiencyVerdict.AMBIGUOUS
                if budget.can_retry_retrieval()
                else SufficiencyVerdict.INSUFFICIENT,
                "No candidate chunks retrieved.",
                "",
                0,
                0,
            )

        coverage = _token_coverage(query, evidence)
        if (
            rerank_used
            and top_score < self._settings.rerank_hard_reject_threshold
            and coverage < 0.3
        ):
            return (
                SufficiencyVerdict.AMBIGUOUS
                if budget.can_retry_retrieval()
                else SufficiencyVerdict.INSUFFICIENT,
                f"Top evidence score ({top_score:.2f}) below threshold.",
                "",
                0,
                0,
            )

        if (rerank_used and top_score >= self._settings.rerank_high_confidence_threshold) or (
            not rerank_used and coverage >= 0.45
        ):
            return (SufficiencyVerdict.SUFFICIENT, "High-confidence evidence found.", "", 0, 0)

        # Gray zone: call fast LLM when configured and within budget
        if self._llm is not None and self._llm.is_configured and budget.consume_llm_call():
            try:
                snippets = "\n\n".join(
                    f"[{ev.chunk_id}] {(ev.expanded_text or ev.raw_text)[:500]}"
                    for ev in evidence[:4]
                )
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Judge whether the retrieved evidence is sufficient to answer the "
                                "question without guessing. Return verdict in "
                                "{sufficient, ambiguous, insufficient} and a suggested_rewrite "
                                "if ambiguous."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Question: {query}\n\nEvidence:\n{snippets}",
                        },
                    ],
                    _SufficiencyDraft,
                    use_fast_model=True,
                    max_tokens=240,
                )
                return (
                    res.parsed.verdict,
                    res.parsed.missing_aspects.strip() or "LLM sufficiency check.",
                    res.parsed.suggested_rewrite.strip(),
                    res.usage.prompt_tokens,
                    res.usage.completion_tokens,
                )
            except Exception as exc:
                logger.warning("crag_sufficiency_llm_fallback", error=str(exc))

        if coverage >= 0.40:
            return (
                SufficiencyVerdict.SUFFICIENT,
                f"Token coverage {coverage:.2f} sufficient.",
                "",
                0,
                0,
            )
        return (
            SufficiencyVerdict.AMBIGUOUS
            if budget.can_retry_retrieval()
            else SufficiencyVerdict.INSUFFICIENT,
            f"Low query term coverage ({coverage:.2f}) across top evidence.",
            "",
            0,
            0,
        )

    async def _rewrite_query(
        self,
        query: str,
        *,
        suggested_rewrite: str,
        budget: ExecutionBudget,
    ) -> tuple[str, int, int]:
        if suggested_rewrite.strip():
            return (suggested_rewrite.strip(), 0, 0)

        if self._llm is not None and self._llm.is_configured and budget.consume_llm_call():
            try:
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the user question into a focused search query using "
                                "key technical nouns, aliases, and error codes."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    _RewriteDraft,
                    use_fast_model=True,
                    max_tokens=120,
                )
                cand = res.parsed.rewritten_query.strip()
                if cand:
                    return (cand, res.usage.prompt_tokens, res.usage.completion_tokens)
            except Exception as exc:
                logger.warning("crag_rewrite_llm_fallback", error=str(exc))

        core = _content_tokens(query)
        if core:
            return (" ".join(core), 0, 0)
        return (query.strip(), 0, 0)

    async def _generate_grounded_draft(
        self,
        question: str,
        evidence: list[RetrievedEvidence],
        *,
        memory_text: str,
        pure_memory_only: bool,
        strict_crag: bool,
        budget: ExecutionBudget,
    ) -> tuple[_GroundedAnswerDraft, int, int]:
        if self._llm is not None and self._llm.is_configured and budget.consume_llm_call():
            try:
                ev_blocks = "\n\n".join(
                    f"[chunk_id={ev.chunk_id} | doc={ev.document_title}]\n"
                    f"{(ev.expanded_text or ev.raw_text)[:900]}"
                    for ev in evidence[:6]
                )
                mem_section = f"\nUser Memory Context:\n{memory_text}\n" if memory_text else ""
                res = await self._llm.complete_structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Answer strictly using the provided evidence chunks (and user "
                                "memory context when provided). Every claim MUST cite one or "
                                "more exact chunk_id UUIDs from the evidence. If the evidence "
                                "does not contain the answer, set cannot_answer=true and explain "
                                "abstention_reason."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}{mem_section}\nEvidence:\n{ev_blocks}",
                        },
                    ],
                    _GroundedAnswerDraft,
                    use_fast_model=False,
                    max_tokens=900,
                )
                return (res.parsed, res.usage.prompt_tokens, res.usage.completion_tokens)
            except Exception as exc:
                logger.warning("grounded_generate_llm_fallback", error=str(exc))

        if pure_memory_only and memory_text:
            return (
                _GroundedAnswerDraft(
                    cannot_answer=False,
                    answer=memory_text,
                    claims=[],
                ),
                0,
                0,
            )

        if not evidence:
            return (
                _GroundedAnswerDraft(
                    cannot_answer=True,
                    abstention_reason="No evidence available.",
                ),
                0,
                0,
            )

        coverage = _token_coverage(question, evidence)
        if strict_crag and coverage < 0.35:
            return (
                _GroundedAnswerDraft(
                    cannot_answer=True,
                    abstention_reason="Retrieved chunks do not adequately match question terms.",
                ),
                0,
                0,
            )

        top_ev = evidence[0]
        raw_snippet = (top_ev.expanded_text or top_ev.raw_text).strip()
        sentences = [
            s.strip() for s in re.split(r"(?<=[。！？.!?\n])\s*", raw_snippet) if s.strip()
        ]
        q_toks = set(_content_tokens(question))
        best_sents = sorted(
            sentences,
            key=lambda s: sum(1 for tok in q_toks if tok in s.lower()),
            reverse=True,
        )
        chosen = best_sents[:2] if best_sents else [raw_snippet[:240]]
        answer_text = " ".join(chosen)[:420]
        claims = [
            _DraftClaim(
                statement=sent[:240],
                cited_chunk_ids=[str(top_ev.chunk_id)],
            )
            for sent in chosen
            if sent.strip()
        ]
        if not claims:
            claims = [_DraftClaim(statement=answer_text, cited_chunk_ids=[str(top_ev.chunk_id)])]

        return (_GroundedAnswerDraft(cannot_answer=False, answer=answer_text, claims=claims), 0, 0)
