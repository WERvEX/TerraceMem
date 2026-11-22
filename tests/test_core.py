"""Unit tests for UUIDv7, MarkdownTreeParser, RRF/decay scoring, budget, verifier & guardrails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from terracemem.api.app import create_app
from terracemem.config import Settings
from terracemem.domain.models import (
    CandidateMemoryFact,
    ClaimWithCitations,
    ConflictJudgment,
    EpistemicModality,
    MemoryResolutionAction,
    MemoryType,
    RetrievedEvidence,
    SemanticMemoryRecord,
)
from terracemem.ingestion.markdown_parser import MarkdownTreeParser
from terracemem.memory.guardrails import GuardrailDecision, evaluate_conflict_guardrails
from terracemem.rag.budget import ExecutionBudget
from terracemem.rag.verifier import CitationVerifier
from terracemem.retrieval.fusion import reciprocal_rank_fusion, score_semantic_memory
from terracemem.utils.ids import new_uuid7


def test_uuid7_is_monotonically_increasing() -> None:
    ids = [new_uuid7() for _ in range(50)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 50


def test_markdown_tree_parser_preserves_headings_and_fences() -> None:
    sample = """# TerraceMem Guide

## 1. Storage
Use PostgreSQL with pgvector.

```python
# Not a markdown heading inside code block
x = 1
```

### 1.1 Hybrid Search
Combine BM25 and dense vectors via RRF.
"""
    parser = MarkdownTreeParser()
    doc = parser.parse_text(sample, title="Fallback", external_key="guide.md", max_chunk_tokens=120)

    assert doc.title == "TerraceMem Guide"
    headings = [s.heading for s in doc.sections]
    assert "Not a markdown heading inside code block" not in headings
    assert headings == ["TerraceMem Guide", "1. Storage", "1.1 Hybrid Search"]
    assert len(doc.chunks) >= 2
    assert any("1. Storage > 1.1 Hybrid Search" in c.contextual_text for c in doc.chunks)


def test_reciprocal_rank_fusion_ranks_consensus_highest() -> None:
    a, b, c = new_uuid7(), new_uuid7(), new_uuid7()
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert fused[0][0] == b
    assert fused[0][1] > fused[1][1]


def test_semantic_memory_scoring_protects_pinned_and_constraints_from_decay() -> None:
    old_time = datetime.now(UTC) - timedelta(days=365)
    normal = SemanticMemoryRecord(
        user_id="u1",
        memory_type=MemoryType.FACT,
        fact_text="Likes tea",
        last_accessed_at=old_time,
    )
    pinned_constraint = SemanticMemoryRecord(
        user_id="u1",
        memory_type=MemoryType.CONSTRAINT,
        fact_text="Never drop production tables",
        is_pinned=True,
        last_accessed_at=old_time,
    )
    assert score_semantic_memory(pinned_constraint, similarity_score=0.8) > score_semantic_memory(
        normal, similarity_score=0.8
    )


def test_execution_budget_blocks_duplicate_queries_and_excess_retries() -> None:
    budget = ExecutionBudget(max_retrieval_retries=2, max_llm_calls=2)
    assert budget.register_query("PostgreSQL   BM25") is True
    assert budget.register_query("postgresql bm25") is False
    assert budget.record_retrieval_retry() is True
    assert budget.record_retrieval_retry() is True
    assert budget.record_retrieval_retry() is False
    assert budget.consume_llm_call() is True
    assert budget.consume_llm_call() is True
    assert budget.consume_llm_call() is False


def test_citation_verifier_rejects_fabricated_chunk_ids() -> None:
    doc_id = new_uuid7()
    valid_chunk_id = new_uuid7()
    fake_chunk_id = new_uuid7()
    evidence = [
        RetrievedEvidence(
            chunk_id=valid_chunk_id,
            document_id=doc_id,
            document_title="Spec",
            raw_text="pgvector 0.8 supports iterative index scans.",
        )
    ]
    verifier = CitationVerifier()

    ok = verifier.verify(
        [
            ClaimWithCitations(
                statement="pgvector 0.8 supports iterative scans.",
                cited_chunk_ids=[valid_chunk_id],
            )
        ],
        evidence,
    )
    assert ok.passed is True
    assert len(ok.valid_citations) == 1

    bad = verifier.verify(
        [ClaimWithCitations(statement="Fabricated claim", cited_chunk_ids=[fake_chunk_id])],
        evidence,
    )
    assert bad.passed is False
    assert fake_chunk_id in bad.invalid_chunk_ids


def test_memory_guardrails_block_hypothetical_old_backfill_and_pinned_overwrites() -> None:
    now = datetime.now(UTC)
    existing = SemanticMemoryRecord(
        user_id="u1",
        slot_key="project.db",
        fact_text="Project uses PostgreSQL",
        valid_from=now,
        is_pinned=False,
    )
    judgment = ConflictJudgment(
        target_memory_id=existing.id,
        action=MemoryResolutionAction.CONTRADICT,
        reason="User switched database",
    )

    # 1) Hypothetical modality blocked
    hypothetical = CandidateMemoryFact(
        memory_type=MemoryType.FACT,
        slot_key="project.db",
        fact_text="Maybe switch to SQLite later?",
        evidence_quote="Maybe switch to SQLite later?",
        confidence=0.95,
        modality=EpistemicModality.HYPOTHETICAL,
        event_time=now + timedelta(minutes=1),
    )
    assert (
        evaluate_conflict_guardrails(hypothetical, existing, judgment).decision
        == GuardrailDecision.REJECT
    )

    # 2) Low confidence downgraded
    low_conf = CandidateMemoryFact(
        memory_type=MemoryType.FACT,
        slot_key="project.db",
        fact_text="Uses MySQL",
        evidence_quote="Uses MySQL",
        confidence=0.5,
        event_time=now + timedelta(minutes=1),
    )
    assert (
        evaluate_conflict_guardrails(low_conf, existing, judgment).decision
        == GuardrailDecision.DOWNGRADE_TO_NOOP
    )

    # 3) Backfill older than existing.valid_from rejected
    stale = CandidateMemoryFact(
        memory_type=MemoryType.FACT,
        slot_key="project.db",
        fact_text="Used MySQL last year",
        evidence_quote="Used MySQL last year",
        confidence=0.95,
        event_time=now - timedelta(days=30),
    )
    assert (
        evaluate_conflict_guardrails(stale, existing, judgment).decision
        == GuardrailDecision.REJECT
    )

    # 4) Pinned existing fact routed to pending_confirmation
    pinned_existing = existing.model_copy(update={"is_pinned": True})
    valid_update = CandidateMemoryFact(
        memory_type=MemoryType.FACT,
        slot_key="project.db",
        fact_text="Migrated to DuckDB",
        evidence_quote="We migrated to DuckDB today.",
        confidence=0.92,
        event_time=now + timedelta(hours=1),
    )
    assert (
        evaluate_conflict_guardrails(valid_update, pinned_existing, judgment).decision
        == GuardrailDecision.PENDING_CONFIRMATION
    )
    assert (
        evaluate_conflict_guardrails(valid_update, existing, judgment).decision
        == GuardrailDecision.ALLOW_COMMIT
    )


def test_fastapi_health_info_and_parse_preview_endpoints() -> None:
    app = create_app(Settings(enable_inspector_ui=True))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service"] == "terracemem"

        info = client.get("/api/v1/info")
        assert info.status_code == 200
        assert info.json()["name"] == "TerraceMem"
        assert info.json()["endpoints"]["mcp_streamable_http"] == "/mcp"

        ui = client.get("/ui")
        assert ui.status_code == 200
        assert "TerraceMem" in ui.text

        preview = client.post(
            "/api/v1/parse-preview",
            json={"title": "Test", "markdown": "# Intro\nHello TerraceMem!"},
        )
        assert preview.status_code == 200
        assert preview.json()["title"] == "Intro"
        assert len(preview.json()["chunks"]) == 1
