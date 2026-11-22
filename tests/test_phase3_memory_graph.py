"""Phase 3 tests: L1/L2/L3 bi-temporal memory, guardrails, Outbox worker & 2-hop RAG."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from terracemem.config import Settings
from terracemem.domain.models import (
    EntityRecord,
    RagDecisionStatus,
    RelationEdgeRecord,
    RetrievalFilter,
)
from terracemem.memory.guardrails import GuardrailDecision
from terracemem.rag.fsm import RagRequest
from terracemem.runtime import TerraceMemRuntime


async def test_l1_working_memory_preserves_pinned_constraints_and_compacts() -> None:
    rt = TerraceMemRuntime(
        Settings(storage_backend="memory", default_session_token_budget=512)
    )
    await rt.startup()
    try:
        session = await rt.memory.create_session(
            user_id="alice",
            token_budget=320,
            pinned_constraints=["CRITICAL: Never modify production database schema directly."],
        )
        for idx in range(14):
            await rt.memory.commit_turn(
                user_id="alice",
                session_id=session.id,
                content=(
                    f"Turn {idx}: Discussing microservice telemetry latency budgets and "
                    "histogram bucket sizes across staging clusters."
                ),
                sync_extract=False,
            )

        ctx = await rt.memory.get_working_context(session.id)
        assert ctx is not None
        assert ctx.pinned_constraints == [
            "CRITICAL: Never modify production database schema directly."
        ]
        assert ctx.used_tokens <= ctx.token_budget
        assert ctx.compaction_summary is not None
        assert len(ctx.recent_messages) <= 4
    finally:
        await rt.shutdown()


async def test_l2_l3_bitemporal_supersession_time_travel_and_guardrails() -> None:
    rt = TerraceMemRuntime(Settings(storage_backend="memory"))
    await rt.startup()
    try:
        t1 = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
        t2 = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

        # 1) Initial asserted fact at t1
        c1 = await rt.memory.commit_turn(
            user_id="bob",
            content="Our primary database is PostgreSQL [slot:project.db].",
            event_time=t1,
            sync_extract=True,
        )
        assert c1.episode is not None
        assert len(c1.audits) == 1
        assert c1.audits[0].new_memory_id is not None
        pg_mem_id = c1.audits[0].new_memory_id

        # 2) Hypothetical & question blocked by Guardrail 1
        hypo = await rt.memory.commit_turn(
            user_id="bob",
            content="Maybe we could switch to SQLite? [slot:project.db]",
            event_time=t1 + timedelta(hours=1),
            sync_extract=True,
        )
        assert hypo.audits[0].guardrail_decision == GuardrailDecision.REJECT.value

        # 3) Historical backfill older than valid_from blocked by Guardrail 3
        backfill = await rt.memory.commit_turn(
            user_id="bob",
            content="Back in 2024 we used MySQL [slot:project.db].",
            event_time=t1 - timedelta(days=30),
            sync_extract=True,
        )
        assert backfill.audits[0].guardrail_decision == GuardrailDecision.REJECT.value

        # 4) Newer asserted fact at t2 softly supersedes PostgreSQL -> DuckDB
        c2 = await rt.memory.commit_turn(
            user_id="bob",
            content="We migrated our primary database to DuckDB [slot:project.db].",
            event_time=t2,
            sync_extract=True,
        )
        assert c2.audits[0].guardrail_decision == GuardrailDecision.ALLOW_COMMIT.value
        assert c2.audits[0].new_memory_id is not None
        duck_mem_id = c2.audits[0].new_memory_id

        # Bi-temporal as_of time-travel inspection
        tl_before = await rt.memory.timeline(
            user_id="bob",
            slot_key="project.db",
            as_of=t1 + timedelta(days=5),
        )
        assert len(tl_before.active_at_as_of) == 1
        assert tl_before.active_at_as_of[0].id == pg_mem_id

        tl_after = await rt.memory.timeline(
            user_id="bob",
            slot_key="project.db",
            as_of=t2 + timedelta(days=1),
        )
        assert len(tl_after.active_at_as_of) == 1
        assert tl_after.active_at_as_of[0].id == duck_mem_id
        assert len(tl_after.history) == 2
        assert tl_after.history[0].superseded_by == duck_mem_id
        assert tl_after.history[0].invalidation_reason

        # 5) Pinning DuckDB protects it via Guardrail 4 (PENDING_CONFIRMATION)
        pinned = await rt.memory.set_pinned(duck_mem_id, user_id="bob", is_pinned=True)
        assert pinned is not None and pinned.is_pinned is True

        blocked_overwrite = await rt.memory.commit_turn(
            user_id="bob",
            content="We switched database to ClickHouse [slot:project.db].",
            event_time=t2 + timedelta(days=2),
            sync_extract=True,
        )
        assert (
            blocked_overwrite.audits[0].guardrail_decision
            == GuardrailDecision.PENDING_CONFIRMATION.value
        )
        tl_latest = await rt.memory.timeline(user_id="bob", slot_key="project.db")
        assert len(tl_latest.active_at_as_of) == 1
        assert tl_latest.active_at_as_of[0].id == duck_mem_id

        # L2 & L3 recall returns active DuckDB + archived episodes
        recalled = await rt.memory.recall(user_id="bob", query="primary database")
        assert any("DuckDB" in s.memory.fact_text for s in recalled.semantic_memories)
        assert len(recalled.episodic_memories) >= 2
    finally:
        await rt.shutdown()


async def test_outbox_queue_idempotency_execution_and_dead_letter() -> None:
    rt = TerraceMemRuntime(Settings(storage_backend="memory"))
    await rt.startup()
    try:
        assert rt.in_memory_outbox is not None
        res = await rt.memory.commit_turn(
            user_id="carol",
            content="I prefer Python over Java [slot:user.language].",
            sync_extract=False,
        )
        assert res.queued_async_job is True
        assert len(rt.in_memory_outbox.list_jobs()) == 1

        # Duplicate idempotency key does not enqueue twice
        dup = rt.in_memory_outbox.enqueue(
            job_type="memory.extract",
            idempotency_key=rt.in_memory_outbox.list_jobs()[0].idempotency_key,
            payload={},
        )
        assert dup is None

        drained = await rt.in_memory_outbox.drain()
        assert drained == 1
        assert rt.in_memory_outbox.list_jobs()[0].status == "succeeded"

        tl = await rt.memory.timeline(user_id="carol")
        assert len(tl.active_at_as_of) == 1
        assert "Python" in tl.active_at_as_of[0].fact_text

        # Failing job moves to dead after max_attempts
        async def _always_fail(_payload: dict[str, object]) -> None:
            raise RuntimeError("boom")

        rt.in_memory_outbox.register("fail.job", _always_fail)
        bad_id = rt.in_memory_outbox.enqueue(
            job_type="fail.job",
            idempotency_key="bad-1",
            payload={},
            max_attempts=2,
        )
        assert bad_id is not None
        await rt.in_memory_outbox.process_one(ignore_backoff=True)
        await rt.in_memory_outbox.process_one(ignore_backoff=True)
        bad_job = rt.in_memory_outbox.get_job(bad_id)
        assert bad_job is not None and bad_job.status == "dead"
    finally:
        await rt.shutdown()


async def test_two_hop_graph_assists_multihop_rag() -> None:
    rt = TerraceMemRuntime(
        Settings(
            storage_backend="memory",
            enable_graph_expansion=True,
            rerank_high_confidence_threshold=0.1,
        )
    )
    await rt.startup()
    try:
        await rt.ingestion.ingest_text(
            "# Topology\n\n## Edge\nServiceGateway forwards auth requests to AuthCore.",
            title="Gateway",
            external_key="gateway",
            enable_graph=False,
        )
        await rt.ingestion.ingest_text(
            "# Cryptography\n\n## Signing\nAuthCore signs tokens with EdDSAJWT at 15m TTL.",
            title="Crypto",
            external_key="crypto",
            enable_graph=False,
        )
        crypto_hits = await rt.doc_repo.retrieve(
            "EdDSAJWT 15m TTL", None, RetrievalFilter(), mode="lexical_only"
        )
        assert crypto_hits
        crypto_chunk_id = crypto_hits[0].chunk_id

        e1 = EntityRecord(canonical_name="ServiceGateway", entity_type="service")
        e2 = EntityRecord(canonical_name="AuthCore", entity_type="service")
        e3 = EntityRecord(canonical_name="EdDSAJWT", entity_type="protocol")
        await rt.graph_repo.upsert_entities_and_edges(
            [e1, e2, e3],
            [
                RelationEdgeRecord(
                    src_entity_id=e1.id,
                    rel_type="calls",
                    dst_entity_id=e2.id,
                    weight=1.0,
                ),
                RelationEdgeRecord(
                    src_entity_id=e2.id,
                    rel_type="uses",
                    dst_entity_id=e3.id,
                    evidence_chunk_id=crypto_chunk_id,
                    weight=1.0,
                ),
            ],
        )

        # Query mentions ServiceGateway (seed) and asks about token TTL (found via 2-hop walk)
        search_res = await rt.retrieval.search(
            "ServiceGateway token TTL",
            RetrievalFilter(),
            top_k=4,
            enable_graph=True,
        )
        assert search_res.seed_entities_matched >= 1
        assert search_res.graph_hit_count >= 1
        assert any(ev.chunk_id == crypto_chunk_id for ev in search_res.evidence)

        rag_res = await rt.rag.run(
            RagRequest(question="What TTL does ServiceGateway token signing use?")
        )
        assert rag_res.status == RagDecisionStatus.ANSWERED
    finally:
        await rt.shutdown()
