"""PostgreSQL integration tests (runs when TERRACEMEM_TEST_DATABASE_URL is configured)."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from terracemem.config import Settings
from terracemem.domain.models import (
    EntityRecord,
    RelationEdgeRecord,
    RetrievalFilter,
)
from terracemem.runtime import TerraceMemRuntime, _probe_tcp
from terracemem.utils.ids import new_uuid7

_PG_DSN = os.getenv("TERRACEMEM_TEST_DATABASE_URL", "").strip()


async def _require_pg_runtime() -> TerraceMemRuntime:
    if not _PG_DSN:
        pytest.skip("TERRACEMEM_TEST_DATABASE_URL not set; skipping PG integration test.")
    if not await _probe_tcp(_PG_DSN, timeout=1.0):
        pytest.skip(f"PostgreSQL unreachable at {_PG_DSN}; skipping PG integration test.")

    lexical = os.getenv("TERRACEMEM_LEXICAL_BACKEND", "native")
    dim = int(os.getenv("TERRACEMEM_EMBEDDING_DIM", "1024"))
    settings = Settings(
        storage_backend="postgres",
        database_url=_PG_DSN,
        lexical_backend="paradedb" if lexical == "paradedb" else "native",
        embedding_dim=dim,
        enable_background_worker=False,
    )
    rt = TerraceMemRuntime(settings)
    await rt.startup()
    return rt


async def test_postgres_full_stack_isolation_graph_memory_and_outbox() -> None:
    rt = await _require_pg_runtime()
    tenant = f"pg_test_{new_uuid7().hex[:10]}"
    try:
        assert rt.db_pool is not None
        # 1) Verify schema tables exist
        async with rt.db_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                      'documents', 'doc_sections', 'doc_chunks',
                      'entities', 'relation_edges',
                      'sessions', 'session_messages', 'session_compactions',
                      'episodic_memories', 'semantic_memories',
                      'async_jobs', 'execution_traces', 'trace_spans'
                  );
                """
            )
            row = await cur.fetchone()
            assert row is not None and int(str(row[0])) == 13

        # 2) Versioned ingestion + soft delete + tenant isolation
        v1 = await rt.ingestion.ingest_text(
            "# PG Spec\n\n## Token\nLegacy token secret ALPHA_111.",
            title="PG Spec",
            external_key="pg-spec",
            tenant_id=tenant,
            namespace="prod",
            enable_graph=False,
        )
        assert v1.version == 1
        v2 = await rt.ingestion.ingest_text(
            "# PG Spec\n\n## Token\nCurrent token protocol EdDSA_999.",
            title="PG Spec",
            external_key="pg-spec",
            tenant_id=tenant,
            namespace="prod",
            enable_graph=False,
        )
        assert v2.version == 2

        hits = await rt.retrieval.search(
            "ALPHA_111",
            RetrievalFilter(tenant_id=tenant, namespace="prod", only_latest=True),
        )
        assert all("ALPHA_111" not in ev.raw_text for ev in hits.evidence)

        isolated = await rt.retrieval.search(
            "EdDSA_999",
            RetrievalFilter(tenant_id=f"{tenant}_other", namespace="prod"),
        )
        assert isolated.evidence == []

        # 3) 2-hop CTE cycle safety
        e1 = EntityRecord(tenant_id=tenant, namespace="prod", canonical_name="NodeA")
        e2 = EntityRecord(tenant_id=tenant, namespace="prod", canonical_name="NodeB")
        c_hits = await rt.doc_repo.retrieve(
            "EdDSA_999",
            None,
            RetrievalFilter(tenant_id=tenant, namespace="prod"),
            mode="lexical_only",
        )
        assert c_hits
        await rt.graph_repo.upsert_entities_and_edges(
            [e1, e2],
            [
                RelationEdgeRecord(
                    tenant_id=tenant,
                    namespace="prod",
                    src_entity_id=e1.id,
                    rel_type="links",
                    dst_entity_id=e2.id,
                    evidence_chunk_id=c_hits[0].chunk_id,
                ),
                RelationEdgeRecord(
                    tenant_id=tenant,
                    namespace="prod",
                    src_entity_id=e2.id,
                    rel_type="links",
                    dst_entity_id=e1.id,
                    evidence_chunk_id=c_hits[0].chunk_id,
                ),
            ],
        )
        hop_res = await rt.graph_repo.expand_two_hop(
            [e1.id], RetrievalFilter(tenant_id=tenant, namespace="prod")
        )
        assert len(hop_res) == 1

        # 4) Bi-temporal memory + Outbox worker SKIP LOCKED
        t1 = datetime(2026, 2, 1, tzinfo=UTC)
        t2 = datetime(2026, 4, 1, tzinfo=UTC)
        await rt.memory.commit_turn(
            user_id="pg-user",
            tenant_id=tenant,
            content="Database is PostgreSQL [slot:project.db].",
            event_time=t1,
            sync_extract=True,
        )
        await rt.memory.commit_turn(
            user_id="pg-user",
            tenant_id=tenant,
            content="Database migrated to DuckDB [slot:project.db].",
            event_time=t2,
            sync_extract=False,
        )
        assert rt.outbox_worker is not None
        processed = await rt.outbox_worker.process_one()
        assert processed is True

        tl_old = await rt.memory.timeline(
            user_id="pg-user",
            tenant_id=tenant,
            slot_key="project.db",
            as_of=t1 + timedelta(days=5),
        )
        assert len(tl_old.active_at_as_of) == 1
        assert "PostgreSQL" in tl_old.active_at_as_of[0].fact_text

        tl_new = await rt.memory.timeline(
            user_id="pg-user",
            tenant_id=tenant,
            slot_key="project.db",
            as_of=t2 + timedelta(days=5),
        )
        assert len(tl_new.active_at_as_of) == 1
        assert "DuckDB" in tl_new.active_at_as_of[0].fact_text

        # Soft delete cleanup verification
        assert await rt.doc_repo.soft_delete_document(
            v2.document_id, tenant_id=tenant, namespace="prod"
        )
        after_del = await rt.retrieval.search(
            "EdDSA_999",
            RetrievalFilter(tenant_id=tenant, namespace="prod", only_latest=False),
        )
        assert after_del.evidence == []
    finally:
        if rt.db_pool is not None:
            async with rt.db_pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                for tbl in (
                    "semantic_memories",
                    "episodic_memories",
                    "sessions",
                    "relation_edges",
                    "entities",
                    "documents",
                    "execution_traces",
                ):
                    await cur.execute(
                        f"DELETE FROM {tbl} WHERE tenant_id = %(tenant)s;",
                        {"tenant": tenant},
                    )
        await rt.shutdown()
        await asyncio.sleep(0)
