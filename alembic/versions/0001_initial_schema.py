"""Initial unified TerraceMem PostgreSQL schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-21
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _embedding_dim() -> int:
    return int(os.getenv("TERRACEMEM_EMBEDDING_DIM", "1024"))


def upgrade() -> None:
    dim = _embedding_dim()

    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN
                CREATE EXTENSION IF NOT EXISTS pg_search;
            END IF;
        END $$;
        """
    )

    # ------------------------------------------------------------------
    # Group 1: Hierarchical Document Tree & Retrieval Chunks
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE documents (
            id              UUID PRIMARY KEY,
            tenant_id       TEXT NOT NULL,
            namespace       TEXT NOT NULL DEFAULT 'default',
            external_key    TEXT NOT NULL,
            title           TEXT NOT NULL,
            source_uri      TEXT,
            content_hash    TEXT NOT NULL,
            version         INT NOT NULL DEFAULT 1,
            is_latest       BOOLEAN NOT NULL DEFAULT TRUE,
            status          TEXT NOT NULL DEFAULT 'ready',
            effective_at    TIMESTAMPTZ,
            summary         TEXT,
            keywords        TEXT[] NOT NULL DEFAULT '{}',
            metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (tenant_id, namespace, external_key, version)
        );

        CREATE INDEX idx_documents_scope_latest
            ON documents (tenant_id, namespace, is_latest)
            WHERE is_latest = TRUE;

        CREATE TABLE doc_sections (
            id                UUID PRIMARY KEY,
            tenant_id         TEXT NOT NULL,
            document_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            parent_section_id UUID REFERENCES doc_sections(id) ON DELETE CASCADE,
            section_path      TEXT NOT NULL,
            heading_path      TEXT[] NOT NULL,
            heading           TEXT NOT NULL,
            level             SMALLINT NOT NULL,
            ordinal           INT NOT NULL,
            summary           TEXT,
            keywords          TEXT[] NOT NULL DEFAULT '{}',
            section_text      TEXT NOT NULL,
            token_count       INT NOT NULL
        );

        CREATE INDEX idx_doc_sections_doc_path
            ON doc_sections (document_id, section_path);
        """
    )

    op.execute(
        f"""
        CREATE TABLE doc_chunks (
            id                UUID PRIMARY KEY,
            tenant_id         TEXT NOT NULL,
            namespace         TEXT NOT NULL DEFAULT 'default',
            document_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            section_id        UUID REFERENCES doc_sections(id) ON DELETE SET NULL,
            chunk_index       INT NOT NULL,
            raw_text          TEXT NOT NULL,
            contextual_text   TEXT NOT NULL,
            enriched_keywords TEXT NOT NULL DEFAULT '',
            lexical_text      TEXT GENERATED ALWAYS AS (
                CASE
                    WHEN enriched_keywords = '' THEN contextual_text
                    ELSE contextual_text || E'\\n[Keywords: ' || enriched_keywords || ']'
                END
            ) STORED,
            search_tokens     TSVECTOR,
            token_count       INT NOT NULL,
            page_numbers      INT[],
            is_active         BOOLEAN NOT NULL DEFAULT TRUE,
            effective_at      TIMESTAMPTZ,
            embedding         halfvec({dim}),
            metadata          JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_doc_chunks_scope_active
            ON doc_chunks (tenant_id, namespace, document_id, chunk_index)
            WHERE is_active = TRUE;

        CREATE INDEX idx_doc_chunks_section
            ON doc_chunks (section_id, chunk_index);

        CREATE INDEX idx_doc_chunks_search_tokens_gin
            ON doc_chunks USING gin (search_tokens);

        CREATE INDEX idx_doc_chunks_embedding_hnsw
            ON doc_chunks USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 128);
        """
    )

    # Conditional ParadeDB BM25 index (Jieba tokenizer + multi-tenant Fast Fields)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
                EXECUTE $bm25$
                    CREATE INDEX idx_doc_chunks_bm25 ON doc_chunks
                    USING bm25 (id, tenant_id, namespace, is_active, document_id, lexical_text)
                    WITH (
                        key_field = 'id',
                        text_fields = '{
                            "lexical_text": { "tokenizer": { "type": "jieba" } },
                            "tenant_id": { "fast": true, "tokenizer": { "type": "raw" } },
                            "namespace": { "fast": true, "tokenizer": { "type": "raw" } }
                        }'
                    );
                $bm25$;
            END IF;
        END $$;
        """
    )

    # ------------------------------------------------------------------
    # Group 2: Lightweight Relation Graph (No External Graph DB)
    # ------------------------------------------------------------------
    op.execute(
        f"""
        CREATE TABLE entities (
            id              UUID PRIMARY KEY,
            tenant_id       TEXT NOT NULL,
            namespace       TEXT NOT NULL DEFAULT 'default',
            canonical_name  TEXT NOT NULL,
            entity_type     TEXT NOT NULL,
            aliases         TEXT[] NOT NULL DEFAULT '{{}}',
            description     TEXT,
            embedding       halfvec({dim}),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE UNIQUE INDEX uq_entities_canonical
            ON entities (tenant_id, namespace, entity_type, lower(canonical_name));

        CREATE TABLE relation_edges (
            id                UUID PRIMARY KEY,
            tenant_id         TEXT NOT NULL,
            namespace         TEXT NOT NULL DEFAULT 'default',
            src_entity_id     UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            rel_type          TEXT NOT NULL,
            dst_entity_id     UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            evidence_chunk_id UUID REFERENCES doc_chunks(id) ON DELETE CASCADE,
            weight            REAL NOT NULL DEFAULT 1.0,
            valid_from        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            valid_to          TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_edges_src
            ON relation_edges (tenant_id, namespace, src_entity_id)
            WHERE valid_to IS NULL;

        CREATE INDEX idx_edges_dst
            ON relation_edges (tenant_id, namespace, dst_entity_id)
            WHERE valid_to IS NULL;
        """
    )

    # ------------------------------------------------------------------
    # Group 3: Bi-temporal Three-Tier Memory (L1 / L2 / L3)
    # ------------------------------------------------------------------
    op.execute(
        f"""
        CREATE TABLE sessions (
            id                 UUID PRIMARY KEY,
            tenant_id          TEXT NOT NULL,
            user_id            TEXT NOT NULL,
            namespace          TEXT NOT NULL DEFAULT 'default',
            token_budget       INT NOT NULL DEFAULT 8000,
            pinned_constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
            last_active_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_sessions_user
            ON sessions (tenant_id, user_id, namespace, last_active_at DESC);

        CREATE TABLE session_messages (
            id              UUID PRIMARY KEY,
            session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            seq_no          BIGINT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            token_count     INT NOT NULL,
            metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (session_id, seq_no)
        );

        CREATE TABLE session_compactions (
            id                  UUID PRIMARY KEY,
            session_id          UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            compacted_up_to_seq BIGINT NOT NULL,
            summary_text        TEXT NOT NULL,
            preserved_facts     JSONB NOT NULL DEFAULT '[]'::jsonb,
            token_count         INT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_session_compactions_latest
            ON session_compactions (session_id, compacted_up_to_seq DESC);

        CREATE TABLE episodic_memories (
            id                 UUID PRIMARY KEY,
            tenant_id          TEXT NOT NULL,
            user_id            TEXT NOT NULL,
            namespace          TEXT NOT NULL DEFAULT 'default',
            session_id         UUID REFERENCES sessions(id) ON DELETE SET NULL,
            event_start_at     TIMESTAMPTZ NOT NULL,
            event_end_at       TIMESTAMPTZ NOT NULL,
            summary            TEXT NOT NULL,
            key_decisions      JSONB NOT NULL DEFAULT '[]'::jsonb,
            entities_mentioned TEXT[] NOT NULL DEFAULT '{{}}',
            importance_score   REAL NOT NULL DEFAULT 0.5,
            embedding          halfvec({dim}),
            source_seq_from    BIGINT,
            source_seq_to      BIGINT,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_episodic_scope_time
            ON episodic_memories (tenant_id, user_id, namespace, event_end_at DESC);

        CREATE INDEX idx_episodic_embedding_hnsw
            ON episodic_memories USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 64);

        CREATE TABLE semantic_memories (
            id                  UUID PRIMARY KEY,
            tenant_id           TEXT NOT NULL,
            user_id             TEXT NOT NULL,
            namespace           TEXT NOT NULL DEFAULT 'default',
            memory_type         TEXT NOT NULL,
            slot_key            TEXT,
            fact_text           TEXT NOT NULL,
            evidence_quote      TEXT,
            confidence          REAL NOT NULL DEFAULT 0.8,
            is_pinned           BOOLEAN NOT NULL DEFAULT FALSE,
            valid_from          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            valid_to            TIMESTAMPTZ,
            recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            invalidated_at      TIMESTAMPTZ,
            superseded_by       UUID REFERENCES semantic_memories(id),
            invalidation_reason TEXT,
            access_count        INT NOT NULL DEFAULT 0,
            last_accessed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            importance_score    REAL NOT NULL DEFAULT 0.6,
            source_episode_id   UUID REFERENCES episodic_memories(id) ON DELETE SET NULL,
            embedding           halfvec({dim})
        );

        CREATE INDEX idx_sem_mem_active
            ON semantic_memories (tenant_id, user_id, namespace, slot_key)
            WHERE valid_to IS NULL AND invalidated_at IS NULL;

        CREATE INDEX idx_sem_mem_bitemporal
            ON semantic_memories (tenant_id, user_id, namespace, valid_from, valid_to);

        CREATE INDEX idx_sem_mem_embedding_hnsw
            ON semantic_memories USING hnsw (embedding halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
    )

    # ------------------------------------------------------------------
    # Group 4: Zero-Redis PG Outbox Jobs & Structured Observability Traces
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE async_jobs (
            id              UUID PRIMARY KEY,
            job_type        TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload         JSONB NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            attempts        INT NOT NULL DEFAULT 0,
            max_attempts    INT NOT NULL DEFAULT 3,
            run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_error      TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_async_jobs_claimable
            ON async_jobs (status, run_at)
            WHERE status = 'pending';

        CREATE TABLE execution_traces (
            id                      UUID PRIMARY KEY,
            tenant_id               TEXT NOT NULL,
            session_id              UUID,
            channel                 TEXT NOT NULL,
            operation               TEXT NOT NULL,
            query_text              TEXT,
            final_status            TEXT NOT NULL,
            retrieval_retries       SMALLINT NOT NULL DEFAULT 0,
            total_latency_ms        INT NOT NULL,
            total_prompt_tokens     INT NOT NULL DEFAULT 0,
            total_completion_tokens INT NOT NULL DEFAULT 0,
            estimated_cost_usd      NUMERIC(10, 6) DEFAULT 0,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_traces_tenant_recent
            ON execution_traces (tenant_id, created_at DESC);

        CREATE TABLE trace_spans (
            id                UUID PRIMARY KEY,
            trace_id          UUID NOT NULL REFERENCES execution_traces(id) ON DELETE CASCADE,
            step_order        SMALLINT NOT NULL,
            node_name         TEXT NOT NULL,
            latency_ms        INT NOT NULL,
            prompt_tokens     INT NOT NULL DEFAULT 0,
            completion_tokens INT NOT NULL DEFAULT 0,
            input_meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
            output_meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX idx_trace_spans_trace
            ON trace_spans (trace_id, step_order);
        """
    )


def downgrade() -> None:
    for table in (
        "trace_spans",
        "execution_traces",
        "async_jobs",
        "semantic_memories",
        "episodic_memories",
        "session_compactions",
        "session_messages",
        "sessions",
        "relation_edges",
        "entities",
        "doc_chunks",
        "doc_sections",
        "documents",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
