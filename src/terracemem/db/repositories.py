"""Document, lightweight relation graph, and trace repositories (PostgreSQL & InMemory)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Literal, Protocol
from uuid import UUID

import jieba
import structlog
from psycopg.types.json import Jsonb

from terracemem.config import Settings
from terracemem.db.pool import DatabasePool
from terracemem.db.queries import (
    DENSE_RETRIEVE_SQL,
    FETCH_ADJACENT_CHUNKS_SQL,
    FETCH_PARENT_SECTIONS_SQL,
    HYBRID_RETRIEVE_NATIVE_SQL,
    HYBRID_RETRIEVE_PARADEDB_SQL,
    HYDRATE_GRAPH_CHUNKS_SQL,
    LEXICAL_RETRIEVE_NATIVE_SQL,
    LEXICAL_RETRIEVE_PARADEDB_SQL,
    TWO_HOP_GRAPH_EXPANSION_SQL,
)
from terracemem.db.row_mappers import (
    as_datetime,
    as_dict,
    as_float,
    as_int,
    as_opt_datetime,
    as_opt_int,
    as_opt_str,
    as_opt_uuid,
    as_str,
    as_str_list,
    as_uuid,
)
from terracemem.domain.models import (
    ChunkRecord,
    DocumentOutline,
    DocumentSummary,
    EntityRecord,
    ExecutionTraceRecord,
    IngestDocumentResult,
    ParsedDocument,
    RelationEdgeRecord,
    RetrievalFilter,
    RetrievedEvidence,
    SectionNode,
    TraceSpanRecord,
    utc_now,
)
from terracemem.llm.providers import cosine_similarity, format_pg_halfvec
from terracemem.retrieval.fusion import reciprocal_rank_fusion
from terracemem.utils.tokens import segment_for_native_tsvector

logger = structlog.get_logger(__name__)

RetrievalMode = Literal["hybrid", "dense_only", "lexical_only"]

_SAFE_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _sanitize_paradedb_query(query_text: str) -> str:
    """Strip Tantivy operator characters so user queries never trigger syntax errors."""
    tokens = _SAFE_QUERY_TOKEN_RE.findall(query_text)
    return " ".join(tokens)


def _tokenize_search(text: str) -> list[str]:
    return [tok.strip().lower() for tok in jieba.cut_for_search(text) if tok.strip()]


class DocumentRepository(Protocol):
    """Repository protocol for versioned documents, outlines, retrieval & context expansion."""

    async def upsert_document(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        embeddings: list[list[float]] | None = None,
        effective_at: datetime | None = None,
        enriched: bool = False,
        entities_indexed: int = 0,
        edges_indexed: int = 0,
    ) -> IngestDocumentResult: ...

    async def soft_delete_document(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> bool: ...

    async def list_documents(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        only_latest: bool = True,
        limit: int = 50,
    ) -> list[DocumentSummary]: ...

    async def get_document_outline(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> DocumentOutline | None: ...

    async def browse_outlines(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        limit: int = 10,
    ) -> list[DocumentOutline]: ...

    async def retrieve(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        filt: RetrievalFilter,
        *,
        mode: RetrievalMode = "hybrid",
        candidate_limit: int = 30,
        final_limit: int = 12,
        rrf_k: int = 60,
    ) -> list[RetrievedEvidence]: ...

    async def expand_context(
        self,
        evidence: list[RetrievedEvidence],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        max_section_tokens: int = 1200,
    ) -> list[RetrievedEvidence]: ...


class GraphRepository(Protocol):
    """Repository protocol for lightweight entity graph & bounded 2-hop traversal."""

    async def upsert_entities_and_edges(
        self,
        entities: list[EntityRecord],
        edges: list[RelationEdgeRecord],
    ) -> tuple[int, int]: ...

    async def find_seed_entities(
        self,
        names: list[str],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        limit: int = 10,
    ) -> list[UUID]: ...

    async def expand_two_hop(
        self,
        seed_entity_ids: list[UUID],
        filt: RetrievalFilter,
        *,
        limit: int = 10,
    ) -> list[RetrievedEvidence]: ...


class TraceRepository(Protocol):
    """Repository protocol for execution traces and span waterfalls."""

    async def save_trace(self, trace: ExecutionTraceRecord) -> None: ...

    async def list_traces(
        self,
        *,
        tenant_id: str = "default",
        limit: int = 30,
    ) -> list[ExecutionTraceRecord]: ...

    async def get_trace(
        self,
        trace_id: UUID,
        *,
        tenant_id: str | None = None,
    ) -> ExecutionTraceRecord | None: ...


# ------------------------------------------------------------------------------
# PostgreSQL Implementations
# ------------------------------------------------------------------------------


def _row_to_evidence(row: tuple[object, ...]) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=as_uuid(row[0]),
        document_id=as_uuid(row[1]),
        document_title=as_str(row[2]),
        section_id=as_opt_uuid(row[3]),
        heading_path=as_str_list(row[4]),
        raw_text=as_str(row[5]),
        dense_rank=as_opt_int(row[6]),
        lexical_rank=as_opt_int(row[7]),
        rrf_score=as_float(row[8]),
    )


def _row_to_doc_summary(row: tuple[object, ...]) -> DocumentSummary:
    return DocumentSummary(
        id=as_uuid(row[0]),
        tenant_id=as_str(row[1]),
        namespace=as_str(row[2]),
        external_key=as_str(row[3]),
        title=as_str(row[4]),
        version=as_int(row[5]),
        is_latest=bool(row[6]),
        status=as_str(row[7]),
        effective_at=as_opt_datetime(row[8]),
        summary=as_opt_str(row[9]),
        keywords=as_str_list(row[10]),
        created_at=as_datetime(row[11]),
        section_count=as_int(row[12]),
        chunk_count=as_int(row[13]),
    )


def _as_channel(val: object) -> Literal["rest", "sse", "mcp", "eval"]:
    raw = as_str(val, default="rest")
    if raw == "sse":
        return "sse"
    if raw == "mcp":
        return "mcp"
    if raw == "eval":
        return "eval"
    return "rest"


class PostgresDocumentRepository:
    """PostgreSQL + pgvector 0.8 + ParadeDB/Native document & chunk repository."""

    def __init__(self, pool: DatabasePool, settings: Settings) -> None:
        self._pool = pool
        self._settings = settings

    async def upsert_document(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        embeddings: list[list[float]] | None = None,
        effective_at: datetime | None = None,
        enriched: bool = False,
        entities_indexed: int = 0,
        edges_indexed: int = 0,
    ) -> IngestDocumentResult:
        if embeddings is not None and len(embeddings) != len(parsed.chunks):
            raise ValueError("embeddings length must match parsed.chunks length")

        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, version, content_hash
                FROM documents
                WHERE tenant_id = %(tenant_id)s
                  AND namespace = %(namespace)s
                  AND external_key = %(external_key)s
                  AND is_latest = TRUE
                FOR UPDATE;
                """,
                {
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                    "external_key": parsed.external_key,
                },
            )
            latest_row = await cur.fetchone()
            if latest_row is not None and as_str(latest_row[2]) == parsed.content_hash:
                return IngestDocumentResult(
                    document_id=as_uuid(latest_row[0]),
                    tenant_id=tenant_id,
                    namespace=namespace,
                    external_key=parsed.external_key,
                    title=parsed.title,
                    version=as_int(latest_row[1]),
                    action="unchanged",
                    section_count=len(parsed.sections),
                    chunk_count=len(parsed.chunks),
                    enriched=enriched,
                    embedded=embeddings is not None,
                    entities_indexed=entities_indexed,
                    edges_indexed=edges_indexed,
                )

            action: Literal["created", "updated"] = "created"
            if latest_row is not None:
                old_doc_id = as_uuid(latest_row[0])
                next_version = as_int(latest_row[1]) + 1
                action = "updated"
                await cur.execute(
                    """
                    UPDATE documents
                    SET is_latest = FALSE, updated_at = NOW()
                    WHERE id = %(old_id)s;
                    """,
                    {"old_id": old_doc_id},
                )
                await cur.execute(
                    """
                    UPDATE doc_chunks
                    SET is_active = FALSE
                    WHERE document_id = %(old_id)s;
                    """,
                    {"old_id": old_doc_id},
                )
            else:
                await cur.execute(
                    """
                    SELECT COALESCE(MAX(version), 0)
                    FROM documents
                    WHERE tenant_id = %(tenant_id)s
                      AND namespace = %(namespace)s
                      AND external_key = %(external_key)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "namespace": namespace,
                        "external_key": parsed.external_key,
                    },
                )
                max_row = await cur.fetchone()
                prev_max = as_int(max_row[0]) if max_row is not None else 0
                next_version = prev_max + 1
                if prev_max > 0:
                    action = "updated"

            await cur.execute(
                """
                INSERT INTO documents (
                    id, tenant_id, namespace, external_key, title, source_uri,
                    content_hash, version, is_latest, status, effective_at,
                    summary, keywords, metadata
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(namespace)s, %(external_key)s, %(title)s,
                    %(source_uri)s, %(content_hash)s, %(version)s, TRUE, 'ready',
                    %(effective_at)s, %(summary)s, %(keywords)s, %(metadata)s
                );
                """,
                {
                    "id": parsed.id,
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                    "external_key": parsed.external_key,
                    "title": parsed.title,
                    "source_uri": parsed.source_uri,
                    "content_hash": parsed.content_hash,
                    "version": next_version,
                    "effective_at": effective_at,
                    "summary": parsed.summary,
                    "keywords": parsed.keywords,
                    "metadata": Jsonb(parsed.metadata),
                },
            )

            for sec in parsed.sections:
                await cur.execute(
                    """
                    INSERT INTO doc_sections (
                        id, tenant_id, document_id, parent_section_id, section_path,
                        heading_path, heading, level, ordinal, summary, keywords,
                        section_text, token_count
                    ) VALUES (
                        %(id)s, %(tenant_id)s, %(document_id)s, %(parent_section_id)s,
                        %(section_path)s, %(heading_path)s, %(heading)s, %(level)s,
                        %(ordinal)s, %(summary)s, %(keywords)s, %(section_text)s,
                        %(token_count)s
                    );
                    """,
                    {
                        "id": sec.id,
                        "tenant_id": tenant_id,
                        "document_id": parsed.id,
                        "parent_section_id": sec.parent_section_id,
                        "section_path": sec.section_path,
                        "heading_path": sec.heading_path,
                        "heading": sec.heading,
                        "level": sec.level,
                        "ordinal": sec.ordinal,
                        "summary": sec.summary,
                        "keywords": sec.keywords,
                        "section_text": sec.section_text,
                        "token_count": sec.token_count,
                    },
                )

            for idx, chunk in enumerate(parsed.chunks):
                lexical_input = chunk.contextual_text
                if chunk.enriched_keywords:
                    lexical_input = f"{lexical_input}\n{chunk.enriched_keywords}"
                segmented = segment_for_native_tsvector(lexical_input)
                emb_literal = (
                    format_pg_halfvec(embeddings[idx]) if embeddings is not None else None
                )
                await cur.execute(
                    """
                    INSERT INTO doc_chunks (
                        id, tenant_id, namespace, document_id, section_id, chunk_index,
                        raw_text, contextual_text, enriched_keywords, search_tokens,
                        token_count, page_numbers, is_active, effective_at,
                        embedding, metadata
                    ) VALUES (
                        %(id)s, %(tenant_id)s, %(namespace)s, %(document_id)s,
                        %(section_id)s, %(chunk_index)s, %(raw_text)s, %(contextual_text)s,
                        %(enriched_keywords)s, to_tsvector('simple', %(segmented)s),
                        %(token_count)s, %(page_numbers)s, TRUE, %(effective_at)s,
                        %(embedding)s::halfvec, %(metadata)s
                    );
                    """,
                    {
                        "id": chunk.id,
                        "tenant_id": tenant_id,
                        "namespace": namespace,
                        "document_id": parsed.id,
                        "section_id": chunk.section_id,
                        "chunk_index": chunk.chunk_index,
                        "raw_text": chunk.raw_text,
                        "contextual_text": chunk.contextual_text,
                        "enriched_keywords": chunk.enriched_keywords,
                        "segmented": segmented,
                        "token_count": chunk.token_count,
                        "page_numbers": chunk.page_numbers or None,
                        "effective_at": effective_at,
                        "embedding": emb_literal,
                        "metadata": Jsonb(chunk.metadata),
                    },
                )

        return IngestDocumentResult(
            document_id=parsed.id,
            tenant_id=tenant_id,
            namespace=namespace,
            external_key=parsed.external_key,
            title=parsed.title,
            version=next_version,
            action=action,
            section_count=len(parsed.sections),
            chunk_count=len(parsed.chunks),
            enriched=enriched,
            embedded=embeddings is not None,
            entities_indexed=entities_indexed,
            edges_indexed=edges_indexed,
        )

    async def soft_delete_document(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> bool:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                SELECT external_key, is_latest
                FROM documents
                WHERE id = %(doc_id)s
                  AND tenant_id = %(tenant_id)s
                  AND namespace = %(namespace)s
                  AND status <> 'deleted'
                FOR UPDATE;
                """,
                {"doc_id": document_id, "tenant_id": tenant_id, "namespace": namespace},
            )
            target = await cur.fetchone()
            if target is None:
                return False
            ext_key = as_str(target[0])
            was_latest = bool(target[1])

            if was_latest:
                await cur.execute(
                    """
                    UPDATE documents
                    SET is_latest = FALSE, status = 'deleted', updated_at = NOW()
                    WHERE tenant_id = %(tenant_id)s
                      AND namespace = %(namespace)s
                      AND external_key = %(external_key)s
                      AND status <> 'deleted';
                    """,
                    {
                        "tenant_id": tenant_id,
                        "namespace": namespace,
                        "external_key": ext_key,
                    },
                )
                await cur.execute(
                    """
                    UPDATE doc_chunks c
                    SET is_active = FALSE
                    FROM documents d
                    WHERE c.document_id = d.id
                      AND d.tenant_id = %(tenant_id)s
                      AND d.namespace = %(namespace)s
                      AND d.external_key = %(external_key)s;
                    """,
                    {
                        "tenant_id": tenant_id,
                        "namespace": namespace,
                        "external_key": ext_key,
                    },
                )
            else:
                await cur.execute(
                    """
                    UPDATE documents
                    SET is_latest = FALSE, status = 'deleted', updated_at = NOW()
                    WHERE id = %(doc_id)s;
                    """,
                    {"doc_id": document_id},
                )
                await cur.execute(
                    """
                    UPDATE doc_chunks
                    SET is_active = FALSE
                    WHERE document_id = %(doc_id)s;
                    """,
                    {"doc_id": document_id},
                )
            return True

    async def list_documents(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        only_latest: bool = True,
        limit: int = 50,
    ) -> list[DocumentSummary]:
        like_pat = f"%{query.strip()}%" if query and query.strip() else None
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    d.id, d.tenant_id, d.namespace, d.external_key, d.title,
                    d.version, d.is_latest, d.status, d.effective_at,
                    d.summary, d.keywords, d.created_at,
                    (SELECT COUNT(*) FROM doc_sections s WHERE s.document_id = d.id) AS sec_cnt,
                    (SELECT COUNT(*) FROM doc_chunks c WHERE c.document_id = d.id) AS chk_cnt
                FROM documents d
                WHERE d.tenant_id = %(tenant_id)s
                  AND d.namespace = %(namespace)s
                  AND d.status <> 'deleted'
                  AND (NOT %(only_latest)s OR d.is_latest = TRUE)
                  AND (
                      %(like_pat)s::text IS NULL
                      OR d.title ILIKE %(like_pat)s
                      OR COALESCE(d.summary, '') ILIKE %(like_pat)s
                      OR array_to_string(d.keywords, ' ') ILIKE %(like_pat)s
                  )
                ORDER BY d.created_at DESC
                LIMIT %(limit)s;
                """,
                {
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                    "only_latest": only_latest,
                    "like_pat": like_pat,
                    "limit": limit,
                },
            )
            rows = await cur.fetchall()
        return [_row_to_doc_summary(row) for row in rows]

    async def get_document_outline(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> DocumentOutline | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    d.id, d.tenant_id, d.namespace, d.external_key, d.title,
                    d.version, d.is_latest, d.status, d.effective_at,
                    d.summary, d.keywords, d.created_at,
                    (SELECT COUNT(*) FROM doc_sections s WHERE s.document_id = d.id) AS sec_cnt,
                    (SELECT COUNT(*) FROM doc_chunks c WHERE c.document_id = d.id) AS chk_cnt
                FROM documents d
                WHERE d.id = %(doc_id)s
                  AND d.tenant_id = %(tenant_id)s
                  AND d.namespace = %(namespace)s
                  AND d.status <> 'deleted';
                """,
                {"doc_id": document_id, "tenant_id": tenant_id, "namespace": namespace},
            )
            doc_row = await cur.fetchone()
            if doc_row is None:
                return None

            await cur.execute(
                """
                SELECT
                    id, parent_section_id, section_path, heading_path, heading,
                    level, ordinal, summary, keywords, section_text, token_count
                FROM doc_sections
                WHERE document_id = %(doc_id)s AND tenant_id = %(tenant_id)s
                ORDER BY ordinal ASC;
                """,
                {"doc_id": document_id, "tenant_id": tenant_id},
            )
            sec_rows = await cur.fetchall()

        sections = [
            SectionNode(
                id=as_uuid(row[0]),
                parent_section_id=as_opt_uuid(row[1]),
                section_path=as_str(row[2]),
                heading_path=as_str_list(row[3]),
                heading=as_str(row[4]),
                level=as_int(row[5]),
                ordinal=as_int(row[6]),
                summary=as_opt_str(row[7]),
                keywords=as_str_list(row[8]),
                section_text=as_str(row[9]),
                token_count=as_int(row[10]),
            )
            for row in sec_rows
        ]
        return DocumentOutline(document=_row_to_doc_summary(doc_row), sections=sections)

    async def browse_outlines(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        limit: int = 10,
    ) -> list[DocumentOutline]:
        docs = await self.list_documents(
            tenant_id=tenant_id,
            namespace=namespace,
            query=query,
            only_latest=True,
            limit=limit,
        )
        if not docs and query:
            docs = await self.list_documents(
                tenant_id=tenant_id,
                namespace=namespace,
                query=None,
                only_latest=True,
                limit=limit,
            )
        outlines: list[DocumentOutline] = []
        for doc in docs:
            outline = await self.get_document_outline(
                doc.id,
                tenant_id=tenant_id,
                namespace=namespace,
            )
            if outline is not None:
                outlines.append(outline)
        return outlines

    async def retrieve(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        filt: RetrievalFilter,
        *,
        mode: RetrievalMode = "hybrid",
        candidate_limit: int = 30,
        final_limit: int = 12,
        rrf_k: int = 60,
    ) -> list[RetrievedEvidence]:
        use_paradedb = self._settings.lexical_backend == "paradedb"
        sanitized_query = _sanitize_paradedb_query(query_text)
        segmented_query = segment_for_native_tsvector(query_text)
        query_vec = format_pg_halfvec(query_embedding) if query_embedding is not None else None

        effective_mode = mode
        if effective_mode == "hybrid" and query_vec is None:
            effective_mode = "lexical_only"
        if effective_mode in ("hybrid", "lexical_only") and not (
            sanitized_query if use_paradedb else segmented_query
        ):
            if effective_mode == "lexical_only" or query_vec is None:
                return []
            effective_mode = "dense_only"
        if effective_mode == "dense_only" and query_vec is None:
            return []

        if effective_mode == "dense_only":
            sql = DENSE_RETRIEVE_SQL
        elif effective_mode == "lexical_only":
            sql = LEXICAL_RETRIEVE_PARADEDB_SQL if use_paradedb else LEXICAL_RETRIEVE_NATIVE_SQL
        else:
            sql = HYBRID_RETRIEVE_PARADEDB_SQL if use_paradedb else HYBRID_RETRIEVE_NATIVE_SQL

        params: dict[str, Any] = {
            "tenant_id": filt.tenant_id,
            "namespace": filt.namespace,
            "only_latest": filt.only_latest,
            "doc_ids": filt.document_ids,
            "section_ids": filt.section_ids,
            "effective_after": filt.effective_after,
            "effective_before": filt.effective_before,
            "query_vec": query_vec,
            "query_text": sanitized_query,
            "segmented_query": segmented_query,
            "candidate_limit": candidate_limit,
            "final_limit": final_limit,
            "rrf_k": rrf_k,
        }

        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if effective_mode in ("hybrid", "dense_only"):
                await cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        return [_row_to_evidence(row) for row in rows]

    async def expand_context(
        self,
        evidence: list[RetrievedEvidence],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        max_section_tokens: int = 1200,
    ) -> list[RetrievedEvidence]:
        if not evidence:
            return []

        section_ids = list({ev.section_id for ev in evidence if ev.section_id is not None})
        doc_ids = list({ev.document_id for ev in evidence})

        section_map: dict[UUID, tuple[str, int]] = {}
        chunk_index_by_id: dict[UUID, tuple[UUID, int]] = {}
        adjacent_map: dict[tuple[UUID, int], str] = {}

        async with self._pool.connection() as conn, conn.cursor() as cur:
            if section_ids:
                await cur.execute(
                    FETCH_PARENT_SECTIONS_SQL,
                    {"tenant_id": tenant_id, "section_ids": section_ids},
                )
                for row in await cur.fetchall():
                    section_map[as_uuid(row[0])] = (as_str(row[1]), as_int(row[2]))

            chunk_ids = [ev.chunk_id for ev in evidence]
            await cur.execute(
                """
                SELECT id, document_id, chunk_index
                FROM doc_chunks
                WHERE tenant_id = %(tenant_id)s
                  AND namespace = %(namespace)s
                  AND id = ANY(%(chunk_ids)s::uuid[]);
                """,
                {"tenant_id": tenant_id, "namespace": namespace, "chunk_ids": chunk_ids},
            )
            needed_indices: set[int] = set()
            for row in await cur.fetchall():
                cid = as_uuid(row[0])
                did = as_uuid(row[1])
                cidx = as_int(row[2])
                chunk_index_by_id[cid] = (did, cidx)
                needed_indices.update({max(0, cidx - 1), cidx, cidx + 1})

            if doc_ids and needed_indices:
                await cur.execute(
                    FETCH_ADJACENT_CHUNKS_SQL,
                    {
                        "tenant_id": tenant_id,
                        "namespace": namespace,
                        "document_ids": doc_ids,
                        "chunk_indices": sorted(needed_indices),
                    },
                )
                for row in await cur.fetchall():
                    adjacent_map[(as_uuid(row[1]), as_int(row[2]))] = as_str(row[3])

        expanded: list[RetrievedEvidence] = []
        for item in evidence:
            exp_text = item.raw_text
            if item.section_id and item.section_id in section_map:
                sec_text, sec_tokens = section_map[item.section_id]
                if sec_tokens <= max_section_tokens and sec_text.strip():
                    expanded.append(item.model_copy(update={"expanded_text": sec_text}))
                    continue

            loc = chunk_index_by_id.get(item.chunk_id)
            if loc is not None:
                did, cidx = loc
                neighbors = [
                    adjacent_map.get((did, cidx - 1)),
                    adjacent_map.get((did, cidx)) or item.raw_text,
                    adjacent_map.get((did, cidx + 1)),
                ]
                parts = [p for p in neighbors if p]
                if parts:
                    exp_text = "\n\n".join(parts)

            expanded.append(item.model_copy(update={"expanded_text": exp_text}))
        return expanded


class PostgresGraphRepository:
    """PostgreSQL lightweight entity & relation edge store with bounded 2-hop CTE walks."""

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    async def upsert_entities_and_edges(
        self,
        entities: list[EntityRecord],
        edges: list[RelationEdgeRecord],
    ) -> tuple[int, int]:
        if not entities and not edges:
            return (0, 0)

        id_remap: dict[UUID, UUID] = {}
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            for ent in entities:
                await cur.execute(
                    """
                    INSERT INTO entities (
                        id, tenant_id, namespace, canonical_name, entity_type, aliases, description
                    ) VALUES (
                        %(id)s, %(tenant_id)s, %(namespace)s, %(canonical_name)s,
                        %(entity_type)s, %(aliases)s, %(description)s
                    )
                    ON CONFLICT (tenant_id, namespace, entity_type, lower(canonical_name))
                    DO UPDATE SET
                        aliases = ARRAY(
                            SELECT DISTINCT unnest(entities.aliases || EXCLUDED.aliases)
                        ),
                        description = COALESCE(EXCLUDED.description, entities.description)
                    RETURNING id;
                    """,
                    {
                        "id": ent.id,
                        "tenant_id": ent.tenant_id,
                        "namespace": ent.namespace,
                        "canonical_name": ent.canonical_name,
                        "entity_type": ent.entity_type,
                        "aliases": ent.aliases,
                        "description": ent.description,
                    },
                )
                row = await cur.fetchone()
                if row is not None:
                    id_remap[ent.id] = as_uuid(row[0])

            edge_count = 0
            for edge in edges:
                src_id = id_remap.get(edge.src_entity_id, edge.src_entity_id)
                dst_id = id_remap.get(edge.dst_entity_id, edge.dst_entity_id)
                if src_id == dst_id:
                    continue
                await cur.execute(
                    """
                    INSERT INTO relation_edges (
                        id, tenant_id, namespace, src_entity_id, rel_type, dst_entity_id,
                        evidence_chunk_id, weight, valid_from, valid_to
                    ) VALUES (
                        %(id)s, %(tenant_id)s, %(namespace)s, %(src_entity_id)s, %(rel_type)s,
                        %(dst_entity_id)s, %(evidence_chunk_id)s, %(weight)s,
                        %(valid_from)s, %(valid_to)s
                    );
                    """,
                    {
                        "id": edge.id,
                        "tenant_id": edge.tenant_id,
                        "namespace": edge.namespace,
                        "src_entity_id": src_id,
                        "rel_type": edge.rel_type,
                        "dst_entity_id": dst_id,
                        "evidence_chunk_id": edge.evidence_chunk_id,
                        "weight": edge.weight,
                        "valid_from": edge.valid_from,
                        "valid_to": edge.valid_to,
                    },
                )
                edge_count += 1

        return (len(id_remap), edge_count)

    async def find_seed_entities(
        self,
        names: list[str],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        limit: int = 10,
    ) -> list[UUID]:
        normalized = [name.strip().lower() for name in names if name.strip()]
        if not normalized:
            return []
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id
                FROM entities
                WHERE tenant_id = %(tenant_id)s
                  AND namespace = %(namespace)s
                  AND (
                      lower(canonical_name) = ANY(%(names)s::text[])
                      OR EXISTS (
                          SELECT 1 FROM unnest(aliases) a
                          WHERE lower(a) = ANY(%(names)s::text[])
                      )
                  )
                LIMIT %(limit)s;
                """,
                {
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                    "names": normalized,
                    "limit": limit,
                },
            )
            rows = await cur.fetchall()
        return [as_uuid(row[0]) for row in rows]

    async def expand_two_hop(
        self,
        seed_entity_ids: list[UUID],
        filt: RetrievalFilter,
        *,
        limit: int = 10,
    ) -> list[RetrievedEvidence]:
        if not seed_entity_ids:
            return []
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                TWO_HOP_GRAPH_EXPANSION_SQL,
                {
                    "tenant_id": filt.tenant_id,
                    "namespace": filt.namespace,
                    "seed_entity_ids": seed_entity_ids,
                    "limit": limit,
                },
            )
            hop_rows = await cur.fetchall()
            if not hop_rows:
                return []

            score_by_chunk: dict[UUID, float] = {
                as_uuid(row[0]): as_float(row[1]) for row in hop_rows
            }
            await cur.execute(
                HYDRATE_GRAPH_CHUNKS_SQL,
                {
                    "tenant_id": filt.tenant_id,
                    "namespace": filt.namespace,
                    "only_latest": filt.only_latest,
                    "doc_ids": filt.document_ids,
                    "section_ids": filt.section_ids,
                    "effective_after": filt.effective_after,
                    "effective_before": filt.effective_before,
                    "chunk_ids": list(score_by_chunk.keys()),
                },
            )
            chunk_rows = await cur.fetchall()

        results = [
            RetrievedEvidence(
                chunk_id=as_uuid(row[0]),
                document_id=as_uuid(row[1]),
                document_title=as_str(row[2]),
                section_id=as_opt_uuid(row[3]),
                heading_path=as_str_list(row[4]),
                raw_text=as_str(row[5]),
                graph_score=score_by_chunk.get(as_uuid(row[0]), 0.0),
            )
            for row in chunk_rows
        ]
        results.sort(key=lambda item: -(item.graph_score or 0.0))
        return results[:limit]


class PostgresTraceRepository:
    """PostgreSQL observability trace & span store."""

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    async def save_trace(self, trace: ExecutionTraceRecord) -> None:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO execution_traces (
                    id, tenant_id, session_id, channel, operation, query_text,
                    final_status, retrieval_retries, total_latency_ms,
                    total_prompt_tokens, total_completion_tokens, created_at
                ) VALUES (
                    %(id)s, %(tenant_id)s, %(session_id)s, %(channel)s, %(operation)s,
                    %(query_text)s, %(final_status)s, %(retrieval_retries)s,
                    %(total_latency_ms)s, %(total_prompt_tokens)s,
                    %(total_completion_tokens)s, %(created_at)s
                )
                ON CONFLICT (id) DO NOTHING;
                """,
                {
                    "id": trace.id,
                    "tenant_id": trace.tenant_id,
                    "session_id": trace.session_id,
                    "channel": trace.channel,
                    "operation": trace.operation,
                    "query_text": trace.query_text,
                    "final_status": trace.final_status,
                    "retrieval_retries": trace.retrieval_retries,
                    "total_latency_ms": trace.total_latency_ms,
                    "total_prompt_tokens": trace.total_prompt_tokens,
                    "total_completion_tokens": trace.total_completion_tokens,
                    "created_at": trace.created_at,
                },
            )
            for span in trace.spans:
                await cur.execute(
                    """
                    INSERT INTO trace_spans (
                        id, trace_id, step_order, node_name, latency_ms,
                        prompt_tokens, completion_tokens, input_meta, output_meta, created_at
                    ) VALUES (
                        %(id)s, %(trace_id)s, %(step_order)s, %(node_name)s, %(latency_ms)s,
                        %(prompt_tokens)s, %(completion_tokens)s, %(input_meta)s,
                        %(output_meta)s, %(created_at)s
                    )
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    {
                        "id": span.id,
                        "trace_id": trace.id,
                        "step_order": span.step_order,
                        "node_name": span.node_name,
                        "latency_ms": span.latency_ms,
                        "prompt_tokens": span.prompt_tokens,
                        "completion_tokens": span.completion_tokens,
                        "input_meta": Jsonb(span.input_meta),
                        "output_meta": Jsonb(span.output_meta),
                        "created_at": span.created_at,
                    },
                )

    async def list_traces(
        self,
        *,
        tenant_id: str = "default",
        limit: int = 30,
    ) -> list[ExecutionTraceRecord]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id, tenant_id, session_id, channel, operation, query_text,
                    final_status, retrieval_retries, total_latency_ms,
                    total_prompt_tokens, total_completion_tokens, created_at
                FROM execution_traces
                WHERE tenant_id = %(tenant_id)s
                ORDER BY created_at DESC
                LIMIT %(limit)s;
                """,
                {"tenant_id": tenant_id, "limit": limit},
            )
            rows = await cur.fetchall()
        return [
            ExecutionTraceRecord(
                id=as_uuid(row[0]),
                tenant_id=as_str(row[1]),
                session_id=as_opt_uuid(row[2]),
                channel=_as_channel(row[3]),
                operation=as_str(row[4]),
                query_text=as_opt_str(row[5]),
                final_status=as_str(row[6]),
                retrieval_retries=as_int(row[7]),
                total_latency_ms=as_int(row[8]),
                total_prompt_tokens=as_int(row[9]),
                total_completion_tokens=as_int(row[10]),
                created_at=as_datetime(row[11]),
            )
            for row in rows
        ]

    async def get_trace(
        self,
        trace_id: UUID,
        *,
        tenant_id: str | None = None,
    ) -> ExecutionTraceRecord | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id, tenant_id, session_id, channel, operation, query_text,
                    final_status, retrieval_retries, total_latency_ms,
                    total_prompt_tokens, total_completion_tokens, created_at
                FROM execution_traces
                WHERE id = %(trace_id)s
                  AND (%(tenant_id)s::text IS NULL OR tenant_id = %(tenant_id)s);
                """,
                {"trace_id": trace_id, "tenant_id": tenant_id},
            )
            row = await cur.fetchone()
            if row is None:
                return None
            await cur.execute(
                """
                SELECT
                    id, step_order, node_name, latency_ms, prompt_tokens,
                    completion_tokens, input_meta, output_meta, created_at
                FROM trace_spans
                WHERE trace_id = %(trace_id)s
                ORDER BY step_order ASC;
                """,
                {"trace_id": trace_id},
            )
            span_rows = await cur.fetchall()

        spans = [
            TraceSpanRecord(
                id=as_uuid(srow[0]),
                step_order=as_int(srow[1]),
                node_name=as_str(srow[2]),
                latency_ms=as_int(srow[3]),
                prompt_tokens=as_int(srow[4]),
                completion_tokens=as_int(srow[5]),
                input_meta=as_dict(srow[6]),
                output_meta=as_dict(srow[7]),
                created_at=as_datetime(srow[8]),
            )
            for srow in span_rows
        ]
        return ExecutionTraceRecord(
            id=as_uuid(row[0]),
            tenant_id=as_str(row[1]),
            session_id=as_opt_uuid(row[2]),
            channel=_as_channel(row[3]),
            operation=as_str(row[4]),
            query_text=as_opt_str(row[5]),
            final_status=as_str(row[6]),
            retrieval_retries=as_int(row[7]),
            total_latency_ms=as_int(row[8]),
            total_prompt_tokens=as_int(row[9]),
            total_completion_tokens=as_int(row[10]),
            spans=spans,
            created_at=as_datetime(row[11]),
        )


# ------------------------------------------------------------------------------
# In-Memory Implementations (for deterministic unit tests, evals & zero-PG dev)
# ------------------------------------------------------------------------------


class _StoredDoc:
    def __init__(
        self,
        *,
        parsed: ParsedDocument,
        tenant_id: str,
        namespace: str,
        version: int,
        is_latest: bool,
        status: str,
        effective_at: datetime | None,
        created_at: datetime,
    ) -> None:
        self.parsed = parsed
        self.tenant_id = tenant_id
        self.namespace = namespace
        self.version = version
        self.is_latest = is_latest
        self.status = status
        self.effective_at = effective_at
        self.created_at = created_at


class _StoredChunk:
    def __init__(
        self,
        *,
        chunk: ChunkRecord,
        tenant_id: str,
        namespace: str,
        document_title: str,
        heading_path: list[str],
        is_active: bool,
        effective_at: datetime | None,
        embedding: list[float] | None,
        tokens: list[str],
    ) -> None:
        self.chunk = chunk
        self.tenant_id = tenant_id
        self.namespace = namespace
        self.document_title = document_title
        self.heading_path = heading_path
        self.is_active = is_active
        self.effective_at = effective_at
        self.embedding = embedding
        self.tokens = tokens


class InMemoryDocumentRepository:
    """In-memory repository enforcing identical filter, version & RRF semantics as PG."""

    def __init__(self) -> None:
        self._docs: dict[UUID, _StoredDoc] = {}
        self._sections: dict[UUID, tuple[str, SectionNode]] = {}
        self._chunks: dict[UUID, _StoredChunk] = {}
        self._chunks_by_doc_idx: dict[tuple[UUID, int], UUID] = {}

    async def upsert_document(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        embeddings: list[list[float]] | None = None,
        effective_at: datetime | None = None,
        enriched: bool = False,
        entities_indexed: int = 0,
        edges_indexed: int = 0,
    ) -> IngestDocumentResult:
        if embeddings is not None and len(embeddings) != len(parsed.chunks):
            raise ValueError("embeddings length must match parsed.chunks length")

        existing_latest: _StoredDoc | None = None
        max_ver = 0
        for stored in self._docs.values():
            if (
                stored.tenant_id == tenant_id
                and stored.namespace == namespace
                and stored.parsed.external_key == parsed.external_key
            ):
                max_ver = max(max_ver, stored.version)
                if stored.is_latest:
                    existing_latest = stored

        if (
            existing_latest is not None
            and existing_latest.parsed.content_hash == parsed.content_hash
        ):
            return IngestDocumentResult(
                document_id=existing_latest.parsed.id,
                tenant_id=tenant_id,
                namespace=namespace,
                external_key=parsed.external_key,
                title=parsed.title,
                version=existing_latest.version,
                action="unchanged",
                section_count=len(parsed.sections),
                chunk_count=len(parsed.chunks),
                enriched=enriched,
                embedded=embeddings is not None,
                entities_indexed=entities_indexed,
                edges_indexed=edges_indexed,
            )

        action: Literal["created", "updated"] = "created"
        if existing_latest is not None:
            existing_latest.is_latest = False
            for sc in self._chunks.values():
                if sc.chunk.document_id == existing_latest.parsed.id:
                    sc.is_active = False
            action = "updated"
        elif max_ver > 0:
            action = "updated"

        next_ver = max_ver + 1
        now = utc_now()
        self._docs[parsed.id] = _StoredDoc(
            parsed=parsed,
            tenant_id=tenant_id,
            namespace=namespace,
            version=next_ver,
            is_latest=True,
            status="ready",
            effective_at=effective_at,
            created_at=now,
        )

        section_headings: dict[UUID, list[str]] = {}
        for sec in parsed.sections:
            self._sections[sec.id] = (tenant_id, sec)
            section_headings[sec.id] = sec.heading_path

        for idx, chunk in enumerate(parsed.chunks):
            lexical_str = chunk.contextual_text
            if chunk.enriched_keywords:
                lexical_str = f"{lexical_str}\n[Keywords: {chunk.enriched_keywords}]"
            emb = embeddings[idx] if embeddings is not None else None
            stored_chunk = _StoredChunk(
                chunk=chunk,
                tenant_id=tenant_id,
                namespace=namespace,
                document_title=parsed.title,
                heading_path=section_headings.get(chunk.section_id, [])
                if chunk.section_id
                else [],
                is_active=True,
                effective_at=effective_at,
                embedding=emb,
                tokens=_tokenize_search(lexical_str),
            )
            self._chunks[chunk.id] = stored_chunk
            self._chunks_by_doc_idx[(parsed.id, chunk.chunk_index)] = chunk.id

        return IngestDocumentResult(
            document_id=parsed.id,
            tenant_id=tenant_id,
            namespace=namespace,
            external_key=parsed.external_key,
            title=parsed.title,
            version=next_ver,
            action=action,
            section_count=len(parsed.sections),
            chunk_count=len(parsed.chunks),
            enriched=enriched,
            embedded=embeddings is not None,
            entities_indexed=entities_indexed,
            edges_indexed=edges_indexed,
        )

    async def soft_delete_document(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> bool:
        stored = self._docs.get(document_id)
        if (
            stored is None
            or stored.tenant_id != tenant_id
            or stored.namespace != namespace
            or stored.status == "deleted"
        ):
            return False

        target_ids: set[UUID]
        if stored.is_latest:
            target_ids = {
                doc.parsed.id
                for doc in self._docs.values()
                if doc.tenant_id == tenant_id
                and doc.namespace == namespace
                and doc.parsed.external_key == stored.parsed.external_key
            }
        else:
            target_ids = {document_id}

        for tid in target_ids:
            doc = self._docs[tid]
            doc.is_latest = False
            doc.status = "deleted"
        for sc in self._chunks.values():
            if sc.chunk.document_id in target_ids:
                sc.is_active = False
        return True

    def _to_summary(self, stored: _StoredDoc) -> DocumentSummary:
        return DocumentSummary(
            id=stored.parsed.id,
            tenant_id=stored.tenant_id,
            namespace=stored.namespace,
            external_key=stored.parsed.external_key,
            title=stored.parsed.title,
            version=stored.version,
            is_latest=stored.is_latest,
            status=stored.status,
            effective_at=stored.effective_at,
            summary=stored.parsed.summary,
            keywords=list(stored.parsed.keywords),
            section_count=len(stored.parsed.sections),
            chunk_count=len(stored.parsed.chunks),
            created_at=stored.created_at,
        )

    async def list_documents(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        only_latest: bool = True,
        limit: int = 50,
    ) -> list[DocumentSummary]:
        q = query.strip().lower() if query and query.strip() else None
        matched: list[_StoredDoc] = []
        for stored in self._docs.values():
            if stored.tenant_id != tenant_id or stored.namespace != namespace:
                continue
            if stored.status == "deleted":
                continue
            if only_latest and not stored.is_latest:
                continue
            if q:
                haystack = " ".join(
                    [
                        stored.parsed.title,
                        stored.parsed.summary or "",
                        " ".join(stored.parsed.keywords),
                    ]
                ).lower()
                if q not in haystack:
                    continue
            matched.append(stored)
        matched.sort(key=lambda d: d.created_at, reverse=True)
        return [self._to_summary(item) for item in matched[:limit]]

    async def get_document_outline(
        self,
        document_id: UUID,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> DocumentOutline | None:
        stored = self._docs.get(document_id)
        if (
            stored is None
            or stored.tenant_id != tenant_id
            or stored.namespace != namespace
            or stored.status == "deleted"
        ):
            return None
        sections = sorted(stored.parsed.sections, key=lambda s: s.ordinal)
        return DocumentOutline(document=self._to_summary(stored), sections=sections)

    async def browse_outlines(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        query: str | None = None,
        limit: int = 10,
    ) -> list[DocumentOutline]:
        docs = await self.list_documents(
            tenant_id=tenant_id,
            namespace=namespace,
            query=query,
            only_latest=True,
            limit=limit,
        )
        if not docs and query:
            docs = await self.list_documents(
                tenant_id=tenant_id,
                namespace=namespace,
                query=None,
                only_latest=True,
                limit=limit,
            )
        outlines: list[DocumentOutline] = []
        for doc in docs:
            outline = await self.get_document_outline(
                doc.id,
                tenant_id=tenant_id,
                namespace=namespace,
            )
            if outline is not None:
                outlines.append(outline)
        return outlines

    def _matches_filter(self, sc: _StoredChunk, filt: RetrievalFilter) -> bool:
        if sc.tenant_id != filt.tenant_id or sc.namespace != filt.namespace:
            return False
        doc = self._docs.get(sc.chunk.document_id)
        if doc is None or doc.status == "deleted":
            return False
        if filt.only_latest and not (sc.is_active and doc.is_latest):
            return False
        if filt.document_ids is not None and sc.chunk.document_id not in filt.document_ids:
            return False
        if filt.section_ids is not None and sc.chunk.section_id not in filt.section_ids:
            return False
        if (
            filt.effective_after is not None
            and sc.effective_at is not None
            and sc.effective_at < filt.effective_after
        ):
            return False
        return not (
            filt.effective_before is not None
            and sc.effective_at is not None
            and sc.effective_at > filt.effective_before
        )

    def hydrate_chunk_ids(
        self,
        chunk_scores: dict[UUID, float],
        filt: RetrievalFilter,
        *,
        limit: int,
    ) -> list[RetrievedEvidence]:
        results: list[RetrievedEvidence] = []
        for cid, gscore in chunk_scores.items():
            sc = self._chunks.get(cid)
            if sc is None or not self._matches_filter(sc, filt):
                continue
            results.append(
                RetrievedEvidence(
                    chunk_id=sc.chunk.id,
                    document_id=sc.chunk.document_id,
                    document_title=sc.document_title,
                    section_id=sc.chunk.section_id,
                    heading_path=sc.heading_path,
                    raw_text=sc.chunk.raw_text,
                    graph_score=gscore,
                )
            )
        results.sort(key=lambda item: -(item.graph_score or 0.0))
        return results[:limit]

    async def retrieve(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        filt: RetrievalFilter,
        *,
        mode: RetrievalMode = "hybrid",
        candidate_limit: int = 30,
        final_limit: int = 12,
        rrf_k: int = 60,
    ) -> list[RetrievedEvidence]:
        candidates = [sc for sc in self._chunks.values() if self._matches_filter(sc, filt)]
        if not candidates:
            return []

        dense_ranked: list[UUID] = []
        if mode in ("hybrid", "dense_only") and query_embedding is not None:
            scored_dense: list[tuple[UUID, float]] = []
            for sc in candidates:
                if sc.embedding is not None:
                    sim = cosine_similarity(query_embedding, sc.embedding)
                    scored_dense.append((sc.chunk.id, sim))
            scored_dense.sort(key=lambda pair: (-pair[1], str(pair[0])))
            dense_ranked = [cid for cid, _ in scored_dense[:candidate_limit]]

        lexical_ranked: list[UUID] = []
        if mode in ("hybrid", "lexical_only"):
            q_tokens = [tok for tok in _tokenize_search(query_text) if len(tok) > 0]
            if q_tokens:
                df: Counter[str] = Counter()
                for sc in candidates:
                    for tok in set(sc.tokens):
                        df[tok] += 1
                num_docs = len(candidates)
                avg_dl = sum(len(sc.tokens) for sc in candidates) / max(1, num_docs)
                scored_lex: list[tuple[UUID, float]] = []
                k1, b = 1.5, 0.75
                for sc in candidates:
                    tf = Counter(sc.tokens)
                    dl = max(1, len(sc.tokens))
                    bm25 = 0.0
                    for qt in q_tokens:
                        freq = tf.get(qt, 0)
                        if freq == 0:
                            continue
                        idf = math.log(1.0 + (num_docs - df[qt] + 0.5) / (df[qt] + 0.5))
                        denom = freq + k1 * (1.0 - b + b * dl / avg_dl)
                        bm25 += idf * (freq * (k1 + 1.0)) / denom
                    if bm25 > 0.0:
                        scored_lex.append((sc.chunk.id, bm25))
                scored_lex.sort(key=lambda pair: (-pair[1], str(pair[0])))
                lexical_ranked = [cid for cid, _ in scored_lex[:candidate_limit]]

        if mode == "dense_only":
            ordered = [(cid, 1.0 / (rrf_k + pos)) for pos, cid in enumerate(dense_ranked, start=1)]
        elif mode == "lexical_only":
            ordered = [
                (cid, 1.0 / (rrf_k + pos)) for pos, cid in enumerate(lexical_ranked, start=1)
            ]
        else:
            lists: list[list[UUID]] = []
            if dense_ranked:
                lists.append(dense_ranked)
            if lexical_ranked:
                lists.append(lexical_ranked)
            if not lists:
                return []
            ordered = reciprocal_rank_fusion(lists, k=rrf_k)

        dense_pos = {cid: idx for idx, cid in enumerate(dense_ranked, start=1)}
        lex_pos = {cid: idx for idx, cid in enumerate(lexical_ranked, start=1)}
        results: list[RetrievedEvidence] = []
        for cid, rrf in ordered[:final_limit]:
            sc = self._chunks[cid]
            results.append(
                RetrievedEvidence(
                    chunk_id=sc.chunk.id,
                    document_id=sc.chunk.document_id,
                    document_title=sc.document_title,
                    section_id=sc.chunk.section_id,
                    heading_path=sc.heading_path,
                    raw_text=sc.chunk.raw_text,
                    dense_rank=dense_pos.get(cid),
                    lexical_rank=lex_pos.get(cid),
                    rrf_score=rrf,
                )
            )
        return results

    async def expand_context(
        self,
        evidence: list[RetrievedEvidence],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        max_section_tokens: int = 1200,
    ) -> list[RetrievedEvidence]:
        del namespace
        expanded: list[RetrievedEvidence] = []
        for item in evidence:
            exp_text = item.raw_text
            if item.section_id and item.section_id in self._sections:
                sec_tenant, sec = self._sections[item.section_id]
                if sec_tenant == tenant_id and sec.token_count <= max_section_tokens:
                    expanded.append(item.model_copy(update={"expanded_text": sec.section_text}))
                    continue

            sc = self._chunks.get(item.chunk_id)
            if sc is not None:
                cidx = sc.chunk.chunk_index
                did = sc.chunk.document_id
                parts: list[str] = []
                for idx in (cidx - 1, cidx, cidx + 1):
                    adj_id = self._chunks_by_doc_idx.get((did, idx))
                    if adj_id is not None and adj_id in self._chunks:
                        parts.append(self._chunks[adj_id].chunk.raw_text)
                if parts:
                    exp_text = "\n\n".join(parts)

            expanded.append(item.model_copy(update={"expanded_text": exp_text}))
        return expanded


class InMemoryGraphRepository:
    """In-memory graph repository implementing bounded 2-hop cycle-safe BFS traversal."""

    def __init__(self, doc_repo: InMemoryDocumentRepository) -> None:
        self._doc_repo = doc_repo
        # key: (tenant_id, namespace, entity_type, lower_name) -> EntityRecord
        self._entities_by_key: dict[tuple[str, str, str, str], EntityRecord] = {}
        self._entities_by_id: dict[UUID, EntityRecord] = {}
        self._edges_by_src: dict[tuple[str, str, UUID], list[RelationEdgeRecord]] = defaultdict(
            list
        )

    async def upsert_entities_and_edges(
        self,
        entities: list[EntityRecord],
        edges: list[RelationEdgeRecord],
    ) -> tuple[int, int]:
        id_remap: dict[UUID, UUID] = {}
        for ent in entities:
            key = (ent.tenant_id, ent.namespace, ent.entity_type, ent.canonical_name.lower())
            existing = self._entities_by_key.get(key)
            if existing is not None:
                merged_aliases = list(dict.fromkeys([*existing.aliases, *ent.aliases]))
                updated = existing.model_copy(
                    update={
                        "aliases": merged_aliases,
                        "description": ent.description or existing.description,
                    }
                )
                self._entities_by_key[key] = updated
                self._entities_by_id[existing.id] = updated
                id_remap[ent.id] = existing.id
            else:
                self._entities_by_key[key] = ent
                self._entities_by_id[ent.id] = ent
                id_remap[ent.id] = ent.id

        edge_count = 0
        for edge in edges:
            src_id = id_remap.get(edge.src_entity_id, edge.src_entity_id)
            dst_id = id_remap.get(edge.dst_entity_id, edge.dst_entity_id)
            if src_id == dst_id:
                continue
            stored_edge = edge.model_copy(update={"src_entity_id": src_id, "dst_entity_id": dst_id})
            self._edges_by_src[(edge.tenant_id, edge.namespace, src_id)].append(stored_edge)
            edge_count += 1
        return (len(id_remap), edge_count)

    async def find_seed_entities(
        self,
        names: list[str],
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        limit: int = 10,
    ) -> list[UUID]:
        wanted = {name.strip().lower() for name in names if name.strip()}
        if not wanted:
            return []
        matched: list[UUID] = []
        for ent in self._entities_by_id.values():
            if ent.tenant_id != tenant_id or ent.namespace != namespace:
                continue
            candidates = {ent.canonical_name.lower(), *(a.lower() for a in ent.aliases)}
            if wanted & candidates:
                matched.append(ent.id)
                if len(matched) >= limit:
                    break
        return matched

    async def expand_two_hop(
        self,
        seed_entity_ids: list[UUID],
        filt: RetrievalFilter,
        *,
        limit: int = 10,
    ) -> list[RetrievedEvidence]:
        if not seed_entity_ids:
            return []

        chunk_scores: dict[UUID, float] = {}
        # Queue items: (entity_id, depth, weight, visited_path)
        frontier: list[tuple[UUID, int, float, tuple[UUID, ...]]] = [
            (seed_id, 0, 1.0, (seed_id,)) for seed_id in seed_entity_ids
        ]

        while frontier:
            curr_id, depth, curr_weight, visited = frontier.pop(0)
            if depth >= 2:
                continue
            for edge in self._edges_by_src.get((filt.tenant_id, filt.namespace, curr_id), []):
                if edge.valid_to is not None:
                    continue
                if edge.dst_entity_id in visited:
                    continue
                next_weight = curr_weight * edge.weight * (1.0 if depth == 0 else 0.7)
                if edge.evidence_chunk_id is not None:
                    prev = chunk_scores.get(edge.evidence_chunk_id, 0.0)
                    if next_weight > prev:
                        chunk_scores[edge.evidence_chunk_id] = next_weight
                frontier.append(
                    (edge.dst_entity_id, depth + 1, next_weight, (*visited, edge.dst_entity_id))
                )

        return self._doc_repo.hydrate_chunk_ids(chunk_scores, filt, limit=limit)


class InMemoryTraceRepository:
    """In-memory execution trace store for tests, evals, and local Inspector UI."""

    def __init__(self) -> None:
        self._traces: dict[UUID, ExecutionTraceRecord] = {}

    async def save_trace(self, trace: ExecutionTraceRecord) -> None:
        self._traces[trace.id] = trace

    async def list_traces(
        self,
        *,
        tenant_id: str = "default",
        limit: int = 30,
    ) -> list[ExecutionTraceRecord]:
        items = [t for t in self._traces.values() if t.tenant_id == tenant_id]
        items.sort(key=lambda t: t.created_at, reverse=True)
        return items[:limit]

    async def get_trace(
        self,
        trace_id: UUID,
        *,
        tenant_id: str | None = None,
    ) -> ExecutionTraceRecord | None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        if tenant_id is not None and trace.tenant_id != tenant_id:
            return None
        return trace
