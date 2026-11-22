"""Parameterized SQL query templates for hybrid retrieval, expansion, and 2-hop graph walks."""

from __future__ import annotations

_FILTER_CLAUSE = """
    c.tenant_id = %(tenant_id)s
    AND c.namespace = %(namespace)s
    AND doc.status <> 'deleted'
    AND (NOT %(only_latest)s OR (c.is_active = TRUE AND doc.is_latest = TRUE))
    AND (%(doc_ids)s::uuid[] IS NULL OR c.document_id = ANY(%(doc_ids)s::uuid[]))
    AND (%(section_ids)s::uuid[] IS NULL OR c.section_id = ANY(%(section_ids)s::uuid[]))
    AND (
        %(effective_after)s::timestamptz IS NULL
        OR c.effective_at IS NULL
        OR c.effective_at >= %(effective_after)s::timestamptz
    )
    AND (
        %(effective_before)s::timestamptz IS NULL
        OR c.effective_at IS NULL
        OR c.effective_at <= %(effective_before)s::timestamptz
    )
"""

# Executed inside a transaction block with:
#   SET LOCAL hnsw.iterative_scan = 'relaxed_order';
HYBRID_RETRIEVE_PARADEDB_SQL = f"""
WITH dense_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vec)s::halfvec) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE {_FILTER_CLAUSE}
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(query_vec)s::halfvec
    LIMIT %(candidate_limit)s
),
lexical_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY paradedb.score(c.id) DESC) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE c.lexical_text @@@ %(query_text)s
      AND c.tenant_id === %(tenant_id)s
      AND c.namespace === %(namespace)s
      AND doc.status <> 'deleted'
      AND (NOT %(only_latest)s OR (c.is_active = TRUE AND doc.is_latest = TRUE))
      AND (%(doc_ids)s::uuid[] IS NULL OR c.document_id = ANY(%(doc_ids)s::uuid[]))
      AND (%(section_ids)s::uuid[] IS NULL OR c.section_id = ANY(%(section_ids)s::uuid[]))
      AND (
          %(effective_after)s::timestamptz IS NULL
          OR c.effective_at IS NULL
          OR c.effective_at >= %(effective_after)s::timestamptz
      )
      AND (
          %(effective_before)s::timestamptz IS NULL
          OR c.effective_at IS NULL
          OR c.effective_at <= %(effective_before)s::timestamptz
      )
    ORDER BY paradedb.score(c.id) DESC
    LIMIT %(candidate_limit)s
),
fused AS (
    SELECT
        COALESCE(d.id, l.id) AS chunk_id,
        d.rank_pos AS dense_rank,
        l.rank_pos AS lexical_rank,
        COALESCE(1.0 / (%(rrf_k)s + d.rank_pos), 0.0)
            + COALESCE(1.0 / (%(rrf_k)s + l.rank_pos), 0.0) AS rrf_score
    FROM dense_hits d
    FULL OUTER JOIN lexical_hits l ON d.id = l.id
)
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text,
    f.dense_rank,
    f.lexical_rank,
    f.rrf_score
FROM fused f
JOIN doc_chunks c ON c.id = f.chunk_id
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
ORDER BY f.rrf_score DESC
LIMIT %(final_limit)s;
"""

HYBRID_RETRIEVE_NATIVE_SQL = f"""
WITH dense_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vec)s::halfvec) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE {_FILTER_CLAUSE}
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(query_vec)s::halfvec
    LIMIT %(candidate_limit)s
),
lexical_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(
                c.search_tokens,
                plainto_tsquery('simple', %(segmented_query)s)
            ) DESC
        ) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE {_FILTER_CLAUSE}
      AND c.search_tokens @@ plainto_tsquery('simple', %(segmented_query)s)
    LIMIT %(candidate_limit)s
),
fused AS (
    SELECT
        COALESCE(d.id, l.id) AS chunk_id,
        d.rank_pos AS dense_rank,
        l.rank_pos AS lexical_rank,
        COALESCE(1.0 / (%(rrf_k)s + d.rank_pos), 0.0)
            + COALESCE(1.0 / (%(rrf_k)s + l.rank_pos), 0.0) AS rrf_score
    FROM dense_hits d
    FULL OUTER JOIN lexical_hits l ON d.id = l.id
)
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text,
    f.dense_rank,
    f.lexical_rank,
    f.rrf_score
FROM fused f
JOIN doc_chunks c ON c.id = f.chunk_id
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
ORDER BY f.rrf_score DESC
LIMIT %(final_limit)s;
"""

DENSE_RETRIEVE_SQL = f"""
WITH dense_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vec)s::halfvec) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE {_FILTER_CLAUSE}
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(query_vec)s::halfvec
    LIMIT %(final_limit)s
)
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text,
    d.rank_pos AS dense_rank,
    NULL::bigint AS lexical_rank,
    (1.0 / (%(rrf_k)s + d.rank_pos))::double precision AS rrf_score
FROM dense_hits d
JOIN doc_chunks c ON c.id = d.id
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
ORDER BY d.rank_pos ASC;
"""

LEXICAL_RETRIEVE_PARADEDB_SQL = """
WITH lexical_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY paradedb.score(c.id) DESC) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE c.lexical_text @@@ %(query_text)s
      AND c.tenant_id === %(tenant_id)s
      AND c.namespace === %(namespace)s
      AND doc.status <> 'deleted'
      AND (NOT %(only_latest)s OR (c.is_active = TRUE AND doc.is_latest = TRUE))
      AND (%(doc_ids)s::uuid[] IS NULL OR c.document_id = ANY(%(doc_ids)s::uuid[]))
      AND (%(section_ids)s::uuid[] IS NULL OR c.section_id = ANY(%(section_ids)s::uuid[]))
      AND (
          %(effective_after)s::timestamptz IS NULL
          OR c.effective_at IS NULL
          OR c.effective_at >= %(effective_after)s::timestamptz
      )
      AND (
          %(effective_before)s::timestamptz IS NULL
          OR c.effective_at IS NULL
          OR c.effective_at <= %(effective_before)s::timestamptz
      )
    ORDER BY paradedb.score(c.id) DESC
    LIMIT %(final_limit)s
)
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text,
    NULL::bigint AS dense_rank,
    l.rank_pos AS lexical_rank,
    (1.0 / (%(rrf_k)s + l.rank_pos))::double precision AS rrf_score
FROM lexical_hits l
JOIN doc_chunks c ON c.id = l.id
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
ORDER BY l.rank_pos ASC;
"""

LEXICAL_RETRIEVE_NATIVE_SQL = f"""
WITH lexical_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(
                c.search_tokens,
                plainto_tsquery('simple', %(segmented_query)s)
            ) DESC
        ) AS rank_pos
    FROM doc_chunks c
    JOIN documents doc ON doc.id = c.document_id
    WHERE {_FILTER_CLAUSE}
      AND c.search_tokens @@ plainto_tsquery('simple', %(segmented_query)s)
    LIMIT %(final_limit)s
)
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text,
    NULL::bigint AS dense_rank,
    l.rank_pos AS lexical_rank,
    (1.0 / (%(rrf_k)s + l.rank_pos))::double precision AS rrf_score
FROM lexical_hits l
JOIN doc_chunks c ON c.id = l.id
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
ORDER BY l.rank_pos ASC;
"""

FETCH_PARENT_SECTIONS_SQL = """
SELECT id, section_text, token_count
FROM doc_sections
WHERE tenant_id = %(tenant_id)s
  AND id = ANY(%(section_ids)s::uuid[]);
"""

FETCH_ADJACENT_CHUNKS_SQL = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.raw_text
FROM doc_chunks c
WHERE c.tenant_id = %(tenant_id)s
  AND c.namespace = %(namespace)s
  AND c.document_id = ANY(%(document_ids)s::uuid[])
  AND c.chunk_index = ANY(%(chunk_indices)s::int[])
ORDER BY c.document_id, c.chunk_index;
"""

TWO_HOP_GRAPH_EXPANSION_SQL = """
WITH RECURSIVE hop_walk AS (
    SELECT
        e.src_entity_id,
        e.dst_entity_id,
        e.rel_type,
        e.evidence_chunk_id,
        e.weight,
        1 AS depth,
        ARRAY[e.src_entity_id, e.dst_entity_id] AS visited_path
    FROM relation_edges e
    WHERE e.tenant_id = %(tenant_id)s
      AND e.namespace = %(namespace)s
      AND e.valid_to IS NULL
      AND e.src_entity_id = ANY(%(seed_entity_ids)s::uuid[])

    UNION ALL

    SELECT
        e.src_entity_id,
        e.dst_entity_id,
        e.rel_type,
        e.evidence_chunk_id,
        (w.weight * e.weight * 0.7)::real AS weight,
        w.depth + 1 AS depth,
        w.visited_path || e.dst_entity_id
    FROM hop_walk w
    JOIN relation_edges e
      ON e.src_entity_id = w.dst_entity_id
     AND e.tenant_id = %(tenant_id)s
     AND e.namespace = %(namespace)s
     AND e.valid_to IS NULL
    WHERE w.depth < 2
      AND NOT (e.dst_entity_id = ANY(w.visited_path))
)
SELECT
    evidence_chunk_id AS chunk_id,
    MAX(weight) AS graph_score
FROM hop_walk
WHERE evidence_chunk_id IS NOT NULL
GROUP BY evidence_chunk_id
ORDER BY graph_score DESC
LIMIT %(limit)s;
"""

HYDRATE_GRAPH_CHUNKS_SQL = f"""
SELECT
    c.id,
    c.document_id,
    doc.title AS document_title,
    c.section_id,
    COALESCE(sec.heading_path, ARRAY[]::text[]) AS heading_path,
    c.raw_text
FROM doc_chunks c
JOIN documents doc ON doc.id = c.document_id
LEFT JOIN doc_sections sec ON sec.id = c.section_id
WHERE {_FILTER_CLAUSE}
  AND c.id = ANY(%(chunk_ids)s::uuid[]);
"""
