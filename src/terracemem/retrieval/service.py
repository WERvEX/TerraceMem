"""Unified retrieval pipeline combining Hybrid RRF, 2-hop graph expansion, Rerank & Expand."""

from __future__ import annotations

import time
from uuid import UUID

from pydantic import BaseModel, Field

from terracemem.config import Settings
from terracemem.db.repositories import DocumentRepository, GraphRepository, RetrievalMode
from terracemem.domain.models import RetrievalFilter, RetrievedEvidence
from terracemem.llm.providers import EmbeddingProvider, RerankerProvider
from terracemem.retrieval.fusion import reciprocal_rank_fusion
from terracemem.retrieval.graph import extract_query_entity_candidates


class RetrievalStageResult(BaseModel):
    """Complete retrieval stage payload including intermediate diagnostics for tracing."""

    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    base_hit_count: int = 0
    graph_hit_count: int = 0
    seed_entities_matched: int = 0
    top_score: float = 0.0
    mode: RetrievalMode = "hybrid"
    retrieve_latency_ms: int = 0
    rerank_expand_latency_ms: int = 0


class RetrievalService:
    """Orchestrates dense+BM25 RRF, optional 2-hop graph walk, reranking & parent expansion."""

    def __init__(
        self,
        *,
        settings: Settings,
        doc_repo: DocumentRepository,
        embedder: EmbeddingProvider,
        reranker: RerankerProvider,
        graph_repo: GraphRepository | None = None,
    ) -> None:
        self._settings = settings
        self._doc_repo = doc_repo
        self._embedder = embedder
        self._reranker = reranker
        self._graph_repo = graph_repo

    async def search(
        self,
        query: str,
        filt: RetrievalFilter,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 6,
        candidate_limit: int = 30,
        enable_graph: bool | None = None,
        enable_rerank: bool = True,
        enable_parent_expand: bool = True,
    ) -> RetrievalStageResult:
        if not query.strip():
            return RetrievalStageResult(mode=mode)

        t0 = time.monotonic()
        query_embedding: list[float] | None = None
        if mode in ("hybrid", "dense_only"):
            vecs = await self._embedder.embed_texts([query])
            query_embedding = vecs[0] if vecs else None

        base_candidates = await self._doc_repo.retrieve(
            query,
            query_embedding,
            filt,
            mode=mode,
            candidate_limit=candidate_limit,
            final_limit=max(top_k * 2, 12),
            rrf_k=self._settings.rrf_k,
        )

        use_graph = (
            enable_graph if enable_graph is not None else self._settings.enable_graph_expansion
        )
        graph_candidates: list[RetrievedEvidence] = []
        seed_count = 0

        if use_graph and self._graph_repo is not None:
            seed_names = extract_query_entity_candidates(query)
            if seed_names:
                seed_ids = await self._graph_repo.find_seed_entities(
                    seed_names,
                    tenant_id=filt.tenant_id,
                    namespace=filt.namespace,
                    limit=8,
                )
                seed_count = len(seed_ids)
                if seed_ids:
                    graph_candidates = await self._graph_repo.expand_two_hop(
                        seed_ids,
                        filt,
                        limit=max(top_k, 8),
                    )

        merged_candidates = self._merge_base_and_graph(
            base_candidates,
            graph_candidates,
            final_limit=max(top_k * 2, 12),
        )
        retrieve_ms = round((time.monotonic() - t0) * 1000)
        if not merged_candidates:
            return RetrievalStageResult(
                evidence=[],
                base_hit_count=0,
                graph_hit_count=0,
                seed_entities_matched=seed_count,
                top_score=0.0,
                mode=mode,
                retrieve_latency_ms=retrieve_ms,
                rerank_expand_latency_ms=0,
            )

        t1 = time.monotonic()
        if enable_rerank:
            passages = [item.raw_text for item in merged_candidates]
            reranked_pairs = await self._reranker.rerank(
                query,
                passages,
                top_n=min(top_k, len(merged_candidates)),
            )
            reranked: list[RetrievedEvidence] = []
            for idx, score in reranked_pairs:
                reranked.append(merged_candidates[idx].model_copy(update={"rerank_score": score}))
            selected = reranked
        else:
            selected = merged_candidates[:top_k]

        if enable_parent_expand and selected:
            selected = await self._doc_repo.expand_context(
                selected,
                tenant_id=filt.tenant_id,
                namespace=filt.namespace,
                max_section_tokens=self._settings.parent_expansion_max_tokens,
            )
        rerank_ms = round((time.monotonic() - t1) * 1000)

        top_score = 0.0
        if selected:
            first = selected[0]
            top_score = first.rerank_score if first.rerank_score is not None else first.rrf_score

        return RetrievalStageResult(
            evidence=selected,
            base_hit_count=len(base_candidates),
            graph_hit_count=len(graph_candidates),
            seed_entities_matched=seed_count,
            top_score=top_score,
            mode=mode,
            retrieve_latency_ms=retrieve_ms,
            rerank_expand_latency_ms=rerank_ms,
        )

    def _merge_base_and_graph(
        self,
        base: list[RetrievedEvidence],
        graph: list[RetrievedEvidence],
        *,
        final_limit: int,
    ) -> list[RetrievedEvidence]:
        if not graph:
            return base[:final_limit]
        if not base:
            return graph[:final_limit]

        by_id: dict[UUID, RetrievedEvidence] = {}
        for item in base:
            by_id[item.chunk_id] = item
        for gitem in graph:
            existing = by_id.get(gitem.chunk_id)
            if existing is not None:
                by_id[gitem.chunk_id] = existing.model_copy(
                    update={"graph_score": gitem.graph_score}
                )
            else:
                by_id[gitem.chunk_id] = gitem

        fused_ids = reciprocal_rank_fusion(
            [[b.chunk_id for b in base], [g.chunk_id for g in graph]],
            k=self._settings.rrf_k,
            weights=[1.0, 0.75],
        )
        merged: list[RetrievedEvidence] = []
        for cid, score in fused_ids[:final_limit]:
            merged.append(by_id[cid].model_copy(update={"rrf_score": score}))
        return merged
