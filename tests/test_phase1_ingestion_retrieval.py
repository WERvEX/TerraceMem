"""Phase 1 tests: versioned ingestion, soft delete, multi-tenant filters, enrichment & 2-hop."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from terracemem.config import Settings
from terracemem.db.repositories import InMemoryDocumentRepository, InMemoryGraphRepository
from terracemem.domain.models import (
    EntityRecord,
    RelationEdgeRecord,
    RetrievalFilter,
)
from terracemem.ingestion.docling_parser import DoclingParser, DoclingTextBlock, ParserRegistry
from terracemem.ingestion.enricher import WriteTimeEnricher
from terracemem.ingestion.pipeline import IngestionService
from terracemem.llm.providers import (
    DeterministicHashEmbeddingProvider,
    HTTPRerankerProvider,
    LexicalOverlapReranker,
)
from terracemem.retrieval.graph import EntityRelationExtractor
from terracemem.retrieval.service import RetrievalService


def _build_services(
    *,
    enable_enrichment: bool = False,
    enable_graph: bool = True,
) -> tuple[
    InMemoryDocumentRepository,
    InMemoryGraphRepository,
    IngestionService,
    RetrievalService,
]:
    settings = Settings(
        enable_write_time_enrichment=enable_enrichment,
        enable_graph_expansion=enable_graph,
    )
    doc_repo = InMemoryDocumentRepository()
    graph_repo = InMemoryGraphRepository(doc_repo)
    embedder = DeterministicHashEmbeddingProvider(dim=256)
    reranker = LexicalOverlapReranker()
    enricher = WriteTimeEnricher()
    extractor = EntityRelationExtractor()
    ingester = IngestionService(
        settings=settings,
        doc_repo=doc_repo,
        embedder=embedder,
        enricher=enricher,
        graph_repo=graph_repo,
        graph_extractor=extractor,
    )
    retriever = RetrievalService(
        settings=settings,
        doc_repo=doc_repo,
        embedder=embedder,
        reranker=reranker,
        graph_repo=graph_repo,
    )
    return doc_repo, graph_repo, ingester, retriever


async def test_idempotent_upsert_version_bump_and_soft_delete_zero_leakage() -> None:
    doc_repo, _, ingester, retriever = _build_services()

    v1 = await ingester.ingest_text(
        "# Auth Spec\n\n## Token Format\nTokens expire after 3600 seconds using HS256.",
        title="Auth Spec",
        external_key="auth-spec",
        tenant_id="acme",
        namespace="prod",
    )
    assert v1.action == "created"
    assert v1.version == 1

    # Same content hash -> unchanged
    v1_dup = await ingester.ingest_text(
        "# Auth Spec\n\n## Token Format\nTokens expire after 3600 seconds using HS256.",
        title="Auth Spec",
        external_key="auth-spec",
        tenant_id="acme",
        namespace="prod",
    )
    assert v1_dup.action == "unchanged"
    assert v1_dup.document_id == v1.document_id

    # Updated content -> version 2, v1 chunks deactivated
    v2 = await ingester.ingest_text(
        "# Auth Spec\n\n## Token Format\nTokens expire after 900 seconds using EdDSA.",
        title="Auth Spec",
        external_key="auth-spec",
        tenant_id="acme",
        namespace="prod",
    )
    assert v2.action == "updated"
    assert v2.version == 2

    res = await retriever.search(
        "HS256 3600",
        RetrievalFilter(tenant_id="acme", namespace="prod", only_latest=True),
        top_k=5,
    )
    assert all("HS256" not in ev.raw_text for ev in res.evidence)

    # Tenant & namespace isolation
    other_tenant = await retriever.search(
        "EdDSA 900",
        RetrievalFilter(tenant_id="other-tenant", namespace="prod"),
    )
    assert other_tenant.evidence == []
    other_ns = await retriever.search(
        "EdDSA 900",
        RetrievalFilter(tenant_id="acme", namespace="staging"),
    )
    assert other_ns.evidence == []

    # Soft delete eliminates all visibility
    deleted = await doc_repo.soft_delete_document(
        v2.document_id, tenant_id="acme", namespace="prod"
    )
    assert deleted is True
    after_del = await retriever.search(
        "EdDSA 900",
        RetrievalFilter(tenant_id="acme", namespace="prod", only_latest=False),
    )
    assert after_del.evidence == []


async def test_effective_time_filter_and_retrieval_modes() -> None:
    _, _, ingester, retriever = _build_services(enable_enrichment=True)
    t_old = datetime(2024, 1, 1, tzinfo=UTC)
    t_new = datetime(2026, 6, 1, tzinfo=UTC)

    await ingester.ingest_text(
        "# Error Codes\n\n## ERR_GATEWAY_42\nLegacy retry interval was 60s.",
        title="Legacy Manual",
        external_key="manual-2024",
        effective_at=t_old,
    )
    await ingester.ingest_text(
        "# Error Codes\n\n## ERR_GATEWAY_42\nCurrent retry interval is 5s with jitter.",
        title="Current Manual",
        external_key="manual-2026",
        effective_at=t_new,
    )

    filt_recent = RetrievalFilter(effective_after=datetime(2026, 1, 1, tzinfo=UTC))
    for mode in ("hybrid", "dense_only", "lexical_only"):
        res = await retriever.search("ERR_GATEWAY_42 retry interval", filt_recent, mode=mode)
        assert len(res.evidence) == 1
        assert "5s with jitter" in res.evidence[0].raw_text


async def test_write_time_enrichment_and_parent_expansion_keep_raw_text_intact() -> None:
    doc_repo, _, ingester, retriever = _build_services(enable_enrichment=True)
    body = (
        "# Architecture\n\n"
        "## Storage Engine\n"
        "Paragraph one describes ParadeDB Tantivy BM25 indexing.\n\n"
        "Paragraph two explains pgvector iterative HNSW scanning with halfvec.\n\n"
        "Paragraph three details transactional outbox jobs."
    )
    res = await ingester.ingest_text(
        body,
        title="Architecture",
        external_key="arch",
        max_chunk_tokens=16,
    )
    assert res.enriched is True
    assert res.chunk_count >= 2

    outline = await doc_repo.get_document_outline(res.document_id)
    assert outline is not None
    assert outline.document.summary
    assert len(outline.document.keywords) > 0
    assert any(sec.summary for sec in outline.sections)

    search_res = await retriever.search("iterative HNSW halfvec", RetrievalFilter(), top_k=1)
    assert len(search_res.evidence) == 1
    ev = search_res.evidence[0]
    assert ev.expanded_text is not None
    assert len(ev.expanded_text) > len(ev.raw_text)
    assert "ParadeDB" in ev.expanded_text


async def test_two_hop_graph_cycle_prevention() -> None:
    doc_repo, graph_repo, ingester, _ = _build_services(enable_graph=False)
    ing_res = await ingester.ingest_text(
        "# Graph Doc\n\n## Hop1\nAlpha connects to Beta.\n\n## Hop2\nBeta connects to Gamma.",
        title="Graph Doc",
        external_key="graph-doc",
    )
    outlines = await doc_repo.browse_outlines()
    assert len(outlines) == 1
    assert ing_res.chunk_count >= 2

    # Retrieve chunk IDs for evidence attachment
    hits = await doc_repo.retrieve(
        "Beta connects to Gamma", None, RetrievalFilter(), mode="lexical_only"
    )
    assert hits
    gamma_chunk_id = hits[0].chunk_id

    alpha = EntityRecord(canonical_name="Alpha", entity_type="service")
    beta = EntityRecord(canonical_name="Beta", entity_type="service")
    gamma = EntityRecord(canonical_name="Gamma", entity_type="service")
    # Build cycle Alpha -> Beta -> Gamma -> Alpha
    edges = [
        RelationEdgeRecord(
            src_entity_id=alpha.id,
            rel_type="calls",
            dst_entity_id=beta.id,
            evidence_chunk_id=None,
            weight=1.0,
        ),
        RelationEdgeRecord(
            src_entity_id=beta.id,
            rel_type="calls",
            dst_entity_id=gamma.id,
            evidence_chunk_id=gamma_chunk_id,
            weight=0.9,
        ),
        RelationEdgeRecord(
            src_entity_id=gamma.id,
            rel_type="calls",
            dst_entity_id=alpha.id,
            evidence_chunk_id=gamma_chunk_id,
            weight=0.8,
        ),
    ]
    await graph_repo.upsert_entities_and_edges([alpha, beta, gamma], edges)
    seeds = await graph_repo.find_seed_entities(["Alpha"])
    assert seeds == [alpha.id]

    expanded = await graph_repo.expand_two_hop(seeds, RetrievalFilter(), limit=5)
    assert len(expanded) == 1
    assert expanded[0].chunk_id == gamma_chunk_id
    assert expanded[0].graph_score == pytest.approx(0.9 * 0.7, rel=1e-4)


async def test_http_reranker_mock_and_fallback(monkeypatch: Any) -> None:
    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.96},
                    {"index": 0, "relevance_score": 0.21},
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> _FakeResp:
            del args, kwargs
            return _FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    reranker = HTTPRerankerProvider(
        Settings(
            reranker_enabled=True,
            reranker_base_url="https://mock-rerank.local/v1",
            reranker_api_key="test-key",
        ),
        fallback=LexicalOverlapReranker(),
    )
    ranked = await reranker.rerank("pgvector", ["unrelated doc", "pgvector HNSW"], top_n=2)
    assert ranked == [(1, 0.96), (0, 0.21)]

    # Disabled reranker falls back cleanly
    disabled = HTTPRerankerProvider(
        Settings(reranker_enabled=False),
        fallback=LexicalOverlapReranker(),
    )
    fallback_ranked = await disabled.rerank("pgvector", ["unrelated doc", "pgvector HNSW"], top_n=2)
    assert fallback_ranked[0][0] == 1


def test_docling_parser_blocks_and_missing_extra(tmp_path: Path) -> None:
    parser = DoclingParser()
    assert parser.supports(Path("manual.pdf")) is True
    assert parser.supports(Path("notes.md")) is False

    parsed = parser.build_from_blocks(
        [
            DoclingTextBlock(
                text="1. Overview", heading="1. Overview", level=1, page_numbers=[1]
            ),
            DoclingTextBlock(text="PostgreSQL unified storage.", page_numbers=[1]),
            DoclingTextBlock(
                text="1.1 Indexing", heading="1.1 Indexing", level=2, page_numbers=[2]
            ),
            DoclingTextBlock(text="ParadeDB BM25 with Jieba.", page_numbers=[2, 3]),
        ],
        title="Design PDF",
        external_key="design.pdf",
    )
    assert len(parsed.sections) == 2
    assert parsed.sections[1].heading_path == ["1. Overview", "1.1 Indexing"]
    assert parsed.chunks[1].page_numbers == [2, 3]

    registry = ParserRegistry()
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello\nWorld", encoding="utf-8")
    md_doc = registry.parse_file(md_file)
    assert md_doc.title == "Hello"
