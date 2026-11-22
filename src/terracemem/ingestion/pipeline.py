"""End-to-end structured document ingestion pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from terracemem.config import Settings
from terracemem.db.repositories import DocumentRepository, GraphRepository
from terracemem.domain.models import IngestDocumentResult, ParsedDocument
from terracemem.ingestion.docling_parser import ParserRegistry
from terracemem.ingestion.enricher import WriteTimeEnricher
from terracemem.llm.providers import EmbeddingProvider
from terracemem.retrieval.graph import EntityRelationExtractor


class IngestionService:
    """Orchestrates parse -> optional enrichment -> graph extraction -> embedding -> upsert."""

    def __init__(
        self,
        *,
        settings: Settings,
        doc_repo: DocumentRepository,
        embedder: EmbeddingProvider,
        parser_registry: ParserRegistry | None = None,
        enricher: WriteTimeEnricher | None = None,
        graph_repo: GraphRepository | None = None,
        graph_extractor: EntityRelationExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._doc_repo = doc_repo
        self._embedder = embedder
        self._parsers = parser_registry or ParserRegistry()
        self._enricher = enricher
        self._graph_repo = graph_repo
        self._graph_extractor = graph_extractor

    async def ingest_text(
        self,
        text: str,
        *,
        title: str,
        external_key: str,
        tenant_id: str = "default",
        namespace: str = "default",
        source_uri: str | None = None,
        effective_at: datetime | None = None,
        max_chunk_tokens: int = 450,
        metadata: dict[str, Any] | None = None,
        enable_enrichment: bool | None = None,
        enable_graph: bool | None = None,
        skip_embeddings: bool = False,
    ) -> IngestDocumentResult:
        parsed = self._parsers.parse_text(
            text,
            title=title,
            external_key=external_key,
            source_uri=source_uri,
            max_chunk_tokens=max_chunk_tokens,
            metadata=metadata,
        )
        return await self.ingest_parsed(
            parsed,
            tenant_id=tenant_id,
            namespace=namespace,
            effective_at=effective_at,
            enable_enrichment=enable_enrichment,
            enable_graph=enable_graph,
            skip_embeddings=skip_embeddings,
        )

    async def ingest_file(
        self,
        path: Path,
        *,
        external_key: str | None = None,
        tenant_id: str = "default",
        namespace: str = "default",
        effective_at: datetime | None = None,
        max_chunk_tokens: int = 450,
        enable_enrichment: bool | None = None,
        enable_graph: bool | None = None,
        skip_embeddings: bool = False,
    ) -> IngestDocumentResult:
        parsed = self._parsers.parse_file(
            path,
            external_key=external_key,
            max_chunk_tokens=max_chunk_tokens,
        )
        return await self.ingest_parsed(
            parsed,
            tenant_id=tenant_id,
            namespace=namespace,
            effective_at=effective_at,
            enable_enrichment=enable_enrichment,
            enable_graph=enable_graph,
            skip_embeddings=skip_embeddings,
        )

    async def ingest_parsed(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        effective_at: datetime | None = None,
        enable_enrichment: bool | None = None,
        enable_graph: bool | None = None,
        skip_embeddings: bool = False,
    ) -> IngestDocumentResult:
        use_enrich = (
            enable_enrichment
            if enable_enrichment is not None
            else self._settings.enable_write_time_enrichment
        )
        use_graph = (
            enable_graph
            if enable_graph is not None
            else self._settings.enable_graph_expansion
        )

        enriched_flag = False
        working_doc = parsed
        if use_enrich and self._enricher is not None:
            working_doc = await self._enricher.enrich(working_doc)
            enriched_flag = True

        embeddings: list[list[float]] | None = None
        if not skip_embeddings and working_doc.chunks:
            texts = [chunk.contextual_text for chunk in working_doc.chunks]
            embeddings = await self._embedder.embed_texts(texts)

        entities_count = 0
        edges_count = 0
        result = await self._doc_repo.upsert_document(
            working_doc,
            tenant_id=tenant_id,
            namespace=namespace,
            embeddings=embeddings,
            effective_at=effective_at,
            enriched=enriched_flag,
        )

        if (
            result.action != "unchanged"
            and use_graph
            and self._graph_repo is not None
            and self._graph_extractor is not None
        ):
            entities, edges = await self._graph_extractor.extract_from_document(
                working_doc,
                tenant_id=tenant_id,
                namespace=namespace,
            )
            if entities or edges:
                entities_count, edges_count = await self._graph_repo.upsert_entities_and_edges(
                    entities, edges
                )

        return result.model_copy(
            update={"entities_indexed": entities_count, "edges_indexed": edges_count}
        )
