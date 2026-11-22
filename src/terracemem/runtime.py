"""Unified TerraceMem runtime wiring storage, LLM/embedding/reranker, RAG & memory."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Literal
from urllib.parse import urlparse

import structlog

from terracemem.config import Settings, get_settings
from terracemem.db.memory_repository import (
    InMemoryMemoryRepository,
    MemoryRepository,
    PostgresMemoryRepository,
)
from terracemem.db.pool import DatabasePool
from terracemem.db.repositories import (
    DocumentRepository,
    GraphRepository,
    InMemoryDocumentRepository,
    InMemoryGraphRepository,
    InMemoryTraceRepository,
    PostgresDocumentRepository,
    PostgresGraphRepository,
    PostgresTraceRepository,
    TraceRepository,
)
from terracemem.ingestion.enricher import WriteTimeEnricher
from terracemem.ingestion.pipeline import IngestionService
from terracemem.llm.providers import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    create_default_embedding_provider,
    create_default_llm_provider,
    create_default_reranker_provider,
)
from terracemem.memory.service import MemoryService
from terracemem.rag.fsm import AgenticRagEngine
from terracemem.retrieval.graph import EntityRelationExtractor
from terracemem.retrieval.service import RetrievalService
from terracemem.workers.outbox import InMemoryOutboxQueue, PostgresOutboxWorker

logger = structlog.get_logger(__name__)

StorageMode = Literal["postgres", "memory", "memory_fallback"]


async def _probe_tcp(database_url: str, *, timeout: float = 0.3) -> bool:
    parsed = urlparse(database_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True
    except Exception:
        return False


class TerraceMemRuntime:
    """Lifecycle-managed dependency container for API, MCP Server, Worker, and Evals."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm: LLMProvider = llm or create_default_llm_provider(self.settings)
        self.embedder: EmbeddingProvider = embedder or create_default_embedding_provider(
            self.settings
        )
        self.reranker: RerankerProvider = reranker or create_default_reranker_provider(
            self.settings
        )

        self.storage_mode: StorageMode = "memory"
        self.db_pool: DatabasePool | None = None
        self.doc_repo: DocumentRepository
        self.graph_repo: GraphRepository
        self.trace_repo: TraceRepository
        self.memory_repo: MemoryRepository
        self.outbox_worker: PostgresOutboxWorker | None = None
        self.in_memory_outbox: InMemoryOutboxQueue | None = None

        self.ingestion: IngestionService
        self.retrieval: RetrievalService
        self.memory: MemoryService
        self.rag: AgenticRagEngine

        self._started = False
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    async def startup(self) -> None:
        if self._started:
            return

        use_pg = False
        if self.settings.storage_backend == "postgres":
            use_pg = True
        elif self.settings.storage_backend == "auto":
            use_pg = await _probe_tcp(self.settings.database_url, timeout=0.3)

        if use_pg:
            pool = DatabasePool(self.settings)
            try:
                await pool.open()
                if await pool.ping():
                    self.db_pool = pool
                    self.storage_mode = "postgres"
                    self.doc_repo = PostgresDocumentRepository(pool, self.settings)
                    self.graph_repo = PostgresGraphRepository(pool)
                    self.trace_repo = PostgresTraceRepository(pool)
                    self.memory_repo = PostgresMemoryRepository(pool)
                    self.outbox_worker = PostgresOutboxWorker(
                        pool,
                        notify_channel=self.settings.outbox_notify_channel,
                        database_url=self.settings.database_url,
                    )
                else:
                    await pool.close()
                    if self.settings.storage_backend == "postgres":
                        raise RuntimeError("PostgreSQL ping failed in strict 'postgres' mode.")
                    use_pg = False
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await pool.close()
                if self.settings.storage_backend == "postgres":
                    raise
                logger.warning("postgres_unreachable_using_memory_fallback", error=str(exc))
                use_pg = False

        if not use_pg:
            self.storage_mode = (
                "memory" if self.settings.storage_backend == "memory" else "memory_fallback"
            )
            in_mem_docs = InMemoryDocumentRepository()
            in_mem_outbox = InMemoryOutboxQueue()
            self.doc_repo = in_mem_docs
            self.graph_repo = InMemoryGraphRepository(in_mem_docs)
            self.trace_repo = InMemoryTraceRepository()
            self.memory_repo = InMemoryMemoryRepository(outbox=in_mem_outbox)
            self.in_memory_outbox = in_mem_outbox

        enricher = WriteTimeEnricher(self.llm)
        extractor = EntityRelationExtractor(self.llm)
        self.ingestion = IngestionService(
            settings=self.settings,
            doc_repo=self.doc_repo,
            embedder=self.embedder,
            enricher=enricher,
            graph_repo=self.graph_repo,
            graph_extractor=extractor,
        )
        self.retrieval = RetrievalService(
            settings=self.settings,
            doc_repo=self.doc_repo,
            embedder=self.embedder,
            reranker=self.reranker,
            graph_repo=self.graph_repo,
        )
        self.memory = MemoryService(
            settings=self.settings,
            repo=self.memory_repo,
            embedder=self.embedder,
            llm=self.llm,
        )
        if self.outbox_worker is not None:
            self.memory.register_worker_handlers(self.outbox_worker)
            if self.settings.enable_background_worker:
                self._worker_task = asyncio.create_task(self.outbox_worker.run_forever())
        elif self.in_memory_outbox is not None:
            self.memory.register_worker_handlers(self.in_memory_outbox)

        self.rag = AgenticRagEngine(
            settings=self.settings,
            doc_repo=self.doc_repo,
            retrieval_service=self.retrieval,
            trace_repo=self.trace_repo,
            llm=self.llm,
            memory_service=self.memory,
        )
        self._started = True
        logger.info(
            "terracemem_runtime_started",
            storage_mode=self.storage_mode,
            lexical_backend=self.settings.lexical_backend,
        )

    async def shutdown(self) -> None:
        if not self._started:
            return
        if self.outbox_worker is not None:
            self.outbox_worker.request_stop()
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        if self.db_pool is not None:
            await self.db_pool.close()
            self.db_pool = None
        self._started = False
