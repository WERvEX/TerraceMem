"""FastAPI application factory unifying REST/SSE, Streamable HTTP MCP, and /ui Inspector."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from terracemem import __version__
from terracemem.api.schemas import (
    AskRequest,
    CreateSessionRequest,
    DocumentIngestRequest,
    MemoryCommitRequest,
    MemoryRecallRequest,
    ParsePreviewRequest,
    PinMemoryRequest,
    SearchRequest,
)
from terracemem.config import Settings, get_settings
from terracemem.domain.models import RetrievalFilter
from terracemem.ingestion.markdown_parser import MarkdownTreeParser
from terracemem.mcp_server.server import create_mcp_server
from terracemem.rag.fsm import RagRequest
from terracemem.runtime import TerraceMemRuntime

_UI_INDEX_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"


def create_app(
    settings: Settings | None = None,
    *,
    runtime: TerraceMemRuntime | None = None,
) -> FastAPI:
    """Build the FastAPI application with mounted MCP Streamable HTTP and /ui Inspector."""
    cfg = runtime.settings if runtime is not None else (settings or get_settings())
    owns_runtime = runtime is None
    rt = runtime or TerraceMemRuntime(cfg)

    mcp = create_mcp_server(runtime=rt)
    mcp_http_app = mcp.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
    )
    parser = MarkdownTreeParser()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if not rt.is_started:
                await rt.startup()
            await stack.enter_async_context(mcp.session_manager.run())
            try:
                yield
            finally:
                if owns_runtime:
                    await rt.shutdown()

    app = FastAPI(
        title="TerraceMem",
        description="Trustworthy Agentic Retrieval & Bi-temporal Tiered Memory Engine",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        db_ok = await rt.db_pool.ping() if rt.db_pool is not None else False
        return {
            "status": "ok",
            "service": "terracemem",
            "version": __version__,
            "storage_mode": rt.storage_mode,
            "db_connected": db_ok,
        }

    @app.get("/api/v1/info")
    async def engine_info() -> dict[str, Any]:
        return {
            "name": "TerraceMem",
            "version": __version__,
            "storage_mode": rt.storage_mode,
            "lexical_backend": cfg.lexical_backend,
            "embedding_dim": cfg.embedding_dim,
            "features": {
                "write_time_enrichment": cfg.enable_write_time_enrichment,
                "graph_expansion": cfg.enable_graph_expansion,
                "strict_citation_mode": cfg.strict_citation_mode,
                "inspector_ui": cfg.enable_inspector_ui,
                "reranker_enabled": cfg.reranker_enabled,
            },
            "budgets": {
                "max_retrieval_retries": cfg.max_retrieval_retries,
                "max_state_transitions": cfg.max_state_transitions,
                "max_llm_calls": cfg.max_llm_calls,
                "request_timeout_seconds": cfg.request_timeout_seconds,
            },
            "endpoints": {
                "rest": "/api/v1",
                "mcp_streamable_http": "/mcp",
                "inspector_ui": "/ui" if cfg.enable_inspector_ui else None,
            },
        }

    @app.post("/api/v1/parse-preview")
    async def parse_preview(payload: ParsePreviewRequest) -> dict[str, Any]:
        parsed = parser.parse_text(
            payload.markdown,
            title=payload.title,
            external_key="preview.md",
            max_chunk_tokens=payload.max_chunk_tokens,
        )
        return parsed.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Documents & Knowledge Outlines
    # ------------------------------------------------------------------

    @app.post("/api/v1/documents")
    async def ingest_document(payload: DocumentIngestRequest) -> dict[str, Any]:
        res = await rt.ingestion.ingest_text(
            payload.content,
            title=payload.title,
            external_key=payload.external_key,
            tenant_id=payload.tenant_id,
            namespace=payload.namespace,
            source_uri=payload.source_uri,
            effective_at=payload.effective_at,
            max_chunk_tokens=payload.max_chunk_tokens,
            metadata=payload.metadata,
            enable_enrichment=payload.enable_enrichment,
            enable_graph=payload.enable_graph,
            skip_embeddings=payload.skip_embeddings,
        )
        return res.model_dump(mode="json")

    @app.get("/api/v1/documents")
    async def list_documents(
        tenant_id: str = Query(default="default"),
        namespace: str = Query(default="default"),
        query: str | None = Query(default=None),
        only_latest: bool = Query(default=True),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        docs = await rt.doc_repo.list_documents(
            tenant_id=tenant_id,
            namespace=namespace,
            query=query,
            only_latest=only_latest,
            limit=limit,
        )
        return {
            "count": len(docs),
            "documents": [d.model_dump(mode="json") for d in docs],
        }

    @app.get("/api/v1/documents/{document_id}/outline")
    async def get_document_outline(
        document_id: UUID,
        tenant_id: str = Query(default="default"),
        namespace: str = Query(default="default"),
    ) -> dict[str, Any]:
        outline = await rt.doc_repo.get_document_outline(
            document_id,
            tenant_id=tenant_id,
            namespace=namespace,
        )
        if outline is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return outline.model_dump(mode="json")

    @app.delete("/api/v1/documents/{document_id}")
    async def delete_document(
        document_id: UUID,
        tenant_id: str = Query(default="default"),
        namespace: str = Query(default="default"),
    ) -> dict[str, Any]:
        deleted = await rt.doc_repo.soft_delete_document(
            document_id,
            tenant_id=tenant_id,
            namespace=namespace,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "document_id": str(document_id)}

    # ------------------------------------------------------------------
    # Retrieval, Agentic RAG (Sync & SSE) & Observability Traces
    # ------------------------------------------------------------------

    @app.post("/api/v1/search")
    async def search_knowledge(payload: SearchRequest) -> dict[str, Any]:
        filt = RetrievalFilter(
            tenant_id=payload.tenant_id,
            namespace=payload.namespace,
            document_ids=payload.document_ids,
            section_ids=payload.section_ids,
            effective_after=payload.effective_after,
            effective_before=payload.effective_before,
            only_latest=payload.only_latest,
        )
        result = await rt.retrieval.search(
            payload.query,
            filt,
            mode=payload.mode,
            top_k=payload.top_k,
            enable_graph=payload.enable_graph,
            enable_rerank=payload.enable_rerank,
            enable_parent_expand=payload.enable_parent_expand,
        )
        return result.model_dump(mode="json")

    def _to_rag_request(payload: AskRequest, *, channel: str) -> RagRequest:
        return RagRequest(
            question=payload.question,
            tenant_id=payload.tenant_id,
            namespace=payload.namespace,
            user_id=payload.user_id,
            session_id=payload.session_id,
            document_ids=payload.document_ids,
            section_ids=payload.section_ids,
            effective_after=payload.effective_after,
            effective_before=payload.effective_before,
            only_latest=payload.only_latest,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            enable_rerank=payload.enable_rerank,
            enable_parent_expand=payload.enable_parent_expand,
            enable_graph_expansion=payload.enable_graph_expansion,
            enable_outline_navigation=payload.enable_outline_navigation,
            enable_crag=payload.enable_crag,
            strict_citation_mode=payload.strict_citation_mode,
            channel="sse" if channel == "sse" else "rest",
        )

    @app.post("/api/v1/ask")
    async def ask_knowledge(payload: AskRequest) -> dict[str, Any]:
        response = await rt.rag.run(_to_rag_request(payload, channel="rest"))
        return response.model_dump(mode="json")

    @app.post("/api/v1/ask/stream")
    async def ask_knowledge_stream(payload: AskRequest) -> StreamingResponse:
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        async def _callback(ev_type: str, data: dict[str, Any]) -> None:
            await queue.put((ev_type, data))

        async def _event_generator() -> AsyncIterator[str]:
            async def _runner() -> None:
                try:
                    await rt.rag.run(
                        _to_rag_request(payload, channel="sse"),
                        event_callback=_callback,
                    )
                except Exception as exc:
                    await queue.put(("error", {"detail": str(exc)}))
                finally:
                    await queue.put(None)

            task = asyncio.create_task(_runner())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    ev_type, data = item
                    encoded = json.dumps(data, ensure_ascii=False)
                    yield f"event: {ev_type}\ndata: {encoded}\n\n"
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    @app.get("/api/v1/traces")
    async def list_traces(
        tenant_id: str = Query(default="default"),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> dict[str, Any]:
        traces = await rt.trace_repo.list_traces(tenant_id=tenant_id, limit=limit)
        return {
            "count": len(traces),
            "traces": [t.model_dump(mode="json") for t in traces],
        }

    @app.get("/api/v1/traces/{trace_id}")
    async def get_trace(
        trace_id: UUID,
        tenant_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        trace = await rt.trace_repo.get_trace(trace_id, tenant_id=tenant_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Bi-temporal Three-Tier Memory (L1 / L2 / L3)
    # ------------------------------------------------------------------

    @app.post("/api/v1/sessions")
    async def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
        session = await rt.memory.create_session(
            session_id=payload.session_id,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            namespace=payload.namespace,
            token_budget=payload.token_budget,
            pinned_constraints=payload.pinned_constraints,
        )
        return session.model_dump(mode="json")

    @app.post("/api/v1/memory/commit")
    async def commit_memory(payload: MemoryCommitRequest) -> dict[str, Any]:
        result = await rt.memory.commit_turn(
            user_id=payload.user_id,
            content=payload.content,
            tenant_id=payload.tenant_id,
            namespace=payload.namespace,
            session_id=payload.session_id,
            role=payload.role,
            event_time=payload.event_time,
            is_pinned=payload.is_pinned,
            pinned_constraints=payload.pinned_constraints,
            sync_extract=payload.sync_extract,
        )
        return result.model_dump(mode="json")

    @app.post("/api/v1/memory/recall")
    async def recall_memory(payload: MemoryRecallRequest) -> dict[str, Any]:
        result = await rt.memory.recall(
            user_id=payload.user_id,
            query=payload.query,
            tenant_id=payload.tenant_id,
            namespace=payload.namespace,
            session_id=payload.session_id,
            limit=payload.limit,
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/memory/timeline")
    async def memory_timeline(
        user_id: str = Query(...),
        tenant_id: str = Query(default="default"),
        namespace: str = Query(default="default"),
        slot_key: str | None = Query(default=None),
        as_of: datetime | None = Query(default=None),
    ) -> dict[str, Any]:
        result = await rt.memory.timeline(
            user_id=user_id,
            tenant_id=tenant_id,
            namespace=namespace,
            slot_key=slot_key,
            as_of=as_of,
        )
        return result.model_dump(mode="json")

    @app.post("/api/v1/memory/{memory_id}/pin")
    async def pin_memory(memory_id: UUID, payload: PinMemoryRequest) -> dict[str, Any]:
        updated = await rt.memory.set_pinned(
            memory_id,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            is_pinned=payload.is_pinned,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Semantic memory not found")
        return updated.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Eval Reports & Embedded Inspector UI
    # ------------------------------------------------------------------

    @app.get("/api/v1/eval-reports")
    async def list_eval_reports() -> dict[str, Any]:
        reports_dir = Path(cfg.eval_reports_dir)
        reports: list[dict[str, Any]] = []
        if reports_dir.exists() and reports_dir.is_dir():
            for file_path in sorted(reports_dir.glob("*.json")):
                try:
                    content = json.loads(file_path.read_text(encoding="utf-8"))
                    if isinstance(content, dict):
                        reports.append({"filename": file_path.name, **content})
                except Exception:
                    continue
        return {"count": len(reports), "reports": reports}

    if cfg.enable_inspector_ui:

        @app.get("/ui", response_class=HTMLResponse)
        async def inspector_ui() -> HTMLResponse:
            html = _UI_INDEX_PATH.read_text(encoding="utf-8")
            return HTMLResponse(content=html)

    app.mount("/mcp", mcp_http_app)
    return app
