"""Phase 2 tests: Agentic RAG FSM, CRAG rewrite, citation verifier, SSE, REST & MCP v2."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import BaseModel

from terracemem.api.app import create_app
from terracemem.config import Settings
from terracemem.domain.models import RagDecisionStatus, SufficiencyVerdict
from terracemem.llm.providers import LLMTextResult, LLMUsage, StructuredLLMResult
from terracemem.mcp_server.server import create_mcp_server
from terracemem.rag.fsm import RagRequest, RagState
from terracemem.runtime import TerraceMemRuntime
from terracemem.utils.ids import new_uuid7


class _MockScriptedLLM:
    """Deterministic mock LLM for testing CRAG rewrites and fabricated citation rejection."""

    def __init__(self, *, mode: str) -> None:
        self._mode = mode
        self.sufficiency_calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    async def complete_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMTextResult:
        del messages, use_fast_model, temperature, max_tokens
        return LLMTextResult(text="mock", usage=LLMUsage(prompt_tokens=10, completion_tokens=5))

    async def complete_structured[TModel: BaseModel](
        self,
        messages: Sequence[Mapping[str, str]],
        response_model: type[TModel],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> StructuredLLMResult[TModel]:
        del use_fast_model, temperature, max_tokens
        name = response_model.__name__
        usage = LLMUsage(prompt_tokens=20, completion_tokens=10, model="mock-llm")

        if name == "_ChunkTripleList":
            return StructuredLLMResult(
                parsed=response_model.model_validate({"triples": []}),
                usage=usage,
            )

        if name == "_SufficiencyDraft":
            self.sufficiency_calls += 1
            if self._mode == "rewrite_then_answer" and self.sufficiency_calls == 1:
                parsed = response_model.model_validate(
                    {
                        "verdict": SufficiencyVerdict.AMBIGUOUS.value,
                        "missing_aspects": "Needs exact protocol token",
                        "suggested_rewrite": "ParadeDB Tantivy jieba BM25",
                    }
                )
            else:
                parsed = response_model.model_validate(
                    {
                        "verdict": SufficiencyVerdict.SUFFICIENT.value,
                        "missing_aspects": "",
                        "suggested_rewrite": "",
                    }
                )
            return StructuredLLMResult(parsed=parsed, usage=usage)

        if name == "_GroundedAnswerDraft":
            if self._mode == "fabricated_citation":
                fake_id = str(new_uuid7())
                parsed = response_model.model_validate(
                    {
                        "cannot_answer": False,
                        "answer": "Claim backed by fabricated UUID.",
                        "claims": [
                            {
                                "statement": "Fabricated claim.",
                                "cited_chunk_ids": [fake_id],
                            }
                        ],
                    }
                )
            else:
                # Extract first valid chunk_id from the prompt
                user_msg = messages[-1]["content"]
                marker = "[chunk_id="
                cid = ""
                if marker in user_msg:
                    cid = user_msg.split(marker, 1)[1].split(" ", 1)[0].strip()
                parsed = response_model.model_validate(
                    {
                        "cannot_answer": False,
                        "answer": "ParadeDB uses Tantivy with built-in jieba tokenization.",
                        "claims": [
                            {
                                "statement": "ParadeDB uses Tantivy with jieba.",
                                "cited_chunk_ids": [cid] if cid else [],
                            }
                        ],
                    }
                )
            return StructuredLLMResult(parsed=parsed, usage=usage)

        raise ValueError(f"Unexpected structured response model: {name}")

    async def stream_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        del messages, use_fast_model, temperature, max_tokens
        yield "mock"


async def test_rag_fsm_happy_path_and_trace_persistence() -> None:
    rt = TerraceMemRuntime(Settings(storage_backend="memory"))
    await rt.startup()
    try:
        await rt.ingestion.ingest_text(
            "# Storage\n\n## BM25\nParadeDB uses Rust Tantivy with built-in jieba tokenizer.",
            title="Storage Spec",
            external_key="storage-spec",
        )
        res = await rt.rag.run(RagRequest(question="What tokenizer does ParadeDB use?"))
        assert res.status == RagDecisionStatus.ANSWERED
        assert len(res.citations) >= 1
        assert "ParadeDB" in res.answer

        trace = await rt.trace_repo.get_trace(res.trace_id)
        assert trace is not None
        assert trace.final_status == "answered"
        span_names = [s.node_name for s in trace.spans]
        assert span_names[0] == RagState.INTAKE_AND_FILTER.value
        assert RagState.HYBRID_RETRIEVE.value in span_names
        assert RagState.CITATION_VERIFY.value in span_names
    finally:
        await rt.shutdown()


async def test_rag_fsm_crag_rewrite_and_fabricated_citation_abstention() -> None:
    # 1) CRAG rewrite triggers retry and answers
    rt_rewrite = TerraceMemRuntime(
        Settings(storage_backend="memory", rerank_high_confidence_threshold=0.99),
        llm=_MockScriptedLLM(mode="rewrite_then_answer"),
    )
    await rt_rewrite.startup()
    try:
        await rt_rewrite.ingestion.ingest_text(
            "# Search\n\n## BM25\nParadeDB Tantivy jieba BM25 handles Chinese segmentation.",
            title="Search",
            external_key="search",
        )
        res = await rt_rewrite.rag.run(RagRequest(question="How does Chinese search work?"))
        assert res.status == RagDecisionStatus.ANSWERED
        assert res.retrieval_retries == 1
    finally:
        await rt_rewrite.shutdown()

    # 2) Fabricated citation forces STRICT_ABSTAIN
    rt_fake = TerraceMemRuntime(
        Settings(
            storage_backend="memory",
            strict_citation_mode=True,
            rerank_high_confidence_threshold=0.1,
        ),
        llm=_MockScriptedLLM(mode="fabricated_citation"),
    )
    await rt_fake.startup()
    try:
        await rt_fake.ingestion.ingest_text(
            "# Security\n\n## Tokens\nTokens use EdDSA signatures.",
            title="Security",
            external_key="security",
        )
        res = await rt_fake.rag.run(RagRequest(question="What signatures do Tokens use?"))
        assert res.status == RagDecisionStatus.ABSTAINED
        assert res.citations == []
        assert res.abstention_reason is not None

        # Blank question abstains immediately
        blank = await rt_fake.rag.run(RagRequest(question="   "))
        assert blank.status == RagDecisionStatus.ABSTAINED
    finally:
        await rt_fake.shutdown()


def test_fastapi_rest_sse_and_mcp_streamable_http_end_to_end() -> None:
    app = create_app(Settings(storage_backend="memory", enable_inspector_ui=True))
    with TestClient(app, base_url="http://localhost:8000") as client:
        # Ingest document
        ingest = client.post(
            "/api/v1/documents",
            json={
                "title": "Outbox Spec",
                "external_key": "outbox-spec",
                "content": (
                    "# Outbox Spec\n\n## Worker Queue\n"
                    "PostgresOutboxWorker claims async_jobs using FOR UPDATE SKIP LOCKED."
                ),
                "enable_enrichment": True,
            },
        )
        assert ingest.status_code == 200
        doc_id = ingest.json()["document_id"]

        # List & Outline
        listed = client.get("/api/v1/documents")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1

        outline = client.get(f"/api/v1/documents/{doc_id}/outline")
        assert outline.status_code == 200
        assert len(outline.json()["sections"]) >= 1

        # Search
        search = client.post(
            "/api/v1/search",
            json={"query": "FOR UPDATE SKIP LOCKED", "top_k": 3},
        )
        assert search.status_code == 200
        assert len(search.json()["evidence"]) >= 1

        # Ask (sync)
        ask = client.post(
            "/api/v1/ask",
            json={"question": "How does PostgresOutboxWorker claim async_jobs?"},
        )
        assert ask.status_code == 200
        ask_body = ask.json()
        assert ask_body["status"] == "answered"
        assert len(ask_body["citations"]) >= 1
        trace_id = ask_body["trace_id"]

        # Ask (SSE stream)
        with client.stream(
            "POST",
            "/api/v1/ask/stream",
            json={"question": "How does PostgresOutboxWorker claim async_jobs?"},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            raw_sse = "".join(chunk for chunk in stream_resp.iter_text())
        assert "event: state" in raw_sse
        assert "event: retrieval" in raw_sse
        assert "event: final" in raw_sse

        # Traces
        traces = client.get("/api/v1/traces")
        assert traces.status_code == 200
        assert traces.json()["count"] >= 2

        trace_detail = client.get(f"/api/v1/traces/{trace_id}")
        assert trace_detail.status_code == 200
        assert len(trace_detail.json()["spans"]) >= 4

        # Streamable HTTP /mcp initialize + tools/list
        mcp_init = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0.1"},
                },
            },
        )
        assert mcp_init.status_code == 200

        mcp_list = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert mcp_list.status_code == 200
        tool_names = [t["name"] for t in mcp_list.json()["result"]["tools"]]
        assert set(tool_names) == {
            "kb_browse_outline",
            "kb_search",
            "kb_ask",
            "memory_recall",
            "memory_commit",
            "memory_timeline",
        }

        # Soft delete
        deleted = client.delete(f"/api/v1/documents/{UUID(doc_id)}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


async def test_mcp_server_all_six_tools_direct() -> None:
    rt = TerraceMemRuntime(Settings(storage_backend="memory", enable_write_time_enrichment=True))
    await rt.startup()
    try:
        await rt.ingestion.ingest_text(
            "# Guide\n\n## Connection Pool\nTerraceMem uses psycopg3 AsyncConnectionPool.",
            title="Guide",
            external_key="guide",
        )
        mcp = create_mcp_server(runtime=rt)
        tools = await mcp.list_tools()
        assert len(tools) == 6

        outlines_res = await mcp.call_tool("kb_browse_outline", {"query": "Guide"})
        assert json.loads(outlines_res.content[0].text)["count"] == 1  # type: ignore[union-attr]

        search_res = await mcp.call_tool("kb_search", {"query": "AsyncConnectionPool"})
        search_data = json.loads(search_res.content[0].text)  # type: ignore[union-attr]
        assert len(search_data["evidence"]) >= 1

        ask_res = await mcp.call_tool(
            "kb_ask", {"question": "What connection pool does TerraceMem use?"}
        )
        ask_data = json.loads(ask_res.content[0].text)  # type: ignore[union-attr]
        assert ask_data["status"] == "answered"

        commit_res = await mcp.call_tool(
            "memory_commit",
            {"user_id": "dev-1", "content": "Our project uses PostgreSQL [slot:project.db]."},
        )
        commit_data = json.loads(commit_res.content[0].text)  # type: ignore[union-attr]
        assert len(commit_data["audits"]) == 1

        recall_res = await mcp.call_tool(
            "memory_recall",
            {"user_id": "dev-1", "query": "PostgreSQL database"},
        )
        recall_data = json.loads(recall_res.content[0].text)  # type: ignore[union-attr]
        assert len(recall_data["semantic_memories"]) == 1

        timeline_res = await mcp.call_tool("memory_timeline", {"user_id": "dev-1"})
        timeline_data = json.loads(timeline_res.content[0].text)  # type: ignore[union-attr]
        assert len(timeline_data["active_at_as_of"]) == 1
    finally:
        await rt.shutdown()
