"""MCP SDK v2 server exposing TerraceMem retrieval and bi-temporal memory tools."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from mcp.server.mcpserver import MCPServer

from terracemem import __version__
from terracemem.config import get_settings
from terracemem.domain.models import RetrievalFilter
from terracemem.rag.fsm import RagRequest
from terracemem.runtime import TerraceMemRuntime


def create_mcp_server(runtime: TerraceMemRuntime | None = None) -> MCPServer:
    """Create the configured TerraceMem MCPServer (MCP Python SDK v2)."""
    mcp = MCPServer(
        name="TerraceMem",
        version=__version__,
        instructions=(
            "Trustworthy Agentic Retrieval & Bi-temporal Tiered Memory Engine. "
            "Use kb_* tools for grounded knowledge lookup and memory_* tools for "
            "cross-session episodic/semantic memory recall and bi-temporal auditing."
        ),
    )

    state: dict[str, TerraceMemRuntime | None] = {"runtime": runtime}
    init_lock = asyncio.Lock()

    async def _get_runtime() -> TerraceMemRuntime:
        rt = state["runtime"]
        if rt is None:
            async with init_lock:
                rt = state["runtime"]
                if rt is None:
                    rt = TerraceMemRuntime(get_settings())
                    state["runtime"] = rt
        if not rt.is_started:
            await rt.startup()
        return rt

    @mcp.tool()
    async def kb_browse_outline(
        namespace: str = "default",
        tenant_id: str = "default",
        query: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Browse document titles, hierarchical sections, summaries, and keywords."""
        rt = await _get_runtime()
        outlines = await rt.doc_repo.browse_outlines(
            tenant_id=tenant_id,
            namespace=namespace,
            query=query,
            limit=limit,
        )
        return {
            "count": len(outlines),
            "tenant_id": tenant_id,
            "namespace": namespace,
            "outlines": [o.model_dump(mode="json") for o in outlines],
        }

    @mcp.tool()
    async def kb_search(
        query: str,
        namespace: str = "default",
        tenant_id: str = "default",
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Run hybrid BM25 + dense vector + RRF search with deterministic SQL filters."""
        rt = await _get_runtime()
        result = await rt.retrieval.search(
            query,
            RetrievalFilter(tenant_id=tenant_id, namespace=namespace),
            top_k=max(1, min(top_k, 20)),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def kb_ask(
        question: str,
        namespace: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ask a question through the Agentic RAG state machine with strict citations/abstention."""
        rt = await _get_runtime()
        parsed_sid = UUID(session_id) if session_id else None
        response = await rt.rag.run(
            RagRequest(
                question=question,
                tenant_id=tenant_id,
                namespace=namespace,
                user_id=user_id,
                session_id=parsed_sid,
                channel="mcp",
            )
        )
        return response.model_dump(mode="json")

    @mcp.tool()
    async def memory_recall(
        user_id: str,
        query: str,
        namespace: str = "default",
        tenant_id: str = "default",
        session_id: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Recall active L2 episodic and L3 bi-temporal semantic memories for a user."""
        rt = await _get_runtime()
        parsed_sid = UUID(session_id) if session_id else None
        result = await rt.memory.recall(
            user_id=user_id,
            query=query,
            tenant_id=tenant_id,
            namespace=namespace,
            session_id=parsed_sid,
            limit=max(1, min(limit, 25)),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def memory_commit(
        user_id: str,
        content: str,
        session_id: str | None = None,
        namespace: str = "default",
        tenant_id: str = "default",
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Commit conversation/decision context to update L1/L2/L3 bi-temporal memory."""
        rt = await _get_runtime()
        parsed_sid = UUID(session_id) if session_id else None
        result = await rt.memory.commit_turn(
            user_id=user_id,
            content=content,
            tenant_id=tenant_id,
            namespace=namespace,
            session_id=parsed_sid,
            is_pinned=is_pinned,
            sync_extract=True,
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def memory_timeline(
        user_id: str,
        slot_key: str | None = None,
        as_of: str | None = None,
        namespace: str = "default",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Audit bi-temporal semantic memory history and supersession chain as of a timestamp."""
        rt = await _get_runtime()
        parsed_as_of = datetime.fromisoformat(as_of) if as_of else None
        result = await rt.memory.timeline(
            user_id=user_id,
            tenant_id=tenant_id,
            namespace=namespace,
            slot_key=slot_key,
            as_of=parsed_as_of,
        )
        return result.model_dump(mode="json")

    return mcp
