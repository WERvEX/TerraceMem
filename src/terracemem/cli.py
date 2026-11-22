"""Command-line interface for running TerraceMem server, MCP bridge, worker & diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys

import uvicorn

from terracemem import __version__
from terracemem.config import get_settings
from terracemem.mcp_server.server import create_mcp_server
from terracemem.runtime import TerraceMemRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terracemem",
        description="TerraceMem: Trustworthy Agentic Retrieval & Bi-temporal Tiered Memory Engine",
    )
    parser.add_argument("--version", action="version", version=f"terracemem {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_cmd = subparsers.add_parser(
        "serve",
        help="Start FastAPI (REST/SSE + Streamable HTTP /mcp + /ui Inspector).",
    )
    serve_cmd.add_argument("--host", type=str, default=None)
    serve_cmd.add_argument("--port", type=int, default=None)
    serve_cmd.add_argument("--reload", action="store_true")

    mcp_cmd = subparsers.add_parser("mcp", help="Run standalone MCP server.")
    mcp_cmd.add_argument(
        "--stdio",
        action="store_true",
        default=True,
        help="Run over stdio transport for Claude Code / Codex local integration.",
    )

    worker_cmd = subparsers.add_parser(
        "worker",
        help="Run standalone PostgreSQL FOR UPDATE SKIP LOCKED Outbox worker.",
    )
    worker_cmd.add_argument("--poll-interval", type=float, default=1.0)

    subparsers.add_parser("info", help="Print resolved runtime configuration summary.")
    return parser


async def _run_worker(poll_interval: float) -> int:
    settings = get_settings().model_copy(update={"enable_background_worker": False})
    rt = TerraceMemRuntime(settings)
    await rt.startup()
    if rt.outbox_worker is None:
        print(
            "Error: PostgreSQL Outbox worker requires an active PostgreSQL connection.",
            file=sys.stderr,
        )
        await rt.shutdown()
        return 1

    rt.outbox_worker._poll_interval_seconds = max(0.1, poll_interval)
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, rt.outbox_worker.request_stop)

    try:
        await rt.outbox_worker.run_forever()
    finally:
        await rt.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "serve":
        uvicorn.run(
            "terracemem.api.app:create_app",
            factory=True,
            host=args.host or settings.host,
            port=args.port or settings.port,
            reload=args.reload,
            log_level=settings.log_level.lower(),
        )
        return 0

    if args.command == "mcp":
        server = create_mcp_server()
        server.run(transport="stdio")
        return 0

    if args.command == "worker":
        return asyncio.run(_run_worker(args.poll_interval))

    if args.command == "info":
        summary = {
            "version": __version__,
            "storage_backend": settings.storage_backend,
            "lexical_backend": settings.lexical_backend,
            "embedding_dim": settings.embedding_dim,
            "enable_write_time_enrichment": settings.enable_write_time_enrichment,
            "enable_graph_expansion": settings.enable_graph_expansion,
            "strict_citation_mode": settings.strict_citation_mode,
            "enable_inspector_ui": settings.enable_inspector_ui,
        }
        print(json.dumps(summary, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
