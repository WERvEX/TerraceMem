"""Model Context Protocol (MCP) server exposing TerraceMem tools via Streamable HTTP & stdio."""

from terracemem.mcp_server.server import create_mcp_server

__all__ = ["create_mcp_server"]
