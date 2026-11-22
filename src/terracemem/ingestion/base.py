"""Abstract protocol for pluggable document parsers (Markdown, Docling, future MinerU)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from terracemem.domain.models import ParsedDocument


class DocumentParser(Protocol):
    """Protocol implemented by all document ingestion parsers."""

    def supports(self, path: Path) -> bool:
        """Return True if this parser can handle the given file extension."""
        ...

    def parse_file(
        self,
        path: Path,
        *,
        external_key: str | None = None,
        max_chunk_tokens: int = 450,
    ) -> ParsedDocument:
        """Parse a file into a hierarchical section tree and context-enriched chunks."""
        ...
