"""Structure-aware document parsing, optional write-time enrichment, and ingestion."""

from terracemem.ingestion.base import DocumentParser
from terracemem.ingestion.docling_parser import DoclingParser, DoclingTextBlock, ParserRegistry
from terracemem.ingestion.enricher import WriteTimeEnricher
from terracemem.ingestion.markdown_parser import MarkdownTreeParser
from terracemem.ingestion.pipeline import IngestionService

__all__ = [
    "DoclingParser",
    "DoclingTextBlock",
    "DocumentParser",
    "IngestionService",
    "MarkdownTreeParser",
    "ParserRegistry",
    "WriteTimeEnricher",
]
