"""Optional IBM Docling document parser adapter and unified parser registry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from terracemem.domain.models import ChunkRecord, ParsedDocument, SectionNode
from terracemem.ingestion.base import DocumentParser
from terracemem.ingestion.markdown_parser import MarkdownTreeParser
from terracemem.utils.ids import new_uuid7
from terracemem.utils.tokens import estimate_tokens


@dataclass
class DoclingTextBlock:
    """Normalized block extracted from a Docling conversion output."""

    text: str
    heading: str | None = None
    level: int = 1
    page_numbers: list[int] = field(default_factory=list)


class DoclingParser:
    """Structure-aware parser for PDF, DOCX, PPTX, and HTML via IBM Docling (MIT)."""

    SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse_file(
        self,
        path: Path,
        *,
        external_key: str | None = None,
        max_chunk_tokens: int = 450,
    ) -> ParsedDocument:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed. Install optional extra: uv sync --extra docling"
            ) from exc

        converter = DocumentConverter()
        conv_result = converter.convert(str(path))
        dl_doc = conv_result.document

        blocks: list[DoclingTextBlock] = []
        iterate_items = getattr(dl_doc, "iterate_items", None)
        if callable(iterate_items):
            for item, item_level in iterate_items():
                text = getattr(item, "text", None)
                if not isinstance(text, str) or not text.strip():
                    export_md = getattr(item, "export_to_markdown", None)
                    if callable(export_md):
                        text = str(export_md())
                if not isinstance(text, str) or not text.strip():
                    continue
                label = str(getattr(item, "label", "")).lower()
                prov = getattr(item, "prov", None) or []
                pages = sorted(
                    {
                        int(p.page_no)
                        for p in prov
                        if isinstance(getattr(p, "page_no", None), int)
                    }
                )
                is_heading = "header" in label or "title" in label
                lvl = max(1, min(6, int(item_level) if isinstance(item_level, int) else 1))
                blocks.append(
                    DoclingTextBlock(
                        text=text.strip(),
                        heading=text.strip() if is_heading else None,
                        level=lvl,
                        page_numbers=pages,
                    )
                )

        if not blocks:
            markdown_export = dl_doc.export_to_markdown()
            return MarkdownTreeParser().parse_text(
                markdown_export,
                title=path.stem,
                external_key=external_key or str(path),
                source_uri=str(path),
                max_chunk_tokens=max_chunk_tokens,
            )

        raw_bytes = path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        return self.build_from_blocks(
            blocks,
            title=path.stem,
            external_key=external_key or str(path),
            source_uri=str(path),
            content_hash=content_hash,
            max_chunk_tokens=max_chunk_tokens,
        )

    def build_from_blocks(
        self,
        blocks: list[DoclingTextBlock],
        *,
        title: str,
        external_key: str,
        source_uri: str | None = None,
        content_hash: str | None = None,
        max_chunk_tokens: int = 450,
    ) -> ParsedDocument:
        """Convert structured Docling blocks into SectionNode tree and breadcrumb chunks."""
        doc_id = new_uuid7()
        joined_text = "\n\n".join(b.text for b in blocks)
        resolved_hash = content_hash or hashlib.sha256(joined_text.encode("utf-8")).hexdigest()

        sections: list[SectionNode] = []
        chunks: list[ChunkRecord] = []
        stack: list[tuple[int, UUID, str, str]] = []
        sibling_counts: dict[str, int] = {}

        grouped: list[tuple[int, str, list[DoclingTextBlock]]] = []
        curr_level = 1
        curr_heading = title
        curr_blocks: list[DoclingTextBlock] = []

        for blk in blocks:
            if blk.heading is not None:
                if curr_blocks or grouped:
                    grouped.append((curr_level, curr_heading, curr_blocks))
                curr_level = max(1, min(6, blk.level))
                curr_heading = blk.heading
                curr_blocks = []
            else:
                curr_blocks.append(blk)

        if curr_blocks or not grouped:
            grouped.append((curr_level, curr_heading, curr_blocks))

        chunk_idx = 0
        for ordinal, (level, heading, sec_blocks) in enumerate(grouped):
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            parent_path = stack[-1][3] if stack else ""
            sib = sibling_counts.get(parent_path, 0) + 1
            sibling_counts[parent_path] = sib
            sec_path = f"{parent_path}.{sib:03d}" if parent_path else f"{sib:03d}"
            heading_path = [item[2] for item in stack] + [heading]

            body = "\n\n".join(b.text for b in sec_blocks).strip()
            sec_id = new_uuid7()
            sec_text = f"{heading}\n\n{body}".strip() if body else heading
            sections.append(
                SectionNode(
                    id=sec_id,
                    parent_section_id=parent_id,
                    section_path=sec_path,
                    heading_path=heading_path,
                    heading=heading,
                    level=level,
                    ordinal=ordinal,
                    section_text=sec_text,
                    token_count=estimate_tokens(sec_text),
                )
            )
            stack.append((level, sec_id, heading, sec_path))

            if not sec_blocks:
                continue

            breadcrumb = " > ".join(heading_path)
            header_prefix = f"[Doc: {title} | Section: {breadcrumb}]\n"

            buf_texts: list[str] = []
            buf_pages: set[int] = set()
            buf_tokens = 0

            def flush_chunk(
                texts: list[str],
                pages: set[int],
                *,
                section_id: UUID = sec_id,
                prefix: str = header_prefix,
            ) -> None:
                nonlocal chunk_idx
                raw_chunk = "\n\n".join(texts).strip()
                if not raw_chunk:
                    return
                ctx = f"{prefix}{raw_chunk}"
                chunks.append(
                    ChunkRecord(
                        document_id=doc_id,
                        section_id=section_id,
                        chunk_index=chunk_idx,
                        raw_text=raw_chunk,
                        contextual_text=ctx,
                        token_count=estimate_tokens(ctx),
                        page_numbers=sorted(pages),
                    )
                )
                chunk_idx += 1

            for blk in sec_blocks:
                btok = estimate_tokens(blk.text)
                if buf_texts and buf_tokens + btok > max_chunk_tokens:
                    flush_chunk(buf_texts, buf_pages)
                    buf_texts = [blk.text]
                    buf_pages = set(blk.page_numbers)
                    buf_tokens = btok
                else:
                    buf_texts.append(blk.text)
                    buf_pages.update(blk.page_numbers)
                    buf_tokens += btok

            if buf_texts:
                flush_chunk(buf_texts, buf_pages)

        return ParsedDocument(
            id=doc_id,
            external_key=external_key,
            title=title,
            content_hash=resolved_hash,
            source_uri=source_uri,
            sections=sections,
            chunks=chunks,
        )


class ParserRegistry:
    """Routes document files or raw text to the appropriate structure-aware parser."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._markdown = MarkdownTreeParser()
        self._parsers: list[DocumentParser] = parsers or [self._markdown, DoclingParser()]

    def parse_file(
        self,
        path: Path,
        *,
        external_key: str | None = None,
        max_chunk_tokens: int = 450,
    ) -> ParsedDocument:
        for parser in self._parsers:
            if parser.supports(path):
                return parser.parse_file(
                    path,
                    external_key=external_key,
                    max_chunk_tokens=max_chunk_tokens,
                )
        return self._markdown.parse_file(
            path,
            external_key=external_key,
            max_chunk_tokens=max_chunk_tokens,
        )

    def parse_text(
        self,
        text: str,
        *,
        title: str,
        external_key: str,
        source_uri: str | None = None,
        max_chunk_tokens: int = 450,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        parsed = self._markdown.parse_text(
            text,
            title=title,
            external_key=external_key,
            source_uri=source_uri,
            max_chunk_tokens=max_chunk_tokens,
        )
        if metadata:
            parsed = parsed.model_copy(update={"metadata": metadata})
        return parsed
