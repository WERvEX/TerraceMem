"""Zero-heavy-dependency Markdown / plain-text structure-aware tree parser & chunker.

Preserves heading hierarchy, keeps fenced code blocks and Markdown tables intact
when possible, and prepends deterministic heading breadcrumbs into `contextual_text`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import UUID

from terracemem.domain.models import ChunkRecord, ParsedDocument, SectionNode
from terracemem.utils.ids import new_uuid7
from terracemem.utils.tokens import estimate_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _split_blocks(body: str) -> list[str]:
    """Split section text into atomic paragraph / fenced-code / table blocks."""
    lines = body.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if current:
            text = "\n".join(current).strip()
            if text:
                blocks.append(text)
            current.clear()

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = marker
                current.append(line)
            elif marker == fence_marker:
                current.append(line)
                flush()
                in_fence = False
                fence_marker = ""
            else:
                current.append(line)
            continue

        if in_fence:
            current.append(line)
            continue

        if not line.strip():
            flush()
        else:
            current.append(line)

    flush()
    return blocks


def _chunk_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    """Pack atomic blocks up to max_tokens without slicing mid-code-block when avoidable."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block)
        if buf and buf_tokens + block_tokens > max_tokens:
            chunks.append("\n\n".join(buf))
            buf = [block]
            buf_tokens = block_tokens
        else:
            buf.append(block)
            buf_tokens += block_tokens

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


class MarkdownTreeParser:
    """Fast native parser for Markdown, source code, and plain text files."""

    SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".rst", ".py", ".ts", ".sql"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse_file(
        self,
        path: Path,
        *,
        external_key: str | None = None,
        max_chunk_tokens: int = 450,
    ) -> ParsedDocument:
        raw_text = path.read_text(encoding="utf-8")
        return self.parse_text(
            raw_text,
            title=path.stem,
            external_key=external_key or str(path),
            source_uri=str(path),
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
    ) -> ParsedDocument:
        doc_id = new_uuid7()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Walk lines while respecting fenced code blocks so '#' inside code is not a heading
        raw_sections: list[tuple[int, str, list[str]]] = []
        current_level = 0
        current_heading = title
        current_lines: list[str] = []
        in_fence = False
        fence_char = ""
        resolved_title = title

        for line in text.splitlines():
            fence_match = _FENCE_RE.match(line)
            if fence_match:
                ch = fence_match.group(1)[0]
                if not in_fence:
                    in_fence = True
                    fence_char = ch
                elif ch == fence_char:
                    in_fence = False
                    fence_char = ""
                current_lines.append(line)
                continue

            if not in_fence:
                heading_match = _HEADING_RE.match(line)
                if heading_match:
                    if current_lines or raw_sections:
                        raw_sections.append((current_level, current_heading, current_lines))
                    current_level = len(heading_match.group(1))
                    current_heading = heading_match.group(2).strip()
                    if current_level == 1 and resolved_title == title:
                        resolved_title = current_heading
                    current_lines = []
                    continue

            current_lines.append(line)

        if current_lines or not raw_sections:
            raw_sections.append((current_level, current_heading, current_lines))

        # Build SectionNode hierarchy with materialized section_path and heading_path
        sections: list[SectionNode] = []
        chunks: list[ChunkRecord] = []
        # stack entries: (level, section_id, heading, path_prefix)
        stack: list[tuple[int, UUID, str, str]] = []
        sibling_counts: dict[str, int] = {}
        chunk_idx = 0

        for ordinal, (level, heading, lines) in enumerate(raw_sections):
            body = "\n".join(lines).strip()
            if not body and level == 0:
                continue

            while stack and stack[-1][0] >= level and level > 0:
                stack.pop()

            parent_id = stack[-1][1] if stack else None
            parent_path = stack[-1][3] if stack else ""
            sibling_num = sibling_counts.get(parent_path, 0) + 1
            sibling_counts[parent_path] = sibling_num
            section_path = (
                f"{parent_path}.{sibling_num:03d}" if parent_path else f"{sibling_num:03d}"
            )
            heading_path = [item[2] for item in stack] + [heading]

            section_id = new_uuid7()
            section_text = f"{heading}\n\n{body}".strip() if body else heading
            node = SectionNode(
                id=section_id,
                parent_section_id=parent_id,
                section_path=section_path,
                heading_path=heading_path,
                heading=heading,
                level=level,
                ordinal=ordinal,
                section_text=section_text,
                token_count=estimate_tokens(section_text),
            )
            sections.append(node)
            if level > 0:
                stack.append((level, section_id, heading, section_path))

            if not body:
                continue

            breadcrumb = " > ".join(heading_path)
            header_prefix = f"[Doc: {resolved_title} | Section: {breadcrumb}]\n"
            for raw_chunk in _chunk_blocks(_split_blocks(body), max_tokens=max_chunk_tokens):
                contextual_text = f"{header_prefix}{raw_chunk}"
                chunks.append(
                    ChunkRecord(
                        document_id=doc_id,
                        section_id=section_id,
                        chunk_index=chunk_idx,
                        raw_text=raw_chunk,
                        contextual_text=contextual_text,
                        token_count=estimate_tokens(contextual_text),
                    )
                )
                chunk_idx += 1

        return ParsedDocument(
            id=doc_id,
            external_key=external_key,
            title=resolved_title,
            content_hash=content_hash,
            source_uri=source_uri,
            sections=sections,
            chunks=chunks,
        )
