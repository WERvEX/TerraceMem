"""Optional write-time LLM summarization & keyword enrichment for outline nav & BM25 boost."""

from __future__ import annotations

import re
from collections import Counter
from uuid import UUID

import jieba
import structlog
from pydantic import BaseModel, Field

from terracemem.domain.models import ChunkRecord, ParsedDocument, SectionNode
from terracemem.llm.providers import LLMProvider

logger = structlog.get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "are",
        "was",
        "were",
        "will",
        "can",
        "use",
        "used",
        "using",
        "not",
        "but",
        "have",
        "has",
        "had",
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "及",
        "为",
        "通过",
        "使用",
        "支持",
        "进行",
    }
)
_SENT_SPLIT_RE = re.compile(r"[。！？\n.!?]+")


class _SectionEnrichmentItem(BaseModel):
    section_path: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)


class _DocumentEnrichmentPayload(BaseModel):
    doc_summary: str = ""
    doc_keywords: list[str] = Field(default_factory=list)
    sections: list[_SectionEnrichmentItem] = Field(default_factory=list)


def _deterministic_keywords(text: str, *, limit: int = 6) -> list[str]:
    counts: Counter[str] = Counter()
    for raw_tok in jieba.cut_for_search(text):
        tok = raw_tok.strip()
        lower = tok.lower()
        if len(tok) <= 1 or lower in _STOPWORDS or tok.isdigit():
            continue
        if all(ch in "`~!@#$%^&*()_+-=[]{}|;':\",./<>?，。！？；：（）【】" for ch in tok):
            continue
        counts[tok] += 1
    return [word for word, _ in counts.most_common(limit)]


def _deterministic_summary(text: str, *, max_chars: int = 180) -> str:
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("```")
    ]
    body = " ".join(lines)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(body) if p.strip()]
    if not parts:
        return body[:max_chars]
    summary = parts[0]
    if len(summary) < 40 and len(parts) > 1:
        summary = f"{summary}; {parts[1]}"
    return summary[:max_chars]


class WriteTimeEnricher:
    """Enriches documents, sections, and chunks with summaries and normalized keywords."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    async def enrich(self, parsed: ParsedDocument) -> ParsedDocument:
        """Populate document/section summaries & keywords and propagate keywords to chunks."""
        if not parsed.sections:
            return parsed

        if self._llm is not None and self._llm.is_configured:
            try:
                return await self._enrich_with_llm(parsed)
            except Exception as exc:
                logger.warning("write_time_enrichment_fallback", error=str(exc))

        return self._enrich_deterministic(parsed)

    async def _enrich_with_llm(self, parsed: ParsedDocument) -> ParsedDocument:
        assert self._llm is not None
        section_previews = "\n".join(
            f"- [{sec.section_path}] {sec.heading}: {sec.section_text[:320]}"
            for sec in parsed.sections[:24]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate concise technical summaries and search keywords (aliases, "
                    "error codes, concepts, bilingual terms) for a document and each section."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document Title: {parsed.title}\n"
                    f"Sections:\n{section_previews}\n\n"
                    "Return doc_summary, doc_keywords (up to 8), and per-section summary + "
                    "keywords (up to 6 per section) keyed by section_path."
                ),
            },
        ]
        result = await self._llm.complete_structured(
            messages,
            _DocumentEnrichmentPayload,
            use_fast_model=True,
            max_tokens=1200,
        )
        payload = result.parsed
        by_path = {item.section_path: item for item in payload.sections}

        updated_sections: list[SectionNode] = []
        kw_by_sec_id: dict[UUID, list[str]] = {}
        for sec in parsed.sections:
            item = by_path.get(sec.section_path)
            sec_summary = (
                item.summary.strip()
                if item and item.summary.strip()
                else _deterministic_summary(sec.section_text)
            )
            sec_keywords = (
                [kw.strip() for kw in item.keywords if kw.strip()][:8]
                if item and item.keywords
                else _deterministic_keywords(sec.section_text)
            )
            kw_by_sec_id[sec.id] = sec_keywords
            updated_sections.append(
                sec.model_copy(update={"summary": sec_summary, "keywords": sec_keywords})
            )

        doc_summary = (
            payload.doc_summary.strip()
            if payload.doc_summary.strip()
            else _deterministic_summary(" ".join(s.summary or "" for s in updated_sections))
        )
        doc_keywords = (
            [kw.strip() for kw in payload.doc_keywords if kw.strip()][:10]
            if payload.doc_keywords
            else _deterministic_keywords(parsed.title + " " + doc_summary, limit=8)
        )

        updated_chunks = self._apply_keywords_to_chunks(
            parsed.chunks,
            doc_keywords=doc_keywords,
            kw_by_sec_id=kw_by_sec_id,
        )
        return parsed.model_copy(
            update={
                "summary": doc_summary,
                "keywords": doc_keywords,
                "sections": updated_sections,
                "chunks": updated_chunks,
            }
        )

    def _enrich_deterministic(self, parsed: ParsedDocument) -> ParsedDocument:
        updated_sections: list[SectionNode] = []
        kw_by_sec_id: dict[UUID, list[str]] = {}
        all_text_parts: list[str] = [parsed.title]

        for sec in parsed.sections:
            sec_summary = _deterministic_summary(sec.section_text)
            sec_keywords = _deterministic_keywords(sec.section_text, limit=6)
            kw_by_sec_id[sec.id] = sec_keywords
            all_text_parts.append(sec_summary)
            updated_sections.append(
                sec.model_copy(update={"summary": sec_summary, "keywords": sec_keywords})
            )

        doc_summary = _deterministic_summary(" ".join(all_text_parts))
        doc_keywords = _deterministic_keywords(
            " ".join(s.section_text for s in parsed.sections[:8]),
            limit=8,
        )
        updated_chunks = self._apply_keywords_to_chunks(
            parsed.chunks,
            doc_keywords=doc_keywords,
            kw_by_sec_id=kw_by_sec_id,
        )
        return parsed.model_copy(
            update={
                "summary": doc_summary,
                "keywords": doc_keywords,
                "sections": updated_sections,
                "chunks": updated_chunks,
            }
        )

    def _apply_keywords_to_chunks(
        self,
        chunks: list[ChunkRecord],
        *,
        doc_keywords: list[str],
        kw_by_sec_id: dict[UUID, list[str]],
    ) -> list[ChunkRecord]:
        updated: list[ChunkRecord] = []
        for chunk in chunks:
            sec_kws = kw_by_sec_id.get(chunk.section_id, []) if chunk.section_id else []
            chunk_kws = _deterministic_keywords(chunk.raw_text, limit=4)
            merged = list(dict.fromkeys([*sec_kws, *chunk_kws, *doc_keywords[:3]]))[:10]
            updated.append(chunk.model_copy(update={"enriched_keywords": ", ".join(merged)}))
        return updated
