"""Lightweight entity & relation extraction for bounded 2-hop PostgreSQL graph retrieval."""

from __future__ import annotations

import re

import structlog
from pydantic import BaseModel, Field

from terracemem.domain.models import EntityRecord, ParsedDocument, RelationEdgeRecord
from terracemem.llm.providers import LLMProvider
from terracemem.utils.ids import new_uuid7

logger = structlog.get_logger(__name__)

_BACKTICK_ENTITY_RE = re.compile(r"`([A-Za-z0-9_.\-/:]{2,48})`")
_PASCAL_OR_ACRONYM_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]{2,}(?:[A-Z][a-zA-Z0-9]+)+|[A-Z]{2,}[A-Z0-9_-]{1,})\b"
)
_CJK_PROPER_RE = re.compile(r"《([^》]{2,24})》|「([^」]{2,24})」")

_ENTITY_STOPWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "AND",
        "OR",
        "NOT",
        "NULL",
        "TRUE",
        "FALSE",
        "ORDER",
        "LIMIT",
        "GROUP",
        "CREATE",
        "TABLE",
        "INDEX",
        "INSERT",
        "UPDATE",
        "DELETE",
        "HTTP",
        "HTTPS",
        "JSON",
        "YAML",
        "REST",
        "API",
        "SQL",
    }
)


class _ExtractedTriple(BaseModel):
    src: str
    src_type: str = "concept"
    rel_type: str = "related_to"
    dst: str
    dst_type: str = "concept"


class _ChunkTripleList(BaseModel):
    triples: list[_ExtractedTriple] = Field(default_factory=list)


def extract_query_entity_candidates(query: str) -> list[str]:
    """Extract candidate entity names/identifiers from a user question."""
    found: list[str] = []
    for m in _BACKTICK_ENTITY_RE.findall(query):
        found.append(m.strip())
    for m in _PASCAL_OR_ACRONYM_RE.findall(query):
        if m not in _ENTITY_STOPWORDS:
            found.append(m.strip())
    for g1, g2 in _CJK_PROPER_RE.findall(query):
        term = (g1 or g2).strip()
        if term:
            found.append(term)
    # Also include capitalized words longer than 3 chars (excluding sentence-initial stop words)
    for token in re.findall(r"\b[A-Z][a-zA-Z0-9_-]{3,30}\b", query):
        if token.upper() not in _ENTITY_STOPWORDS and token not in {
            "What",
            "When",
            "Where",
            "Which",
            "Does",
            "How",
            "Why",
        }:
            found.append(token)
    return list(dict.fromkeys(found))[:10]


def _extract_chunk_entities_deterministic(text: str) -> list[str]:
    candidates: list[str] = []
    for item in _BACKTICK_ENTITY_RE.findall(text):
        if len(item) >= 3 and item.upper() not in _ENTITY_STOPWORDS:
            candidates.append(item)
    for item in _PASCAL_OR_ACRONYM_RE.findall(text):
        if item.upper() not in _ENTITY_STOPWORDS:
            candidates.append(item)
    for g1, g2 in _CJK_PROPER_RE.findall(text):
        term = (g1 or g2).strip()
        if term:
            candidates.append(term)
    return list(dict.fromkeys(candidates))[:8]


class EntityRelationExtractor:
    """Extracts canonical entities and evidenced relation edges from parsed documents."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    async def extract_from_document(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> tuple[list[EntityRecord], list[RelationEdgeRecord]]:
        entity_map: dict[tuple[str, str], EntityRecord] = {}
        edges: list[RelationEdgeRecord] = []

        def ensure_entity(name: str, entity_type: str = "concept") -> EntityRecord:
            cleaned = name.strip()
            key = (entity_type.lower(), cleaned.lower())
            existing = entity_map.get(key)
            if existing is not None:
                return existing
            rec = EntityRecord(
                id=new_uuid7(),
                tenant_id=tenant_id,
                namespace=namespace,
                canonical_name=cleaned,
                entity_type=entity_type.lower(),
            )
            entity_map[key] = rec
            return rec

        for chunk in parsed.chunks[:40]:
            used_llm = False
            if self._llm is not None and self._llm.is_configured:
                try:
                    triples = await self._extract_chunk_triples_llm(chunk.raw_text)
                    for triple in triples:
                        if not triple.src.strip() or not triple.dst.strip():
                            continue
                        src_ent = ensure_entity(triple.src, triple.src_type or "concept")
                        dst_ent = ensure_entity(triple.dst, triple.dst_type or "concept")
                        if src_ent.id != dst_ent.id:
                            edges.append(
                                RelationEdgeRecord(
                                    tenant_id=tenant_id,
                                    namespace=namespace,
                                    src_entity_id=src_ent.id,
                                    rel_type=triple.rel_type.strip().lower() or "related_to",
                                    dst_entity_id=dst_ent.id,
                                    evidence_chunk_id=chunk.id,
                                    weight=1.0,
                                )
                            )
                    used_llm = bool(triples)
                except Exception as exc:
                    logger.warning("graph_extract_llm_fallback", error=str(exc))

            if not used_llm:
                names = _extract_chunk_entities_deterministic(chunk.raw_text)
                if len(names) >= 2:
                    anchor = ensure_entity(names[0], "concept")
                    for target_name in names[1:]:
                        target = ensure_entity(target_name, "concept")
                        if anchor.id != target.id:
                            edges.append(
                                RelationEdgeRecord(
                                    tenant_id=tenant_id,
                                    namespace=namespace,
                                    src_entity_id=anchor.id,
                                    rel_type="co_occurs_with",
                                    dst_entity_id=target.id,
                                    evidence_chunk_id=chunk.id,
                                    weight=0.85,
                                )
                            )

        return (list(entity_map.values()), edges)

    async def _extract_chunk_triples_llm(self, text: str) -> list[_ExtractedTriple]:
        assert self._llm is not None
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract up to 5 concise directed entity-relation triples from the snippet "
                    "(e.g. Service -> depends_on -> Database). Return JSON only."
                ),
            },
            {"role": "user", "content": text[:1600]},
        ]
        result = await self._llm.complete_structured(
            messages,
            _ChunkTripleList,
            use_fast_model=True,
            max_tokens=400,
        )
        return result.parsed.triples[:5]
