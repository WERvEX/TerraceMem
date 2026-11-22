"""Deterministic citation whitelist verifier blocking fabricated or ungrounded references."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from terracemem.domain.models import Citation, ClaimWithCitations, RetrievedEvidence


class CitationVerificationResult(BaseModel):
    """Audit output of deterministic citation verification."""

    passed: bool
    valid_citations: list[Citation] = Field(default_factory=list)
    invalid_chunk_ids: list[UUID] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    reason: str | None = None


class CitationVerifier:
    """Validates that every cited chunk ID belongs strictly to the retrieved evidence whitelist."""

    def verify(
        self,
        claims: Sequence[ClaimWithCitations],
        evidence: Sequence[RetrievedEvidence],
        *,
        require_all_claims_cited: bool = True,
    ) -> CitationVerificationResult:
        evidence_map: dict[UUID, RetrievedEvidence] = {item.chunk_id: item for item in evidence}

        invalid_ids: list[UUID] = []
        uncited_claims: list[str] = []
        valid_citation_map: dict[UUID, Citation] = {}

        for claim in claims:
            if not claim.cited_chunk_ids:
                uncited_claims.append(claim.statement)
                continue
            for cid in claim.cited_chunk_ids:
                ev = evidence_map.get(cid)
                if ev is None:
                    if cid not in invalid_ids:
                        invalid_ids.append(cid)
                elif cid not in valid_citation_map:
                    valid_citation_map[cid] = Citation(
                        chunk_id=ev.chunk_id,
                        document_id=ev.document_id,
                        document_title=ev.document_title,
                        heading_path=ev.heading_path,
                    )

        if invalid_ids:
            return CitationVerificationResult(
                passed=False,
                valid_citations=list(valid_citation_map.values()),
                invalid_chunk_ids=invalid_ids,
                uncited_claims=uncited_claims,
                reason="Answer cited chunk IDs outside the retrieved evidence whitelist.",
            )

        if require_all_claims_cited and (not claims or uncited_claims):
            return CitationVerificationResult(
                passed=False,
                valid_citations=list(valid_citation_map.values()),
                invalid_chunk_ids=[],
                uncited_claims=uncited_claims,
                reason="One or more claims lacked verifiable chunk citations.",
            )

        return CitationVerificationResult(
            passed=True,
            valid_citations=list(valid_citation_map.values()),
            invalid_chunk_ids=[],
            uncited_claims=uncited_claims,
        )
