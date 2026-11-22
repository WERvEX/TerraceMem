"""Deterministic execution budgets and query deduplication guards for Agentic RAG."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")


def fingerprint_query(query: str) -> str:
    """Normalize and hash a retrieval query to prevent duplicate rewrite loops."""
    normalized = _WHITESPACE_RE.sub(" ", query.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExecutionBudget:
    """Hard deterministic boundaries preventing infinite agent loops or cost blowups."""

    max_retrieval_retries: int = 2
    max_state_transitions: int = 10
    max_llm_calls: int = 5
    timeout_seconds: float = 30.0

    retrieval_retries: int = 0
    state_transitions: int = 0
    llm_calls: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)
    seen_query_fingerprints: set[str] = field(default_factory=set)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    @property
    def is_timed_out(self) -> bool:
        return self.elapsed_seconds >= self.timeout_seconds

    def record_transition(self) -> bool:
        """Advance state transition counter; return False when transition ceiling is exceeded."""
        self.state_transitions += 1
        return self.state_transitions <= self.max_state_transitions and not self.is_timed_out

    def consume_llm_call(self) -> bool:
        """Consume one LLM call budget slot; return False if budget or deadline is exhausted."""
        if self.llm_calls >= self.max_llm_calls or self.is_timed_out:
            return False
        self.llm_calls += 1
        return True

    def register_query(self, query: str) -> bool:
        """Record a query fingerprint; return False if an identical query was already tried."""
        fp = fingerprint_query(query)
        if fp in self.seen_query_fingerprints:
            return False
        self.seen_query_fingerprints.add(fp)
        return True

    def can_retry_retrieval(self) -> bool:
        return self.retrieval_retries < self.max_retrieval_retries and not self.is_timed_out

    def record_retrieval_retry(self) -> bool:
        if not self.can_retry_retrieval():
            return False
        self.retrieval_retries += 1
        return True
