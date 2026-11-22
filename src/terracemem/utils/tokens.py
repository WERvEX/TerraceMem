"""Token estimation and bilingual CJK/English segmentation helpers."""

from __future__ import annotations

import re

import jieba

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """Estimate LLM token count accurately across Chinese and English mixed text.

    Approximation: ~0.7 tokens per CJK character and ~1.3 tokens per ASCII word,
    plus minimal overhead for punctuation/whitespace.
    """
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    ascii_words = len(_WORD_RE.findall(text))
    other_chars = max(0, len(text) - cjk_chars - sum(len(w) for w in _WORD_RE.findall(text)))
    estimated = int(cjk_chars * 0.7 + ascii_words * 1.3 + other_chars * 0.25)
    return max(1, estimated)


def segment_for_native_tsvector(text: str) -> str:
    """Pre-segment Chinese/English text using jieba search mode for PG 'simple' tsvector."""
    tokens = [tok.strip().lower() for tok in jieba.cut_for_search(text) if tok.strip()]
    return " ".join(tokens)
