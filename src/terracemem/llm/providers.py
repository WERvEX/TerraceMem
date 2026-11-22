"""Async OpenAI-compatible LLM, Embedding, and Reranker provider protocols & implementations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

import httpx
import jieba
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from terracemem.config import Settings

logger = structlog.get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def format_pg_halfvec(vector: Sequence[float]) -> str:
    """Serialize a dense float sequence to pgvector halfvec literal syntax."""
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity clamped to [0.0, 1.0] for two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _is_placeholder_key(api_key: str) -> bool:
    stripped = api_key.strip()
    return not stripped or stripped in {"your-api-key-here", "not-configured", "changeme"}


class LLMUsage(BaseModel):
    """Token consumption metadata reported by an LLM call."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model: str = ""


class LLMTextResult(BaseModel):
    """Unstructured text completion paired with token usage."""

    text: str
    usage: LLMUsage = Field(default_factory=LLMUsage)


class StructuredLLMResult[TModel: BaseModel](BaseModel):
    """Schema-validated structured completion paired with token usage."""

    parsed: TModel
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw_text: str = ""


class LLMProvider(Protocol):
    """Async chat/structured generation provider protocol."""

    @property
    def is_configured(self) -> bool: ...

    async def complete_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMTextResult: ...

    async def complete_structured[TModel: BaseModel](
        self,
        messages: Sequence[Mapping[str, str]],
        response_model: type[TModel],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> StructuredLLMResult[TModel]: ...

    def stream_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


class EmbeddingProvider(Protocol):
    """Computes dense embedding vectors for documents and queries."""

    @property
    def is_configured(self) -> bool: ...

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankerProvider(Protocol):
    """Scores candidate passages jointly against a query."""

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]: ...


def _extract_json_blob(text: str) -> str:
    """Extract raw JSON object/array from plain or markdown-fenced LLM output."""
    stripped = text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        stripped = match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


class OpenAICompatibleLLMProvider:
    """OpenAI-compatible async chat & structured JSON provider.

    Works across OpenAI, DeepSeek, Qwen/DashScope, OpenRouter, vLLM, and Ollama.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._configured = not _is_placeholder_key(settings.llm_api_key)
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key or "not-configured",
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout_seconds,
        )

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _select_model(self, use_fast_model: bool) -> str:
        return self._settings.fast_llm_model if use_fast_model else self._settings.llm_model

    async def complete_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMTextResult:
        model = self._select_model(use_fast_model)
        payload_messages: list[Any] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            model=model,
        )
        return LLMTextResult(text=content, usage=usage)

    async def complete_structured[TModel: BaseModel](
        self,
        messages: Sequence[Mapping[str, str]],
        response_model: type[TModel],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> StructuredLLMResult[TModel]:
        model = self._select_model(use_fast_model)
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        schema_instruction = (
            "Respond ONLY with a valid JSON object matching this exact JSON Schema "
            f"(no prose, no markdown fences):\n{schema_json}"
        )

        payload_messages: list[Any] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        if payload_messages and payload_messages[0]["role"] == "system":
            payload_messages[0]["content"] = (
                f"{payload_messages[0]['content']}\n\n{schema_instruction}"
            )
        else:
            payload_messages.insert(0, {"role": "system", "content": schema_instruction})

        total_prompt = 0
        total_completion = 0
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(2):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": payload_messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = await self._client.chat.completions.create(**kwargs)
            last_raw = response.choices[0].message.content or ""
            if response.usage:
                total_prompt += response.usage.prompt_tokens
                total_completion += response.usage.completion_tokens

            blob = _extract_json_blob(last_raw)
            try:
                parsed = response_model.model_validate_json(blob)
                return StructuredLLMResult(
                    parsed=parsed,
                    usage=LLMUsage(
                        prompt_tokens=total_prompt,
                        completion_tokens=total_completion,
                        model=model,
                    ),
                    raw_text=last_raw,
                )
            except ValidationError as exc:
                last_error = exc
                if attempt == 0:
                    payload_messages.append({"role": "assistant", "content": last_raw})
                    payload_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your JSON failed schema validation: {exc.errors()[:3]}. "
                                "Return corrected JSON only."
                            ),
                        }
                    )

        raise ValueError(
            f"Structured LLM output failed validation for {response_model.__name__}: {last_error}"
        )

    async def stream_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        model = self._select_model(use_fast_model)
        payload_messages: list[Any] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible embedding client (works with OpenAI, DashScope, SiliconFlow, TEI)."""

    def __init__(self, settings: Settings, *, batch_size: int = 64) -> None:
        self._settings = settings
        self._batch_size = max(1, batch_size)
        effective_key = settings.embedding_api_key or settings.llm_api_key
        self._configured = not _is_placeholder_key(effective_key)
        self._client = AsyncOpenAI(
            api_key=effective_key or "not-configured",
            base_url=settings.embedding_base_url,
            timeout=settings.request_timeout_seconds,
        )

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = await self._client.embeddings.create(
                model=self._settings.embedding_model,
                input=batch,
                dimensions=self._settings.embedding_dim,
            )
            vectors.extend(list(item.embedding) for item in response.data)
        return vectors


class DeterministicHashEmbeddingProvider:
    """Deterministic bilingual feature-hashing embedder for offline tests & local dev.

    Produces unit-normalized dense vectors matching `embedding_dim` using Jieba search
    tokens plus ASCII word bi-grams so lexical/semantic overlap yields smooth cosine scores.
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    @property
    def is_configured(self) -> bool:
        return True

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = [tok.strip().lower() for tok in jieba.cut_for_search(text) if tok.strip()]
        if not tokens:
            return vec

        features = list(tokens)
        for idx in range(len(tokens) - 1):
            features.append(f"{tokens[idx]}_{tokens[idx + 1]}")

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            raw_int = int.from_bytes(digest[:4], "big")
            bucket = raw_int % self._dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class PassthroughReranker:
    """Deterministic fallback reranker used when external reranker API is disabled."""

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        del query
        count = min(len(passages), top_n)
        if count == 0:
            return []
        return [(idx, round(1.0 - (idx / max(count, 1)) * 0.5, 4)) for idx in range(count)]


class LexicalOverlapReranker:
    """Zero-dependency bilingual overlap reranker providing calibrated [0, 1] scores locally."""

    def _tokenize(self, text: str) -> set[str]:
        return {
            tok.strip().lower()
            for tok in jieba.cut_for_search(text)
            if len(tok.strip()) > 0 and not tok.strip().isspace()
        }

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not passages or top_n <= 0:
            return []
        q_tokens = self._tokenize(query)
        q_lower = query.strip().lower()
        scored: list[tuple[int, float]] = []

        for idx, passage in enumerate(passages):
            p_tokens = self._tokenize(passage)
            if not q_tokens or not p_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & p_tokens) / len(q_tokens)
            phrase_bonus = (
                0.15 if (q_lower and len(q_lower) >= 4 and q_lower in passage.lower()) else 0.0
            )
            rank_prior = 0.05 * (1.0 / (1.0 + idx))
            score = min(1.0, round(0.85 * overlap + phrase_bonus + rank_prior, 4))
            scored.append((idx, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_n]


class HTTPRerankerProvider:
    """Cross-encoder REST reranker compatible with Jina, SiliconFlow, DashScope, Cohere, and TEI."""

    def __init__(
        self,
        settings: Settings,
        *,
        fallback: RerankerProvider | None = None,
    ) -> None:
        self._settings = settings
        self._fallback = fallback or PassthroughReranker()

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not passages or top_n <= 0:
            return []
        if not self._settings.reranker_enabled or not self._settings.reranker_base_url:
            return await self._fallback.rerank(query, passages, top_n=top_n)

        url = self._settings.reranker_base_url.rstrip("/")
        if not url.endswith("/rerank"):
            url = f"{url}/rerank"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if not _is_placeholder_key(self._settings.reranker_api_key):
            headers["Authorization"] = f"Bearer {self._settings.reranker_api_key}"

        payload = {
            "model": self._settings.reranker_model,
            "query": query,
            "documents": list(passages),
            "top_n": min(top_n, len(passages)),
        }
        try:
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
        except Exception as exc:
            logger.warning("reranker_http_fallback", error=str(exc), url=url)
            return await self._fallback.rerank(query, passages, top_n=top_n)

        raw_results = body.get("results") or body.get("data") or []
        parsed: list[tuple[int, float]] = []
        for item in raw_results:
            if not isinstance(item, Mapping):
                continue
            raw_idx = item.get("index")
            raw_score = item.get("relevance_score", item.get("score"))
            if (
                isinstance(raw_idx, int)
                and isinstance(raw_score, (int, float))
                and 0 <= raw_idx < len(passages)
            ):
                parsed.append((raw_idx, float(raw_score)))

        if not parsed:
            return await self._fallback.rerank(query, passages, top_n=top_n)
        parsed.sort(key=lambda pair: (-pair[1], pair[0]))
        return parsed[:top_n]


def create_default_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Use OpenAI-compatible embeddings when configured, else deterministic hash embeddings."""
    remote = OpenAICompatibleEmbeddingProvider(settings)
    if remote.is_configured:
        return remote
    return DeterministicHashEmbeddingProvider(dim=settings.embedding_dim)


def create_default_llm_provider(settings: Settings) -> LLMProvider:
    """Return configured OpenAI-compatible LLM provider."""
    return OpenAICompatibleLLMProvider(settings)


def create_default_reranker_provider(settings: Settings) -> RerankerProvider:
    """Use HTTP reranker when enabled, else calibrated local overlap reranker."""
    local_fallback = LexicalOverlapReranker()
    if settings.reranker_enabled and settings.reranker_base_url:
        return HTTPRerankerProvider(settings, fallback=local_fallback)
    return local_fallback
