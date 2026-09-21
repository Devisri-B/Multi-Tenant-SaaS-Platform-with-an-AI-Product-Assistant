"""Cross-Encoder Reranker for deep query-passage interaction scoring.

Takes top candidate chunks from hybrid retrieval (dense + sparse) and computes
deep semantic cross-attention scores using a local CrossEncoder model
(e.g., cross-encoder/ms-marco-MiniLM-L-6-v2), elevating true contextual relevance
and eliminating false positives.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import structlog

from app.core.config import settings
from app.core.exceptions import ProviderError

if TYPE_CHECKING:
    from app.rag.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class RerankerProvider(ABC):
    """Abstract base class for cross-encoder passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Score (query, chunk) pairs and return chunks reordered by relevance."""
        ...


class CrossEncoderReranker(RerankerProvider):
    """Local Cross-Encoder model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device_str = device or settings.EMBEDDING_DEVICE
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device_str)
        except Exception as exc:
            raise ProviderError(
                f"Failed to load CrossEncoder model '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        from app.rag.retriever import RetrievedChunk

        if not chunks:
            return []
        self._ensure_loaded()
        pairs = [[query, f"{c.document_title}: {c.content}"] for c in chunks]
        try:
            raw_scores = self._model.predict(pairs)
            import numpy as np

            scores_arr = np.array(raw_scores, dtype=float)
            # Map unconstrained logits to (0.0, 1.0) probability range
            if scores_arr.ndim > 0 and (scores_arr.min() < 0.0 or scores_arr.max() > 1.0):
                scores_arr = 1.0 / (1.0 + np.exp(-scores_arr))

            reranked: list[RetrievedChunk] = []
            for chunk, score in zip(chunks, scores_arr.tolist(), strict=False):
                reranked.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_title=chunk.document_title,
                        ordinal=chunk.ordinal,
                        content=chunk.content,
                        score=round(float(score), 4),
                    )
                )
            reranked.sort(key=lambda c: c.score, reverse=True)
            limit = top_k or len(reranked)
            return reranked[:limit]
        except Exception as exc:
            raise ProviderError(f"CrossEncoder reranking failed: {exc}") from exc


class FakeReranker(RerankerProvider):
    """Fast deterministic cross-encoder simulator for unit tests and CI."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        from app.rag.retriever import RetrievedChunk

        if not chunks:
            return []

        q_tokens = set(_tokenize(query))
        meaningful_q = {t for t in q_tokens if len(t) > 2}
        reranked: list[RetrievedChunk] = []

        for chunk in chunks:
            doc_text = f"{chunk.document_title} {chunk.content}".lower()
            doc_tokens = set(_tokenize(doc_text))

            if meaningful_q:
                overlap = len(meaningful_q & doc_tokens) / len(meaningful_q)
            else:
                overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0

            exact_match = query.lower().strip() in doc_text

            if exact_match:
                score = min(1.0, 0.85 + 0.15 * overlap)
            elif overlap > 0:
                score = min(1.0, 0.50 + 0.40 * overlap + 0.10 * chunk.score)
            else:
                # Disjoint: weak relevance below RAG_MIN_SCORE
                score = min(0.25, round(chunk.score * 0.25, 4))

            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    ordinal=chunk.ordinal,
                    content=chunk.content,
                    score=round(score, 4),
                )
            )

        reranked.sort(key=lambda c: c.score, reverse=True)
        limit = top_k or len(reranked)
        return reranked[:limit]


@lru_cache
def get_reranker_provider() -> RerankerProvider:
    if settings.LLM_PROVIDER == "fake" or settings.RERANKER_PROVIDER == "fake":
        return FakeReranker()
    if settings.RERANKER_PROVIDER == "cross_encoder":
        return CrossEncoderReranker()
    return FakeReranker()


def reset_reranker_cache() -> None:
    get_reranker_provider.cache_clear()


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank candidate chunks using the active cross-encoder provider."""
    provider = get_reranker_provider()
    return provider.rerank(query, chunks, top_k=top_k)
