"""LLM and embedding providers.

Production uses Anthropic Claude (via SyncAnthropic) for chat generation and
local SentenceTransformers (all-MiniLM-L6-v2) for zero-API-cost vector embeddings.
Tests and offline development use FakeEmbeddings and FakeChat — the pipeline
exercises exactly the same code paths without a network call or an API key.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.config import settings
from app.core.exceptions import ProviderError


class EmbeddingProvider(ABC):
    dimensions: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Deterministic offline provider
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbeddings(EmbeddingProvider):
    """Hashed bag-of-words embeddings.

    Not semantic, but stable and genuinely similarity-bearing: documents that
    share vocabulary land near each other, which is all the retrieval tests
    need to assert against.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [component / norm for component in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeChat(ChatProvider):
    """Extractive stand-in for a chat model.

    It echoes the most relevant context lines so that assertions about
    grounding and citation plumbing stay meaningful offline.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # 1. Document relevance grading prompt
        is_doc_grading = (
            "assessing whether retrieved documentation excerpts are relevant" in system_prompt
            or "assessing whether retrieved documentation excerpts are relevant" in user_prompt
        )
        if is_doc_grading:
            question_match = re.search(r"Question:\s*(.*?)\n", user_prompt)
            context_match = re.search(r"Context Excerpts:\s*(.*?)$", user_prompt, flags=re.DOTALL)
            if question_match and context_match:
                q_words = {token for token in _tokenize(question_match.group(1)) if len(token) > 2}
                c_words = set(_tokenize(context_match.group(1)))
                return "yes" if len(q_words & c_words) > 0 else "no"
            return "yes"

        # 2. Hallucination / groundedness grading prompt
        is_hallucination_grading = (
            "assessing whether an answer is grounded in facts" in system_prompt
            or "assessing whether an answer is grounded in facts" in user_prompt
        )
        if is_hallucination_grading:
            facts_match = re.search(
                r"Facts:\s*(.*?)\s*Candidate Answer:", user_prompt, flags=re.DOTALL
            )
            ans_match = re.search(
                r"Candidate Answer:\s*(.*?)(?:\n\nRespond with|\Z)", user_prompt, flags=re.DOTALL
            )
            if facts_match and ans_match:
                f_words = set(_tokenize(facts_match.group(1)))
                a_words = {token for token in _tokenize(ans_match.group(1)) if len(token) > 3}
                if not a_words:
                    return "yes"
                overlap = len(a_words & f_words) / len(a_words)
                # If more than 30% of significant answer words come from facts, consider grounded
                return "yes" if overlap >= 0.3 else "no"
            return "yes"

        # 3. Answer question relevance grading prompt
        is_answer_grading = (
            "assessing whether an answer resolves a user question" in system_prompt
            or "assessing whether an answer resolves a user question" in user_prompt
        )
        if is_answer_grading:
            if "could not find anything" in user_prompt.lower():
                return "no"
            return "yes"

        # 4. Question contextualization / rewrite prompt
        is_query_rewrite = (
            "rephrase the follow-up question" in system_prompt
            or "rephrase the follow-up question" in user_prompt
            or "Standalone Query:" in user_prompt
        )
        if is_query_rewrite:
            q_match = re.search(
                r"Follow-up Question:\s*(.*?)(?:\n|Standalone Query:|$)", user_prompt
            )
            h_match = re.search(
                r"<conversation_history>(.*?)</conversation_history>", user_prompt, flags=re.DOTALL
            )
            follow_up = q_match.group(1).strip() if q_match else ""
            if h_match and h_match.group(1).strip():
                history_text = h_match.group(1).strip()
                h_words = [w for w in _tokenize(history_text) if len(w) > 3]
                if any(
                    p in follow_up.lower()
                    for p in ("it", "this", "that", "they", "them", "these", "those")
                ):
                    topic = h_words[-1] if h_words else ""
                    return f"{follow_up} regarding {topic}".strip()
            return follow_up or "general inquiry"

        # Check for web search results
        web_match = re.search(
            r"<web_search_results>(.*?)</web_search_results>", user_prompt, flags=re.DOTALL
        )
        question_match = re.search(
            r"<question>(.*?)</question>", user_prompt, flags=re.DOTALL
        )
        question = (question_match.group(1) if question_match else user_prompt).strip()

        if web_match and web_match.group(1).strip():
            web_context = web_match.group(1)
            keywords = {token for token in _tokenize(question) if len(token) > 3}
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", web_context) if s.strip()]
            scored = sorted(
                sentences,
                key=lambda s: len(keywords & set(_tokenize(s))),
                reverse=True,
            )
            best = [s for s in scored[:3] if s]
            body = " ".join(best) if best else (sentences[0] if sentences else web_context[:200])
            return (
                f"This answer was found via online search (not in workspace documentation): {body}"
            )

        # Standard workspace docs context
        context_match = re.search(
            r"<context>(.*?)</context>", user_prompt, flags=re.DOTALL
        )

        if not context_match or not context_match.group(1).strip():
            return (
                "I could not find anything in this workspace's documentation that "
                "answers that question."
            )

        context = context_match.group(1)
        keywords = {token for token in _tokenize(question) if len(token) > 3}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        scored = sorted(
            sentences,
            key=lambda s: len(keywords & set(_tokenize(s))),
            reverse=True,
        )
        best = [s for s in scored[:3] if s]
        body = " ".join(best) if best else sentences[0]
        return f"Based on the workspace documentation: {body}"


# ---------------------------------------------------------------------------
# SentenceTransformers Local Embedding Provider
# ---------------------------------------------------------------------------
class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers (e.g. all-MiniLM-L6-v2)."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.SENTENCE_TRANSFORMER_MODEL
        self.device_str = device or settings.EMBEDDING_DEVICE
        self._model: Any = None
        self.dimensions = settings.EMBEDDING_DIMENSIONS

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device_str)
            self.dimensions = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise ProviderError(
                f"Failed to load SentenceTransformer model '{self.model_name}': {exc}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [vec.tolist() for vec in embeddings]
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"SentenceTransformer embed_documents failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"SentenceTransformer embed_query failed: {exc}") from exc


class AnthropicChat(ChatProvider):
    """Synchronous Anthropic Claude client for chat completions using SyncAnthropic."""

    def __init__(self) -> None:
        if not settings.ANTHROPIC_API_KEY:
            raise ProviderError("ANTHROPIC_API_KEY is not configured.")
        from anthropic import Anthropic as SyncAnthropic

        self._client = SyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=45.0,
            max_retries=2,
        )
        self.model = settings.ANTHROPIC_CHAT_MODEL

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts = [block.text for block in message.content if hasattr(block, "text")]
            return "".join(parts).strip()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"Anthropic chat completion failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    if settings.LLM_PROVIDER == "fake" or settings.EMBEDDING_PROVIDER == "fake":
        return FakeEmbeddings()
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        return SentenceTransformerEmbeddings()
    return FakeEmbeddings()


@lru_cache
def get_chat_provider() -> ChatProvider:
    if settings.LLM_PROVIDER == "anthropic":
        return AnthropicChat()
    return FakeChat()


def reset_provider_cache() -> None:
    """Drop cached providers (used by tests that flip ``LLM_PROVIDER``)."""
    from app.rag.nli import reset_nli_cache

    get_embedding_provider.cache_clear()
    get_chat_provider.cache_clear()
    reset_nli_cache()
