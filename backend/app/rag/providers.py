"""LLM and embedding providers.

Production uses OpenAI through LangChain.  Tests and offline development use
``FakeProvider``, a deterministic hash-based embedder plus an extractive
"generator" — the pipeline exercises exactly the same code paths without a
network call or an API key.
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
            ans_match = re.search(r"Candidate Answer:\s*(.*?)$", user_prompt, flags=re.DOTALL)
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
# OpenAI provider
# ---------------------------------------------------------------------------
class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ProviderError("OPENAI_API_KEY is not configured.")
        from langchain_openai import OpenAIEmbeddings as LCEmbeddings

        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self._client = LCEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"Embedding request failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(text)
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"Embedding request failed: {exc}") from exc


class OpenAIChat(ChatProvider):
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ProviderError("OPENAI_API_KEY is not configured.")
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self._prompt = ChatPromptTemplate.from_messages(
            [("system", "{system_prompt}"), ("human", "{user_prompt}")]
        )
        self._model = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
            timeout=45,
            max_retries=2,
        )
        self._chain = self._prompt | self._model | StrOutputParser()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._chain.invoke(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            ).strip()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"Chat completion failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    if settings.LLM_PROVIDER == "openai":
        return OpenAIEmbeddings()
    return FakeEmbeddings()


@lru_cache
def get_chat_provider() -> ChatProvider:
    if settings.LLM_PROVIDER == "openai":
        return OpenAIChat()
    return FakeChat()


def reset_provider_cache() -> None:
    """Drop cached providers (used by tests that flip ``LLM_PROVIDER``)."""
    get_embedding_provider.cache_clear()
    get_chat_provider.cache_clear()
