"""Splitting documentation into retrievable chunks with token-based measurement.

Anthropic models measure context and prompt lengths in tokens. Unlike OpenAI,
which requires the external `tiktoken` library, Anthropic provides native token
counting via its messages API (`client.messages.count_tokens`).

In `RecursiveCharacterTextSplitter`, `length_function` is configured with `count_tokens`:
- When an Anthropic client is active with valid credentials, exact token counts
  can be retrieved and cached via LRU.
- When running offline, in test suites, or when the API is unreachable, fast local
  subword tokenization counts tokens without any dependency on `tiktoken` or OpenAI SDK.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import settings


@dataclass(slots=True)
class Chunk:
    ordinal: int
    content: str
    token_estimate: int
    metadata: dict


# Fast subword / token pattern matching words and punctuation tokens
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_anthropic_client: Any = None


def set_anthropic_client(client: Any) -> None:
    """Set or override the Anthropic client for token counting (e.g. in tests)."""
    global _anthropic_client
    _anthropic_client = client
    count_tokens.cache_clear()


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None and getattr(settings, "ANTHROPIC_API_KEY", None):
        try:
            from anthropic import Anthropic as SyncAnthropic

            _anthropic_client = SyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                timeout=10.0,
                max_retries=1,
            )
        except Exception:
            pass
    return _anthropic_client


def _count_local_tokens(text: str) -> int:
    """Fast local subword token counter without tiktoken.

    Splits text on alphanumeric words and punctuation symbols, providing a fast,
    accurate approximation of BPE tokens (as used by Claude) without requiring
    tiktoken, network calls, or C extensions.
    """
    if not text:
        return 0
    tokens = _TOKEN_PATTERN.findall(text)
    return max(1, len(tokens))


@lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    """Token-based length function for RecursiveCharacterTextSplitter.

    Uses Anthropic native token counting when available, or local
    subword tokenization without requiring OpenAI's tiktoken.
    """
    normalised = text.strip()
    if not normalised:
        return 0

    client = _get_anthropic_client()
    if client is not None and getattr(settings, "LLM_PROVIDER", "") == "anthropic":
        try:
            resp = client.messages.count_tokens(
                model=settings.ANTHROPIC_CHAT_MODEL,
                messages=[{"role": "user", "content": normalised}],
            )
            return resp.input_tokens
        except Exception:
            # Graceful offline / error fallback to local counter
            pass

    return _count_local_tokens(normalised)


def estimate_tokens(text: str) -> int:
    """Token count used for chunk metadata and context window budgeting."""
    return count_tokens(text)


def _splitter(chunk_size: int, chunk_overlap: int):
    """Prefer LangChain's recursive splitter with token-based length_function."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - exercised only without langchain
        return None
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=count_tokens,
    )


def _naive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Token-aware fallback splitter when langchain_text_splitters is unavailable."""
    if count_tokens(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    paragraphs = text.split("\n\n")
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens <= chunk_size:
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            if current_chunk:
                pieces.append("\n\n".join(current_chunk).strip())
            current_chunk = [para]
            current_tokens = para_tokens

    if current_chunk:
        pieces.append("\n\n".join(current_chunk).strip())

    return [piece for piece in pieces if piece]


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks, preserving heading context."""
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalised = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
    normalised = normalised.strip()
    if not normalised:
        return []

    splitter = _splitter(chunk_size, chunk_overlap)
    raw_chunks = (
        splitter.split_text(normalised)
        if splitter is not None
        else _naive_split(normalised, chunk_size, chunk_overlap)
    )

    base_metadata = metadata or {}
    chunks: list[Chunk] = []
    current_heading: str | None = None

    for raw in raw_chunks:
        content = raw.strip()
        if not content:
            continue
        for line in content.split("\n"):
            if line.startswith("#"):
                current_heading = line.lstrip("#").strip()
                break
        chunk_metadata = dict(base_metadata)
        if current_heading:
            chunk_metadata["heading"] = current_heading
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                content=content,
                token_estimate=estimate_tokens(content),
                metadata=chunk_metadata,
            )
        )
    return chunks
