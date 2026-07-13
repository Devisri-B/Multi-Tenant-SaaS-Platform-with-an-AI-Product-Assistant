"""Splitting documentation into retrievable chunks."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass(slots=True)
class Chunk:
    ordinal: int
    content: str
    token_estimate: int
    metadata: dict


def estimate_tokens(text: str) -> int:
    """Rough token count — good enough for budgeting the context window."""
    return max(1, len(text) // 4)


def _splitter(chunk_size: int, chunk_overlap: int):
    """Prefer LangChain's recursive splitter, fall back to a local one."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - exercised only without langchain
        return None
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _naive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    stride = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        window = text[start : start + chunk_size]
        boundary = window.rfind("\n\n")
        if boundary < chunk_size // 2:
            boundary = window.rfind(". ")
        if boundary > chunk_size // 2 and start + chunk_size < len(text):
            window = window[: boundary + 1]
        pieces.append(window.strip())
        start += max(stride, len(window) - chunk_overlap)
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
