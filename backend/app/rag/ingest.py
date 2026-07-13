"""Turn an uploaded document into embedded, retrievable chunks."""

from __future__ import annotations

import hashlib
import io
import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ProviderError, ValidationError
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.rag.chunking import chunk_text
from app.rag.providers import get_embedding_provider

log = get_logger(__name__)

EMBED_BATCH_SIZE = 64
SUPPORTED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "text/csv",
}


def checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_text(payload: bytes, content_type: str, filename: str) -> str:
    """Decode an upload into plain text."""
    lowered = (content_type or "").split(";")[0].strip().lower()

    if lowered == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("PDF support requires the 'pypdf' package.") from exc
        reader = PdfReader(io.BytesIO(payload))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(page for page in pages if page)

    if lowered and lowered not in SUPPORTED_TEXT_TYPES and not lowered.startswith("text/"):
        raise ValidationError(f"Unsupported document type '{content_type}'.")

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="replace")


def _embed_in_batches(texts: list[str]) -> list[list[float]]:
    provider = get_embedding_provider()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(provider.embed_documents(texts[start : start + EMBED_BATCH_SIZE]))
    return vectors


def index_document(db: Session, document: Document, text: str) -> Document:
    """Chunk, embed and persist a document. Idempotent — re-indexing replaces."""
    document.status = DocumentStatus.PROCESSING
    document.error_message = None
    db.flush()

    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))

    try:
        chunks = chunk_text(
            text,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            metadata={"source": document.source_name, "title": document.title},
        )
        if not chunks:
            raise ValidationError("The document contains no extractable text.")

        embeddings = _embed_in_batches([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ProviderError("Embedding provider returned a mismatched batch size.")

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            db.add(
                DocumentChunk(
                    tenant_id=document.tenant_id,
                    document_id=document.id,
                    ordinal=chunk.ordinal,
                    content=chunk.content,
                    token_estimate=chunk.token_estimate,
                    embedding=embedding,
                    chunk_metadata=chunk.metadata,
                )
            )

        document.chunk_count = len(chunks)
        document.status = DocumentStatus.INDEXED
        db.flush()
        log.info(
            "document.indexed",
            document_id=str(document.id),
            tenant_id=str(document.tenant_id),
            chunks=len(chunks),
        )
    except Exception as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:500]
        document.chunk_count = 0
        db.flush()
        log.warning("document.index_failed", document_id=str(document.id), error=str(exc))
        raise

    return document


def purge_document_vectors(db: Session, document_id: uuid.UUID) -> int:
    result = db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    db.flush()
    return int(result.rowcount or 0)
