"""Tenant-scoped vector retrieval.

On Postgres the nearest-neighbour search is pushed into pgvector via the ``<=>``
cosine-distance operator.  On SQLite (tests, offline dev) the same interface is
served by an in-Python cosine scan.  Either way the tenant predicate is applied
inside the SQL, so a retrieval can never surface another workspace's content.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    content: str
    score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def build_pgvector_query(
    tenant_id: uuid.UUID, query_embedding: list[float], top_k: int
) -> Select:
    """Build the nearest-neighbour statement pushed down to pgvector.

    Kept separate from execution so it can be compiled against the Postgres
    dialect in tests without a live server — the SQLite test path never
    exercises this branch, and a missing operator only surfaces at query-build
    time.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    return (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.title,
            DocumentChunk.ordinal,
            DocumentChunk.content,
            distance.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding.is_not(None),
            Document.status == DocumentStatus.INDEXED,
        )
        .order_by(distance)
        .limit(top_k)
    )


def _retrieve_pgvector(
    db: Session, tenant_id: uuid.UUID, query_embedding: list[float], top_k: int
) -> list[RetrievedChunk]:
    stmt = build_pgvector_query(tenant_id, query_embedding, top_k)
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            ordinal=row.ordinal,
            content=row.content,
            score=1.0 - float(row.distance),
        )
        for row in db.execute(stmt).all()
    ]


def _retrieve_in_python(
    db: Session, tenant_id: uuid.UUID, query_embedding: list[float], top_k: int
) -> list[RetrievedChunk]:
    stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding.is_not(None),
            Document.status == DocumentStatus.INDEXED,
        )
    )
    scored: list[RetrievedChunk] = []
    for chunk, title in db.execute(stmt).all():
        scored.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=title,
                ordinal=chunk.ordinal,
                content=chunk.content,
                score=cosine_similarity(query_embedding, chunk.embedding or []),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def retrieve(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` most similar chunks belonging to ``tenant_id``."""
    top_k = top_k or settings.RAG_TOP_K
    threshold = settings.RAG_MIN_SCORE if min_score is None else min_score

    if db.bind is not None and db.bind.dialect.name == "postgresql":
        results = _retrieve_pgvector(db, tenant_id, query_embedding, top_k)
    else:
        results = _retrieve_in_python(db, tenant_id, query_embedding, top_k)

    return [chunk for chunk in results if chunk.score >= threshold]
