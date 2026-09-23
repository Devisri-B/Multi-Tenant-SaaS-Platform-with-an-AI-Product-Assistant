"""Tenant-scoped hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF).

Combines semantic dense vector search (pgvector cosine distance) and lexical sparse search
(PostgreSQL Full-Text Search / BM25) executed concurrently via thread pooling.
Both retrieval branches are tenant-scoped in SQL to guarantee multi-tenant isolation.
"""

from __future__ import annotations

import math
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import engine
from app.db.tenancy import set_tenant_context
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(name: str | None = None, run_type: str | None = None, **kwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


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


# ---------------------------------------------------------------------------
# Dense Vector Search (pgvector & SQLite fallback)
# ---------------------------------------------------------------------------
def build_pgvector_query(
    tenant_id: uuid.UUID, query_embedding: list[float], top_k: int
) -> Select:
    """Build the nearest-neighbour statement pushed down to pgvector."""
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
    set_tenant_context(db, tenant_id)
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
    set_tenant_context(db, tenant_id)
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


# ---------------------------------------------------------------------------
# Sparse Lexical Search (PostgreSQL FTS & In-Python BM25 fallback)
# ---------------------------------------------------------------------------
def build_sparse_query(tenant_id: uuid.UUID, query: str, top_k: int) -> Select:
    """Build the PostgreSQL Full-Text Search (FTS) statement using plainto_tsquery."""
    ts_query = func.plainto_tsquery("english", query)
    ts_vector = func.to_tsvector("english", DocumentChunk.content)
    rank = func.ts_rank_cd(ts_vector, ts_query)

    return (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.title,
            DocumentChunk.ordinal,
            DocumentChunk.content,
            rank.label("rank"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            Document.status == DocumentStatus.INDEXED,
            ts_vector.op("@@")(ts_query),
        )
        .order_by(rank.desc())
        .limit(top_k)
    )


def _retrieve_sparse_pg(
    db: Session, tenant_id: uuid.UUID, query: str, top_k: int
) -> list[RetrievedChunk]:
    set_tenant_context(db, tenant_id)
    stmt = build_sparse_query(tenant_id, query, top_k)
    results = db.execute(stmt).all()
    if not results:
        return []

    # Normalize ts_rank scores to [0.0, 1.0]
    max_rank = max(float(row.rank) for row in results) or 1.0
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            ordinal=row.ordinal,
            content=row.content,
            score=round(float(row.rank) / max_rank, 4),
        )
        for row in results
    ]


def _retrieve_sparse_python(
    db: Session, tenant_id: uuid.UUID, query: str, top_k: int
) -> list[RetrievedChunk]:
    """Pure-Python BM25 ranker for SQLite tests and offline local dev."""
    set_tenant_context(db, tenant_id)
    q_terms = [t for t in _tokenize(query) if len(t) > 1]
    if not q_terms:
        return []

    stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            Document.status == DocumentStatus.INDEXED,
        )
    )
    rows = db.execute(stmt).all()
    if not rows:
        return []

    corpus_docs: list[tuple[Any, str, list[str]]] = []
    doc_freqs: dict[str, int] = {t: 0 for t in q_terms}

    for chunk, title in rows:
        tokens = _tokenize(f"{title} {chunk.content}")
        token_set = set(tokens)
        for t in q_terms:
            if t in token_set:
                doc_freqs[t] += 1
        corpus_docs.append((chunk, title, tokens))

    total_docs = len(corpus_docs)
    avg_dl = sum(len(doc[2]) for doc in corpus_docs) / total_docs if total_docs else 1.0
    k1 = 1.5
    b = 0.75

    idfs: dict[str, float] = {}
    for t in q_terms:
        df = doc_freqs[t]
        idfs[t] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0) if df > 0 else 0.0

    scored: list[RetrievedChunk] = []
    for chunk, title, doc_tokens in corpus_docs:
        doc_len = len(doc_tokens)
        counts: dict[str, int] = {}
        for tok in doc_tokens:
            counts[tok] = counts.get(tok, 0) + 1

        bm25_score = 0.0
        for t in q_terms:
            tf = counts.get(t, 0)
            if tf > 0:
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                bm25_score += idfs[t] * (numerator / denominator)

        if bm25_score > 0.0:
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=title,
                    ordinal=chunk.ordinal,
                    content=chunk.content,
                    score=bm25_score,
                )
            )

    scored.sort(key=lambda c: c.score, reverse=True)
    if scored:
        max_score = scored[0].score or 1.0
        for c in scored:
            c.score = round(c.score / max_score, 4)

    return scored[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
@traceable(name="reciprocal_rank_fusion", run_type="parser")
def reciprocal_rank_fusion(
    dense_hits: list[RetrievedChunk],
    sparse_hits: list[RetrievedChunk],
    top_k: int,
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse dense semantic rankings and sparse lexical rankings using RRF.

    Score(d) = (dense_weight / (rrf_k + rank_dense)) + (sparse_weight / (rrf_k + rank_sparse))
    """
    rrf_scores: dict[uuid.UUID, float] = {}
    chunk_map: dict[uuid.UUID, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_hits, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
            dense_weight / (rrf_k + rank)
        )
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_hits, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
            sparse_weight / (rrf_k + rank)
        )
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    if not rrf_scores:
        return []

    # Max possible score occurs when a chunk is rank 1 in both dense and sparse
    max_theoretical = (dense_weight / (rrf_k + 1)) + (sparse_weight / (rrf_k + 1))
    scale = max_theoretical if max_theoretical > 0 else 1.0

    # Sort candidates by combined RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_k]:
        base_chunk = chunk_map[cid]
        normalized_score = min(1.0, round(rrf_scores[cid] / scale, 4))
        fused_results.append(
            RetrievedChunk(
                chunk_id=base_chunk.chunk_id,
                document_id=base_chunk.document_id,
                document_title=base_chunk.document_title,
                ordinal=base_chunk.ordinal,
                content=base_chunk.content,
                score=normalized_score,
            )
        )

    return fused_results


# ---------------------------------------------------------------------------
# Concurrent Hybrid Retrieval
# ---------------------------------------------------------------------------
def retrieve_hybrid_concurrent(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    query: str,
    query_embedding: list[float] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    rrf_k: int | None = None,
    enable_reranker: bool | None = None,
) -> list[RetrievedChunk]:
    """Execute dense semantic and sparse lexical retrieval concurrently, followed by
    cross-encoder reranking.

    Runs dense embedding + vector search and sparse full-text search in parallel
    threads, completely hiding the sparse search time behind dense embedding latency.
    Fuses hits using Reciprocal Rank Fusion, then passes top candidates through a
    Cross-Encoder Reranker for deep query-passage interaction scoring.
    """
    from app.rag.providers import get_embedding_provider
    from app.rag.reranker import rerank_chunks

    top_k = top_k or settings.RAG_TOP_K
    rrf_k_val = rrf_k or settings.RAG_RRF_K
    w_dense = dense_weight if dense_weight is not None else settings.RAG_DENSE_WEIGHT
    w_sparse = sparse_weight if sparse_weight is not None else settings.RAG_SPARSE_WEIGHT

    should_rerank = (
        settings.RAG_ENABLE_RERANKER if enable_reranker is None else enable_reranker
    )
    candidate_k = (
        max(top_k * 3, settings.RERANKER_CANDIDATE_POOL)
        if should_rerank
        else max(top_k * 2, 10)
    )

    is_pg = db.bind is not None and db.bind.dialect.name == "postgresql"
    bind_target = db.bind or engine

    if not is_pg:
        # On SQLite / StaticPool (unit test suite):
        # StaticPool shares a single underlying DBAPI connection across threads.
        # Opening/closing child sessions in background threads resets/rolls back
        # active transactions on the caller's session.
        # Instead, generate dense embedding in background thread (concurrency)
        # while querying sparse lexical hits on the caller's session.
        @traceable(name="sparse_lexical_search", run_type="retriever")
        def _sparse_local() -> list[RetrievedChunk]:
            return _retrieve_sparse_python(db, tenant_id, query, candidate_k)

        @traceable(name="dense_vector_search", run_type="retriever")
        def _dense_local(embedding: list[float]) -> list[RetrievedChunk]:
            return _retrieve_in_python(db, tenant_id, embedding, candidate_k)

        if query_embedding is None:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder") as executor:
                emb_future = executor.submit(get_embedding_provider().embed_query, query)
                sparse_hits = _sparse_local()
                emb = emb_future.result()
        else:
            emb = query_embedding
            sparse_hits = _sparse_local()

        dense_hits = _dense_local(emb)
    else:
        @traceable(name="dense_vector_search", run_type="retriever")
        def _dense_worker() -> list[RetrievedChunk]:
            # Dedicated session for thread safety and independent RLS context
            with Session(bind=bind_target, expire_on_commit=False) as sess:
                set_tenant_context(sess, tenant_id)
                emb = query_embedding
                if emb is None:
                    emb = get_embedding_provider().embed_query(query)
                return _retrieve_pgvector(sess, tenant_id, emb, candidate_k)

        @traceable(name="sparse_lexical_search", run_type="retriever")
        def _sparse_worker() -> list[RetrievedChunk]:
            # Dedicated session for thread safety and independent RLS context
            with Session(bind=bind_target, expire_on_commit=False) as sess:
                set_tenant_context(sess, tenant_id)
                try:
                    return _retrieve_sparse_pg(sess, tenant_id, query, candidate_k)
                except Exception as exc:
                    log.warning("retriever.sparse_worker_failed", error=str(exc))
                    return []

        # Execute dense and sparse retrieval concurrently with dedicated PostgreSQL connections
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retriever") as executor:
            dense_future = executor.submit(_dense_worker)
            sparse_future = executor.submit(_sparse_worker)

            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()

    fused = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        top_k=candidate_k,
        rrf_k=rrf_k_val,
        dense_weight=w_dense,
        sparse_weight=w_sparse,
    )

    if should_rerank and fused:
        results = rerank_chunks(query=query, chunks=fused, top_k=top_k)
    else:
        results = fused[:top_k]

    if min_score is not None:
        return [c for c in results if c.score >= min_score]
    return results


def retrieve(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    query_embedding: list[float] | None = None,
    query: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    enable_reranker: bool | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top_k document chunks for a tenant.

    If ``query`` is provided and hybrid search is enabled, executes concurrent
    dense + sparse hybrid retrieval with Reciprocal Rank Fusion and Cross-Encoder Reranking.
    Otherwise falls back to vector-only retrieval for backward compatibility.
    """
    top_k = top_k or settings.RAG_TOP_K

    if query and settings.RAG_ENABLE_HYBRID_SEARCH:
        return retrieve_hybrid_concurrent(
            db,
            tenant_id=tenant_id,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            enable_reranker=enable_reranker,
        )

    # Legacy vector-only fallback path
    from app.rag.providers import get_embedding_provider
    from app.rag.reranker import rerank_chunks

    emb = query_embedding
    if emb is None and query:
        emb = get_embedding_provider().embed_query(query)

    if emb is None:
        return []

    threshold = settings.RAG_MIN_SCORE if min_score is None else min_score

    should_rerank = (
        settings.RAG_ENABLE_RERANKER if enable_reranker is None else enable_reranker
    )
    fetch_k = (
        max(top_k * 3, settings.RERANKER_CANDIDATE_POOL)
        if (should_rerank and query)
        else top_k
    )

    if db.bind is not None and db.bind.dialect.name == "postgresql":
        results = _retrieve_pgvector(db, tenant_id, emb, fetch_k)
    else:
        results = _retrieve_in_python(db, tenant_id, emb, fetch_k)

    if should_rerank and query and results:
        results = rerank_chunks(query=query, chunks=results, top_k=top_k)

    return [chunk for chunk in results if chunk.score >= threshold]
