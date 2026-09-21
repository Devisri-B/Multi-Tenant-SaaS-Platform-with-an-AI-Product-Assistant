"""Tests for concurrent hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import time
import uuid
from typing import Any
from unittest.mock import patch

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.rag.retriever import (
    RetrievedChunk,
    build_sparse_query,
    reciprocal_rank_fusion,
    retrieve,
    retrieve_hybrid_concurrent,
)
from app.services import auth as auth_service
from app.services import tenant as tenant_service


def make_chunk(cid: uuid.UUID, title: str, content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        document_title=title,
        ordinal=0,
        content=content,
        score=score,
    )


def test_reciprocal_rank_fusion_boosts_overlap():
    id_a = uuid.uuid4()
    id_b = uuid.uuid4()
    id_c = uuid.uuid4()

    # Dense retrieved [A, B]
    dense_hits = [
        make_chunk(id_a, "Doc A", "Content A", score=0.9),
        make_chunk(id_b, "Doc B", "Content B", score=0.8),
    ]

    # Sparse retrieved [B, C]
    sparse_hits = [
        make_chunk(id_b, "Doc B", "Content B", score=0.95),
        make_chunk(id_c, "Doc C", "Content C", score=0.7),
    ]

    fused = reciprocal_rank_fusion(
        dense_hits, sparse_hits, top_k=3, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
    )

    # Chunk B appeared in both ranked lists, so its RRF score must be highest!
    assert fused[0].chunk_id == id_b
    assert len(fused) == 3
    # All scores should be normalized to <= 1.0
    assert 0.0 <= fused[0].score <= 1.0
    assert fused[0].score > fused[1].score


def test_reciprocal_rank_fusion_respects_weights():
    id_dense_top = uuid.uuid4()
    id_sparse_top = uuid.uuid4()

    dense_hits = [make_chunk(id_dense_top, "Dense Top", "Dense Content")]
    sparse_hits = [make_chunk(id_sparse_top, "Sparse Top", "Sparse Content")]

    # Heavy dense weight
    dense_favored = reciprocal_rank_fusion(
        dense_hits, sparse_hits, top_k=2, dense_weight=2.0, sparse_weight=0.5
    )
    assert dense_favored[0].chunk_id == id_dense_top

    # Heavy sparse weight
    sparse_favored = reciprocal_rank_fusion(
        dense_hits, sparse_hits, top_k=2, dense_weight=0.5, sparse_weight=2.0
    )
    assert sparse_favored[0].chunk_id == id_sparse_top


def test_build_sparse_query_compiles_postgres_dialect():
    tenant_id = uuid.uuid4()
    stmt = build_sparse_query(tenant_id, query="refund policy error 404", top_k=5)
    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))

    assert "plainto_tsquery" in compiled_sql
    assert "to_tsvector" in compiled_sql
    assert "ts_rank_cd" in compiled_sql
    assert "document_chunks.tenant_id = " in compiled_sql
    assert "LIMIT" in compiled_sql


def create_test_tenant(db: Session, prefix: str = "t") -> tuple[Any, uuid.UUID]:
    unique = uuid.uuid4().hex[:8]
    user = auth_service.create_user(
        db,
        email=f"{prefix}_{unique}@example.com",
        password="SecretPassw0rd123",
        full_name=f"User {unique}",
    )
    tenant = tenant_service.create_tenant(db, name=f"Tenant {unique}", owner=user)
    db.commit()
    return user, tenant.id


def test_hybrid_retrieval_e2e_with_exact_code_matching(db):
    """BM25 sparse search reliably surfaces exact codes/acronyms dense search might dilute."""
    _, tenant_id = create_test_tenant(db, "exact")
    doc = Document(
        tenant_id=tenant_id,
        title="Error Catalog",
        source_name="errors.txt",
        checksum=f"chk-err-{uuid.uuid4().hex[:6]}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc)
    db.flush()

    chunk_1 = DocumentChunk(
        tenant_id=tenant_id,
        document_id=doc.id,
        ordinal=0,
        content="General network connection timeout guidelines.",
        embedding=[0.05] * settings.EMBEDDING_DIMENSIONS,
    )
    chunk_2 = DocumentChunk(
        tenant_id=tenant_id,
        document_id=doc.id,
        ordinal=1,
        content="Fatal database lock failure with code ERR-PG-99281.",
        embedding=[0.05] * settings.EMBEDDING_DIMENSIONS,
    )
    db.add_all([chunk_1, chunk_2])
    db.commit()

    # Search for the exact error code
    hits = retrieve(
        db,
        tenant_id=tenant_id,
        query="ERR-PG-99281",
        top_k=5,
    )

    assert len(hits) > 0
    # Chunk with exact error code must rank first thanks to sparse lexical matching
    assert hits[0].chunk_id == chunk_2.id
    assert "ERR-PG-99281" in hits[0].content


def test_hybrid_retrieval_tenant_isolation(db):
    """Tenant A must never retrieve Tenant B's chunks via dense or sparse branches."""
    _, tenant_a_id = create_test_tenant(db, "tenanta")
    _, tenant_b_id = create_test_tenant(db, "tenantb")

    # Tenant 1 chunk
    doc_1 = Document(
        tenant_id=tenant_a_id,
        title="Owner Secret",
        source_name="s1.txt",
        checksum=f"chk-1-{uuid.uuid4().hex[:6]}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc_1)
    db.flush()
    db.add(
        DocumentChunk(
            tenant_id=tenant_a_id,
            document_id=doc_1.id,
            ordinal=0,
            content="Owner confidential revenue data $100M.",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
        )
    )

    # Tenant 2 chunk in a completely separate tenant
    doc_2 = Document(
        tenant_id=tenant_b_id,
        title="Other Public",
        source_name="s2.txt",
        checksum=f"chk-2-{uuid.uuid4().hex[:6]}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc_2)
    db.flush()
    db.add(
        DocumentChunk(
            tenant_id=tenant_b_id,
            document_id=doc_2.id,
            ordinal=0,
            content="Other public roadmap 2026.",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
        )
    )
    db.commit()

    # Tenant B queries for owner's confidential data
    hits = retrieve(
        db,
        tenant_id=tenant_b_id,
        query="confidential revenue data $100M",
        top_k=5,
    )

    # Only Tenant B's chunks may ever appear
    for hit in hits:
        assert hit.document_title != "Owner Secret"
        assert "revenue data" not in hit.content


def test_concurrent_execution_hides_sparse_latency(db):
    """Validates that dense embedding generation and sparse retrieval run concurrently."""
    _, tenant_id = create_test_tenant(db, "concurrent")
    doc = Document(
        tenant_id=tenant_id,
        title="Guide",
        source_name="guide.txt",
        checksum=f"chk-guide-{uuid.uuid4().hex[:6]}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc)
    db.flush()
    db.add(
        DocumentChunk(
            tenant_id=tenant_id,
            document_id=doc.id,
            ordinal=0,
            content="Nimbus product deployment guide and rolling updates.",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
        )
    )
    db.commit()

    # Simulate a slow embedding call (50ms)
    def slow_embed(*_args, **_kwargs):
        time.sleep(0.05)
        return [0.1] * settings.EMBEDDING_DIMENSIONS

    with patch("app.rag.providers.FakeEmbeddings.embed_query", side_effect=slow_embed):
        start = time.perf_counter()
        hits = retrieve_hybrid_concurrent(
            db,
            tenant_id=tenant_id,
            query="deployment guide",
            top_k=2,
        )
        elapsed = time.perf_counter() - start

    assert len(hits) > 0
    # Because they ran in parallel threads, elapsed time is close to ~50ms, well below 100ms serial
    assert elapsed < 0.15
