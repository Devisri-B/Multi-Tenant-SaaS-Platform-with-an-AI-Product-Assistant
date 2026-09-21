"""Tests for Cross-Encoder Reranker."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.rag.reranker import (
    CrossEncoderReranker,
    FakeReranker,
    get_reranker_provider,
    rerank_chunks,
    reset_reranker_cache,
)
from app.rag.retriever import RetrievedChunk, retrieve
from app.services import auth as auth_service
from app.services import tenant as tenant_service


def make_chunk(title: str, content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        ordinal=0,
        content=content,
        score=score,
    )


def test_fake_reranker_reorders_by_relevance():
    """Verify that FakeReranker elevates highly relevant passages over generic ones."""
    reranker = FakeReranker()
    c1 = make_chunk(
        "General Info", "General platform overview and security protocols.", score=0.9
    )
    c2 = make_chunk(
        "Kubernetes", "Pod autoscaling and ingress controllers in Kubernetes.", score=0.8
    )
    c3 = make_chunk(
        "Billing & Invoices",
        "Invoices are generated on the 1st of every month for billing.",
        score=0.4,
    )

    query = "When are billing invoices generated?"
    reranked = reranker.rerank(query, [c1, c2, c3], top_k=2)

    assert len(reranked) == 2
    # c3 had lower initial score (0.4) but has exact terms ("billing", "invoices", "generated")
    assert reranked[0].chunk_id == c3.chunk_id
    assert reranked[0].score >= 0.70
    assert reranked[1].chunk_id == c1.chunk_id


def test_fake_reranker_exact_phrase_match():
    """Exact phrase occurrence in passage receives significant score boost."""
    reranker = FakeReranker()
    c1 = make_chunk("Doc A", "Standard support ticket response SLA.", score=0.8)
    c2 = make_chunk(
        "Doc B", "For rapid assistance contact priority-support-hotline immediately.", score=0.5
    )

    query = "priority-support-hotline"
    reranked = reranker.rerank(query, [c1, c2])

    assert reranked[0].chunk_id == c2.chunk_id
    assert reranked[0].score >= 0.85


def test_fake_reranker_empty_input():
    reranker = FakeReranker()
    assert reranker.rerank("any query", []) == []


def test_cross_encoder_reranker_with_mock():
    """CrossEncoderReranker uses sentence_transformers.CrossEncoder and maps logits via sigmoid."""
    with patch("sentence_transformers.CrossEncoder") as mock_cls:
        mock_instance = MagicMock()
        # Mock predict to return raw unbounded logits [-1.5, 3.2]
        mock_instance.predict.return_value = [-1.5, 3.2]
        mock_cls.return_value = mock_instance

        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        c1 = make_chunk("Doc 1", "Low relevance content", score=0.5)
        c2 = make_chunk("Doc 2", "High relevance content", score=0.5)

        reranked = reranker.rerank("target query", [c1, c2], top_k=2)

        mock_instance.predict.assert_called_once()
        pairs = mock_instance.predict.call_args[0][0]
        assert len(pairs) == 2
        assert pairs[0] == ["target query", "Doc 1: Low relevance content"]
        assert pairs[1] == ["target query", "Doc 2: High relevance content"]

        # Logit 3.2 maps to sigmoid ~0.9608 -> Doc 2 ranks 1st
        assert reranked[0].chunk_id == c2.chunk_id
        assert 0.95 <= reranked[0].score <= 1.0

        # Logit -1.5 maps to sigmoid ~0.1824 -> Doc 1 ranks 2nd
        assert reranked[1].chunk_id == c1.chunk_id
        assert 0.15 <= reranked[1].score <= 0.25


def test_cross_encoder_reranker_model_load_failure():
    """Raises ProviderError if model cannot be loaded."""
    from app.core.exceptions import ProviderError

    with patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("Network offline")):
        reranker = CrossEncoderReranker(model_name="non-existent-model")
        with pytest.raises(ProviderError, match="Failed to load CrossEncoder model"):
            reranker.rerank("query", [make_chunk("T", "C")])


def test_get_reranker_provider_factory():
    """Factory respects settings and caches instance."""
    reset_reranker_cache()

    with patch.object(settings, "LLM_PROVIDER", "fake"):
        p1 = get_reranker_provider()
        assert isinstance(p1, FakeReranker)
        p2 = get_reranker_provider()
        assert p1 is p2

    reset_reranker_cache()
    with patch.object(settings, "LLM_PROVIDER", "anthropic"), patch.object(
        settings, "RERANKER_PROVIDER", "cross_encoder"
    ):
        p3 = get_reranker_provider()
        assert isinstance(p3, CrossEncoderReranker)

    reset_reranker_cache()


def test_rerank_chunks_convenience_function():
    c = make_chunk("Nimbus FAQ", "Nimbus provides automated daily backups.")
    results = rerank_chunks("automated backups", [c], top_k=1)
    assert len(results) == 1
    assert results[0].score >= 0.50


def test_retrieve_end_to_end_with_reranker(db: Session):
    """retrieve() executes hybrid search, fuses with RRF, and applies Cross-Encoder reranking."""
    unique = uuid.uuid4().hex[:8]
    user = auth_service.create_user(
        db, email=f"rerank_{unique}@acme.io", password="SecretPassw0rd123", full_name="User"
    )
    tenant = tenant_service.create_tenant(db, name=f"Rerank WS {unique}", owner=user)
    db.commit()

    doc = Document(
        tenant_id=tenant.id,
        title="Security Compliance",
        source_name="sec.txt",
        checksum=f"chk-{unique}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc)
    db.flush()

    chunk_1 = DocumentChunk(
        tenant_id=tenant.id,
        document_id=doc.id,
        ordinal=0,
        content="General guidelines for office security and badges.",
        embedding=[0.05] * settings.EMBEDDING_DIMENSIONS,
    )
    chunk_2 = DocumentChunk(
        tenant_id=tenant.id,
        document_id=doc.id,
        ordinal=1,
        content="SOC2 Type II compliance audit reports and penetration testing records.",
        embedding=[0.05] * settings.EMBEDDING_DIMENSIONS,
    )
    db.add_all([chunk_1, chunk_2])
    db.commit()

    # Query with reranker enabled
    hits = retrieve(
        db,
        tenant_id=tenant.id,
        query="SOC2 compliance audit",
        top_k=2,
        enable_reranker=True,
    )

    assert len(hits) > 0
    # Chunk 2 matches SOC2 compliance and should be rank 1 with high score
    assert hits[0].chunk_id == chunk_2.id
    assert hits[0].score >= 0.60


def test_retrieve_with_reranker_disabled_uses_raw_rrf(db: Session):
    """When enable_reranker=False, results retain raw RRF fusion scores."""
    unique = uuid.uuid4().hex[:8]
    user = auth_service.create_user(
        db, email=f"norerank_{unique}@acme.io", password="SecretPassw0rd123", full_name="User"
    )
    tenant = tenant_service.create_tenant(db, name=f"NoRerank WS {unique}", owner=user)
    db.commit()

    doc = Document(
        tenant_id=tenant.id,
        title="Doc",
        source_name="doc.txt",
        checksum=f"chk-{unique}",
        status=DocumentStatus.INDEXED,
    )
    db.add(doc)
    db.flush()

    c = DocumentChunk(
        tenant_id=tenant.id,
        document_id=doc.id,
        ordinal=0,
        content="Some simple text for retrieval testing.",
        embedding=[0.1] * settings.EMBEDDING_DIMENSIONS,
    )
    db.add(c)
    db.commit()

    hits_reranked = retrieve(
        db,
        tenant_id=tenant.id,
        query="retrieval testing",
        top_k=1,
        enable_reranker=True,
    )

    hits_no_rerank = retrieve(
        db,
        tenant_id=tenant.id,
        query="retrieval testing",
        top_k=1,
        enable_reranker=False,
    )

    assert len(hits_reranked) == 1
    assert len(hits_no_rerank) == 1
    assert hits_reranked[0].chunk_id == c.id
    assert hits_no_rerank[0].chunk_id == c.id
