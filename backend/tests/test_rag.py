"""Chunking, embedding and retrieval unit tests."""

from __future__ import annotations

import pytest

from app.rag.chunking import chunk_text, count_tokens, estimate_tokens
from app.rag.providers import FakeChat, FakeEmbeddings
from app.rag.retriever import cosine_similarity

SAMPLE = """\
# Billing

## Invoices
Invoices are generated on the first of every month and emailed to the billing
contact. You can download past invoices from Settings.

## Refunds
Refunds are issued to the original payment method within ten business days.
"""


def test_chunking_returns_ordered_chunks():
    chunks = chunk_text(SAMPLE, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_chunking_preserves_all_words():
    chunks = chunk_text(SAMPLE, chunk_size=20, chunk_overlap=5)
    joined = " ".join(chunk.content for chunk in chunks)
    assert "Refunds are issued" in joined
    assert "Invoices are generated" in joined


def test_chunking_attaches_heading_metadata():
    chunks = chunk_text(SAMPLE, chunk_size=20, chunk_overlap=5)
    assert any("heading" in chunk.metadata for chunk in chunks)


def test_empty_text_produces_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text(SAMPLE, chunk_size=20, chunk_overlap=20)


def test_token_estimate_is_positive():
    assert estimate_tokens("hello world") >= 1


def test_count_tokens_local_and_estimate():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 2
    assert estimate_tokens("hello world") == count_tokens("hello world")


def test_count_tokens_anthropic_client_integration():
    from unittest.mock import MagicMock

    from app.core.config import settings
    from app.rag.chunking import set_anthropic_client

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.input_tokens = 42
    mock_client.messages.count_tokens.return_value = mock_resp

    old_provider = settings.LLM_PROVIDER
    try:
        settings.LLM_PROVIDER = "anthropic"
        set_anthropic_client(mock_client)
        tokens = count_tokens("test custom token count")
        assert tokens == 42
        mock_client.messages.count_tokens.assert_called_once()
    finally:
        settings.LLM_PROVIDER = old_provider
        set_anthropic_client(None)


def test_count_tokens_anthropic_fallback_on_error():
    from unittest.mock import MagicMock

    from app.core.config import settings
    from app.rag.chunking import set_anthropic_client

    mock_client = MagicMock()
    mock_client.messages.count_tokens.side_effect = RuntimeError("Anthropic rate limit")

    old_provider = settings.LLM_PROVIDER
    try:
        settings.LLM_PROVIDER = "anthropic"
        set_anthropic_client(mock_client)
        tokens = count_tokens("fallback text token count")
        assert tokens > 0
    finally:
        settings.LLM_PROVIDER = old_provider
        set_anthropic_client(None)


def test_chunking_chunks_respect_token_limits():
    chunks = chunk_text(SAMPLE, chunk_size=20, chunk_overlap=5)
    for chunk in chunks:
        assert chunk.token_estimate <= 20


def test_fake_embeddings_are_deterministic():
    embedder = FakeEmbeddings(dimensions=64)
    assert embedder.embed_query("reset my password") == embedder.embed_query(
        "reset my password"
    )


def test_fake_embeddings_are_unit_length():
    vector = FakeEmbeddings(dimensions=64).embed_query("anything at all")
    assert abs(sum(component**2 for component in vector) - 1.0) < 1e-6


def test_similar_text_scores_higher_than_unrelated():
    embedder = FakeEmbeddings(dimensions=256)
    query = embedder.embed_query("how do refunds work")
    related = embedder.embed_query("refunds are issued to the original payment method")
    unrelated = embedder.embed_query("kubernetes ingress controller configuration")
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_cosine_handles_degenerate_input():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_fake_chat_refuses_without_context():
    answer = FakeChat().complete("sys", "<context></context><question>hi</question>")
    assert "could not find" in answer.lower()


def test_fake_chat_extracts_from_context():
    prompt = (
        "<context>Refunds are issued within ten business days. "
        "Invoices go out monthly.</context><question>refunds</question>"
    )
    answer = FakeChat().complete("sys", prompt)
    assert "refunds" in answer.lower()
