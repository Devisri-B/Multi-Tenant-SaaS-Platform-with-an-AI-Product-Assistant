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


# ---------------------------------------------------------------------------
# Provider: AnthropicChat
# ---------------------------------------------------------------------------
def test_anthropic_chat_requires_api_key():
    from unittest.mock import patch

    from app.core.config import settings
    from app.core.exceptions import ProviderError
    from app.rag.providers import AnthropicChat

    with (
        patch.object(settings, "ANTHROPIC_API_KEY", None),
        pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not configured"),
    ):
        AnthropicChat()


def test_anthropic_chat_completion():
    from unittest.mock import MagicMock, patch

    from app.core.config import settings
    from app.rag.providers import AnthropicChat

    mock_client = MagicMock()
    mock_block_1 = MagicMock()
    mock_block_1.text = "Nimbus provides single-schema multi-tenancy. "
    mock_block_2 = MagicMock()
    mock_block_2.text = "Every query is scoped by tenant_id."
    mock_response = MagicMock()
    mock_response.content = [mock_block_1, mock_block_2]

    mock_client.messages.create.return_value = mock_response

    with (
        patch.object(settings, "ANTHROPIC_API_KEY", "test-anthropic-key"),
        patch("anthropic.Anthropic", return_value=mock_client),
    ):
        chat = AnthropicChat()
        res = chat.complete(
            system_prompt="You are a helpful assistant.",
            user_prompt="Explain tenant isolation in Nimbus.",
        )

    assert "Nimbus provides single-schema multi-tenancy." in res
    assert "Every query is scoped by tenant_id." in res

    mock_client.messages.create.assert_called_once_with(
        model=settings.ANTHROPIC_CHAT_MODEL,
        max_tokens=1024,
        temperature=0.1,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Explain tenant isolation in Nimbus."}],
    )


def test_anthropic_chat_handles_api_error():
    from unittest.mock import MagicMock, patch

    from app.core.config import settings
    from app.core.exceptions import ProviderError
    from app.rag.providers import AnthropicChat

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("Rate limit exceeded")

    with (
        patch.object(settings, "ANTHROPIC_API_KEY", "test-anthropic-key"),
        patch("anthropic.Anthropic", return_value=mock_client),
        pytest.raises(ProviderError, match="Anthropic chat completion failed"),
    ):
        chat = AnthropicChat()
        chat.complete("sys", "user")


def test_get_chat_provider_resolves_anthropic():
    from unittest.mock import MagicMock, patch

    from app.core.config import settings
    from app.rag.providers import AnthropicChat, get_chat_provider, reset_provider_cache

    reset_provider_cache()
    with (
        patch.object(settings, "LLM_PROVIDER", "anthropic"),
        patch.object(settings, "ANTHROPIC_API_KEY", "test-key"),
        patch("anthropic.Anthropic", return_value=MagicMock()),
    ):
        provider = get_chat_provider()
        assert isinstance(provider, AnthropicChat)
    reset_provider_cache()


# ---------------------------------------------------------------------------
# Provider: SentenceTransformerEmbeddings
# ---------------------------------------------------------------------------
def test_sentence_transformer_empty_texts():
    from app.rag.providers import SentenceTransformerEmbeddings

    provider = SentenceTransformerEmbeddings(model_name="mock-model", device="cpu")
    assert provider.embed_documents([]) == []


def test_sentence_transformer_embed_documents_mocked():
    from unittest.mock import MagicMock

    import numpy as np

    from app.rag.providers import SentenceTransformerEmbeddings

    provider = SentenceTransformerEmbeddings(model_name="mock-model", device="cpu")

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.array([
        [0.1] * 384,
        [0.2] * 384,
    ])

    provider._model = mock_model
    provider.dimensions = 384

    texts = ["First document text", "Second document text"]
    embeddings = provider.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    mock_model.encode.assert_called_once_with(texts, normalize_embeddings=True)


def test_sentence_transformer_embed_query_mocked():
    from unittest.mock import MagicMock

    import numpy as np

    from app.rag.providers import SentenceTransformerEmbeddings

    provider = SentenceTransformerEmbeddings(model_name="mock-model", device="cpu")

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.array([0.5] * 384)

    provider._model = mock_model
    provider.dimensions = 384

    query_embedding = provider.embed_query("billing policy")
    assert len(query_embedding) == 384
    mock_model.encode.assert_called_once_with("billing policy", normalize_embeddings=True)


def test_sentence_transformer_load_failure():
    from unittest.mock import patch

    from app.core.exceptions import ProviderError
    from app.rag.providers import SentenceTransformerEmbeddings

    provider = SentenceTransformerEmbeddings(model_name="non-existent-model", device="cpu")

    err = RuntimeError("Download failed")
    with (
        patch("sentence_transformers.SentenceTransformer", side_effect=err),
        pytest.raises(ProviderError, match="Failed to load SentenceTransformer model"),
    ):
        provider.embed_query("test query")


def test_get_embedding_provider_factory():
    from unittest.mock import MagicMock, patch

    from app.core.config import settings
    from app.rag.providers import (
        FakeEmbeddings,
        SentenceTransformerEmbeddings,
        get_embedding_provider,
        reset_provider_cache,
    )

    reset_provider_cache()
    provider = get_embedding_provider()
    assert isinstance(provider, FakeEmbeddings)

    with (
        patch.object(settings, "LLM_PROVIDER", "anthropic"),
        patch.object(settings, "EMBEDDING_PROVIDER", "sentence_transformers"),
    ):
        reset_provider_cache()
        with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
            prod_provider = get_embedding_provider()
            assert isinstance(prod_provider, SentenceTransformerEmbeddings)

    reset_provider_cache()


