"""Tests for SentenceTransformerEmbeddings provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.rag.providers import (
    FakeEmbeddings,
    SentenceTransformerEmbeddings,
    get_embedding_provider,
    reset_provider_cache,
)


def test_sentence_transformer_empty_texts():
    """SentenceTransformerEmbeddings returns empty list for empty input."""
    provider = SentenceTransformerEmbeddings(model_name="mock-model", device="cpu")
    assert provider.embed_documents([]) == []


def test_sentence_transformer_embed_documents_mocked():
    """SentenceTransformerEmbeddings encodes documents and normalizes output."""
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
    """SentenceTransformerEmbeddings encodes single query."""
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
    """Raises ProviderError when model cannot be loaded."""
    provider = SentenceTransformerEmbeddings(model_name="non-existent-model", device="cpu")

    with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Download failed")):
        with pytest.raises(ProviderError, match="Failed to load SentenceTransformer model"):
            provider.embed_query("test query")


def test_get_embedding_provider_factory():
    """get_embedding_provider returns FakeEmbeddings in test env and SentenceTransformerEmbeddings when configured."""
    reset_provider_cache()
    # In test environment with LLM_PROVIDER=fake, returns FakeEmbeddings
    provider = get_embedding_provider()
    assert isinstance(provider, FakeEmbeddings)

    # In production with EMBEDDING_PROVIDER=sentence_transformers and non-fake LLM
    with patch.object(settings, "LLM_PROVIDER", "anthropic"):
        with patch.object(settings, "EMBEDDING_PROVIDER", "sentence_transformers"):
            reset_provider_cache()
            with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
                prod_provider = get_embedding_provider()
                assert isinstance(prod_provider, SentenceTransformerEmbeddings)

    reset_provider_cache()
