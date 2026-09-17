"""Tests for SyncAnthropic (Claude) ChatProvider integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.rag.providers import AnthropicChat, get_chat_provider, reset_provider_cache


def test_anthropic_chat_requires_api_key():
    """AnthropicChat raises ProviderError if ANTHROPIC_API_KEY is missing."""
    with patch.object(settings, "ANTHROPIC_API_KEY", None):
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not configured"):
            AnthropicChat()


def test_anthropic_chat_completion():
    """AnthropicChat invokes SyncAnthropic messages.create and extracts text blocks."""
    mock_client = MagicMock()
    mock_block_1 = MagicMock()
    mock_block_1.text = "Nimbus provides single-schema multi-tenancy. "
    mock_block_2 = MagicMock()
    mock_block_2.text = "Every query is scoped by tenant_id."
    mock_response = MagicMock()
    mock_response.content = [mock_block_1, mock_block_2]

    mock_client.messages.create.return_value = mock_response

    with patch.object(settings, "ANTHROPIC_API_KEY", "test-anthropic-key"):
        with patch("anthropic.Anthropic", return_value=mock_client):
            chat = AnthropicChat()
            res = chat.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Explain tenant isolation in Nimbus.",
            )

    assert "Nimbus provides single-schema multi-tenancy." in res
    assert "Every query is scoped by tenant_id." in res

    # Verify messages.create call args
    mock_client.messages.create.assert_called_once_with(
        model=settings.ANTHROPIC_CHAT_MODEL,
        max_tokens=1024,
        temperature=0.1,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Explain tenant isolation in Nimbus."}],
    )


def test_anthropic_chat_handles_api_error():
    """AnthropicChat wraps API errors in ProviderError."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("Rate limit exceeded")

    with patch.object(settings, "ANTHROPIC_API_KEY", "test-anthropic-key"):
        with patch("anthropic.Anthropic", return_value=mock_client):
            chat = AnthropicChat()
            with pytest.raises(ProviderError, match="Anthropic chat completion failed"):
                chat.complete("sys", "user")


def test_get_chat_provider_resolves_anthropic():
    """get_chat_provider resolves AnthropicChat when LLM_PROVIDER=anthropic."""
    reset_provider_cache()
    with patch.object(settings, "LLM_PROVIDER", "anthropic"):
        with patch.object(settings, "ANTHROPIC_API_KEY", "test-key"):
            with patch("anthropic.Anthropic", return_value=MagicMock()):
                provider = get_chat_provider()
                assert isinstance(provider, AnthropicChat)
    reset_provider_cache()
