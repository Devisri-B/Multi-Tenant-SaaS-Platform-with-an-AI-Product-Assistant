"""Tests for Local NLI entailment scoring and hallucination verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.rag.graph import grade_hallucination_node
from app.rag.nli import (
    DebertaNLIProvider,
    FakeNLIProvider,
    LLMNLIProvider,
    get_nli_provider,
    reset_nli_cache,
)
from app.rag.retriever import RetrievedChunk


def test_fake_nli_provider_grounded():
    """Grounded text with matching facts scores high entailment."""
    provider = FakeNLIProvider()
    context = "Nimbus supports billing via Stripe. Payments can be made with credit cards or ACH."
    answer = "Nimbus supports billing with Stripe and accepts credit cards."

    result = provider.check_groundedness(context, answer)
    assert result.is_grounded is True
    assert result.entailment_score >= 0.5
    assert result.contradiction_score < 0.3


def test_fake_nli_provider_hallucination():
    """Unsupported claims without factual basis score high contradiction."""
    provider = FakeNLIProvider()
    context = "Nimbus supports billing via Stripe. Payments can be made with credit cards or ACH."
    answer = "Nimbus has built-in cryptocurrency blockchain mining and quantum quantum telemetry."

    result = provider.check_groundedness(context, answer)
    assert result.is_grounded is False
    assert result.contradiction_score >= 0.5


def test_fake_nli_provider_empty_answer():
    """Empty answer is not grounded."""
    provider = FakeNLIProvider()
    result = provider.check_groundedness("Some context", "")
    assert result.is_grounded is False
    assert result.contradiction_score == 1.0


def test_llm_nli_provider_fallback():
    """LLMNLIProvider delegates to chat provider."""
    provider = LLMNLIProvider()
    context = "Refunds are processed within 5 business days."
    answer = "Refunds take 5 business days."
    result = provider.check_groundedness(context, answer)
    assert result.is_grounded is True


def test_get_nli_provider_factory():
    """get_nli_provider respects settings and caches instance."""
    reset_nli_cache()
    provider = get_nli_provider()
    assert isinstance(provider, FakeNLIProvider)  # LLM_PROVIDER=fake in test env
    provider2 = get_nli_provider()
    assert provider is provider2


def test_deberta_nli_provider_mocked():
    """DebertaNLIProvider tokenizes and computes softmax probabilities across logits."""
    provider = DebertaNLIProvider(model_name="mock-model", device="cpu")

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value.to.return_value = {"input_ids": [1, 2, 3]}

    import torch

    # Simulated logits: batch of 2 sentences, 3 classes [contradiction, entailment, neutral]
    # sentence 1: high entailment [0.1, 4.5, 0.2] -> softmax entailment ~ 0.98
    # sentence 2: high entailment [0.0, 3.8, 0.5] -> softmax entailment ~ 0.94
    mock_logits = torch.tensor([[0.1, 4.5, 0.2], [0.0, 3.8, 0.5]])
    mock_model = MagicMock()
    mock_model.return_value.logits = mock_logits
    mock_model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    provider._tokenizer = mock_tokenizer
    provider._model = mock_model
    provider._device = torch.device("cpu")
    provider._label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

    context = "Nimbus provides single-schema multi-tenancy with PostgreSQL."
    answer = "Nimbus isolates tenants in a single schema. PostgreSQL is used for storage."

    result = provider.check_groundedness(context, answer)

    assert result.is_grounded is True
    assert result.entailment_score > 0.8
    assert result.contradiction_score < 0.2
    assert len(result.sentence_scores) == 2


def test_grade_hallucination_node_integration():
    """grade_hallucination_node uses NLI provider and attaches scores to state."""
    import uuid

    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Pricing & Billing",
        ordinal=0,
        content="Enterprise plan costs $99/month and includes unlimited seats.",
        score=0.9,
    )

    state = {
        "tenant_id": uuid.uuid4(),
        "question": "How much does the Enterprise plan cost?",
        "answer": "The Enterprise plan costs $99/month with unlimited seats.",
        "documents": [chunk],
        "is_grounded": False,
        "answers_question": False,
        "retry_count": 0,
        "allow_web_search": True,
    }

    result = grade_hallucination_node(state)
    assert result["is_grounded"] is True
    assert result["answers_question"] is True
    assert "entailment_score" in result
    assert result["entailment_score"] > 0.5


def test_grade_hallucination_node_disabled():
    """When ENABLE_HALLUCINATION_CHECK is False, returns grounded immediately."""
    with patch.object(settings, "ENABLE_HALLUCINATION_CHECK", False):
        state = {
            "answer": "Fabricated text",
            "documents": [],
            "question": "Any question",
        }
        result = grade_hallucination_node(state)
        assert result["is_grounded"] is True
        assert result["entailment_score"] == 1.0
