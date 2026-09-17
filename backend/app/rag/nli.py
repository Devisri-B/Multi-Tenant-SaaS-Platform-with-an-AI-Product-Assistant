"""Natural Language Inference (NLI) entailment scoring for hallucination verification.

Uses local DeBERTa-v3 cross-encoder models to verify whether generated candidate
answers are factually entailed by retrieved context passages.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import structlog

from app.core.config import settings
from app.core.exceptions import ProviderError

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class NLIResult:
    is_grounded: bool
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    sentence_scores: list[dict[str, Any]] = field(default_factory=list)


class NLIProvider(ABC):
    """Abstract interface for factual consistency and entailment verification."""

    @abstractmethod
    def check_groundedness(
        self,
        context: str,
        answer: str,
        *,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> NLIResult:
        """Evaluate whether candidate answer is entailed by context without hallucination."""
        ...


class DebertaNLIProvider(NLIProvider):
    """Local DeBERTa-v3 cross-encoder running zero-API-cost entailment classification.

    Tokenizes (premise=context, hypothesis=sentence) pairs and calculates calibrated
    probabilities across [entailment, neutral, contradiction].
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.DEBERTA_MODEL_NAME
        self.device_str = device or settings.NLI_DEVICE
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._label_map: dict[str, int] = {}

    def _ensure_loaded(self) -> None:
        """Lazily load tokenizer and model to avoid import-time overhead."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ProviderError(
                "DeBERTa NLI requires 'torch' and 'transformers'. "
                "Install them via 'pip install torch transformers'."
            ) from exc

        try:
            logger.info(
                "rag.nli.loading_model",
                model=self.model_name,
                device=self.device_str,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Determine device
            if self.device_str == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(self.device_str)

            self._model.to(self._device)
            self._model.eval()

            # Map model label IDs to semantic classes
            id2label = getattr(self._model.config, "id2label", {}) or {}
            self._label_map = {}
            for idx, name in id2label.items():
                clean_name = str(name).lower()
                if "entail" in clean_name:
                    self._label_map["entailment"] = int(idx)
                elif "contra" in clean_name:
                    self._label_map["contradiction"] = int(idx)
                elif "neut" in clean_name:
                    self._label_map["neutral"] = int(idx)

            # Fallback label ordering for standard MNLI if id2label is missing
            if "entailment" not in self._label_map:
                self._label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

        except Exception as exc:
            msg = f"Failed to load DeBERTa NLI model '{self.model_name}': {exc}"
            raise ProviderError(msg) from exc

    def check_groundedness(
        self,
        context: str,
        answer: str,
        *,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> NLIResult:
        self._ensure_loaded()
        import torch

        e_thresh = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.NLI_ENTAILMENT_THRESHOLD
        )
        c_thresh = (
            contradiction_threshold
            if contradiction_threshold is not None
            else settings.NLI_CONTRADICTION_THRESHOLD
        )

        clean_ans = answer.strip()
        if not clean_ans:
            return NLIResult(
                is_grounded=False,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
            )

        # Decompose answer into individual sentences / claims
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) > 5
        ]
        if not sentences:
            sentences = [clean_ans]

        pairs = [(context, s) for s in sentences]
        sentence_scores: list[dict[str, Any]] = []

        try:
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)

                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu()

                ent_idx = self._label_map.get("entailment", 1)
                con_idx = self._label_map.get("contradiction", 0)
                neu_idx = self._label_map.get("neutral", 2)

                ent_scores = probs[:, ent_idx].tolist()
                con_scores = probs[:, con_idx].tolist()
                neu_scores = probs[:, neu_idx].tolist()

                for i, s in enumerate(sentences):
                    sentence_scores.append(
                        {
                            "sentence": s,
                            "entailment": float(ent_scores[i]),
                            "contradiction": float(con_scores[i]),
                            "neutral": float(neu_scores[i]),
                        }
                    )

        except Exception as exc:
            raise ProviderError(f"DeBERTa NLI inference failed: {exc}") from exc

        avg_entailment = sum(s["entailment"] for s in sentence_scores) / len(sentence_scores)
        max_contradiction = max(s["contradiction"] for s in sentence_scores)
        avg_neutral = sum(s["neutral"] for s in sentence_scores) / len(sentence_scores)

        # Grounded if average entailment meets threshold and no claim sharply contradicts context
        is_grounded = avg_entailment >= e_thresh and max_contradiction < c_thresh

        return NLIResult(
            is_grounded=is_grounded,
            entailment_score=round(avg_entailment, 4),
            contradiction_score=round(max_contradiction, 4),
            neutral_score=round(avg_neutral, 4),
            sentence_scores=sentence_scores,
        )


class FakeNLIProvider(NLIProvider):
    """Deterministic token-overlap NLI stand-in for tests and CI."""

    def check_groundedness(
        self,
        context: str,
        answer: str,
        *,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> NLIResult:
        e_thresh = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.NLI_ENTAILMENT_THRESHOLD
        )

        clean_ans = answer.strip()
        if not clean_ans:
            return NLIResult(
                is_grounded=False,
                entailment_score=0.0,
                contradiction_score=1.0,
                neutral_score=0.0,
            )

        f_words = set(_tokenize(context))
        a_words = {token for token in _tokenize(clean_ans) if len(token) > 3}

        if not a_words:
            return NLIResult(
                is_grounded=True,
                entailment_score=0.95,
                contradiction_score=0.02,
                neutral_score=0.03,
            )

        overlap = len(a_words & f_words) / len(a_words)
        is_grounded = overlap >= 0.30 and overlap >= e_thresh

        entailment = round(max(0.05, min(0.99, overlap)), 4)
        contradiction = round(0.05 if is_grounded else 0.85, 4)
        neutral = round(max(0.0, 1.0 - entailment - contradiction), 4)

        return NLIResult(
            is_grounded=is_grounded,
            entailment_score=entailment,
            contradiction_score=contradiction,
            neutral_score=neutral,
            sentence_scores=[
                {
                    "sentence": clean_ans,
                    "entailment": entailment,
                    "contradiction": contradiction,
                    "neutral": neutral,
                }
            ],
        )


class LLMNLIProvider(NLIProvider):
    """Legacy prompt-based LLM factual consistency evaluator using ChatProvider."""

    def check_groundedness(
        self,
        context: str,
        answer: str,
        *,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> NLIResult:
        from app.rag import prompts
        from app.rag.providers import get_chat_provider

        chat = get_chat_provider()
        prompt = prompts.HALLUCINATION_GRADER_PROMPT.format(
            context=context, generation=answer
        )
        res = chat.complete(
            "You are an evaluator assessing factual consistency.", prompt
        ).strip().lower()

        is_grounded = "yes" in res and "no" not in res
        return NLIResult(
            is_grounded=is_grounded,
            entailment_score=0.90 if is_grounded else 0.10,
            contradiction_score=0.05 if is_grounded else 0.85,
            neutral_score=0.05,
        )


@lru_cache
def get_nli_provider() -> NLIProvider:
    """Factory returning the configured NLI hallucination verification provider."""
    # Deterministic offline provider for test/fake environments
    if settings.HALLUCINATION_PROVIDER == "fake" or settings.LLM_PROVIDER == "fake":
        return FakeNLIProvider()

    if settings.HALLUCINATION_PROVIDER == "llm":
        return LLMNLIProvider()

    return DebertaNLIProvider()


def reset_nli_cache() -> None:
    """Clear cached NLI provider instances."""
    get_nli_provider.cache_clear()
