"""Sliding window memory management and conversation history conditioning."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.config import settings


def trim_sliding_window(
    history: Sequence[tuple[str, str]] | None,
    window_size: int | None = None,
) -> list[tuple[str, str]]:
    """Return the most recent K dialogue messages within the sliding window."""
    if not history:
        return []
    limit = window_size or settings.RAG_MEMORY_WINDOW_SIZE
    return list(history[-limit:])


def format_sliding_window_history(
    history: Sequence[tuple[str, str]] | None,
    window_size: int | None = None,
) -> str:
    """Format dialogue history into a structured transcript for prompt conditioning."""
    recent_history = trim_sliding_window(history, window_size)
    if not recent_history:
        return ""

    formatted_lines: list[str] = []
    for role, content in recent_history:
        role_label = "User" if role.lower() in ("user", "human") else "Assistant"
        cleaned_content = " ".join(content.strip().split())
        formatted_lines.append(f"{role_label}: {cleaned_content}")

    return "\n".join(formatted_lines)


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


def extract_contextual_keywords(history: Sequence[tuple[str, str]]) -> list[str]:
    """Extract key nouns and topics from recent conversation history."""
    combined_text = " ".join(content for _, content in history[-4:])
    words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", combined_text)
    # Exclude common stopwords
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "have",
        "what", "when", "where", "which", "will", "would", "about",
        "your", "user", "assistant", "they", "them", "some", "more",
    }
    return [w for w in words if w.lower() not in stopwords]
