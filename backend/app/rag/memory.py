"""Sliding window memory management and conversation history conditioning."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.config import settings
from app.core.logging import get_logger
from app.rag import prompts

log = get_logger(__name__)


def trim_sliding_window(
    history: Sequence[tuple[str, str]] | None,
    window_size: int | None = None,
) -> list[tuple[str, str]]:
    """Return the most recent K dialogue messages within the sliding window."""
    if not history:
        return []
    limit = window_size or settings.RAG_MEMORY_WINDOW_SIZE
    return list(history[-limit:])


def split_history_and_pruned(
    history: Sequence[tuple[str, str]] | None,
    window_size: int | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Partition conversation history into pruned (older) and active sliding window messages.

    Returns:
        tuple of (older_messages, active_window_messages)
    """
    if not history:
        return [], []
    limit = window_size or settings.RAG_MEMORY_WINDOW_SIZE
    if len(history) <= limit:
        return [], list(history)
    return list(history[:-limit]), list(history[-limit:])


def format_messages_transcript(messages: Sequence[tuple[str, str]]) -> str:
    """Format dialogue messages into clean 'Role: Content' lines."""
    formatted_lines: list[str] = []
    for role, content in messages:
        role_label = "User" if role.lower() in ("user", "human") else "Assistant"
        cleaned_content = " ".join(content.strip().split())
        formatted_lines.append(f"{role_label}: {cleaned_content}")
    return "\n".join(formatted_lines)


def _deterministic_summary(
    messages: Sequence[tuple[str, str]],
    existing_summary: str | None = None,
) -> str:
    """Fast deterministic summary extractor for offline environments, mock mode, and CI."""
    user_topics: list[str] = []
    for role, content in messages:
        if role.lower() in ("user", "human"):
            clean_q = " ".join(content.strip().split())
            if clean_q:
                user_topics.append(clean_q.rstrip("?."))
    topics_str = "; ".join(user_topics[:4]) if user_topics else "various topics"
    new_summary = f"Earlier discussion covered questions regarding: {topics_str}."
    if existing_summary and existing_summary.strip():
        return f"{existing_summary.strip()} {new_summary}"
    return new_summary


def summarize_messages(
    messages: Sequence[tuple[str, str]],
    existing_summary: str | None = None,
) -> str:
    """Summarize older messages outside the active sliding window."""
    if not messages:
        return existing_summary or ""

    transcript = format_messages_transcript(messages)
    if not transcript:
        return existing_summary or ""

    existing_clause = (
        f"Existing context from previous summary:\n{existing_summary.strip()}\n"
        if existing_summary and existing_summary.strip()
        else ""
    )

    if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        try:
            from app.rag.providers import get_chat_provider

            chat = get_chat_provider()
            prompt = prompts.SUMMARIZE_CONVERSATION_PROMPT.format(
                existing_summary_clause=existing_clause,
                dialogue=transcript,
            )
            result = chat.complete("You are a concise conversation summarizer.", prompt).strip()
            if result:
                return result
        except Exception as exc:
            log.warning("memory.summarize_messages.error", error=str(exc))

    return _deterministic_summary(messages, existing_summary)


def format_sliding_window_history(
    history: Sequence[tuple[str, str]] | None,
    window_size: int | None = None,
    summary: str | None = None,
) -> str:
    """Format dialogue history into a structured transcript for prompt conditioning.

    If older messages exist beyond the window and no summary was provided, older turns
    are summarized automatically. If summary is provided, it is prepended.
    """
    if not history and not summary:
        return ""

    older_msgs, recent_history = split_history_and_pruned(history, window_size)

    effective_summary = summary
    if not effective_summary and older_msgs:
        effective_summary = summarize_messages(older_msgs)

    parts: list[str] = []
    if effective_summary and effective_summary.strip():
        parts.append(
            f"<earlier_conversation_summary>\n{effective_summary.strip()}\n"
            "</earlier_conversation_summary>"
        )

    if recent_history:
        parts.append(format_messages_transcript(recent_history))

    return "\n\n".join(parts)


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
