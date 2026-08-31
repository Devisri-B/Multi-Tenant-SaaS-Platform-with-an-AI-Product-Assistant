"""Tests for sliding window memory and conversation history conditioning."""

from __future__ import annotations

from app.rag import memory


def test_trim_sliding_window_empty():
    assert memory.trim_sliding_window(None) == []
    assert memory.trim_sliding_window([]) == []


def test_trim_sliding_window_limits():
    history = [
        ("user", "Hello 1"),
        ("assistant", "Hi 1"),
        ("user", "Question 2"),
        ("assistant", "Answer 2"),
        ("user", "Question 3"),
        ("assistant", "Answer 3"),
        ("user", "Question 4"),
        ("assistant", "Answer 4"),
    ]
    trimmed = memory.trim_sliding_window(history, window_size=4)
    assert len(trimmed) == 4
    assert trimmed[0] == ("user", "Question 3")
    assert trimmed[-1] == ("assistant", "Answer 4")


def test_format_sliding_window_history():
    history = [
        ("user", "What is Nimbus?"),
        ("assistant", "Nimbus is a multi-tenant platform."),
    ]
    formatted = memory.format_sliding_window_history(history)
    assert "User: What is Nimbus?" in formatted
    assert "Assistant: Nimbus is a multi-tenant platform." in formatted


def test_estimate_tokens():
    tokens = memory.estimate_tokens("Hello world, this is a test string.")
    assert tokens >= 1


def test_extract_contextual_keywords():
    history = [
        ("user", "What is the policy regarding database backups and replication?"),
        ("assistant", "Backups are performed daily."),
    ]
    keywords = memory.extract_contextual_keywords(history)
    assert any("backup" in k.lower() for k in keywords)
