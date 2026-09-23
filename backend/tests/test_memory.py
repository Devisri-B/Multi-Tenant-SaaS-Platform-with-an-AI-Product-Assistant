"""Tests for sliding window memory, conversation history conditioning, and summarization."""

from __future__ import annotations

from app.models.enums import MessageRole
from app.rag import memory
from app.services import conversation as conversation_service


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


def test_split_history_and_pruned():
    assert memory.split_history_and_pruned(None) == ([], [])
    assert memory.split_history_and_pruned([]) == ([], [])

    history = [
        ("user", "Q1"),
        ("assistant", "A1"),
        ("user", "Q2"),
        ("assistant", "A2"),
    ]
    # When history <= window_size, older should be empty
    older, window = memory.split_history_and_pruned(history, window_size=4)
    assert older == []
    assert len(window) == 4

    # When history > window_size, older contains preceding messages
    older, window = memory.split_history_and_pruned(history, window_size=2)
    assert older == [("user", "Q1"), ("assistant", "A1")]
    assert window == [("user", "Q2"), ("assistant", "A2")]


def test_summarize_messages_deterministic():
    pruned = [
        ("user", "How do I configure database replication?"),
        ("assistant", "Navigate to Settings -> Database -> Replication."),
        ("user", "What is the failover timeout?"),
        ("assistant", "The default failover timeout is 30 seconds."),
    ]
    summary = memory.summarize_messages(pruned)
    assert "replication" in summary.lower()
    assert "failover timeout" in summary.lower()

    # Appending to existing summary
    merged = memory.summarize_messages(
        [("user", "Can we set custom alerts?")],
        existing_summary=summary,
    )
    assert summary in merged
    assert "custom alerts" in merged.lower()


def test_format_sliding_window_history():
    history = [
        ("user", "What is Nimbus?"),
        ("assistant", "Nimbus is a multi-tenant platform."),
    ]
    formatted = memory.format_sliding_window_history(history)
    assert "User: What is Nimbus?" in formatted
    assert "Assistant: Nimbus is a multi-tenant platform." in formatted


def test_format_sliding_window_history_with_summary():
    history = [
        ("user", "Can I upgrade my tier?"),
        ("assistant", "Yes, in Settings -> Billing."),
    ]
    summary = "User earlier inquired about database backup retention schedules."
    formatted = memory.format_sliding_window_history(history, summary=summary)
    assert "<earlier_conversation_summary>" in formatted
    assert summary in formatted
    assert "</earlier_conversation_summary>" in formatted
    assert "User: Can I upgrade my tier?" in formatted


def test_format_sliding_window_history_auto_summarizes():
    history = [
        ("user", "Question 1 about authentication"),
        ("assistant", "Answer 1"),
        ("user", "Question 2 about authorization"),
        ("assistant", "Answer 2"),
        ("user", "Question 3 about audit logs"),
        ("assistant", "Answer 3"),
    ]
    # With window_size=2, questions 1 and 2 are outside sliding window and should be summarized
    formatted = memory.format_sliding_window_history(history, window_size=2)
    assert "<earlier_conversation_summary>" in formatted
    assert "authentication" in formatted.lower()
    assert "User: Question 3 about audit logs" in formatted


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


def test_sync_conversation_summary_database(db, owner):
    conversation = conversation_service.get_or_create_conversation(
        db,
        tenant_id=owner.tenant_id,
        user_id=owner.user.id,
        conversation_id=None,
        first_question="How do I invite team members?",
    )

    # Add 6 turns (12 messages)
    for i in range(1, 7):
        conversation_service.append_message(
            db,
            conversation=conversation,
            role=MessageRole.USER,
            content=f"Question {i} about feature {i}",
        )
        conversation_service.append_message(
            db,
            conversation=conversation,
            role=MessageRole.ASSISTANT,
            content=f"Answer {i} for feature {i}",
        )

    # Window size 4 means messages 1-8 are outside window and must be summarized
    summary = conversation_service.sync_conversation_summary(db, conversation, window_size=4)
    assert summary is not None
    assert "feature" in summary.lower()
    assert conversation.summary == summary
