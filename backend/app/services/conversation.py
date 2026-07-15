"""Assistant conversations."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models.conversation import Conversation, Message
from app.models.enums import MessageRole
from app.services.base import TenantScopedRepository

TITLE_MAX = 60


class ConversationRepository(TenantScopedRepository[Conversation]):
    model = Conversation

    def for_user(
        self, user_id: uuid.UUID, *, offset: int = 0, limit: int = 20
    ) -> tuple[list[Conversation], int]:
        stmt = (
            self._scoped()
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        items = list(self.db.execute(stmt).scalars().unique().all())
        total = int(
            self.db.execute(
                select(func.count())
                .select_from(Conversation)
                .where(
                    Conversation.tenant_id == self.tenant_id,
                    Conversation.user_id == user_id,
                )
            ).scalar_one()
        )
        return items, total


def derive_title(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= TITLE_MAX:
        return cleaned or "New conversation"
    return cleaned[: TITLE_MAX - 1].rstrip() + "…"


def get_or_create_conversation(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    first_question: str,
) -> Conversation:
    repo = ConversationRepository(db, tenant_id)
    if conversation_id is not None:
        conversation = repo.get(conversation_id)
        if conversation is None:
            raise NotFoundError("Conversation not found in this workspace.")
        return conversation

    conversation = Conversation(
        tenant_id=tenant_id, user_id=user_id, title=derive_title(first_question)
    )
    repo.add(conversation)
    return conversation


def append_message(
    db: Session,
    *,
    conversation: Conversation,
    role: MessageRole,
    content: str,
    citations: list | None = None,
    latency_ms: int | None = None,
) -> Message:
    message = Message(
        tenant_id=conversation.tenant_id,
        conversation_id=conversation.id,
        role=role,
        content=content,
        citations=citations or [],
        latency_ms=latency_ms,
    )
    db.add(message)
    db.flush()
    return message


def history_pairs(conversation: Conversation, limit: int = 6) -> list[tuple[str, str]]:
    """Recent turns as ``(role, content)`` for prompt conditioning."""
    return [
        (str(message.role), message.content) for message in conversation.messages[-limit:]
    ]
