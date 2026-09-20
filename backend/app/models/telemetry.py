"""Telemetry events for tracking real-world user behavior and RAG answer quality."""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.tenancy import attach_rls_policies
from app.db.types import GUID, JSONType
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class TelemetryEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Production telemetry: copy events, citation clicks, friction, and evaluations."""

    __tablename__ = "telemetry_events"
    __table_args__ = (
        Index("ix_telemetry_tenant_event", "tenant_id", "event_type"),
        Index("ix_telemetry_conversation", "conversation_id"),
        Index("ix_telemetry_created_at", "created_at"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_data: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)


attach_rls_policies(TelemetryEvent.__table__)
