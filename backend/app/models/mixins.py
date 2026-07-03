"""Reusable column mixins."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from app.db.types import GUID


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4, index=True
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        self.deleted_at = utcnow()


class TenantScopedMixin:
    """Every row of a tenant-scoped table carries the owning tenant id.

    Combined with the query helpers in ``app.services.base`` this gives us
    shared-schema multi-tenancy: one set of tables, hard isolation enforced at
    the repository boundary and re-checked by an index-backed foreign key.
    """

    @declared_attr
    def tenant_id(cls) -> Mapped[uuid.UUID]:  # noqa: N805
        return mapped_column(
            GUID(),
            ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )

    @declared_attr.directive
    def __table_args__(cls):  # noqa: N805
        return (Index(f"ix_{cls.__tablename__}_tenant_created", "tenant_id", "created_at"),)
