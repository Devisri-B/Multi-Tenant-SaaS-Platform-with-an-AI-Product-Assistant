"""Tenant (workspace) — the isolation boundary for every other resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import JSONType
from app.models.enums import TenantPlan
from app.models.mixins import SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.membership import Membership


class Tenant(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A customer workspace.

    All product data hangs off a tenant.  A single deployment serves many
    tenants from one schema; the ``tenant_id`` foreign key on every child table
    plus the scoped repositories are what keep them apart.
    """

    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    plan: Mapped[TenantPlan] = mapped_column(
        String(32), nullable=False, default=TenantPlan.FREE
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    seat_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    document_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    settings: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan", lazy="selectin"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )

    @property
    def member_count(self) -> int:
        return len(self.memberships)
