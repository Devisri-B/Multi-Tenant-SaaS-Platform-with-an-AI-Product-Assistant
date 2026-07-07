"""Workspace lifecycle: creation, slug allocation, stats."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from secrets import token_hex

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, NotFoundError, PermissionDeniedError
from app.models.conversation import Conversation
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus, MembershipStatus, Role
from app.models.membership import Membership
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.tenant import TenantStats

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _SLUG_STRIP.sub("-", value.strip().lower()).strip("-")
    return slug or f"workspace-{token_hex(3)}"


def allocate_slug(db: Session, desired: str) -> str:
    """Return ``desired`` or the first free ``desired-N`` variant."""
    base = slugify(desired)[:70]
    candidate = base
    suffix = 2
    while db.execute(select(Tenant.id).where(Tenant.slug == candidate)).first():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def create_tenant(
    db: Session, *, name: str, owner: User, slug: str | None = None
) -> Tenant:
    """Create a workspace and make ``owner`` its first OWNER member."""
    if slug:
        existing = db.execute(select(Tenant.id).where(Tenant.slug == slug)).first()
        if existing:
            raise ConflictError(f"The slug '{slug}' is already taken.")
        final_slug = slug
    else:
        final_slug = allocate_slug(db, name)

    tenant = Tenant(name=name.strip(), slug=final_slug)
    db.add(tenant)
    db.flush()

    membership = Membership(
        tenant_id=tenant.id,
        user_id=owner.id,
        role=Role.OWNER,
        status=MembershipStatus.ACTIVE,
        accepted_at=datetime.now(timezone.utc),
    )
    db.add(membership)
    db.flush()
    db.refresh(tenant)
    return tenant


def get_tenant(db: Session, tenant_id: uuid.UUID) -> Tenant:
    tenant = db.get(Tenant, tenant_id)
    if tenant is None or tenant.is_deleted:
        raise NotFoundError("Workspace not found.")
    return tenant


def get_tenant_by_slug(db: Session, slug: str) -> Tenant:
    tenant = db.execute(select(Tenant).where(Tenant.slug == slug)).scalars().first()
    if tenant is None or tenant.is_deleted:
        raise NotFoundError("Workspace not found.")
    return tenant


def list_tenants_for_user(db: Session, user_id: uuid.UUID) -> list[tuple[Tenant, Role]]:
    stmt = (
        select(Tenant, Membership.role)
        .join(Membership, Membership.tenant_id == Tenant.id)
        .where(
            Membership.user_id == user_id,
            Membership.status == MembershipStatus.ACTIVE,
            Tenant.deleted_at.is_(None),
        )
        .order_by(Tenant.created_at.asc())
    )
    return [(row[0], Role(row[1])) for row in db.execute(stmt).all()]


def update_tenant(
    db: Session, tenant: Tenant, *, name: str | None = None, settings: dict | None = None
) -> Tenant:
    if name is not None:
        tenant.name = name.strip()
    if settings is not None:
        tenant.settings = {**tenant.settings, **settings}
    db.flush()
    return tenant


def archive_tenant(db: Session, tenant: Tenant) -> None:
    """Soft-delete a workspace; data is retained for the retention window."""
    if tenant.is_deleted:
        raise ConflictError("Workspace is already archived.")
    tenant.soft_delete()
    tenant.is_active = False
    db.flush()


def _scalar_count(db: Session, model, tenant_id: uuid.UUID, *extra) -> int:
    stmt = (
        select(func.count())
        .select_from(model)
        .where(model.tenant_id == tenant_id, *extra)
    )
    return int(db.execute(stmt).scalar_one())


def tenant_stats(db: Session, tenant: Tenant) -> TenantStats:
    return TenantStats(
        tenant_id=tenant.id,
        member_count=_scalar_count(db, Membership, tenant.id),
        document_count=_scalar_count(db, Document, tenant.id),
        indexed_document_count=_scalar_count(
            db, Document, tenant.id, Document.status == DocumentStatus.INDEXED
        ),
        chunk_count=_scalar_count(db, DocumentChunk, tenant.id),
        conversation_count=_scalar_count(db, Conversation, tenant.id),
    )


def assert_seat_available(db: Session, tenant: Tenant) -> None:
    used = _scalar_count(db, Membership, tenant.id)
    if used >= tenant.seat_limit:
        raise PermissionDeniedError(
            f"Workspace has reached its seat limit of {tenant.seat_limit}. "
            "Upgrade the plan to invite more members."
        )


def assert_document_quota(db: Session, tenant: Tenant) -> None:
    used = _scalar_count(db, Document, tenant.id)
    if used >= tenant.document_limit:
        raise PermissionDeniedError(
            f"Workspace has reached its document limit of {tenant.document_limit}."
        )
