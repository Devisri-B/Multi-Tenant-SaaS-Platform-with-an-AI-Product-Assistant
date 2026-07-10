"""Membership management: invite, list, change role, remove."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from secrets import token_urlsafe

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, NotFoundError, PermissionDeniedError
from app.models.enums import MembershipStatus, Role
from app.models.membership import Membership
from app.models.tenant import Tenant
from app.models.user import User
from app.services import auth as auth_service
from app.services import tenant as tenant_service


def list_members(
    db: Session, tenant_id: uuid.UUID, *, offset: int = 0, limit: int = 50
) -> tuple[list[Membership], int]:
    stmt = (
        select(Membership)
        .where(Membership.tenant_id == tenant_id)
        .order_by(Membership.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    items = list(db.execute(stmt).scalars().unique().all())
    total = int(
        db.execute(
            select(func.count()).select_from(Membership).where(Membership.tenant_id == tenant_id)
        ).scalar_one()
    )
    return items, total


def get_membership(db: Session, tenant_id: uuid.UUID, membership_id: uuid.UUID) -> Membership:
    stmt = select(Membership).where(
        Membership.id == membership_id, Membership.tenant_id == tenant_id
    )
    membership = db.execute(stmt).scalars().first()
    if membership is None:
        raise NotFoundError("Member not found in this workspace.")
    return membership


def get_membership_for_user(
    db: Session, tenant_id: uuid.UUID, user_id: uuid.UUID
) -> Membership | None:
    stmt = select(Membership).where(
        Membership.tenant_id == tenant_id, Membership.user_id == user_id
    )
    return db.execute(stmt).scalars().first()


def count_owners(db: Session, tenant_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(Membership)
        .where(Membership.tenant_id == tenant_id, Membership.role == Role.OWNER)
    )
    return int(db.execute(stmt).scalar_one())


def invite_member(
    db: Session,
    *,
    tenant: Tenant,
    email: str,
    role: Role,
    full_name: str | None,
    invited_by: User,
) -> tuple[Membership, bool, str | None]:
    """Invite a user to a workspace, creating the account if it is new.

    Returns ``(membership, created_new_user, temporary_password)``.
    """
    tenant_service.assert_seat_available(db, tenant)

    user = auth_service.get_user_by_email(db, email)
    created_new_user = False
    temporary_password = None

    if user is None:
        temporary_password = token_urlsafe(12) + "aA1"
        user = auth_service.create_user(
            db, email=email, password=temporary_password, full_name=full_name
        )
        created_new_user = True

    if get_membership_for_user(db, tenant.id, user.id):
        raise ConflictError("That user is already a member of this workspace.")

    membership = Membership(
        tenant_id=tenant.id,
        user_id=user.id,
        role=role,
        status=MembershipStatus.INVITED if created_new_user else MembershipStatus.ACTIVE,
        invited_by_id=invited_by.id,
        accepted_at=None if created_new_user else datetime.now(timezone.utc),
    )
    db.add(membership)
    db.flush()
    db.refresh(membership)
    return membership, created_new_user, temporary_password


def change_role(
    db: Session, *, membership: Membership, new_role: Role, actor: Membership
) -> Membership:
    """Change a member's role, guarding privilege escalation and owner loss."""
    if membership.id == actor.id:
        raise PermissionDeniedError("You cannot change your own role.")

    actor_role = Role(actor.role)
    target_role = Role(membership.role)

    if not actor_role.satisfies(Role.ADMIN):
        raise PermissionDeniedError("Only admins and owners can change roles.")
    if target_role.level >= actor_role.level and actor_role is not Role.OWNER:
        raise PermissionDeniedError("You cannot modify a member at or above your role.")
    if new_role is Role.OWNER and actor_role is not Role.OWNER:
        raise PermissionDeniedError("Only an owner can promote another owner.")
    if (
        target_role is Role.OWNER
        and new_role is not Role.OWNER
        and count_owners(db, membership.tenant_id) <= 1
    ):
        raise ConflictError("A workspace must always have at least one owner.")

    membership.role = new_role
    db.flush()
    return membership


def remove_member(db: Session, *, membership: Membership, actor: Membership) -> None:
    if membership.id == actor.id:
        raise PermissionDeniedError("Use 'leave workspace' to remove yourself.")

    actor_role = Role(actor.role)
    target_role = Role(membership.role)
    if not actor_role.satisfies(Role.ADMIN):
        raise PermissionDeniedError("Only admins and owners can remove members.")
    if target_role.level >= actor_role.level and actor_role is not Role.OWNER:
        raise PermissionDeniedError("You cannot remove a member at or above your role.")
    if target_role is Role.OWNER and count_owners(db, membership.tenant_id) <= 1:
        raise ConflictError("A workspace must always have at least one owner.")

    db.delete(membership)
    db.flush()


def activate_membership(db: Session, membership: Membership) -> Membership:
    membership.status = MembershipStatus.ACTIVE
    membership.accepted_at = datetime.now(timezone.utc)
    db.flush()
    return membership
