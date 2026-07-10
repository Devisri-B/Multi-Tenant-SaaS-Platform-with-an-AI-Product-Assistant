"""Workspace membership endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, RequireAdmin, RequireViewer
from app.models.membership import Membership
from app.schemas.common import MessageResponse, Page
from app.schemas.member import (
    MemberInvite,
    MemberInviteResult,
    MemberRead,
    MemberRoleUpdate,
)
from app.services import audit as audit_service
from app.services import member as member_service

router = APIRouter(prefix="/workspaces/{tenant_id}/members", tags=["members"])


def _to_read(membership: Membership) -> MemberRead:
    return MemberRead(
        id=membership.id,
        user_id=membership.user_id,
        email=membership.user.email,
        full_name=membership.user.full_name,
        role=membership.role,
        status=membership.status,
        created_at=membership.created_at,
    )


@router.get("", response_model=Page[MemberRead])
def list_members(
    tenant_id: uuid.UUID,
    db: DbSession,
    context: RequireViewer,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> Page[MemberRead]:
    items, total = member_service.list_members(
        db, context.tenant_id, offset=(page - 1) * size, limit=size
    )
    return Page(items=[_to_read(m) for m in items], total=total, page=page, size=size)


@router.post("", response_model=MemberInviteResult, status_code=status.HTTP_201_CREATED)
def invite_member(
    tenant_id: uuid.UUID, payload: MemberInvite, db: DbSession, context: RequireAdmin
) -> MemberInviteResult:
    membership, created, temporary_password = member_service.invite_member(
        db,
        tenant=context.tenant,
        email=payload.email,
        role=payload.role,
        full_name=payload.full_name,
        invited_by=context.user,
    )
    audit_service.record(
        db,
        action="member.invited",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="membership",
        target_id=membership.id,
        context={"email": payload.email, "role": payload.role.value},
    )
    return MemberInviteResult(
        member=_to_read(membership),
        invited_new_user=created,
        temporary_password=temporary_password,
    )


@router.patch("/{membership_id}", response_model=MemberRead)
def update_member_role(
    tenant_id: uuid.UUID,
    membership_id: uuid.UUID,
    payload: MemberRoleUpdate,
    db: DbSession,
    context: RequireAdmin,
) -> MemberRead:
    membership = member_service.get_membership(db, context.tenant_id, membership_id)
    updated = member_service.change_role(
        db, membership=membership, new_role=payload.role, actor=context.membership
    )
    audit_service.record(
        db,
        action="member.role_changed",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="membership",
        target_id=membership_id,
        context={"role": payload.role.value},
    )
    return _to_read(updated)


@router.delete("/{membership_id}", response_model=MessageResponse)
def remove_member(
    tenant_id: uuid.UUID,
    membership_id: uuid.UUID,
    db: DbSession,
    context: RequireAdmin,
) -> MessageResponse:
    membership = member_service.get_membership(db, context.tenant_id, membership_id)
    member_service.remove_member(db, membership=membership, actor=context.membership)
    audit_service.record(
        db,
        action="member.removed",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="membership",
        target_id=membership_id,
    )
    return MessageResponse(message="Member removed from workspace.")
