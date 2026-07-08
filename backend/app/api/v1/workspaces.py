"""Workspace (tenant) endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.api.deps import CurrentUser, DbSession, RequireAdmin, RequireOwner, RequireViewer
from app.schemas.common import MessageResponse
from app.schemas.tenant import TenantCreate, TenantRead, TenantStats, TenantUpdate
from app.services import audit as audit_service
from app.services import tenant as tenant_service

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("", response_model=list[TenantRead])
def list_my_workspaces(db: DbSession, user: CurrentUser) -> list[TenantRead]:
    pairs = tenant_service.list_tenants_for_user(db, user.id)
    return [TenantRead.model_validate(tenant) for tenant, _ in pairs]


@router.post("", response_model=TenantRead, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: TenantCreate, db: DbSession, user: CurrentUser
) -> TenantRead:
    tenant = tenant_service.create_tenant(
        db, name=payload.name, owner=user, slug=payload.slug
    )
    audit_service.record(
        db,
        action="workspace.created",
        tenant_id=tenant.id,
        actor_id=user.id,
        target_type="tenant",
        target_id=tenant.id,
    )
    return TenantRead.model_validate(tenant)


@router.get("/{tenant_id}", response_model=TenantRead)
def get_workspace(tenant_id: uuid.UUID, context: RequireViewer) -> TenantRead:
    return TenantRead.model_validate(context.tenant)


@router.patch("/{tenant_id}", response_model=TenantRead)
def update_workspace(
    tenant_id: uuid.UUID, payload: TenantUpdate, db: DbSession, context: RequireAdmin
) -> TenantRead:
    tenant = tenant_service.update_tenant(
        db, context.tenant, name=payload.name, settings=payload.settings
    )
    audit_service.record(
        db,
        action="workspace.updated",
        tenant_id=tenant.id,
        actor_id=context.user.id,
        target_type="tenant",
        target_id=tenant.id,
        context=payload.model_dump(exclude_none=True),
    )
    return TenantRead.model_validate(tenant)


@router.get("/{tenant_id}/stats", response_model=TenantStats)
def workspace_stats(
    tenant_id: uuid.UUID, db: DbSession, context: RequireViewer
) -> TenantStats:
    return tenant_service.tenant_stats(db, context.tenant)


@router.delete("/{tenant_id}", response_model=MessageResponse)
def archive_workspace(
    tenant_id: uuid.UUID, db: DbSession, context: RequireOwner
) -> MessageResponse:
    tenant_service.archive_tenant(db, context.tenant)
    audit_service.record(
        db,
        action="workspace.archived",
        tenant_id=context.tenant.id,
        actor_id=context.user.id,
        target_type="tenant",
        target_id=context.tenant.id,
    )
    return MessageResponse(message="Workspace archived.")
