"""Authentication and session endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import CurrentUser, DbSession
from app.schemas.auth import (
    LoginRequest,
    MembershipSummary,
    PasswordChangeRequest,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    SessionResponse,
    TokenPair,
    UserRead,
)
from app.schemas.common import MessageResponse
from app.services import audit as audit_service
from app.services import auth as auth_service
from app.services import tenant as tenant_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: DbSession) -> RegisterResponse:
    """Create an account and its first workspace in a single transaction."""
    user = auth_service.create_user(
        db,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
    )
    tenant = tenant_service.create_tenant(db, name=payload.workspace_name, owner=user)
    audit_service.record(
        db,
        action="auth.register",
        tenant_id=tenant.id,
        actor_id=user.id,
        target_type="user",
        target_id=user.id,
    )
    return RegisterResponse(
        user=UserRead.model_validate(user),
        tenant_id=tenant.id,
        tokens=auth_service.issue_tokens(user),
    )


@router.post("/login", response_model=TokenPair)
def login(payload: LoginRequest, db: DbSession) -> TokenPair:
    user = auth_service.authenticate(db, email=payload.email, password=payload.password)
    audit_service.record(db, action="auth.login", actor_id=user.id)
    return auth_service.issue_tokens(user)


@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: DbSession) -> TokenPair:
    return auth_service.refresh_tokens(db, payload.refresh_token)


@router.get("/me", response_model=SessionResponse)
def me(db: DbSession, user: CurrentUser) -> SessionResponse:
    memberships = [
        MembershipSummary(
            tenant_id=tenant.id,
            tenant_name=tenant.name,
            tenant_slug=tenant.slug,
            role=role.value,
        )
        for tenant, role in tenant_service.list_tenants_for_user(db, user.id)
    ]
    return SessionResponse(user=UserRead.model_validate(user), memberships=memberships)


@router.post("/password", response_model=MessageResponse)
def change_password(
    payload: PasswordChangeRequest, db: DbSession, user: CurrentUser
) -> MessageResponse:
    auth_service.change_password(
        db,
        user,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    audit_service.record(db, action="auth.password_changed", actor_id=user.id)
    return MessageResponse(message="Password updated.")
