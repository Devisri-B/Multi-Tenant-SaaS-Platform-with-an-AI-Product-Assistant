"""FastAPI dependencies: authentication, tenant resolution, role guards.

The dependency chain is deliberately linear:

    bearer token -> current_user -> current_tenant -> require_role(...)

Route handlers therefore never touch a tenant id supplied by the client
without it first having been validated against the caller's memberships.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.core.security import decode_token
from app.db.session import get_db
from app.models.enums import MembershipStatus, Role
from app.models.membership import Membership
from app.models.tenant import Tenant
from app.models.user import User
from app.services import member as member_service
from app.services import tenant as tenant_service

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token.")

    payload = decode_token(credentials.credentials, expected_type="access")
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise AuthenticationError("Token subject is not a valid user id.")

    user = db.get(User, user_id)
    if user is None or user.is_deleted:
        raise AuthenticationError("The account for this token no longer exists.")
    if not user.is_active:
        raise AuthenticationError("This account has been deactivated.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


@dataclass(slots=True)
class TenantContext:
    """Everything a handler needs about the caller's position in a workspace."""

    tenant: Tenant
    membership: Membership
    user: User

    @property
    def tenant_id(self) -> uuid.UUID:
        return self.tenant.id

    @property
    def role(self) -> Role:
        return Role(self.membership.role)

    def require(self, minimum: Role) -> None:
        if not self.role.satisfies(minimum):
            raise PermissionDeniedError(
                f"This action requires the '{minimum.value}' role or higher; "
                f"you have '{self.role.value}'."
            )


def _resolve_tenant_id(request: Request, header_tenant_id: str | None) -> uuid.UUID:
    """Prefer the workspace in the path, fall back to the X-Workspace-Id header.

    The path value is read from ``request.path_params`` rather than declared as
    a dependency parameter, because this dependency is also used by routes that
    carry no ``{tenant_id}`` segment.
    """
    raw = request.path_params.get("tenant_id") or header_tenant_id
    if not raw:
        raise PermissionDeniedError(
            "No workspace selected. Pass an X-Workspace-Id header or use a "
            "workspace-scoped route."
        )
    if isinstance(raw, uuid.UUID):
        return raw
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        raise PermissionDeniedError("The workspace identifier is not a valid UUID.")


def get_tenant_context(
    request: Request,
    db: DbSession,
    user: CurrentUser,
    x_workspace_id: Annotated[str | None, Header()] = None,
) -> TenantContext:
    resolved = _resolve_tenant_id(request, x_workspace_id)
    tenant = tenant_service.get_tenant(db, resolved)

    if not tenant.is_active:
        raise PermissionDeniedError("This workspace has been deactivated.")

    membership = member_service.get_membership_for_user(db, tenant.id, user.id)
    if membership is None and not user.is_superuser:
        # Deliberately a 403 rather than a 404 leak of workspace existence.
        raise PermissionDeniedError("You are not a member of this workspace.")
    if membership is not None and membership.status == MembershipStatus.SUSPENDED:
        raise PermissionDeniedError("Your access to this workspace is suspended.")

    if membership is None:  # superuser acting on any workspace
        membership = Membership(
            tenant_id=tenant.id,
            user_id=user.id,
            role=Role.OWNER,
            status=MembershipStatus.ACTIVE,
        )

    request.state.tenant_id = str(tenant.id)
    return TenantContext(tenant=tenant, membership=membership, user=user)


Tenancy = Annotated[TenantContext, Depends(get_tenant_context)]


def require_role(minimum: Role) -> Callable[[TenantContext], TenantContext]:
    """Build a dependency that enforces a minimum workspace role."""

    def _guard(context: Tenancy) -> TenantContext:
        context.require(minimum)
        return context

    return _guard


RequireViewer = Annotated[TenantContext, Depends(require_role(Role.VIEWER))]
RequireMember = Annotated[TenantContext, Depends(require_role(Role.MEMBER))]
RequireAdmin = Annotated[TenantContext, Depends(require_role(Role.ADMIN))]
RequireOwner = Annotated[TenantContext, Depends(require_role(Role.OWNER))]
