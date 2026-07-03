"""Enumerations shared by the ORM models and the API schemas."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Workspace roles, ordered from least to most privileged."""

    VIEWER = "viewer"
    MEMBER = "member"
    ADMIN = "admin"
    OWNER = "owner"

    @property
    def level(self) -> int:
        return _ROLE_LEVELS[self]

    def satisfies(self, required: Role) -> bool:
        """True when this role is at least as privileged as ``required``."""
        return self.level >= required.level


_ROLE_LEVELS: dict[Role, int] = {
    Role.VIEWER: 10,
    Role.MEMBER: 20,
    Role.ADMIN: 30,
    Role.OWNER: 40,
}


class TenantPlan(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class MembershipStatus(str, Enum):
    INVITED = "invited"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
