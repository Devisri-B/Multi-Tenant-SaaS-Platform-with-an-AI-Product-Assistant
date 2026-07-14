"""Importing this package registers every model on the shared metadata."""

from app.db.base import Base
from app.models.audit import AuditLog
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentChunk
from app.models.enums import (
    DocumentStatus,
    MembershipStatus,
    MessageRole,
    Role,
    TenantPlan,
)
from app.models.membership import Membership
from app.models.tenant import Tenant
from app.models.user import User

__all__ = [
    "AuditLog",
    "Base",
    "Conversation",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "Membership",
    "MembershipStatus",
    "Message",
    "MessageRole",
    "Role",
    "Tenant",
    "TenantPlan",
    "User",
]
