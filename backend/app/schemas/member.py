"""Membership payloads."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import MembershipStatus, Role
from app.schemas.common import ORMModel


class MemberInvite(BaseModel):
    email: EmailStr
    role: Role = Role.MEMBER
    full_name: str | None = Field(default=None, max_length=160)


class MemberRoleUpdate(BaseModel):
    role: Role


class MemberRead(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    email: EmailStr
    full_name: str | None
    role: Role
    status: MembershipStatus
    created_at: datetime


class MemberInviteResult(BaseModel):
    member: MemberRead
    invited_new_user: bool
    temporary_password: str | None = None
