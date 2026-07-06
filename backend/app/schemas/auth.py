"""Auth request/response payloads."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.schemas.common import ORMModel

MIN_PASSWORD_LENGTH = 10


def _validate_password_strength(value: str) -> str:
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if value.lower() == value or value.upper() == value:
        raise ValueError("Password must mix upper and lower case characters.")
    if not any(char.isdigit() for char in value):
        raise ValueError("Password must contain at least one digit.")
    return value


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = Field(default=None, max_length=160)
    workspace_name: str = Field(min_length=2, max_length=120)

    _check_password = field_validator("password")(_validate_password_strength)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserRead(ORMModel):
    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    is_superuser: bool


class MembershipSummary(BaseModel):
    tenant_id: uuid.UUID
    tenant_name: str
    tenant_slug: str
    role: str


class SessionResponse(BaseModel):
    user: UserRead
    memberships: list[MembershipSummary]


class RegisterResponse(BaseModel):
    user: UserRead
    tenant_id: uuid.UUID
    tokens: TokenPair


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

    _check_password = field_validator("new_password")(_validate_password_strength)
