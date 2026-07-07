"""Workspace (tenant) payloads."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.enums import TenantPlan
from app.schemas.common import ORMModel

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class TenantCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    slug: str | None = Field(default=None, max_length=80)

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if not SLUG_RE.match(value):
            raise ValueError(
                "Slug may only contain lowercase letters, digits and single hyphens."
            )
        return value


class TenantUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    settings: dict | None = None


class TenantRead(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: TenantPlan
    is_active: bool
    seat_limit: int
    document_limit: int
    settings: dict
    created_at: datetime


class TenantStats(BaseModel):
    tenant_id: uuid.UUID
    member_count: int
    document_count: int
    indexed_document_count: int
    chunk_count: int
    conversation_count: int
