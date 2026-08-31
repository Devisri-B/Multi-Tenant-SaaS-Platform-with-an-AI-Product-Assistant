"""Document payloads."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import DocumentStatus
from app.schemas.common import ORMModel


class DocumentCreate(BaseModel):
    """Create a document from raw text rather than a file upload."""

    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class DocumentRead(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    source_name: str
    content_type: str
    status: DocumentStatus
    byte_size: int
    chunk_count: int
    error_message: str | None
    doc_metadata: dict
    created_at: datetime


class DocumentChunkRead(ORMModel):
    id: uuid.UUID
    document_id: uuid.UUID
    ordinal: int
    content: str
    token_estimate: int


class DocumentDetail(DocumentRead):
    content: str
    chunks: list[DocumentChunkRead] = Field(default_factory=list)


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    content: str | None = Field(default=None, min_length=1)


class ReindexResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    chunk_count: int
