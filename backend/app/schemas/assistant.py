"""Assistant / RAG payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.enums import MessageRole
from app.schemas.common import ORMModel


class Citation(BaseModel):
    document_id: uuid.UUID | None = None
    document_title: str
    chunk_id: uuid.UUID | None = None
    ordinal: int = 0
    score: float = 0.0
    excerpt: str
    url: str | None = None
    source_type: Literal["document", "web"] = "document"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: uuid.UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    allow_web_search: bool = True


class AskResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    latency_ms: int
    used_context: bool
    source_type: Literal["workspace_docs", "online_search", "none"] = "workspace_docs"


class MessageRead(ORMModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    citations: list
    created_at: datetime


class ConversationRead(ORMModel):
    id: uuid.UUID
    title: str
    created_at: datetime


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    score: float
    content: str
