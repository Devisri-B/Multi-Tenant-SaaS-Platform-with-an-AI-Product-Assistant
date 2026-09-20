"""Assistant / RAG payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

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


class EvaluationSummary(BaseModel):
    numeric_accuracy_score: float = 1.0
    entity_accuracy_score: float = 1.0
    citation_verification_rate: float = 1.0
    total_numbers: int = 0
    unsupported_numbers: list[str] = Field(default_factory=list)
    total_entities: int = 0
    unsupported_entities: list[str] = Field(default_factory=list)
    total_citations: int = 0
    verified_citations: int = 0


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
    evaluation: EvaluationSummary | None = None
    friction_detected: bool = False
    friction_reason: str | None = None
    ground_truth_score: float = 1.0


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


class TelemetryEventCreate(BaseModel):
    event_type: Literal[
        "copy",
        "citation_click",
        "feedback_positive",
        "feedback_negative",
        "friction_requery",
    ]
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    event_data: dict[str, Any] = Field(default_factory=dict)


class TelemetryEventRead(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID | None
    conversation_id: uuid.UUID | None
    message_id: uuid.UUID | None
    event_type: str
    event_data: dict[str, Any]
    created_at: datetime


class RAGQualityMetrics(BaseModel):
    total_evaluations: int
    numeric_accuracy_avg: float
    entity_accuracy_avg: float
    citation_verification_rate_avg: float
    copy_count: int
    copy_rate: float
    citation_clicks_count: int
    citation_ctr: float
    friction_requeries_count: int
    friction_rate: float
    successful_followups_count: int
    positive_feedback_count: int
    negative_feedback_count: int
    ground_truth_score: float
