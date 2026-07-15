"""AI product-assistant endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.api.deps import DbSession, RequireViewer
from app.models.enums import MessageRole
from app.rag import chain as rag_chain
from app.schemas.assistant import (
    AskRequest,
    AskResponse,
    Citation,
    ConversationDetail,
    ConversationRead,
    MessageRead,
    SearchHit,
    SearchRequest,
)
from app.schemas.common import Page
from app.services import audit as audit_service
from app.services import conversation as conversation_service

router = APIRouter(prefix="/workspaces/{tenant_id}/assistant", tags=["assistant"])


@router.post("/ask", response_model=AskResponse)
def ask(
    tenant_id: uuid.UUID, payload: AskRequest, db: DbSession, context: RequireViewer
) -> AskResponse:
    """Answer a question from this workspace's documentation."""
    conversation = conversation_service.get_or_create_conversation(
        db,
        tenant_id=context.tenant_id,
        user_id=context.user.id,
        conversation_id=payload.conversation_id,
        first_question=payload.question,
    )
    history = conversation_service.history_pairs(conversation)

    conversation_service.append_message(
        db, conversation=conversation, role=MessageRole.USER, content=payload.question
    )

    result = rag_chain.answer_question(
        db,
        tenant=context.tenant,
        question=payload.question,
        top_k=payload.top_k,
        history=history,
    )

    message = conversation_service.append_message(
        db,
        conversation=conversation,
        role=MessageRole.ASSISTANT,
        content=result.answer,
        citations=result.citations,
        latency_ms=result.latency_ms,
    )

    audit_service.record(
        db,
        action="assistant.ask",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="conversation",
        target_id=conversation.id,
        context={"used_context": result.used_context, "latency_ms": result.latency_ms},
    )

    return AskResponse(
        conversation_id=conversation.id,
        message_id=message.id,
        answer=result.answer,
        citations=[Citation(**citation) for citation in _strip_index(result.citations)],
        latency_ms=result.latency_ms,
        used_context=result.used_context,
    )


def _strip_index(citations: list[dict]) -> list[dict]:
    """Drop the prompt-only ``index`` key before serialising."""
    return [{k: v for k, v in citation.items() if k != "index"} for citation in citations]


@router.post("/search", response_model=list[SearchHit])
def search(
    tenant_id: uuid.UUID, payload: SearchRequest, db: DbSession, context: RequireViewer
) -> list[SearchHit]:
    hits = rag_chain.semantic_search(
        db, tenant_id=context.tenant_id, query=payload.query, top_k=payload.top_k
    )
    return [
        SearchHit(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            ordinal=hit.ordinal,
            score=round(hit.score, 4),
            content=hit.content,
        )
        for hit in hits
    ]


@router.get("/conversations", response_model=Page[ConversationRead])
def list_conversations(
    tenant_id: uuid.UUID,
    db: DbSession,
    context: RequireViewer,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> Page[ConversationRead]:
    repo = conversation_service.ConversationRepository(db, context.tenant_id)
    items, total = repo.for_user(context.user.id, offset=(page - 1) * size, limit=size)
    return Page(
        items=[ConversationRead.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    tenant_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: DbSession,
    context: RequireViewer,
) -> ConversationDetail:
    repo = conversation_service.ConversationRepository(db, context.tenant_id)
    conversation = repo.get_or_404(conversation_id)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[MessageRead.model_validate(m) for m in conversation.messages],
    )
