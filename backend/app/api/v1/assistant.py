"""AI product-assistant endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from app.api.deps import DbSession, RequireViewer
from app.models.enums import MessageRole
from app.rag import chain as rag_chain
from app.rag.eval import (
    detect_query_friction,
    verify_citation_rate,
    verify_entity_accuracy,
    verify_numeric_accuracy,
)
from app.schemas.assistant import (
    AskRequest,
    AskResponse,
    Citation,
    ConversationDetail,
    ConversationRead,
    EvaluationSummary,
    MessageRead,
    RAGQualityMetrics,
    SearchHit,
    SearchRequest,
    TelemetryEventCreate,
    TelemetryEventRead,
)
from app.schemas.common import Page
from app.services import audit as audit_service
from app.services import conversation as conversation_service
from app.services import telemetry as telemetry_service

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

    # 1. Detect query friction against the prior turn before appending new message
    last_user_msg = None
    last_asst_msg = None
    for msg in reversed(conversation.messages):
        if msg.role == MessageRole.USER and last_user_msg is None:
            last_user_msg = msg
        elif msg.role == MessageRole.ASSISTANT and last_asst_msg is None:
            last_asst_msg = msg

    elapsed_seconds: float | None = None
    ref_msg = last_asst_msg or last_user_msg
    ref_time = ref_msg.created_at if ref_msg else None
    if ref_time is not None:
        now = datetime.now(timezone.utc)
        dt = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        elapsed_seconds = max(0.0, (now - dt).total_seconds())

    friction = detect_query_friction(
        current_q=payload.question,
        prev_q=last_user_msg.content if last_user_msg else None,
        elapsed_seconds=elapsed_seconds,
    )

    if friction.get("is_friction"):
        telemetry_service.record_event(
            db,
            tenant_id=context.tenant_id,
            user_id=context.user.id,
            conversation_id=conversation.id,
            message_id=last_asst_msg.id if last_asst_msg else None,
            event_type="friction_requery",
            event_data=friction,
        )
    elif friction.get("is_successful_followup"):
        telemetry_service.record_event(
            db,
            tenant_id=context.tenant_id,
            user_id=context.user.id,
            conversation_id=conversation.id,
            message_id=last_asst_msg.id if last_asst_msg else None,
            event_type="successful_followup",
            event_data=friction,
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
        allow_web_search=payload.allow_web_search,
        conversation_id=conversation.id,
    )

    message = conversation_service.append_message(
        db,
        conversation=conversation,
        role=MessageRole.ASSISTANT,
        content=result.answer,
        citations=result.citations,
        latency_ms=result.latency_ms,
    )

    # 2. Run deterministic zero-cost evaluation (Numeric, Entity, Citation)
    num_eval = verify_numeric_accuracy(result.answer, result.context_text)
    ent_eval = verify_entity_accuracy(result.answer, result.context_text)
    cit_eval = verify_citation_rate(result.citations, full_context=result.context_text)

    eval_summary = EvaluationSummary(
        numeric_accuracy_score=num_eval["numeric_accuracy_score"],
        entity_accuracy_score=ent_eval["entity_accuracy_score"],
        citation_verification_rate=cit_eval["citation_verification_rate"],
        total_numbers=num_eval["total_numbers"],
        unsupported_numbers=num_eval["unsupported_numbers"],
        total_entities=ent_eval["total_entities"],
        unsupported_entities=ent_eval["unsupported_entities"],
        total_citations=cit_eval["total_citations"],
        verified_citations=cit_eval["verified_citations"],
    )

    telemetry_service.record_event(
        db,
        tenant_id=context.tenant_id,
        user_id=context.user.id,
        conversation_id=conversation.id,
        message_id=message.id,
        event_type="eval_answer",
        event_data={
            **num_eval,
            **ent_eval,
            **cit_eval,
            "latency_ms": result.latency_ms,
            "used_context": result.used_context,
            "source_type": result.source_type,
        },
    )

    audit_service.record(
        db,
        action="assistant.ask",
        tenant_id=context.tenant_id,
        actor_id=context.user.id,
        target_type="conversation",
        target_id=conversation.id,
        context={
            "used_context": result.used_context,
            "source_type": result.source_type,
            "latency_ms": result.latency_ms,
            "numeric_accuracy": num_eval["numeric_accuracy_score"],
            "entity_accuracy": ent_eval["entity_accuracy_score"],
            "citation_rate": cit_eval["citation_verification_rate"],
        },
    )

    return AskResponse(
        conversation_id=conversation.id,
        message_id=message.id,
        answer=result.answer,
        citations=[Citation(**citation) for citation in _strip_index(result.citations)],
        latency_ms=result.latency_ms,
        used_context=result.used_context,
        source_type=result.source_type,
        evaluation=eval_summary,
        friction_detected=friction.get("is_friction", False),
        friction_reason=friction.get("friction_reason"),
        ground_truth_score=friction.get("ground_truth_score", 1.0),
    )


def _strip_index(citations: list[dict]) -> list[dict]:
    """Drop the prompt-only ``index`` key before serialising."""
    return [{k: v for k, v in citation.items() if k != "index"} for citation in citations]


@router.post("/telemetry", response_model=TelemetryEventRead, status_code=201)
def record_telemetry(
    tenant_id: uuid.UUID,
    payload: TelemetryEventCreate,
    db: DbSession,
    context: RequireViewer,
) -> TelemetryEventRead:
    """Record user production telemetry signals (copy button, citation click, feedback rating)."""
    event = telemetry_service.record_event(
        db,
        tenant_id=context.tenant_id,
        user_id=context.user.id,
        conversation_id=payload.conversation_id,
        message_id=payload.message_id,
        event_type=payload.event_type,
        event_data=payload.event_data,
    )
    return TelemetryEventRead.model_validate(event)


@router.get("/metrics", response_model=RAGQualityMetrics)
def get_metrics(
    tenant_id: uuid.UUID,
    db: DbSession,
    context: RequireViewer,
) -> RAGQualityMetrics:
    """Get aggregated deterministic evaluation scores and user behavioral telemetry metrics."""
    metrics = telemetry_service.get_rag_metrics(db, tenant_id=context.tenant_id)
    return RAGQualityMetrics(**metrics)


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
