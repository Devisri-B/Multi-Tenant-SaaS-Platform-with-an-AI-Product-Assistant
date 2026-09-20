"""Service for recording production telemetry events and computing RAG quality metrics."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.telemetry import TelemetryEvent
from app.services.base import TenantScopedRepository


class TelemetryRepository(TenantScopedRepository[TelemetryEvent]):
    model = TelemetryEvent

    def list_events(
        self,
        event_type: str | None = None,
        conversation_id: uuid.UUID | None = None,
        limit: int = 1000,
    ) -> list[TelemetryEvent]:
        stmt = self._scoped()
        if event_type:
            stmt = stmt.where(TelemetryEvent.event_type == event_type)
        if conversation_id:
            stmt = stmt.where(TelemetryEvent.conversation_id == conversation_id)
        stmt = stmt.order_by(TelemetryEvent.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())


def record_event(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    event_type: str,
    user_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    message_id: uuid.UUID | None = None,
    event_data: dict[str, Any] | None = None,
) -> TelemetryEvent:
    """Record a production telemetry event scoped to a tenant."""
    event = TelemetryEvent(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        event_type=event_type,
        event_data=event_data or {},
    )
    db.add(event)
    db.flush()
    return event


def get_rag_metrics(db: Session, *, tenant_id: uuid.UUID) -> dict[str, Any]:
    """Compute aggregated deterministic accuracy scores and user behavioral telemetry metrics."""
    stmt = (
        select(TelemetryEvent)
        .where(TelemetryEvent.tenant_id == tenant_id)
        .order_by(TelemetryEvent.created_at.desc())
        .limit(2000)
    )
    events = list(db.execute(stmt).scalars().all())

    eval_events: list[dict[str, Any]] = []
    copy_count = 0
    citation_clicks_count = 0
    positive_feedback_count = 0
    negative_feedback_count = 0
    friction_requeries_count = 0
    successful_followups_count = 0

    for ev in events:
        etype = ev.event_type
        if etype == "eval_answer":
            eval_events.append(ev.event_data)
        elif etype == "copy":
            copy_count += 1
        elif etype == "citation_click":
            citation_clicks_count += 1
        elif etype == "feedback_positive":
            positive_feedback_count += 1
        elif etype == "feedback_negative":
            negative_feedback_count += 1
        elif etype == "friction_requery":
            friction_requeries_count += 1
        elif etype == "successful_followup":
            successful_followups_count += 1

    total_evals = len(eval_events)

    if total_evals > 0:
        numeric_scores = [
            e.get("numeric_accuracy_score", 1.0)
            for e in eval_events
            if "numeric_accuracy_score" in e
        ]
        entity_scores = [
            e.get("entity_accuracy_score", 1.0)
            for e in eval_events
            if "entity_accuracy_score" in e
        ]
        citation_scores = [
            e.get("citation_verification_rate", 1.0)
            for e in eval_events
            if "citation_verification_rate" in e
        ]

        avg_numeric = (
            round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else 1.0
        )
        avg_entity = (
            round(sum(entity_scores) / len(entity_scores), 4) if entity_scores else 1.0
        )
        avg_citation = (
            round(sum(citation_scores) / len(citation_scores), 4) if citation_scores else 1.0
        )
    else:
        avg_numeric = 1.0
        avg_entity = 1.0
        avg_citation = 1.0

    copy_rate = round(copy_count / total_evals, 4) if total_evals > 0 else 0.0
    citation_ctr = round(citation_clicks_count / total_evals, 4) if total_evals > 0 else 0.0

    followup_total = friction_requeries_count + successful_followups_count
    friction_rate = (
        round(friction_requeries_count / followup_total, 4) if followup_total > 0 else 0.0
    )

    # Real ground truth satisfaction: positive user signals (+1) vs friction/negative (0 or -1)
    positive_signals = (
        copy_count
        + citation_clicks_count
        + positive_feedback_count
        + successful_followups_count
    )
    negative_signals = friction_requeries_count + negative_feedback_count
    total_signals = positive_signals + negative_signals
    ground_truth_score = (
        round(positive_signals / total_signals, 4) if total_signals > 0 else 1.0
    )

    return {
        "total_evaluations": total_evals,
        "numeric_accuracy_avg": avg_numeric,
        "entity_accuracy_avg": avg_entity,
        "citation_verification_rate_avg": avg_citation,
        "copy_count": copy_count,
        "copy_rate": copy_rate,
        "citation_clicks_count": citation_clicks_count,
        "citation_ctr": citation_ctr,
        "friction_requeries_count": friction_requeries_count,
        "friction_rate": friction_rate,
        "successful_followups_count": successful_followups_count,
        "positive_feedback_count": positive_feedback_count,
        "negative_feedback_count": negative_feedback_count,
        "ground_truth_score": ground_truth_score,
    }
