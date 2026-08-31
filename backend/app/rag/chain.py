"""The retrieval-augmented answering chain powered by LangGraph."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.tenant import Tenant
from app.rag import prompts
from app.rag.graph import assistant_graph
from app.rag.providers import get_embedding_provider
from app.rag.retriever import RetrievedChunk, retrieve

log = get_logger(__name__)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    used_context: bool = False
    source_type: str = "workspace_docs"


def answer_question(
    db: Session,
    *,
    tenant: Tenant,
    question: str,
    top_k: int | None = None,
    history: list[tuple[str, str]] | None = None,
    allow_web_search: bool = True,
) -> AnswerResult:
    """Execute the LangGraph adaptive RAG workflow with online fallback."""
    started = time.perf_counter()

    initial_state = {
        "db": db,
        "tenant_id": tenant.id,
        "tenant_name": tenant.name,
        "question": question,
        "history": history,
        "top_k": top_k or settings.RAG_TOP_K,
        "allow_web_search": allow_web_search,
    }

    final_state = assistant_graph.invoke(initial_state)

    latency_ms = int((time.perf_counter() - started) * 1000)
    source_type = final_state.get("source_type", "none")
    log.info(
        "assistant.answered",
        tenant_id=str(tenant.id),
        source_type=source_type,
        citations=len(final_state.get("citations", [])),
        latency_ms=latency_ms,
    )

    return AnswerResult(
        answer=final_state.get("answer", prompts.NO_CONTEXT_ANSWER),
        citations=final_state.get("citations", []),
        latency_ms=latency_ms,
        used_context=final_state.get("used_context", False),
        source_type=source_type,
    )


def semantic_search(
    db: Session, *, tenant_id: uuid.UUID, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """Retrieval without generation — powers the docs search box."""
    query_vector = get_embedding_provider().embed_query(query)
    return retrieve(db, tenant_id=tenant_id, query_embedding=query_vector, top_k=top_k)
