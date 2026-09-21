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
from app.rag.retriever import RetrievedChunk, retrieve

log = get_logger(__name__)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    used_context: bool = False
    source_type: str = "workspace_docs"
    context_text: str = ""


def answer_question(
    db: Session,
    *,
    tenant: Tenant,
    question: str,
    top_k: int | None = None,
    history: list[tuple[str, str]] | None = None,
    allow_web_search: bool = True,
    conversation_id: uuid.UUID | None = None,
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

    # Pass LangSmith / LangChain RunnableConfig for versioning & observability
    active_prompt_version = prompts.get_prompt_version()
    tenant_slug = tenant.name.lower().replace(" ", "-") if tenant.name else "workspace"
    run_config = {
        "run_name": f"rag-assistant:{tenant_slug}",
        "tags": [
            f"tenant:{tenant.id}",
            f"prompt:{active_prompt_version}",
            f"env:{settings.ENVIRONMENT}",
            f"provider:{settings.LLM_PROVIDER}",
        ],
        "metadata": {
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "prompt_version": active_prompt_version,
            "app_name": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "hybrid_search": settings.RAG_ENABLE_HYBRID_SEARCH,
            "reranker_enabled": settings.RAG_ENABLE_RERANKER,
            "top_k": top_k or settings.RAG_TOP_K,
        },
    }

    final_state = assistant_graph.invoke(initial_state, config=run_config)

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
        context_text=final_state.get("context_text", ""),
    )


def semantic_search(
    db: Session, *, tenant_id: uuid.UUID, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """Retrieval without generation - powers the docs search box via concurrent hybrid search."""
    return retrieve(db, tenant_id=tenant_id, query=query, top_k=top_k)
