"""The retrieval-augmented answering chain."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.tenant import Tenant
from app.rag import prompts
from app.rag.providers import get_chat_provider, get_embedding_provider
from app.rag.retriever import RetrievedChunk, retrieve

log = get_logger(__name__)

MAX_CONTEXT_CHARS = 8000
EXCERPT_CHARS = 320


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    used_context: bool = False


def _trim_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep chunks until the context budget is spent."""
    kept: list[RetrievedChunk] = []
    budget = MAX_CONTEXT_CHARS
    for chunk in chunks:
        if len(chunk.content) > budget:
            if not kept:
                truncated = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    ordinal=chunk.ordinal,
                    content=chunk.content[:budget],
                    score=chunk.score,
                )
                kept.append(truncated)
            break
        kept.append(chunk)
        budget -= len(chunk.content)
    return kept


def _to_citation(index: int, chunk: RetrievedChunk) -> dict:
    excerpt = chunk.content.strip().replace("\n", " ")
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[: EXCERPT_CHARS - 1].rstrip() + "…"
    return {
        "index": index,
        "document_id": str(chunk.document_id),
        "document_title": chunk.document_title,
        "chunk_id": str(chunk.chunk_id),
        "ordinal": chunk.ordinal,
        "score": round(chunk.score, 4),
        "excerpt": excerpt,
    }


def answer_question(
    db: Session,
    *,
    tenant: Tenant,
    question: str,
    top_k: int | None = None,
    history: list[tuple[str, str]] | None = None,
) -> AnswerResult:
    """Embed the question, retrieve tenant-scoped context, and generate an answer."""
    started = time.perf_counter()

    embeddings = get_embedding_provider()
    query_vector = embeddings.embed_query(question)

    hits = retrieve(
        db,
        tenant_id=tenant.id,
        query_embedding=query_vector,
        top_k=top_k or settings.RAG_TOP_K,
    )

    if not hits:
        return AnswerResult(
            answer=prompts.NO_CONTEXT_ANSWER,
            citations=[],
            latency_ms=int((time.perf_counter() - started) * 1000),
            used_context=False,
        )

    selected = _trim_context(hits)
    passages = [
        (index, chunk.document_title, chunk.content)
        for index, chunk in enumerate(selected, start=1)
    ]
    context_block = prompts.build_context_block(passages)

    question_block = question
    if history:
        recent = "\n".join(f"{role}: {content}" for role, content in history[-4:])
        question_block = f"Earlier in this conversation:\n{recent}\n\nNow: {question}"

    system_prompt = prompts.SYSTEM_PROMPT.format(workspace_name=tenant.name)
    user_prompt = prompts.USER_PROMPT.format(
        context=context_block, question=question_block
    )

    chat = get_chat_provider()
    answer = chat.complete(system_prompt, user_prompt)

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "assistant.answered",
        tenant_id=str(tenant.id),
        hits=len(selected),
        latency_ms=latency_ms,
    )

    return AnswerResult(
        answer=answer,
        citations=[_to_citation(i, chunk) for i, chunk in enumerate(selected, start=1)],
        latency_ms=latency_ms,
        used_context=True,
    )


def semantic_search(
    db: Session, *, tenant_id: uuid.UUID, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """Retrieval without generation — powers the docs search box."""
    query_vector = get_embedding_provider().embed_query(query)
    return retrieve(db, tenant_id=tenant_id, query_embedding=query_vector, top_k=top_k)
