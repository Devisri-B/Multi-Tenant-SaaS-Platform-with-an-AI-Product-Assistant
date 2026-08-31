"""LangGraph workflow for adaptive RAG with sliding window memory and online search routing."""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.rag import prompts
from app.rag.memory import format_sliding_window_history
from app.rag.providers import get_chat_provider, get_embedding_provider
from app.rag.retriever import RetrievedChunk, retrieve
from app.rag.web_search import WebSearchResult, get_web_search_provider

log = get_logger(__name__)

MAX_CONTEXT_CHARS = 8000
EXCERPT_CHARS = 320


class AssistantState(TypedDict, total=False):
    db: Session
    tenant_id: uuid.UUID
    tenant_name: str
    question: str
    search_query: str
    history: list[tuple[str, str]] | None
    formatted_history: str
    top_k: int
    allow_web_search: bool
    retry_count: int

    # Pipeline data
    documents: list[RetrievedChunk]
    web_results: list[WebSearchResult]
    has_relevant_docs: bool

    # Quality & Hallucination verification flags
    is_grounded: bool
    answers_question: bool

    # Output
    answer: str
    citations: list[dict[str, Any]]
    used_context: bool
    source_type: str  # "workspace_docs" | "online_search" | "none"
    latency_ms: int


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


def _to_doc_citation(index: int, chunk: RetrievedChunk) -> dict[str, Any]:
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
        "url": None,
        "source_type": "document",
    }


def _to_web_citation(index: int, result: WebSearchResult) -> dict[str, Any]:
    excerpt = result.snippet.strip().replace("\n", " ")
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[: EXCERPT_CHARS - 1].rstrip() + "…"
    return {
        "index": index,
        "document_id": None,
        "document_title": result.title,
        "chunk_id": None,
        "ordinal": index,
        "score": round(result.score, 4),
        "excerpt": excerpt,
        "url": result.url,
        "source_type": "web",
    }


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------
def contextualize_query_node(state: AssistantState) -> dict[str, Any]:
    """Sliding window memory manager: contextualize query and resolve coreferences."""
    history = state.get("history")
    question = state["question"]
    formatted_history = format_sliding_window_history(history)

    if not history or not settings.RAG_ENABLE_QUERY_REWRITE:
        return {
            "search_query": question,
            "formatted_history": formatted_history,
        }

    try:
        chat = get_chat_provider()
        prompt = prompts.REWRITE_QUESTION_PROMPT.format(
            history=formatted_history,
            question=question,
        )
        rewritten = chat.complete("You are a query reformulation assistant.", prompt).strip()
        search_query = rewritten if rewritten else question
        log.info(
            "langgraph.contextualize_query",
            original=question,
            rewritten=search_query,
            history_turns=len(history),
        )
        return {
            "search_query": search_query,
            "formatted_history": formatted_history,
        }
    except Exception as exc:
        log.warning("langgraph.contextualize_query.error", error=str(exc))
        return {
            "search_query": question,
            "formatted_history": formatted_history,
        }


def retrieve_node(state: AssistantState) -> dict[str, Any]:
    """Retrieve candidate document chunks from tenant vector store."""
    embeddings = get_embedding_provider()
    search_query = state.get("search_query") or state["question"]
    query_vector = embeddings.embed_query(search_query)
    db = state["db"]
    top_k = state.get("top_k") or settings.RAG_TOP_K

    hits = retrieve(
        db,
        tenant_id=state["tenant_id"],
        query_embedding=query_vector,
        top_k=top_k,
    )
    return {
        "documents": hits,
        "retry_count": state.get("retry_count", 0),
    }


def grade_documents_node(state: AssistantState) -> dict[str, Any]:
    """Evaluate whether retrieved documents are relevant to answer the question."""
    hits = state.get("documents", [])
    if not hits:
        return {"has_relevant_docs": False}

    best_score = max(chunk.score for chunk in hits)
    if best_score < max(settings.RAG_MIN_SCORE, 0.0):
        log.info(
            "langgraph.grade_documents.low_score",
            tenant_id=str(state["tenant_id"]),
            best_score=best_score,
            min_score=settings.RAG_MIN_SCORE,
        )
        return {"has_relevant_docs": False}

    from app.rag.providers import _tokenize

    effective_query = state.get("search_query") or state["question"]
    q_words = {w for w in _tokenize(effective_query) if len(w) > 3}
    doc_words = set(_tokenize(" ".join(f"{c.document_title} {c.content}" for c in hits)))
    has_keyword_overlap = bool(q_words & doc_words) if q_words else True

    if settings.LLM_PROVIDER == "openai":
        try:
            chat = get_chat_provider()
            context_preview = "\n\n".join(
                f"[{i}] {c.document_title}: {c.content[:400]}"
                for i, c in enumerate(hits[:3], start=1)
            )
            prompt = prompts.GRADE_DOCUMENTS_PROMPT.format(
                question=effective_query, context=context_preview
            )
            grade = chat.complete("You are a relevance grader.", prompt).strip().lower()
            if "no" in grade and "yes" not in grade:
                log.info(
                    "langgraph.grade_documents.llm_rejected",
                    tenant_id=str(state["tenant_id"]),
                    question=state["question"],
                )
                return {"has_relevant_docs": False}
        except Exception as exc:
            log.warning("langgraph.grade_documents.error", error=str(exc))
            return {"has_relevant_docs": has_keyword_overlap}

    return {"has_relevant_docs": has_keyword_overlap}


def generate_from_docs_node(state: AssistantState) -> dict[str, Any]:
    """Generate answer grounded in retrieved workspace documents with conversation memory."""
    selected = _trim_context(state["documents"])
    passages = [
        (index, chunk.document_title, chunk.content)
        for index, chunk in enumerate(selected, start=1)
    ]
    context_block = prompts.build_context_block(passages)

    formatted_history = state.get("formatted_history", "")
    history_block = (
        f"<conversation_history>\n{formatted_history}\n</conversation_history>\n\n"
        if formatted_history
        else ""
    )

    system_prompt = prompts.SYSTEM_PROMPT.format(workspace_name=state["tenant_name"])
    user_prompt = prompts.USER_PROMPT.format(
        context=context_block,
        history_block=history_block,
        question=state["question"],
    )

    chat = get_chat_provider()
    answer = chat.complete(system_prompt, user_prompt)

    citations = [_to_doc_citation(i, chunk) for i, chunk in enumerate(selected, start=1)]
    return {
        "answer": answer,
        "citations": citations,
        "used_context": True,
        "source_type": "workspace_docs",
    }


def regenerate_strict_node(state: AssistantState) -> dict[str, Any]:
    """Regenerate with strict anti-hallucination constraints after detection."""
    retry_count = state.get("retry_count", 0) + 1
    log.info(
        "langgraph.hallucination_reducer.regenerating",
        tenant_id=str(state["tenant_id"]),
        retry_count=retry_count,
    )
    selected = _trim_context(state["documents"])
    passages = [
        (index, chunk.document_title, chunk.content)
        for index, chunk in enumerate(selected, start=1)
    ]
    context_block = prompts.build_context_block(passages)

    formatted_history = state.get("formatted_history", "")
    history_block = (
        f"<conversation_history>\n{formatted_history}\n</conversation_history>\n\n"
        if formatted_history
        else ""
    )

    system_prompt = prompts.STRICT_GROUNDING_SYSTEM_PROMPT.format(
        workspace_name=state["tenant_name"]
    )
    user_prompt = prompts.USER_PROMPT.format(
        context=context_block,
        history_block=history_block,
        question=state["question"],
    )

    chat = get_chat_provider()
    answer = chat.complete(system_prompt, user_prompt)
    citations = [_to_doc_citation(i, chunk) for i, chunk in enumerate(selected, start=1)]

    return {
        "answer": answer,
        "citations": citations,
        "retry_count": retry_count,
        "used_context": True,
        "source_type": "workspace_docs",
    }


def grade_hallucination_node(state: AssistantState) -> dict[str, Any]:
    """Hallucination Reductor: evaluate whether candidate answer is grounded in facts."""
    if not settings.ENABLE_HALLUCINATION_CHECK:
        return {"is_grounded": True, "answers_question": True}

    answer = state.get("answer", "")
    if not answer or prompts.NO_CONTEXT_ANSWER in answer:
        return {"is_grounded": False, "answers_question": False}

    selected = state.get("documents", [])
    passages = [
        (index, chunk.document_title, chunk.content)
        for index, chunk in enumerate(selected, start=1)
    ]
    context_block = prompts.build_context_block(passages)

    chat = get_chat_provider()

    # 1. Check Hallucination (Groundedness in context)
    hallucination_prompt = prompts.HALLUCINATION_GRADER_PROMPT.format(
        context=context_block, generation=answer
    )
    hallucination_res = chat.complete(
        "You are an evaluator assessing factual consistency.", hallucination_prompt
    ).strip().lower()
    is_grounded = "yes" in hallucination_res and "no" not in hallucination_res

    # 2. Check Answer Relevance (Does it resolve the question?)
    answer_prompt = prompts.ANSWER_GRADER_PROMPT.format(
        question=state["question"], generation=answer
    )
    answer_res = chat.complete(
        "You are an evaluator assessing answer relevance.", answer_prompt
    ).strip().lower()
    answers_question = "yes" in answer_res and "no" not in answer_res

    log.info(
        "langgraph.hallucination_reducer.evaluated",
        tenant_id=str(state["tenant_id"]),
        is_grounded=is_grounded,
        answers_question=answers_question,
    )

    return {
        "is_grounded": is_grounded,
        "answers_question": answers_question,
    }


def web_search_node(state: AssistantState) -> dict[str, Any]:
    """Search the web for queries when workspace documentation is insufficient or ungrounded."""
    search_query = state.get("search_query") or state["question"]
    log.info(
        "langgraph.routing_to_web_search",
        tenant_id=str(state["tenant_id"]),
        question=search_query,
    )
    provider = get_web_search_provider()
    results = provider.search(
        search_query, max_results=settings.WEB_SEARCH_MAX_RESULTS
    )
    return {"web_results": results}


def generate_from_web_node(state: AssistantState) -> dict[str, Any]:
    """Generate answer synthesized from web search results with conversation memory."""
    web_results = state.get("web_results", [])
    web_context_block = prompts.build_web_context_block(web_results)

    formatted_history = state.get("formatted_history", "")
    history_block = (
        f"<conversation_history>\n{formatted_history}\n</conversation_history>\n\n"
        if formatted_history
        else ""
    )

    system_prompt = prompts.WEB_SEARCH_SYSTEM_PROMPT.format(
        workspace_name=state["tenant_name"]
    )
    user_prompt = prompts.WEB_SEARCH_USER_PROMPT.format(
        web_results=web_context_block,
        history_block=history_block,
        question=state["question"],
    )

    chat = get_chat_provider()
    answer = chat.complete(system_prompt, user_prompt)

    citations = [_to_web_citation(i, res) for i, res in enumerate(web_results, start=1)]
    return {
        "answer": answer,
        "citations": citations,
        "used_context": True,
        "source_type": "online_search",
    }


def no_context_node(state: AssistantState) -> dict[str, Any]:
    """Return polite fallback when neither docs nor web search can answer."""
    if state.get("allow_web_search", True) and settings.WEB_SEARCH_ENABLED:
        answer = prompts.NO_SEARCH_RESULTS_ANSWER
    else:
        answer = prompts.NO_CONTEXT_ANSWER

    return {
        "answer": answer,
        "citations": [],
        "used_context": False,
        "source_type": "none",
    }


# ---------------------------------------------------------------------------
# Routing Edges
# ---------------------------------------------------------------------------
def decide_doc_route(state: AssistantState) -> str:
    """Route to doc generation if relevant, else route to online search or fallback."""
    if state.get("has_relevant_docs"):
        return "generate_from_docs"
    if state.get("allow_web_search", True) and settings.WEB_SEARCH_ENABLED:
        return "web_search"
    return "no_context"


def decide_generation_quality(state: AssistantState) -> str:
    """Self-RAG Evaluator edge: verify grounding and question resolution."""
    is_grounded = state.get("is_grounded", True)
    answers_question = state.get("answers_question", True)
    retry_count = state.get("retry_count", 0)

    # 1. Perfect verified answer
    if is_grounded and answers_question:
        return "finish"

    # 2. Hallucination detected: attempt strict self-correction up to max retries
    if not is_grounded and retry_count < settings.MAX_REGENERATE_RETRIES:
        return "regenerate_strict"

    # 3. Grounded or retried, but doesn't resolve question -> fallback to web search if enabled
    if state.get("allow_web_search", True) and settings.WEB_SEARCH_ENABLED:
        return "web_search"

    return "no_context"


def decide_web_route(state: AssistantState) -> str:
    """Route to web generation if results were found, else no context."""
    if state.get("web_results"):
        return "generate_from_web"
    return "no_context"


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
def create_assistant_graph():
    """Build and compile the Self-RAG LangGraph workflow with sliding window memory."""
    workflow = StateGraph(AssistantState)

    # Add Nodes
    workflow.add_node("contextualize_query", contextualize_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_from_docs", generate_from_docs_node)
    workflow.add_node("grade_hallucination", grade_hallucination_node)
    workflow.add_node("regenerate_strict", regenerate_strict_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_from_web", generate_from_web_node)
    workflow.add_node("no_context", no_context_node)

    # Graph Flow
    workflow.set_entry_point("contextualize_query")
    workflow.add_edge("contextualize_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_doc_route,
        {
            "generate_from_docs": "generate_from_docs",
            "web_search": "web_search",
            "no_context": "no_context",
        },
    )

    workflow.add_edge("generate_from_docs", "grade_hallucination")
    workflow.add_edge("regenerate_strict", "grade_hallucination")

    workflow.add_conditional_edges(
        "grade_hallucination",
        decide_generation_quality,
        {
            "finish": END,
            "regenerate_strict": "regenerate_strict",
            "web_search": "web_search",
            "no_context": "no_context",
        },
    )

    workflow.add_conditional_edges(
        "web_search",
        decide_web_route,
        {
            "generate_from_web": "generate_from_web",
            "no_context": "no_context",
        },
    )

    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("no_context", END)

    return workflow.compile()


assistant_graph = create_assistant_graph()
