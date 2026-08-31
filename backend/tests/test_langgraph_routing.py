"""Unit and integration tests for the LangGraph adaptive RAG workflow."""

from __future__ import annotations

import uuid
from unittest.mock import patch

from app.core.config import settings
from app.rag.graph import (
    AssistantState,
    decide_doc_route,
    decide_web_route,
    grade_documents_node,
)
from app.rag.retriever import RetrievedChunk
from app.rag.web_search import (
    DuckDuckGoWebSearch,
    FakeWebSearch,
    TavilyWebSearch,
    WebSearchResult,
    get_web_search_provider,
    reset_web_search_provider_cache,
)


def seed_document(client, actor, title: str, content: str):
    response = client.post(
        f"/api/v1/workspaces/{actor.tenant_id}/documents",
        headers=actor.headers,
        json={"title": title, "content": content},
    )
    assert response.status_code == 201


def test_langgraph_routes_to_docs_when_matching_docs_exist(client, owner):
    """When relevant documents exist, LangGraph routes to generate_from_docs."""
    seed_document(
        client,
        owner,
        "Refund Policy",
        "Refunds are processed within 5 business days of request submission.",
    )
    response = client.post(
        f"/api/v1/workspaces/{owner.tenant_id}/assistant/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take to process?", "allow_web_search": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "workspace_docs"
    assert data["used_context"] is True
    assert len(data["citations"]) > 0
    assert data["citations"][0]["document_title"] == "Refund Policy"
    assert data["citations"][0]["source_type"] == "document"


def test_langgraph_routes_to_web_search_when_no_docs_exist(client, owner):
    """When no workspace documents exist, LangGraph routes to online search."""
    response = client.post(
        f"/api/v1/workspaces/{owner.tenant_id}/assistant/ask",
        headers=owner.headers,
        json={"question": "What is the capital of France?", "allow_web_search": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "online_search"
    assert data["used_context"] is True
    assert len(data["citations"]) > 0
    assert data["citations"][0]["source_type"] == "web"
    assert data["citations"][0]["url"] is not None
    assert "online search" in data["answer"].lower()


def test_langgraph_routes_to_web_search_when_docs_are_irrelevant(client, owner, db):
    """When documents in the workspace have no relevance, LangGraph routes to web."""
    seed_document(
        client,
        owner,
        "Kubernetes Deployment",
        "Kubernetes cluster configuration, pods, replica sets, and ingress controllers.",
    )
    response = client.post(
        f"/api/v1/workspaces/{owner.tenant_id}/assistant/ask",
        headers=owner.headers,
        json={
            "question": "How to brew espresso coffee with an aeropress?",
            "allow_web_search": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # If the score or keyword match is low/irrelevant, it routes to online search
    assert data["source_type"] == "online_search"
    assert data["citations"][0]["source_type"] == "web"


def test_langgraph_fallback_when_web_search_disabled(client, owner):
    """When web search is disabled and no docs exist, LangGraph routes to no_context."""
    response = client.post(
        f"/api/v1/workspaces/{owner.tenant_id}/assistant/ask",
        headers=owner.headers,
        json={"question": "What is the capital of France?", "allow_web_search": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "none"
    assert data["used_context"] is False
    assert data["citations"] == []
    assert "could not find anything" in data["answer"].lower()


def test_decide_doc_route_logic():
    """Unit test for the doc routing conditional edge."""
    # Relevant docs present
    state_with_docs: AssistantState = {"has_relevant_docs": True, "allow_web_search": True}
    assert decide_doc_route(state_with_docs) == "generate_from_docs"

    # No relevant docs, web search enabled
    state_no_docs: AssistantState = {"has_relevant_docs": False, "allow_web_search": True}
    assert decide_doc_route(state_no_docs) == "web_search"

    # No relevant docs, web search disabled
    state_web_disabled: AssistantState = {"has_relevant_docs": False, "allow_web_search": False}
    assert decide_doc_route(state_web_disabled) == "no_context"


def test_decide_web_route_logic():
    """Unit test for the web routing conditional edge."""
    # Results present
    state_with_web: AssistantState = {
        "web_results": [WebSearchResult(title="T", url="U", snippet="S")]
    }
    assert decide_web_route(state_with_web) == "generate_from_web"

    # No results
    state_empty_web: AssistantState = {"web_results": []}
    assert decide_web_route(state_empty_web) == "no_context"


def test_grade_documents_low_score():
    """grade_documents_node rejects chunks below relevance threshold."""
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Title",
        ordinal=0,
        content="Some irrelevant content",
        score=0.01,
    )
    state: AssistantState = {
        "tenant_id": uuid.uuid4(),
        "question": "Unrelated question",
        "documents": [chunk],
    }
    with patch.object(settings, "DOCUMENT_RELEVANCE_THRESHOLD", 0.5):
        result = grade_documents_node(state)
        assert result["has_relevant_docs"] is False


def test_fake_web_search_provider():
    """FakeWebSearch returns deterministic, structured results."""
    provider = FakeWebSearch()
    results = provider.search("PostgreSQL connection pooling", max_results=3)
    assert len(results) > 0
    assert all(isinstance(r, WebSearchResult) for r in results)
    assert results[0].url.startswith("https://")
    assert "PostgreSQL connection pooling" in results[0].title


def test_duckduckgo_web_search_empty_query():
    """DuckDuckGoWebSearch safely handles empty or whitespace queries."""
    provider = DuckDuckGoWebSearch()
    assert provider.search("   ") == []


def test_tavily_web_search_without_key_falls_back():
    """Tavily search without API key falls back to DuckDuckGo / Fake."""
    provider = TavilyWebSearch(api_key=None)
    results = provider.search("test query", max_results=2)
    assert isinstance(results, list)


def test_web_search_provider_cache():
    """Provider cache returns consistent instance and can be reset."""
    p1 = get_web_search_provider()
    p2 = get_web_search_provider()
    assert p1 is p2
    reset_web_search_provider_cache()


def test_decide_generation_quality_grounded():
    """Self-RAG finishes when answer is grounded and answers question."""
    from app.rag.graph import decide_generation_quality

    state: AssistantState = {
        "is_grounded": True,
        "answers_question": True,
        "retry_count": 0,
        "allow_web_search": True,
    }
    assert decide_generation_quality(state) == "finish"


def test_decide_generation_quality_hallucination_retries():
    """Self-RAG triggers strict regeneration when hallucination is detected."""
    from app.rag.graph import decide_generation_quality

    state: AssistantState = {
        "is_grounded": False,
        "answers_question": True,
        "retry_count": 0,
        "allow_web_search": True,
    }
    assert decide_generation_quality(state) == "regenerate_strict"


def test_decide_generation_quality_retries_exhausted_routes_to_web():
    """When retries are exhausted, Self-RAG routes to online search."""
    from app.rag.graph import decide_generation_quality

    state: AssistantState = {
        "is_grounded": False,
        "answers_question": True,
        "retry_count": 2,
        "allow_web_search": True,
    }
    assert decide_generation_quality(state) == "web_search"
