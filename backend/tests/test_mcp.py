"""Tests for the Model Context Protocol (MCP) Self-RAG server."""

from __future__ import annotations

import json
import uuid

import pytest

from app.mcp import server as mcp_server
from app.rag.chain import AnswerResult
from app.rag.retriever import RetrievedChunk


def test_mcp_tools_registered():
    """Verify that all three expected tools are registered with FastMCP."""
    tools = mcp_server.mcp._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    assert "list_workspaces" in tool_names
    assert "semantic_search_chunks" in tool_names
    assert "self_rag_query" in tool_names


def test_list_workspaces_empty():
    """Verify list_workspaces returns empty list when no tenants exist."""
    res_str = mcp_server.list_workspaces()
    data = json.loads(res_str)
    assert "total_workspaces" in data
    assert "workspaces" in data
    assert "guidance" in data


def test_list_workspaces_with_tenant(owner):
    """Verify list_workspaces includes active tenants."""
    res_str = mcp_server.list_workspaces()
    data = json.loads(res_str)
    assert data["total_workspaces"] >= 1
    names = [w["name"] for w in data["workspaces"]]
    assert "Acme Docs" in names


def test_semantic_search_chunks_tenant_not_found():
    """Verify semantic_search_chunks returns structured error when tenant does not exist."""
    non_existent = str(uuid.uuid4())
    res_str = mcp_server.semantic_search_chunks(
        query="test query", tenant_id=non_existent
    )
    data = json.loads(res_str)
    assert data.get("error") == "TenantNotFound"


def test_semantic_search_chunks_success(owner, monkeypatch):
    """Verify semantic_search_chunks returns formatted vector chunks."""
    fake_chunk = RetrievedChunk(
        chunk_id=10,
        document_id=20,
        document_title="Security Policy",
        ordinal=1,
        content="All data is encrypted at rest and in transit.",
        score=0.9123,
    )
    monkeypatch.setattr(
        "app.rag.chain.semantic_search", lambda *args, **kwargs: [fake_chunk]
    )

    res_str = mcp_server.semantic_search_chunks(
        query="encryption", tenant_id=str(owner.tenant_id), top_k=3
    )
    data = json.loads(res_str)
    assert data["total_hits"] == 1
    assert data["workspace_id"] == str(owner.tenant_id)
    assert data["chunks"][0]["document_title"] == "Security Policy"
    assert data["chunks"][0]["score"] == 0.9123
    assert "encrypted" in data["chunks"][0]["content"]


def test_self_rag_query_tenant_not_found():
    """Verify self_rag_query returns structured error when tenant does not exist."""
    non_existent = str(uuid.uuid4())
    res_str = mcp_server.self_rag_query(
        question="What is the refund policy?", tenant_id=non_existent
    )
    data = json.loads(res_str)
    assert data.get("error") == "TenantNotFound"


def test_self_rag_query_success(owner, monkeypatch):
    """Verify self_rag_query returns grounded answer and citations."""
    fake_answer = AnswerResult(
        answer="Refunds are processed within 10 business days.",
        citations=[
            {
                "document_title": "Billing FAQ",
                "excerpt": "Refunds are processed within 10 business days.",
                "url": None,
                "score": 0.88,
                "source_type": "document",
            }
        ],
        latency_ms=120,
        used_context=True,
        source_type="workspace_docs",
    )
    monkeypatch.setattr(
        "app.rag.chain.answer_question", lambda *args, **kwargs: fake_answer
    )

    res_str = mcp_server.self_rag_query(
        question="What is the refund timeline?",
        tenant_id=str(owner.tenant_id),
    )
    data = json.loads(res_str)
    assert data["answer"] == "Refunds are processed within 10 business days."
    assert data["source_type"] == "workspace_docs"
    assert data["used_context"] is True
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_title"] == "Billing FAQ"


@pytest.mark.asyncio
async def test_mcp_server_async_tool_call(owner):
    """Verify calling the tool through FastMCP's call_tool protocol method."""
    results = await mcp_server.mcp.call_tool("list_workspaces", {})
    assert len(results) == 1
    payload = json.loads(results[0].text)
    assert payload["total_workspaces"] >= 1


# ---------------------------------------------------------------------------
# Tenant allowlist (MCP_ALLOWED_TENANT_IDS)
# ---------------------------------------------------------------------------
@pytest.fixture
def allow_only_owner(owner, monkeypatch):
    """Restrict the MCP server to the ``owner`` fixture's workspace."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "MCP_ALLOWED_TENANT_IDS", [owner.tenant_id])
    return owner


def test_allowlist_unset_exposes_every_active_tenant(owner, other_owner):
    data = json.loads(mcp_server.list_workspaces())
    names = {w["name"] for w in data["workspaces"]}
    assert {"Acme Docs", "Globex Docs"} <= names


def test_allowlist_hides_other_tenants_from_list_workspaces(allow_only_owner, other_owner):
    data = json.loads(mcp_server.list_workspaces())
    assert data["total_workspaces"] == 1
    assert data["workspaces"][0]["id"] == str(allow_only_owner.tenant_id)
    assert "Globex Docs" not in json.dumps(data)


def test_allowlist_blocks_explicit_id_outside_allowlist(allow_only_owner, other_owner, monkeypatch):
    """A real but disallowed tenant is indistinguishable from a nonexistent one."""
    monkeypatch.setattr(
        "app.rag.chain.semantic_search",
        lambda *a, **k: pytest.fail("retrieval must not run for a disallowed tenant"),
    )
    monkeypatch.setattr(
        "app.rag.chain.answer_question",
        lambda *a, **k: pytest.fail("pipeline must not run for a disallowed tenant"),
    )

    search = json.loads(
        mcp_server.semantic_search_chunks(query="q", tenant_id=str(other_owner.tenant_id))
    )
    query = json.loads(
        mcp_server.self_rag_query(question="q", tenant_id=str(other_owner.tenant_id))
    )
    assert search["error"] == "TenantNotFound"
    assert query["error"] == "TenantNotFound"
    assert "Globex" not in search["message"] + query["message"]


def test_allowlist_fallback_resolves_to_allowed_tenant_not_oldest(
    db, other_owner, owner, monkeypatch
):
    """With no tenant_id passed, fall back to the first *allowed* tenant."""
    from app.core.config import settings

    # ``other_owner`` was created first, so it is the oldest tenant; the
    # allowlist must still win over creation order.
    monkeypatch.setattr(settings, "MCP_ALLOWED_TENANT_IDS", [owner.tenant_id])
    monkeypatch.setattr("app.rag.chain.semantic_search", lambda *a, **k: [])

    data = json.loads(mcp_server.semantic_search_chunks(query="q"))
    assert data["workspace_id"] == str(owner.tenant_id)
    assert data["workspace_name"] == "Acme Docs"


def test_allowlist_still_allows_listed_tenant(allow_only_owner, monkeypatch):
    monkeypatch.setattr("app.rag.chain.semantic_search", lambda *a, **k: [])
    data = json.loads(
        mcp_server.semantic_search_chunks(query="q", tenant_id=str(allow_only_owner.tenant_id))
    )
    assert data["workspace_id"] == str(allow_only_owner.tenant_id)
