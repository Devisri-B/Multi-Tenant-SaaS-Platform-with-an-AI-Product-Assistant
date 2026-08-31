"""End-to-end tests for the RAG product assistant."""

from __future__ import annotations

BILLING_DOC = """\
# Billing and Refunds

Refunds are issued to the original payment method within ten business days of
the request. Contact billing support to start a refund.

Invoices are generated on the first of every month and emailed to the billing
contact listed in workspace settings.
"""

DEPLOY_DOC = """\
# Deployment Guide

Deployments run through GitHub Actions. Every merge to main builds a container
image, runs the migration job, and performs a rolling restart of the API pods.
"""


def assistant_url(tenant_id) -> str:
    return f"/api/v1/workspaces/{tenant_id}/assistant"


def seed_docs(client, actor, docs):
    for title, content in docs.items():
        response = client.post(
            f"/api/v1/workspaces/{actor.tenant_id}/documents",
            headers=actor.headers,
            json={"title": title, "content": content},
        )
        assert response.status_code == 201


def test_ask_without_documents_and_no_web_says_so(client, owner):
    response = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How do refunds work?", "allow_web_search": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["used_context"] is False
    assert body["citations"] == []
    assert body["source_type"] == "none"


def test_ask_without_documents_routes_online(client, owner):
    response = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "What is Python asyncio?", "allow_web_search": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["used_context"] is True
    assert body["source_type"] == "online_search"
    assert len(body["citations"]) > 0
    assert body["citations"][0]["url"] is not None
    assert body["citations"][0]["source_type"] == "web"


def test_ask_returns_grounded_answer_with_citations(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC, "Deployment": DEPLOY_DOC})
    response = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    )
    body = response.json()
    assert body["used_context"] is True
    assert body["citations"]
    assert body["citations"][0]["document_title"] in {"Billing", "Deployment"}
    assert body["latency_ms"] >= 0


def test_citations_carry_scores_and_excerpts(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    body = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "refunds payment method"},
    ).json()
    citation = body["citations"][0]
    assert 0.0 <= citation["score"] <= 1.0
    assert citation["excerpt"]
    assert "index" not in citation


def test_viewer_can_use_the_assistant(client, owner, viewer):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    response = client.post(
        f"{assistant_url(viewer.tenant_id)}/ask",
        headers=viewer.headers,
        json={"question": "How long do refunds take?"},
    )
    assert response.status_code == 200


def test_conversation_is_created_and_reused(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    first = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    ).json()
    second = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={
            "question": "And who do I contact?",
            "conversation_id": first["conversation_id"],
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]

    detail = client.get(
        f"{assistant_url(owner.tenant_id)}/conversations/{first['conversation_id']}",
        headers=owner.headers,
    ).json()
    assert len(detail["messages"]) == 4
    assert detail["messages"][0]["role"] == "user"


def test_conversation_title_is_derived_from_first_question(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    conversation_id = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    ).json()["conversation_id"]
    detail = client.get(
        f"{assistant_url(owner.tenant_id)}/conversations/{conversation_id}",
        headers=owner.headers,
    ).json()
    assert detail["title"].startswith("How long do refunds take")


def test_conversations_are_listed_per_user(client, owner, member):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    )
    owner_view = client.get(
        f"{assistant_url(owner.tenant_id)}/conversations", headers=owner.headers
    ).json()
    member_view = client.get(
        f"{assistant_url(member.tenant_id)}/conversations", headers=member.headers
    ).json()
    assert owner_view["total"] == 1
    assert member_view["total"] == 0


def test_search_returns_ranked_hits(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC, "Deployment": DEPLOY_DOC})
    hits = client.post(
        f"{assistant_url(owner.tenant_id)}/search",
        headers=owner.headers,
        json={"query": "github actions rolling restart", "top_k": 3},
    ).json()
    assert hits
    assert hits[0]["document_title"] == "Deployment"
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_never_crosses_tenants(client, owner, other_owner):
    """The isolation test that matters: same question, different workspace."""
    seed_docs(client, owner, {"Billing": BILLING_DOC})

    hits = client.post(
        f"{assistant_url(other_owner.tenant_id)}/search",
        headers=other_owner.headers,
        json={"query": "refunds original payment method", "top_k": 5},
    ).json()
    assert hits == []

    answer = client.post(
        f"{assistant_url(other_owner.tenant_id)}/ask",
        headers=other_owner.headers,
        json={"question": "How long do refunds take?", "allow_web_search": False},
    ).json()
    assert answer["used_context"] is False
    assert "ten business days" not in answer["answer"]


def test_conversation_from_another_tenant_is_404(client, owner, other_owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    conversation_id = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    ).json()["conversation_id"]
    response = client.get(
        f"{assistant_url(other_owner.tenant_id)}/conversations/{conversation_id}",
        headers=other_owner.headers,
    )
    assert response.status_code == 404


def test_question_is_length_validated(client, owner):
    response = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "hi"},
    )
    assert response.status_code == 422


def test_assistant_requires_membership(client, owner, other_owner):
    response = client.post(
        f"{assistant_url(other_owner.tenant_id)}/ask",
        headers={"Authorization": f"Bearer {owner.token}"},
        json={"question": "How long do refunds take?"},
    )
    assert response.status_code == 403


def test_multi_turn_conversation_with_sliding_window_memory(client, owner):
    seed_docs(client, owner, {"Billing": BILLING_DOC})
    # Turn 1
    res1 = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "Tell me about the refund policy"},
    ).json()
    conv_id = res1["conversation_id"]

    # Turn 2: Follow-up question relying on memory / query contextualization
    res2 = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How many days does it take?", "conversation_id": conv_id},
    ).json()
    assert res2["conversation_id"] == conv_id
    assert res2["used_context"] is True
    assert "ten business days" in res2["answer"]

    # Check conversation history has all messages stored in order
    conv = client.get(
        f"{assistant_url(owner.tenant_id)}/conversations/{conv_id}",
        headers=owner.headers,
    ).json()
    assert len(conv["messages"]) == 4
