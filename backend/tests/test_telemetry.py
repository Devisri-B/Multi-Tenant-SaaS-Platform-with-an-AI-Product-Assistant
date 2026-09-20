"""Integration tests for telemetry events, RAG evaluation, and quality metrics."""

from __future__ import annotations


def assistant_url(tenant_id) -> str:
    return f"/api/v1/workspaces/{tenant_id}/assistant"


def test_record_copy_telemetry_event(client, owner):
    url = f"{assistant_url(owner.tenant_id)}/telemetry"
    response = client.post(
        url,
        headers=owner.headers,
        json={
            "event_type": "copy",
            "event_data": {"char_count": 150},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["event_type"] == "copy"
    assert body["tenant_id"] == str(owner.tenant_id)
    assert body["event_data"]["char_count"] == 150


def test_record_citation_click_event(client, owner):
    url = f"{assistant_url(owner.tenant_id)}/telemetry"
    response = client.post(
        url,
        headers=owner.headers,
        json={
            "event_type": "citation_click",
            "event_data": {"document_title": "Billing FAQ", "ordinal": 1},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["event_type"] == "citation_click"
    assert body["event_data"]["document_title"] == "Billing FAQ"


def test_record_feedback_events(client, owner):
    url = f"{assistant_url(owner.tenant_id)}/telemetry"
    r1 = client.post(
        url,
        headers=owner.headers,
        json={"event_type": "feedback_positive", "event_data": {"rating": 1}},
    )
    assert r1.status_code == 201

    r2 = client.post(
        url,
        headers=owner.headers,
        json={"event_type": "feedback_negative", "event_data": {"reason": "too long"}},
    )
    assert r2.status_code == 201


def test_ask_returns_deterministic_evaluation(client, owner):
    doc_res = client.post(
        f"/api/v1/workspaces/{owner.tenant_id}/documents",
        headers=owner.headers,
        json={
            "title": "Refunds",
            "content": "Refunds are processed within 10 business days for $25.",
        },
    )
    assert doc_res.status_code == 201

    ask_res = client.post(
        f"{assistant_url(owner.tenant_id)}/ask",
        headers=owner.headers,
        json={"question": "How long do refunds take?"},
    )
    assert ask_res.status_code == 200
    body = ask_res.json()
    assert "evaluation" in body
    eval_data = body["evaluation"]
    assert eval_data is not None
    assert "numeric_accuracy_score" in eval_data
    assert "entity_accuracy_score" in eval_data
    assert "citation_verification_rate" in eval_data
    assert eval_data["numeric_accuracy_score"] >= 0.0


def test_ask_detects_friction_sentiment(client, owner):
    ask_url = f"{assistant_url(owner.tenant_id)}/ask"
    # First question
    r1 = client.post(
        ask_url,
        headers=owner.headers,
        json={"question": "What is the refund turnaround time?"},
    )
    assert r1.status_code == 200
    conv_id = r1.json()["conversation_id"]

    # Immediate follow-up with frustration phrase
    r2 = client.post(
        ask_url,
        headers=owner.headers,
        json={
            "question": "That's not what I asked, answer about refunds",
            "conversation_id": conv_id,
        },
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["friction_detected"] is True
    assert body2["friction_reason"] == "frustration_phrase"
    assert body2["ground_truth_score"] == 0.0


def test_metrics_endpoint_aggregates_stats(client, owner):
    telemetry_url = f"{assistant_url(owner.tenant_id)}/telemetry"
    # Record copy event
    client.post(telemetry_url, headers=owner.headers, json={"event_type": "copy"})
    # Record citation click
    client.post(telemetry_url, headers=owner.headers, json={"event_type": "citation_click"})

    # Query metrics
    metrics_url = f"{assistant_url(owner.tenant_id)}/metrics"
    metrics_res = client.get(metrics_url, headers=owner.headers)
    assert metrics_res.status_code == 200
    m = metrics_res.json()
    assert m["copy_count"] >= 1
    assert m["citation_clicks_count"] >= 1
    assert "numeric_accuracy_avg" in m
    assert "ground_truth_score" in m
