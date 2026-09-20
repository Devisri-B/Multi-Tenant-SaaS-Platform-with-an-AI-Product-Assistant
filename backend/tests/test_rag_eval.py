"""Unit tests for deterministic RAG evaluation and user friction detection."""

from __future__ import annotations

from app.rag.eval import (
    detect_query_friction,
    extract_entities,
    extract_numbers,
    verify_citation_rate,
    verify_entity_accuracy,
    verify_numeric_accuracy,
)


def test_extract_numbers():
    text = "We offer a 14-day trial, $25 monthly subscription, and 99.9% SLA uptime."
    numbers = extract_numbers(text)
    assert "14" in numbers
    assert "25" in numbers
    assert "99.9" in numbers


def test_numeric_accuracy_grounded():
    context = "Refunds are processed within 10 business days. Late fees are $15 or 5%."
    answer = "Your refund will arrive in 10 days. The fee is $15 or 5%."
    res = verify_numeric_accuracy(answer, context)
    assert res["numeric_accuracy_score"] == 1.0
    assert len(res["unsupported_numbers"]) == 0
    assert "10" in res["supported_numbers"]
    assert "15" in res["supported_numbers"]


def test_numeric_accuracy_hallucinated():
    context = "Refunds are processed within 10 business days."
    answer = "Your refund will arrive within 30 days and cost $50."
    res = verify_numeric_accuracy(answer, context)
    assert res["numeric_accuracy_score"] == 0.0
    assert "30" in res["unsupported_numbers"]
    assert "50" in res["unsupported_numbers"]


def test_numeric_accuracy_no_numbers():
    context = "We support credit cards and wire transfers."
    answer = "You can pay with credit card."
    res = verify_numeric_accuracy(answer, context)
    assert res["numeric_accuracy_score"] == 1.0
    assert res["total_numbers"] == 0


def test_extract_entities_filters_stopwords():
    text = "However, PostgreSQL and GitHub Actions are used by Nimbus."
    entities = extract_entities(text)
    assert "However" not in entities
    assert "PostgreSQL" in entities
    assert "GitHub Actions" in entities
    assert "Nimbus" in entities


def test_entity_accuracy_grounded():
    context = "Nimbus runs on PostgreSQL with Row Level Security (RLS) in AWS."
    answer = "Nimbus uses PostgreSQL and RLS for secure multi-tenancy on AWS."
    res = verify_entity_accuracy(answer, context)
    assert res["entity_accuracy_score"] == 1.0
    assert "PostgreSQL" in res["supported_entities"]
    assert "RLS" in res["supported_entities"]
    assert len(res["unsupported_entities"]) == 0


def test_entity_accuracy_hallucinated():
    context = "Nimbus runs on PostgreSQL with Row Level Security in AWS."
    answer = "Nimbus uses MongoDB, GraphQL, and Azure Kubernetes Service."
    res = verify_entity_accuracy(answer, context)
    assert res["entity_accuracy_score"] < 0.5
    assert "MongoDB" in res["unsupported_entities"]
    assert "GraphQL" in res["unsupported_entities"]


def test_citation_verification_rate_authentic():
    context = "Refunds are issued to the original payment method within ten business days."
    citations = [
        {
            "chunk_id": "c1",
            "document_title": "Billing",
            "excerpt": "Refunds are issued to the original payment method...",
        }
    ]
    res = verify_citation_rate(citations, full_context=context)
    assert res["citation_verification_rate"] == 1.0
    assert res["verified_citations"] == 1
    assert len(res["unverified_citations"]) == 0


def test_citation_verification_rate_fabricated():
    context = "Refunds are issued to the original payment method within ten business days."
    citations = [
        {
            "chunk_id": "c1",
            "document_title": "Billing",
            "excerpt": "Completely made up text that does not exist in the document.",
        }
    ]
    res = verify_citation_rate(citations, full_context=context)
    assert res["citation_verification_rate"] == 0.0
    assert res["verified_citations"] == 0
    assert len(res["unverified_citations"]) == 1


def test_citation_verification_empty():
    res = verify_citation_rate([])
    assert res["citation_verification_rate"] == 1.0
    assert res["total_citations"] == 0


def test_detect_query_friction_frustration_phrases():
    res = detect_query_friction(
        current_q="That's not what I asked, please answer my question",
        prev_q="How do refunds work?",
        elapsed_seconds=15.0,
    )
    assert res["is_friction"] is True
    assert res["friction_reason"] == "frustration_phrase"
    assert res["ground_truth_score"] == 0.0


def test_detect_query_friction_rapid_rephrase():
    res = detect_query_friction(
        current_q="How do refunds work in this system?",
        prev_q="How do refunds work?",
        elapsed_seconds=12.0,
    )
    assert res["is_friction"] is True
    assert res["friction_reason"] == "rapid_rephrase"
    assert res["similarity"] >= 0.50
    assert res["ground_truth_score"] == 0.0


def test_detect_query_friction_constructive_followup():
    res = detect_query_friction(
        current_q="How do I configure SAML SSO authentication?",
        prev_q="How do refunds work?",
        elapsed_seconds=18.0,
    )
    assert res["is_friction"] is False
    assert res["is_successful_followup"] is True
    assert res["ground_truth_score"] == 1.0


def test_detect_query_friction_delayed_query():
    # If same query is asked after 120 seconds, it's not a rapid frustrated rephrase
    res = detect_query_friction(
        current_q="How do refunds work in this system?",
        prev_q="How do refunds work?",
        elapsed_seconds=120.0,
    )
    assert res["is_friction"] is False
    assert res["ground_truth_score"] == 1.0
