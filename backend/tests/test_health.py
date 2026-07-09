"""Smoke tests for the service probes."""

from __future__ import annotations


def test_root_returns_service_metadata(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert body["api"].startswith("/api/")


def test_health_is_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_database_up(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "up"
    assert body["llm_provider"] == "fake"


def test_request_id_header_is_echoed(client):
    response = client.get("/api/v1/health", headers={"X-Request-Id": "abc-123"})
    assert response.headers["X-Request-Id"] == "abc-123"
    assert "X-Response-Time-Ms" in response.headers


def test_security_headers_present(client):
    response = client.get("/api/v1/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
