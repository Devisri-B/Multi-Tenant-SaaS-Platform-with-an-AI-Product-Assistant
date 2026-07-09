"""Registration, login, refresh and session endpoints."""

from __future__ import annotations

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"


def _payload(**overrides) -> dict:
    body = {
        "email": "new@acme.io",
        "password": "Sup3rSecret!",
        "full_name": "New User",
        "workspace_name": "New Workspace",
    }
    body.update(overrides)
    return body


def test_register_creates_user_and_workspace(client):
    response = client.post(REGISTER, json=_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["user"]["email"] == "new@acme.io"
    assert body["tokens"]["token_type"] == "bearer"
    assert body["tenant_id"]


def test_register_rejects_duplicate_email(client):
    client.post(REGISTER, json=_payload())
    response = client.post(REGISTER, json=_payload(workspace_name="Another"))
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


def test_register_rejects_weak_password(client):
    response = client.post(REGISTER, json=_payload(password="short"))
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_login_returns_token_pair(client, owner):
    response = client.post(
        LOGIN, json={"email": "owner@acme.io", "password": "OwnerPassw0rd"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_with_bad_password_is_401(client, owner):
    response = client.post(
        LOGIN, json={"email": "owner@acme.io", "password": "nope"}
    )
    assert response.status_code == 401


def test_login_for_unknown_email_is_401(client):
    response = client.post(
        LOGIN, json={"email": "ghost@acme.io", "password": "whatever"}
    )
    assert response.status_code == 401


def test_refresh_issues_new_access_token(client, owner):
    login = client.post(
        LOGIN, json={"email": "owner@acme.io", "password": "OwnerPassw0rd"}
    ).json()
    response = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_refresh_rejects_an_access_token(client, owner):
    response = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": owner.token}
    )
    assert response.status_code == 401


def test_me_lists_memberships(client, owner):
    response = client.get("/api/v1/auth/me", headers=owner.headers)
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["email"] == "owner@acme.io"
    assert body["memberships"][0]["role"] == "owner"


def test_me_requires_a_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_me_rejects_a_garbage_token(client):
    response = client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert response.status_code == 401


def test_password_change_then_login(client, owner):
    response = client.post(
        "/api/v1/auth/password",
        headers=owner.headers,
        json={"current_password": "OwnerPassw0rd", "new_password": "N3wPassword!"},
    )
    assert response.status_code == 200
    assert (
        client.post(
            LOGIN, json={"email": "owner@acme.io", "password": "N3wPassword!"}
        ).status_code
        == 200
    )


def test_password_change_requires_current_password(client, owner):
    response = client.post(
        "/api/v1/auth/password",
        headers=owner.headers,
        json={"current_password": "wrong", "new_password": "N3wPassword!"},
    )
    assert response.status_code == 401
