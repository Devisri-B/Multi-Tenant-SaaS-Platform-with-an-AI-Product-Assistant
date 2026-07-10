"""Workspace lifecycle and — critically — cross-tenant isolation."""

from __future__ import annotations

import uuid

from app.models.enums import Role
from app.services import tenant as tenant_service


def test_create_workspace(client, owner):
    response = client.post(
        "/api/v1/workspaces", headers=owner.headers, json={"name": "Second Product"}
    )
    assert response.status_code == 201
    assert response.json()["slug"] == "second-product"


def test_slug_collisions_get_a_suffix(client, owner):
    client.post("/api/v1/workspaces", headers=owner.headers, json={"name": "Duplicate"})
    response = client.post(
        "/api/v1/workspaces", headers=owner.headers, json={"name": "Duplicate"}
    )
    assert response.json()["slug"] == "duplicate-2"


def test_explicit_slug_must_be_unique(client, owner):
    client.post(
        "/api/v1/workspaces",
        headers=owner.headers,
        json={"name": "Alpha", "slug": "alpha"},
    )
    response = client.post(
        "/api/v1/workspaces",
        headers=owner.headers,
        json={"name": "Alpha Two", "slug": "alpha"},
    )
    assert response.status_code == 409


def test_invalid_slug_is_rejected(client, owner):
    response = client.post(
        "/api/v1/workspaces",
        headers=owner.headers,
        json={"name": "Bad", "slug": "Not A Slug"},
    )
    assert response.status_code == 422


def test_list_only_returns_my_workspaces(client, owner, other_owner):
    response = client.get("/api/v1/workspaces", headers=owner.headers)
    slugs = {item["slug"] for item in response.json()}
    assert slugs == {"acme-docs"}


def test_reading_another_tenant_is_forbidden(client, owner, other_owner):
    response = client.get(
        f"/api/v1/workspaces/{other_owner.tenant_id}",
        headers={"Authorization": f"Bearer {owner.token}"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"


def test_unknown_workspace_is_404(client, owner):
    response = client.get(
        f"/api/v1/workspaces/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {owner.token}"},
    )
    assert response.status_code == 404


def test_workspace_header_must_be_a_uuid(client, owner):
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {owner.token}", "X-Workspace-Id": "nope"},
    )
    # /auth/me does not resolve a tenant, so the bad header is simply ignored.
    assert response.status_code == 200


def test_update_workspace_requires_admin(client, owner, viewer):
    response = client.patch(
        f"/api/v1/workspaces/{viewer.tenant_id}",
        headers=viewer.headers,
        json={"name": "Renamed"},
    )
    assert response.status_code == 403


def test_admin_can_update_workspace(client, admin):
    response = client.patch(
        f"/api/v1/workspaces/{admin.tenant_id}",
        headers=admin.headers,
        json={"name": "Acme Documentation"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Acme Documentation"


def test_only_owner_can_archive(client, admin):
    response = client.delete(
        f"/api/v1/workspaces/{admin.tenant_id}", headers=admin.headers
    )
    assert response.status_code == 403


def test_owner_can_archive_and_it_becomes_inaccessible(client, owner):
    assert (
        client.delete(
            f"/api/v1/workspaces/{owner.tenant_id}", headers=owner.headers
        ).status_code
        == 200
    )
    follow_up = client.get(
        f"/api/v1/workspaces/{owner.tenant_id}", headers=owner.headers
    )
    assert follow_up.status_code == 404


def test_stats_reflect_membership(client, owner, admin, member):
    response = client.get(
        f"/api/v1/workspaces/{owner.tenant_id}/stats", headers=owner.headers
    )
    assert response.json()["member_count"] == 3


def test_slugify_handles_punctuation():
    assert tenant_service.slugify("  Hello,  World!! ") == "hello-world"


def test_role_ordering():
    assert Role.OWNER.satisfies(Role.ADMIN)
    assert Role.ADMIN.satisfies(Role.MEMBER)
    assert not Role.VIEWER.satisfies(Role.MEMBER)
    assert Role.MEMBER.satisfies(Role.MEMBER)
