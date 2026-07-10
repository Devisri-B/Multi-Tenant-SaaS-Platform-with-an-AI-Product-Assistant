"""Membership management and RBAC guards."""

from __future__ import annotations


def members_url(tenant_id) -> str:
    return f"/api/v1/workspaces/{tenant_id}/members"


def test_owner_can_invite(client, owner):
    response = client.post(
        members_url(owner.tenant_id),
        headers=owner.headers,
        json={"email": "invitee@acme.io", "role": "member"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["invited_new_user"] is True
    assert body["temporary_password"]
    assert body["member"]["role"] == "member"


def test_member_cannot_invite(client, member):
    response = client.post(
        members_url(member.tenant_id),
        headers=member.headers,
        json={"email": "nope@acme.io", "role": "member"},
    )
    assert response.status_code == 403


def test_duplicate_invite_is_conflict(client, owner, member):
    response = client.post(
        members_url(owner.tenant_id),
        headers=owner.headers,
        json={"email": "member@acme.io", "role": "member"},
    )
    assert response.status_code == 409


def test_seat_limit_is_enforced(client, db, owner):
    owner.tenant.seat_limit = 2
    db.commit()
    client.post(
        members_url(owner.tenant_id),
        headers=owner.headers,
        json={"email": "one@acme.io", "role": "member"},
    )
    response = client.post(
        members_url(owner.tenant_id),
        headers=owner.headers,
        json={"email": "two@acme.io", "role": "member"},
    )
    assert response.status_code == 403
    assert "seat limit" in response.json()["message"]


def test_list_members_is_paginated(client, owner, admin, member, viewer):
    response = client.get(
        members_url(owner.tenant_id), headers=owner.headers, params={"size": 2}
    )
    body = response.json()
    assert body["total"] == 4
    assert len(body["items"]) == 2


def test_viewer_can_list_but_not_mutate(client, owner, viewer):
    assert client.get(members_url(viewer.tenant_id), headers=viewer.headers).status_code == 200
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = next(i for i in listing["items"] if i["email"] == "viewer@acme.io")
    response = client.patch(
        f"{members_url(viewer.tenant_id)}/{target['id']}",
        headers=viewer.headers,
        json={"role": "owner"},
    )
    assert response.status_code == 403


def test_admin_can_promote_a_member(client, owner, admin, member):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = next(i for i in listing["items"] if i["email"] == "member@acme.io")
    response = client.patch(
        f"{members_url(admin.tenant_id)}/{target['id']}",
        headers=admin.headers,
        json={"role": "admin"},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_admin_cannot_create_an_owner(client, owner, admin, member):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = next(i for i in listing["items"] if i["email"] == "member@acme.io")
    response = client.patch(
        f"{members_url(admin.tenant_id)}/{target['id']}",
        headers=admin.headers,
        json={"role": "owner"},
    )
    assert response.status_code == 403


def test_admin_cannot_demote_an_owner(client, owner, admin):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = next(i for i in listing["items"] if i["email"] == "owner@acme.io")
    response = client.patch(
        f"{members_url(admin.tenant_id)}/{target['id']}",
        headers=admin.headers,
        json={"role": "member"},
    )
    assert response.status_code == 403


def test_last_owner_cannot_be_demoted(client, owner):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = listing["items"][0]
    response = client.patch(
        f"{members_url(owner.tenant_id)}/{target['id']}",
        headers=owner.headers,
        json={"role": "admin"},
    )
    # Owners cannot change their own role at all.
    assert response.status_code == 403


def test_removing_a_member_works(client, owner, member):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = next(i for i in listing["items"] if i["email"] == "member@acme.io")
    assert (
        client.delete(
            f"{members_url(owner.tenant_id)}/{target['id']}", headers=owner.headers
        ).status_code
        == 200
    )
    remaining = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    assert all(i["email"] != "member@acme.io" for i in remaining["items"])


def test_cannot_remove_yourself(client, owner):
    listing = client.get(members_url(owner.tenant_id), headers=owner.headers).json()
    target = listing["items"][0]
    response = client.delete(
        f"{members_url(owner.tenant_id)}/{target['id']}", headers=owner.headers
    )
    assert response.status_code == 403


def test_members_of_another_tenant_are_invisible(client, owner, other_owner):
    response = client.get(
        members_url(other_owner.tenant_id),
        headers={"Authorization": f"Bearer {owner.token}"},
    )
    assert response.status_code == 403
