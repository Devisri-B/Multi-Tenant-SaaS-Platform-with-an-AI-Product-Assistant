"""Document upload, indexing and tenant scoping."""

from __future__ import annotations

import io
import uuid

DOC = """\
# Password Reset

To reset a password, open Settings, choose Security, then Reset password.
A reset link is emailed to the account address and expires after one hour.
"""


def docs_url(tenant_id) -> str:
    return f"/api/v1/workspaces/{tenant_id}/documents"


def create_doc(client, actor, title="Password Reset", content=DOC):
    return client.post(
        docs_url(actor.tenant_id),
        headers=actor.headers,
        json={"title": title, "content": content},
    )


def test_member_can_create_a_document(client, member):
    response = create_doc(client, member)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "indexed"
    assert body["chunk_count"] >= 1


def test_viewer_cannot_create_a_document(client, viewer):
    assert create_doc(client, viewer).status_code == 403


def test_duplicate_content_is_rejected(client, owner):
    create_doc(client, owner)
    assert create_doc(client, owner).status_code == 409


def test_document_quota_is_enforced(client, db, owner):
    owner.tenant.document_limit = 1
    db.commit()
    create_doc(client, owner)
    response = create_doc(client, owner, title="Second", content="Different content here.")
    assert response.status_code == 403
    assert "document limit" in response.json()["message"]


def test_upload_endpoint_indexes_a_file(client, member):
    response = client.post(
        f"{docs_url(member.tenant_id)}/upload",
        headers=member.headers,
        files={"file": ("guide.md", io.BytesIO(DOC.encode()), "text/markdown")},
        data={"title": "Uploaded Guide"},
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Uploaded Guide"
    assert response.json()["status"] == "indexed"


def test_empty_upload_is_rejected(client, member):
    response = client.post(
        f"{docs_url(member.tenant_id)}/upload",
        headers=member.headers,
        files={"file": ("empty.md", io.BytesIO(b""), "text/markdown")},
    )
    assert response.status_code == 422


def test_unsupported_content_type_is_rejected(client, member):
    response = client.post(
        f"{docs_url(member.tenant_id)}/upload",
        headers=member.headers,
        files={"file": ("app.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")},
    )
    assert response.status_code == 422


def test_list_documents_paginates(client, owner):
    create_doc(client, owner, title="One", content="Content one about billing.")
    create_doc(client, owner, title="Two", content="Content two about refunds.")
    response = client.get(
        docs_url(owner.tenant_id), headers=owner.headers, params={"size": 1}
    )
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1


def test_filter_by_status(client, owner):
    create_doc(client, owner)
    response = client.get(
        docs_url(owner.tenant_id), headers=owner.headers, params={"status": "failed"}
    )
    assert response.json()["total"] == 0


def test_chunks_endpoint_returns_ordered_chunks(client, owner):
    document_id = create_doc(client, owner).json()["id"]
    response = client.get(
        f"{docs_url(owner.tenant_id)}/{document_id}/chunks", headers=owner.headers
    )
    chunks = response.json()
    assert chunks
    assert [c["ordinal"] for c in chunks] == sorted(c["ordinal"] for c in chunks)


def test_reindex_rebuilds_chunks(client, owner):
    document_id = create_doc(client, owner).json()["id"]
    response = client.post(
        f"{docs_url(owner.tenant_id)}/{document_id}/reindex", headers=owner.headers
    )
    assert response.status_code == 200
    assert response.json()["status"] == "indexed"


def test_delete_document(client, owner):
    document_id = create_doc(client, owner).json()["id"]
    assert (
        client.delete(
            f"{docs_url(owner.tenant_id)}/{document_id}", headers=owner.headers
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"{docs_url(owner.tenant_id)}/{document_id}", headers=owner.headers
        ).status_code
        == 404
    )


def test_missing_document_is_404(client, owner):
    response = client.get(
        f"{docs_url(owner.tenant_id)}/{uuid.uuid4()}", headers=owner.headers
    )
    assert response.status_code == 404


def test_documents_do_not_leak_across_tenants(client, owner, other_owner):
    document_id = create_doc(client, owner).json()["id"]

    # The rival owner cannot read it even with the correct document id, because
    # the repository scopes every lookup by tenant.
    response = client.get(
        f"{docs_url(other_owner.tenant_id)}/{document_id}", headers=other_owner.headers
    )
    assert response.status_code == 404

    listing = client.get(
        docs_url(other_owner.tenant_id), headers=other_owner.headers
    ).json()
    assert listing["total"] == 0


def test_identical_content_in_two_tenants_is_allowed(client, owner, other_owner):
    assert create_doc(client, owner).status_code == 201
    assert create_doc(client, other_owner).status_code == 201
