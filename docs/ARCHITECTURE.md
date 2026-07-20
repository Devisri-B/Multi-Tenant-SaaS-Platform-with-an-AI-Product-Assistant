# Architecture

## Tenancy model

Three isolation strategies were considered.

| Strategy | Isolation | Cost per tenant | Migration story |
| --- | --- | --- | --- |
| Database per tenant | Strongest | High | N migrations to run |
| Schema per tenant | Strong | Moderate | Search-path juggling |
| **Shared schema + `tenant_id`** | Enforced in code | Near zero | One migration |

Nimbus takes the third option. The trade-off it accepts is that isolation is a
property of the code rather than of the storage engine, so the code has to earn
it:

1. **One choke point.** Every tenant-owned read and write goes through
   `TenantScopedRepository`, which owns the `WHERE tenant_id = :tenant_id`
   predicate. `add()` and `delete()` raise if handed a row belonging to another
   tenant, so a mistake fails loudly instead of leaking.
2. **Resolution before use.** `get_tenant_context` turns the workspace id from
   the path or the `X-Workspace-Id` header into a `TenantContext` only after
   confirming the caller has an active membership. Handlers receive the
   validated context, never the raw id.
3. **Retrieval filtered in SQL.** The vector search carries the same predicate,
   so similarity ranking cannot surface a neighbouring tenant's chunk even if
   the embedding is a perfect match.
4. **Tests that try to break it.** `test_documents.py`,
   `test_repositories.py` and `test_assistant.py` each attempt a cross-tenant
   read with a valid id and a valid token, and assert it fails.

Where the database can help, it does: `(tenant_id, checksum)` is unique per
tenant, so the same document can exist in two workspaces but not twice in one.

## Request lifecycle

```
HTTP request
  → SecurityHeadersMiddleware      conservative response headers
  → RequestContextMiddleware       request id, structlog binding, latency
  → CORSMiddleware
  → route
      → get_current_user           decode JWT, load user, check active
      → get_tenant_context         resolve workspace, check membership
      → require_role(minimum)      compare against the role lattice
      → handler                    services + tenant-scoped repositories
  → AppError handler               domain error → {code, message} + status
```

`AppError` subclasses carry both an HTTP status and a stable machine-readable
`code`, so the frontend can branch on `code` rather than parsing prose.

## Authentication

- Passwords: bcrypt via passlib. Login verifies against a real placeholder hash
  when the account does not exist, keeping failure timing independent of
  whether the email is registered.
- Tokens: short-lived access JWT (30 min) plus a refresh JWT (14 days). Every
  token carries `type` and `jti`; `decode_token` rejects a refresh token
  presented where an access token is required, and vice versa.
- The frontend keeps tokens in `sessionStorage` and retries a 401 exactly once
  after refreshing, which prevents refresh loops.

## Authorization

Roles form a lattice: `viewer(10) < member(20) < admin(30) < owner(40)`.
`require_role` compares levels rather than matching exact strings, so adding a
role between two existing ones does not require touching every route.

Guardrails beyond the lattice, all tested:

- You cannot change your own role or remove yourself.
- An admin cannot modify a member at or above their own level.
- Only an owner can mint another owner.
- The last owner cannot be demoted or removed.

## Data model

```
Tenant 1───* Membership *───1 User
  │
  ├──* Document 1───* DocumentChunk   (embedding vector)
  ├──* Conversation 1───* Message     (citations JSON)
  └──* AuditLog
```

`AuditLog` is append-only and records privileged actions (invites, role
changes, archives, assistant queries) with the acting user and workspace.

## Portability

`GUID`, `JSONType` and `Vector` are `TypeDecorator`s that bind to native
Postgres types (`uuid`, `jsonb`, `vector`) and degrade to `CHAR(32)`, `JSON`
and a JSON array on SQLite. This is what lets the full test-suite — including
the RAG pipeline — run in-memory in CI with no services, while production uses
real pgvector with an `ivfflat` cosine index.
