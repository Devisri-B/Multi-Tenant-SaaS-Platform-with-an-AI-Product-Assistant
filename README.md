# Nimbus - Multi-Tenant SaaS Platform with an Adaptive AI Assistant

A production-grade SaaS platform featuring isolated customer workspaces served
from a single shared Postgres schema, paired with an **adaptive Self-RAG AI assistant**
powered by **LangGraph**, **FastAPI**, **PostgreSQL (`pgvector`)**, and **React/TypeScript**.

**Stack** - Python | FastAPI | LangGraph | Anthropic Claude (SyncAnthropic) | FastMCP | SentenceTransformers | PostgreSQL (`pgvector`) | SQLAlchemy 2.0 | Alembic | React 18 | TypeScript | Vite | Docker | GitHub Actions

> **Live Demo**: [https://multi-tenant-saas-platform-with-an-ai.onrender.com](https://multi-tenant-saas-platform-with-an-ai.onrender.com)  
> **Seeded Demo Account**: Email: `owner@nimbus.dev` | Password: `DemoPassw0rd`

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Clients ["1. Client & Host Layer"]
        direction LR
        SPA["React 18 SPA<br/>(Vite / TS / Theme Switcher)"]
        IDE["Claude Desktop / Cursor<br/>(External AI Host)"]
        AgentCLI["Autonomous MCP Agent<br/>(Local Reasoning CLI)"]
    end

    subgraph Gateway ["2. FastAPI Gateway & Control Plane"]
        direction TB
        MW["Middleware Pipeline<br/>(Security Headers / Request Context / CORS)"]
        AuthTenancy["Auth & Tenancy Guard<br/>(JWT Bearer / Tenant Context / RBAC Lattice)"]
        REST["REST API Endpoints<br/>(/workspaces /documents /assistant)"]
        FastMCP["FastMCP Server (JSON-RPC)<br/>(list_workspaces / semantic_search / self_rag_query)"]
    end

    subgraph DataPlane ["3. Multi-Tenant Storage Layer (PostgreSQL 16)"]
        direction TB
        Repo["TenantScopedRepository<br/>(Mandatory WHERE tenant_id = :id)"]
        RelationalDB[("Relational Data<br/>Tenants / Users / Memberships<br/>Conversations / AuditLogs")]
        PGVectorDB[("Vector Storage (pgvector)<br/>Document Chunks (384-dim)<br/>ivfflat Cosine Index")]
    end

    subgraph RAGWorkflow ["4. Adaptive Self-RAG Engine (LangGraph)"]
        direction TB
        Memory["Sliding Window Memory<br/>(Coreference Reformulator)"]
        Retriever["pgvector Retrieval<br/>(Cosine Similarity Threshold 0.40)"]
        DocGrader{"Document Grader<br/>(Relevance Filter)"}
        Generator["Contextual Generator<br/>(Anthropic Claude 3.5 Haiku)"]
        HallucinationGrader{"Hallucination Reductor<br/>(DeBERTa-v3 NLI Entailment)"}
        WebSearch["Dynamic Web Fallback<br/>(DuckDuckGo / Tavily Citations)"]
        FinalAnswer["Verified Response + Citations"]
    end

    %% Client Connections
    SPA -->|"HTTP / REST + JWT"| MW
    IDE <-->|"stdio Transport"| FastMCP
    AgentCLI <-->|"stdio Transport"| FastMCP

    %% Gateway Pipeline
    MW --> AuthTenancy
    AuthTenancy --> REST

    %% Gateway to Storage & RAG
    REST --> Repo
    FastMCP --> Repo
    FastMCP --> RAGWorkflow
    REST --> RAGWorkflow

    Repo --> RelationalDB
    Repo --> PGVectorDB

    %% RAG Workflow Steps
    RAGWorkflow --> PGVectorDB
    Memory --> Retriever
    Retriever --> DocGrader
    DocGrader -->|"Relevant Chunks"| Generator
    DocGrader -->|"Out of Scope"| WebSearch
    Generator --> HallucinationGrader
    HallucinationGrader -->|"Grounded"| FinalAnswer
    HallucinationGrader -->|"Unverified (Retry)"| Generator
    WebSearch --> FinalAnswer
```

---

## The AI Assistant: Adaptive LangGraph & Self-RAG

The assistant uses an adaptive **LangGraph state graph** with **sliding window conversational memory**, **Self-RAG hallucination reduction**, and **dynamic online search fallback**:
- **Sliding Window Conversation Memory**: Preserves context across multi-turn dialogues with automated conversational query reformulation and pronoun coreference resolution.
- **Multi-Tenant Vector Search**: Answers grounded in tenant-isolated documentation chunks with similarity scores and excerpts.
- **Local NLI Hallucination Verification (DeBERTa-v3)**: Evaluates candidate answers against retrieved context passages using local cross-encoder Natural Language Inference scoring on CPU (~20-40ms latency), calculating calibrated entailment probabilities and triggering strict regeneration loops when unsupported claims are detected.
- **Dynamic Online Routing**: If workspace documents lack context or fail to resolve the query, the graph routes to online web search (DuckDuckGo / Tavily) to synthesize verified answers with external web citations.

| Grounded Workspace Document Citations | Dynamic Online Search Fallback (Out-of-Scope) |
| --- | --- |
| ![The product assistant answering from workspace documents with scored citations](docs/screenshots/assistant.png) | ![The assistant dynamically routing to online search with external citations](docs/screenshots/assistant-web-search.png) |

---

## Model Context Protocol (MCP) Server & Autonomous Agent

Nimbus exposes its multi-tenant Self-RAG knowledge base to external AI tooling and desktop assistants via a standard **FastMCP Server** (`app.mcp.server`) and includes an **Autonomous Tool-Calling Agent** (`app.mcp.agent`) acting as an MCP client over standard I/O (`stdio`).

### MCP Architecture

```mermaid
flowchart LR
    subgraph Host ["Host / Client Layer"]
        Agent["Autonomous Agent<br/>(app/mcp/agent.py)"]
        Claude["Claude Desktop / Cursor"]
    end

    subgraph MCP ["Model Context Protocol (JSON-RPC)"]
        Stdio["Standard I/O (stdio) Transport"]
        Discovery["Dynamic Tool Discovery<br/>(list_tools)"]
    end

    subgraph Server ["FastMCP Server (app/mcp/server.py)"]
        T1["list_workspaces"]
        T2["semantic_search_chunks"]
        T3["self_rag_query"]
    end

    subgraph Pipeline ["Self-RAG Pipeline (app/rag/)"]
        Graph["LangGraph Workflow"]
        PGVector["pgvector Store"]
        Graders["Hallucination & Relevance Graders"]
        Web["DuckDuckGo Fallback"]
    end

    Agent <--> Stdio
    Claude <--> Stdio
    Stdio <--> Server
    Server --> T1 & T2 & T3
    T2 --> PGVector
    T3 --> Graph
    Graph --> PGVector & Graders & Web
```

### Exposed MCP Tools

The FastMCP server exposes three standardized tools via standard JSON-RPC:

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `list_workspaces` | *(none)* | Discovers all active tenant workspaces (UUIDs, names, slugs, subscription tiers) so external agents can dynamically target tenants without hardcoded IDs. |
| `semantic_search_chunks` | `query: str`, `tenant_id?: str`, `top_k?: int` | Performs direct semantic vector retrieval against PostgreSQL `pgvector` without LLM answer generation. Returns chunk excerpts, document titles, and cosine similarity scores. |
| `self_rag_query` | `question: str`, `tenant_id?: str`, `allow_web_search?: bool`, `top_k?: int` | Executes the complete LangGraph Self-RAG workflow: query reformulation, document grading, anti-hallucination verification, iterative self-correction, and web search fallback. |

### Running the Autonomous Agent (MCP Client)

The autonomous agent automatically launches the MCP server as a subprocess over `stdio`, dynamically inspects available tools, and runs an autonomous reasoning and tool-calling loop:

```bash
# 1. Offline demonstration mode (no API key required, verifies MCP handshake & tool execution):
cd backend
python -m app.mcp.agent --offline

# 2. Ask a specific query in offline mode:
python -m app.mcp.agent --question "What is the refund policy?" --offline

# 3. Live LLM mode with SyncAnthropic Claude tool calling over MCP:
export ANTHROPIC_API_KEY="sk-ant-..."
python -m app.mcp.agent --question "Explain the document upload limits."
```

### IDE Integration (Claude Desktop & Cursor)

Connect Claude Desktop or Cursor directly to your private tenant knowledge base by adding the server configuration to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nimbus-self-rag": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/SAAS/backend",
      "env": {
        "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/saas_db"
      }
    }
  }
}
```

---

## Document Ingestion, Chunk Inspection & Live Editor

Uploads are chunked, embedded and indexed on arrival (`pending -> processing -> indexed`). Users can inspect individual vector chunks, view full source text, and live-edit documentation with automatic re-indexing.

![The documentation page listing indexed documents with chunk counts and sizes](docs/screenshots/documentation.png)

### Document Viewer & Live Editor Modal

| 1. Full Document Text | 2. Vector Chunk Inspector | 3. Live Editor & Reindexer |
| --- | --- | --- |
| ![Full document text view](docs/screenshots/edit-document-text.png) | ![Vector chunk inspection](docs/screenshots/edit-document-chunks.png) | ![Live markdown editor and reindexing](docs/screenshots/edit-document-reindex.png) |

---

## Workspace Overview

Usage counters for the selected tenant: members, documents, how many are indexed, chunks embedded, and conversations held.

![Workspace overview showing member, document, chunk and conversation counts](docs/screenshots/overview.png)

---

## Members, RBAC & Settings

Roles form a cumulative lattice (`viewer < member < admin < owner`). Settings allows tenant creation, workspace renaming, appearance theme selection (Bright / Dark Mode), and credentials management.

| Role-Based Access Control | Workspace Settings & Bright/Dark Mode |
| --- | --- |
| ![Member list with role and status columns and an invite form](docs/screenshots/members.png) | ![Settings page with workspace rename, workspace creation and appearance theme switcher](docs/screenshots/settings.png) |

---

## Why it is built this way

**Shared-schema multi-tenancy.** Every tenant-owned table carries a
`tenant_id` foreign key. Rather than trusting each route to remember the
predicate, all reads and writes go through `TenantScopedRepository`
(`backend/app/services/base.py`), which applies `WHERE tenant_id = :tenant_id`
and refuses to persist or delete a row belonging to another tenant. Retrieval
is filtered the same way inside SQL, so the assistant physically cannot quote
another workspace's documents - `tests/test_assistant.py` asserts exactly that.

**Authorization as a dependency chain.** `bearer token -> current_user ->
tenant_context -> require_role(...)`. A handler never sees a tenant id from the
client that has not already been checked against the caller's memberships.
Requesting a workspace you are not a member of returns `403`, not `404`, and a
non-existent workspace returns `404` - the split is deliberate and tested.

**Provider seam for LLM & Embeddings.** `app/rag/providers.py` defines
`EmbeddingProvider` and `ChatProvider`. Production binds Chat to Anthropic Claude
(via `SyncAnthropic`) and Embeddings to local `SentenceTransformers`
(`all-MiniLM-L6-v2`); `LLM_PROVIDER=fake` binds them to a deterministic hashed
bag-of-words embedder and an extractive generator. The whole pipeline
(chunking, embedding, retrieval, prompt assembly, citation building) runs
identically in CI with zero external API keys and zero cost.

**Portable column types.** `Vector` is a real `pgvector` column on Postgres and
a JSON array on SQLite, and the retriever pushes the nearest-neighbour search
into `pgvector`'s `<=>` operator when available and falls back to an in-Python
cosine scan otherwise. One set of models, two environments, no test doubles for
the database.

---

## Quick start

### Docker (everything)

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY, or leave LLM_PROVIDER=fake
docker compose up --build
```

| Service | URL |
| --- | --- |
| Frontend | http://localhost:8080 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

### Local development

```bash
make install      # backend venv + npm install
make migrate      # alembic upgrade head
make seed         # demo workspaces, users and docs
make run          # API on :8000
make web          # Vite dev server on :5173
```

Seeded login: `owner@nimbus.dev` / `DemoPassw0rd`.

### Tests

```bash
make test         # 160 tests, SQLite in-memory, no external services
make lint
```

---

## Layout

```
backend/
  app/
    core/         config, security (bcrypt + JWT), logging, error taxonomy
    db/           engine, session, portable GUID/JSON/Vector column types
    models/       Tenant, User, Membership, Document, DocumentChunk,
                  Conversation, Message, AuditLog
    schemas/      Pydantic request/response contracts
    api/          dependencies (auth, tenancy, RBAC), middleware, v1 routers
    services/     tenant-scoped repositories and domain logic
    rag/          chunking, memory, web search, graph, ingestion, retrieval, answering chain, DeBERTa NLI, SentenceTransformers
    mcp/          FastMCP server (stdio), autonomous tool-calling client agent
  alembic/        three migrations, including the pgvector ivfflat index
  tests/          160 tests across auth, tenancy, RBAC, LangGraph Self-RAG, MCP, DeBERTa NLI, Claude, SentenceTransformers, memory and isolation
frontend/
  src/
    api/          typed fetch client with one-shot token refresh
    context/      session + active-workspace state
    components/   layout, route guards, UI primitives
    pages/        login, register, overview, assistant, documents, members,
                  settings
```

---

## API surface

| Method | Path | Minimum role |
| --- | --- | --- |
| `POST` | `/api/v1/auth/register` | - |
| `POST` | `/api/v1/auth/login` | - |
| `POST` | `/api/v1/auth/refresh` | - |
| `GET` | `/api/v1/auth/me` | authenticated |
| `POST` | `/api/v1/auth/password` | authenticated |
| `GET` `POST` | `/api/v1/workspaces` | authenticated |
| `GET` | `/api/v1/workspaces/{id}` | viewer |
| `PATCH` | `/api/v1/workspaces/{id}` | admin |
| `DELETE` | `/api/v1/workspaces/{id}` | owner |
| `GET` | `/api/v1/workspaces/{id}/stats` | viewer |
| `GET` | `/api/v1/workspaces/{id}/members` | viewer |
| `POST` | `/api/v1/workspaces/{id}/members` | admin |
| `PATCH` `DELETE` | `/api/v1/workspaces/{id}/members/{mid}` | admin |
| `GET` | `/api/v1/workspaces/{id}/documents` | viewer |
| `POST` | `/api/v1/workspaces/{id}/documents` | member |
| `POST` | `/api/v1/workspaces/{id}/documents/upload` | member |
| `POST` | `/api/v1/workspaces/{id}/documents/{did}/reindex` | member |
| `DELETE` | `/api/v1/workspaces/{id}/documents/{did}` | member |
| `POST` | `/api/v1/workspaces/{id}/assistant/ask` | viewer |
| `POST` | `/api/v1/workspaces/{id}/assistant/search` | viewer |
| `GET` | `/api/v1/workspaces/{id}/assistant/conversations` | viewer |

The active workspace can be supplied either in the path or via an
`X-Workspace-Id` header; the frontend uses the header.

---

## Configuration

See `.env.example` for the full list. The ones that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `SECRET_KEY` | dev placeholder | **Must** be replaced in production |
| `DATABASE_URL` | assembled from `POSTGRES_*` | Overrides the parts |
| `LLM_PROVIDER` | `anthropic` | `anthropic` (Claude via SyncAnthropic) or `fake` |
| `ANTHROPIC_API_KEY` | - | Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_CHAT_MODEL` | `claude-3-5-haiku-20241022` | Anthropic Claude model checkpoint |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` (local embeddings) or `fake` |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Local Hugging Face sentence embedding model |
| `EMBEDDING_DIMENSIONS` | `384` | Embedding vector width |
| `HALLUCINATION_PROVIDER` | `deberta` | `deberta` (local DeBERTa-v3 cross-encoder), `llm`, or `fake` |
| `DEBERTA_MODEL_NAME` | `cross-encoder/nli-deberta-v3-small` | Hugging Face cross-encoder model checkpoint |
| `NLI_ENTAILMENT_THRESHOLD` | `0.5` | Minimum entailment probability for factual grounding |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `250` / `40` | Tokens (token-based via `count_tokens`) |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `5` / `0.40` | Retrieval budget and tuned cosine similarity floor |

Further reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
[`docs/RAG.md`](docs/RAG.md).
