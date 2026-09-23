# Nimbus - Multi-Tenant SaaS Platform with an Adaptive AI Assistant

A production-grade SaaS platform featuring isolated customer workspaces served from a single shared PostgreSQL schema, paired with an **adaptive Self-RAG AI assistant** powered by **LangGraph**, **FastAPI**, **PostgreSQL (`pgvector`)**, and **React / TypeScript**.

**Stack**: Python | FastAPI | LangGraph | Anthropic Claude | FastMCP | SentenceTransformers | DeBERTa-v3 | PostgreSQL (`pgvector`) | React 18 | TypeScript | Vite | Docker

> **Live Demo**: [https://multi-tenant-saas-platform-with-an-ai.onrender.com](https://multi-tenant-saas-platform-with-an-ai.onrender.com)  
> **Seeded Demo Account**: Email: `owner@nimbus.dev` | Password: `DemoPassw0rd`

---

## Key Highlights

- **Shared-Schema Multi-Tenancy with PostgreSQL RLS**: Every tenant's data is strictly partitioned using PostgreSQL Row-Level Security (`FORCE ROW LEVEL SECURITY`) and repository-level scoping.
- **Adaptive Self-RAG Knowledge Pipeline**: LangGraph state graph combining dense vector search (`pgvector`), sparse lexical search (PostgreSQL FTS), cross-encoder reranking, and local DeBERTa-v3 NLI anti-hallucination verification.
- **Dynamic Online Search Fallback**: Automatically evaluates context relevance and routes out-of-scope queries to external web search (DuckDuckGo / Tavily) with external citations.
- **Model Context Protocol (FastMCP) Integration**: Built-in FastMCP server (`stdio`) and autonomous tool-calling agent enabling Claude Desktop and Cursor to interface directly with tenant knowledge bases.
- **Zero-Cost Local & CI Testing**: Pluggable provider seams allow running all 220+ tests against SQLite and deterministic mock models with zero API costs and no external dependencies.

---

## Architecture & Engineering Design

### 1. Adaptive Self-RAG Pipeline

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 45, "curve": "basis"}, "themeVariables": {"fontSize": "16px"}}}%%
flowchart LR
    Q(["User question"]) --> R["Rewrite query<br/>using chat memory"]
    R --> H["Hybrid search + rerank<br/>(this tenant only)"]
    H --> G1{{"Relevant<br/>docs?"}}
    G1 -- yes --> GEN["Generate answer"]
    GEN --> G2{{"Grounded?<br/>(NLI check)"}}
    G2 -- yes --> A1(["Answer +<br/>citations"])
    G2 -- "no, retry ≤ 2" --> GEN
    G2 -- "still no" --> W["Web search"]
    G1 -- no --> W
    W --> A2(["Web answer or<br/>'I don't know'"])

    classDef io fill:#1e3a8a,stroke:#1e3a8a,color:#ffffff
    classDef step fill:#ffffff,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef guard fill:#fde68a,stroke:#b45309,color:#451a03,stroke-width:2px
    classDef out fill:#bbf7d0,stroke:#15803d,color:#052e16,stroke-width:2px
    class Q io
    class R,H,GEN,W step
    class G1,G2 guard
    class A1,A2 out
```

The RAG engine is implemented as an adaptive **LangGraph state graph**:
- **Sliding-Window Memory & Reformulation**: Resolves pronouns and conversational coreferences across turns before retrieval.
- **Tenant-Isolated Hybrid Search**: Executes dense vector cosine similarity (`pgvector`) concurrently with sparse lexical search (Postgres FTS / BM25), combined using Reciprocal Rank Fusion (RRF) and scored through a Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`).
- **Local NLI Hallucination Verification**: Evaluates generated candidate answers against retrieved passages using a local DeBERTa-v3 cross-encoder (~20–40ms latency on CPU). If unsupported claims are detected, a self-correction loop regenerates the response with stricter constraints.
- **Dynamic Web Routing**: If retrieved documents fail relevance grading (score < 0.40), the pipeline automatically routes to live web search to synthesize answers with verified web citations.

### 2. Multi-Tenancy & Security Architecture

- **PostgreSQL Row-Level Security (RLS)**: Every tenant-owned table carries a `tenant_id` foreign key. On PostgreSQL, RLS policies (`FORCE ROW LEVEL SECURITY` with `current_setting('app.tenant_id')::uuid`) are applied to `documents`, `document_chunks`, `conversations`, and `messages`. Even an arbitrary SQL injection (`OR 1=1`) cannot cross tenant boundaries.
- **Authorization Lattice**: Roles form a cumulative lattice (`viewer < member < admin < owner`). Enforced via FastAPI dependency injection: `bearer token -> current_user -> tenant_context -> require_role(...)`. Requesting a workspace the user does not belong to yields `403 Forbidden`, while a non-existent workspace yields `404 Not Found`.
- **Pluggable Provider Seams**: `app/rag/providers.py` abstracts LLM generation (`SyncAnthropic` or deterministic `fake`) and vector embeddings (`SentenceTransformers` or hash-based `fake`). This allows end-to-end ingestion, chunking, retrieval, and evaluation to run in CI with zero API keys.
- **Dual Vector Portability**: Transparently executes native `<=>` cosine distance on PostgreSQL `pgvector` in production and in-Python cosine similarity over JSON arrays on SQLite for development and fast testing.

---

## Model Context Protocol (MCP)

Nimbus exposes tenant knowledge to external AI agents and IDEs via a standardized **FastMCP Server** (`app.mcp.server`) and includes an **Autonomous Tool-Calling Agent** (`app.mcp.agent`).

### MCP Architecture

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 40, "rankSpacing": 45, "curve": "basis"}, "themeVariables": {"fontSize": "16px"}}}%%
flowchart LR
    C["AI client<br/>Claude Desktop / agent.py"] <-- "JSON-RPC<br/>over stdio" --> S["FastMCP server"]
    S --> G{{"Tenant<br/>allowed?"}}
    G -- no --> E(["TenantNotFound"])
    G -- yes --> T["3 tools<br/>list_workspaces<br/>semantic_search_chunks<br/>self_rag_query"]
    T --> P["Self-RAG pipeline"]
    T --> DB[("Postgres + pgvector")]
    P --> DB

    classDef io fill:#1e3a8a,stroke:#1e3a8a,color:#ffffff
    classDef step fill:#ffffff,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef guard fill:#fde68a,stroke:#b45309,color:#451a03,stroke-width:2px
    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px
    classDef data fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px
    class C io
    class S,T,P step
    class G guard
    class E bad
    class DB data
```

### Exposed MCP Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `list_workspaces` | *(none)* | Discovers active tenant workspaces (UUIDs, names, slugs, subscription tiers) for dynamic target selection. |
| `semantic_search_chunks` | `query: str`, `tenant_id?: str`, `top_k?: int` | Direct vector retrieval against `pgvector` returning chunks, titles, and cosine similarity scores without LLM generation. |
| `self_rag_query` | `question: str`, `tenant_id?: str`, `allow_web_search?: bool`, `top_k?: int` | Executes the complete LangGraph Self-RAG workflow: query reformulation, document grading, anti-hallucination NLI checks, and web search fallback. |

### Running the Autonomous Agent CLI

The agent launches the MCP server as a subprocess over `stdio`, discovers available tools, and runs an autonomous ReAct loop:

```bash
cd backend

# Offline demonstration (verifies MCP handshake & tool execution without API keys):
python -m app.mcp.agent --offline

# Ask a specific question offline:
python -m app.mcp.agent --question "What is the refund policy?" --offline

# Live Claude tool-calling loop (AsyncAnthropic) over MCP:
export ANTHROPIC_API_KEY="sk-ant-..."
python -m app.mcp.agent --question "Explain the document upload limits."
```

### Desktop & IDE Integration (Claude Desktop / Cursor)

Add the server definition to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nimbus-self-rag": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/SAAS/backend",
      "env": {
        "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/saas_db",
        "MCP_ALLOWED_TENANT_IDS": "<workspace-uuid>,<workspace-uuid>"
      }
    }
  }
}
```

> **Note**: `MCP_ALLOWED_TENANT_IDS` scopes access to explicit workspaces. Requests for unauthorized workspaces return `TenantNotFound` before any retrieval occurs.

---

## Product Tour & Visual Walkthrough

### 1. AI Assistant & Dynamic Fallback
| Grounded Workspace Citations | Dynamic Online Search Fallback |
| :---: | :---: |
| ![The product assistant answering from workspace documents with scored citations](docs/screenshots/assistant.png) | ![The assistant dynamically routing to online search with external citations](docs/screenshots/assistant-web-search.png) |

### 2. Knowledge Base, Chunk Inspector & Live Editor
Uploads are automatically chunked, embedded, and indexed (`pending -> processing -> indexed`). Users can inspect individual vector embeddings, read raw source text, and live-edit documents with automatic re-indexing.

| Document Indexing View | Vector Chunk Inspector | Live Markdown Editor & Re-indexer |
| :---: | :---: | :---: |
| ![Documentation view](docs/screenshots/documentation.png) | ![Vector chunk inspection](docs/screenshots/edit-document-chunks.png) | ![Live editor and reindexing](docs/screenshots/edit-document-reindex.png) |

### 3. Multi-Tenant Governance & Administration
| Workspace Metrics & Usage Counters | Role-Based Access Control (RBAC) | Appearance & Workspace Settings |
| :---: | :---: | :---: |
| ![Workspace overview](docs/screenshots/overview.png) | ![Member list and roles](docs/screenshots/members.png) | ![Settings and theme switcher](docs/screenshots/settings.png) |

---

## Quick Start

### Option A: Docker Compose (Full Stack)

```bash
# 1. Clone repository and set environment variables
cp .env.example .env

# 2. Launch PostgreSQL (pgvector), FastAPI backend, and React frontend
docker compose up --build
```

| Service | Endpoint | Description |
| :--- | :--- | :--- |
| **Frontend** | [http://localhost:8080](http://localhost:8080) | React 18 / TypeScript SPA |
| **API Backend** | [http://localhost:8000](http://localhost:8000) | FastAPI REST API |
| **Interactive Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI for all REST endpoints |

*Default Seeded Login*: `owner@nimbus.dev` / `DemoPassw0rd`

### Option B: Local Development

```bash
make install      # Creates backend venv, installs dependencies & frontend packages
make migrate      # Runs Alembic migrations (alembic upgrade head)
make seed         # Seeds demo workspaces, users, and documents
make run          # Runs FastAPI on :8000 with hot reload
make web          # Runs Vite dev server on :5173
```

### Automated Testing & Quality Checks

```bash
make test         # Runs 220+ backend unit and integration tests (SQLite in-memory, no API keys needed)
make lint         # Lints Python (Ruff) and TypeScript / React (ESLint)
make format       # Auto-formats backend code with Ruff
```

---

## Project Structure

```
.
├── backend/
│   ├── alembic/          # 6 database migrations (initial schema, pgvector, conversations, RLS, telemetry, FTS)
│   ├── app/
│   │   ├── api/          # Dependencies (auth, tenancy, RBAC), middleware, and v1 routers
│   │   ├── core/         # Config, security (bcrypt + JWT), logging, error taxonomy
│   │   ├── db/           # Session management and portable column types (Vector, GUID)
│   │   ├── mcp/          # FastMCP stdio server and autonomous client agent
│   │   ├── models/       # SQLAlchemy models (Tenant, User, Document, DocumentChunk, Message, etc.)
│   │   ├── rag/          # LangGraph state graph, DeBERTa NLI, chunking, hybrid retrieval, web search
│   │   ├── schemas/      # Pydantic request and response schemas
│   │   └── services/     # Tenant-scoped repositories and domain logic
│   └── tests/            # 220+ tests covering auth, RLS, Self-RAG, MCP, NLI, and tenancy
├── frontend/
│   └── src/
│       ├── api/          # Typed API fetch client with automatic token refresh
│       ├── components/   # Layout, navigation, modals, and route guards
│       ├── context/      # Auth session and active workspace state providers
│       └── pages/        # Overview, Assistant, Documents, Members, Settings, Auth
└── docs/                 # Architectural specifications, deep dives, and screenshots
```

---

## Configuration Reference

Key environment variables from [`.env.example`](.env.example):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `SECRET_KEY` | *(dev placeholder)* | JWT signing secret (**must** be set in production) |
| `DATABASE_URL` | `postgresql+psycopg://...` | Connection URI for PostgreSQL with `pgvector` |
| `LLM_PROVIDER` | `anthropic` | `anthropic` (Claude via SyncAnthropic) or `fake` (deterministic mock) |
| `ANTHROPIC_API_KEY` | *(empty)* | Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_CHAT_MODEL`| `claude-haiku-4-5` | Model checkpoint for answer synthesis |
| `EMBEDDING_PROVIDER` | `sentence_transformers`| `sentence_transformers` (local embeddings) or `fake` |
| `HALLUCINATION_PROVIDER`| `deberta` | `deberta` (local DeBERTa-v3 cross-encoder), `llm`, or `fake` |
| `NLI_ENTAILMENT_THRESHOLD`| `0.5` | Minimum entailment probability required for factual grounding |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `5` / `0.40` | Retrieval candidate count and minimum similarity score threshold |
| `MCP_ALLOWED_TENANT_IDS` | *(empty = all)* | Comma-separated workspace UUIDs permitted for MCP client queries |

Further technical reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | [`docs/RAG.md`](docs/RAG.md).
