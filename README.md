# Nimbus - Multi-Tenant SaaS Platform with an Adaptive AI Assistant

A production-grade SaaS platform featuring isolated customer workspaces served
from a single shared Postgres schema, paired with an **adaptive Self-RAG AI assistant**
powered by **LangGraph**, **FastAPI**, **PostgreSQL (`pgvector`)**, and **React/TypeScript**.

**Stack** - Python | FastAPI | LangGraph | Anthropic Claude (SyncAnthropic) | FastMCP | SentenceTransformers | PostgreSQL (`pgvector`) | React 18 | TypeScript | Vite | Docker | GitHub Actions

> **Live Demo**: [https://multi-tenant-saas-platform-with-an-ai.onrender.com](https://multi-tenant-saas-platform-with-an-ai.onrender.com)  
> **Seeded Demo Account**: Email: `owner@nimbus.dev` | Password: `DemoPassw0rd`

---

## Architecture Overview: Adaptive Self-RAG Pipeline

```mermaid
flowchart LR
    %% Adaptive Self-RAG Architecture (Landscape View)

    subgraph InputStage ["1. Query & Contextualization"]
        direction TB
        UserQuery(["User Question"])
        Memory["Sliding Window History<br/>(Last 6 Turns)"]
        Rewrite["Query Reformulator<br/>(Coreference Resolution)"]
        UserQuery --> Rewrite
        Memory --> Rewrite
    end

    subgraph RetrievalStage ["2. Hybrid Retrieval & Reranking"]
        direction TB
        subgraph ConcurrentRetrieval ["Tenant-Scoped Search"]
            direction TB
            DenseSearch["Dense Vector Search<br/>(pgvector Cosine Sim)"]
            SparseSearch["Sparse Lexical Search<br/>(PostgreSQL FTS / BM25)"]
        end
        RRF["Reciprocal Rank Fusion<br/>(Score Fusion: k=60)"]
        Reranker["Cross-Encoder Reranker<br/>(ms-marco-MiniLM-L-6-v2)"]
        
        DenseSearch --> RRF
        SparseSearch --> RRF
        RRF --> Reranker
    end

    subgraph RoutingStage ["3. Relevance Evaluation"]
        direction TB
        DocGrader{"Relevance Filter<br/>Score >= 0.40?"}
        WebSearch["Dynamic Web Search<br/>(DuckDuckGo / Tavily)"]
    end

    subgraph GenerationStage ["4. Self-RAG Verification Loop"]
        direction TB
        Generator["Contextual Generator<br/>(Claude Haiku / Strict Prompt)"]
        NLIEval{"DeBERTa-v3 NLI<br/>Entailment Check"}
        StrictRetry["Anti-Hallucination<br/>Regeneration Prompt"]
        
        Generator --> NLIEval
        NLIEval -- "Contradiction / Hallucination<br/>(Retry <= 2)" --> StrictRetry
        StrictRetry --> Generator
    end

    subgraph OutputStage ["5. Observability & Delivery"]
        direction TB
        FinalAnswer(["Verified Response<br/>+ Scored Citations [1][2]"])
        Telemetry["LangSmith Observability<br/>(Spans, Traces, Prompt v1.0.0)"]
    end

    %% Cross-stage transitions (Left to Right)
    Rewrite -->|"Standalone Query"| ConcurrentRetrieval
    Reranker -->|"Top-K Candidates"| DocGrader
    DocGrader -- "Relevant Chunks" --> Generator
    DocGrader -- "Out of Scope / Low Score" --> WebSearch
    WebSearch --> FinalAnswer
    NLIEval -- "Entailment Score >= 0.50" --> FinalAnswer
    FinalAnswer -. "Live Telemetry" .-> Telemetry

    %% Styling
    classDef startEnd fill:#0284c7,stroke:#0369a1,color:#ffffff,stroke-width:2px;
    classDef step fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:1.5px;
    classDef decision fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px;
    classDef retry fill:#fee2e2,stroke:#dc2626,color:#991b1b,stroke-width:1.5px;
    classDef obs fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px;

    class UserQuery,FinalAnswer startEnd;
    class Memory,Rewrite,DenseSearch,SparseSearch,RRF,Reranker,Generator,WebSearch step;
    class DocGrader,NLIEval decision;
    class StrictRetry retry;
    class Telemetry obs;
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

### MCP Architecture: Autonomous Agent & FastMCP Server

```mermaid
flowchart LR
    %% Model Context Protocol (MCP) Architecture (Landscape Interview View)

    subgraph Clients ["1. AI Host & Client Layer"]
        direction TB
        AgentCLI["Autonomous MCP Agent<br/>(app/mcp/agent.py)"]
        IDE["Claude Desktop / Cursor<br/>(External AI Host)"]
        ReActLoop["ReAct Reasoning Loop<br/>(Prompt -> Tool Use -> Synthesize)"]
        AgentCLI --> ReActLoop
        IDE --> ReActLoop
    end

    subgraph Protocol ["2. Protocol & Transport (JSON-RPC)"]
        direction TB
        StdioPipe["Bidirectional stdio Stream<br/>(Clean Subprocess Pipe)"]
        Handshake["Dynamic Handshake<br/>(initialize + tools/list)"]
        ToolCall["JSON-RPC Execution<br/>(tools/call)"]
        StdioPipe --> Handshake
        StdioPipe --> ToolCall
    end

    subgraph Server ["3. FastMCP Server & Tenancy Guard"]
        direction TB
        MCPEngine["FastMCP Server<br/>(app/mcp/server.py)"]
        TenantGuard["Tenant Isolation Guard<br/>(MCP_ALLOWED_TENANT_IDS)"]
        MCPEngine --> TenantGuard
    end

    subgraph Tools ["4. Exposed MCP Tool APIs"]
        direction TB
        T1["list_workspaces<br/>(Discover Active Tenants)"]
        T2["semantic_search_chunks<br/>(Raw pgvector Vectors)"]
        T3["self_rag_query<br/>(LangGraph Self-RAG Engine)"]
    end

    subgraph Backends ["5. Knowledge Plane & Observability"]
        direction TB
        PGVector[("PostgreSQL 16 (pgvector)<br/>Tenant-Isolated Chunks")]
        LangGraphPipe["LangGraph Self-RAG<br/>(NLI Anti-Hallucination)"]
        LangSmithTrace["LangSmith Tracing<br/>(@traceable Spans & Metrics)"]
    end

    %% Flow connections (Left to Right)
    ReActLoop <-->|"stdio JSON-RPC"| StdioPipe
    ToolCall -->|"Dispatch Tool Call"| MCPEngine
    TenantGuard --> T1 & T2 & T3
    
    T1 -. "Tenant List JSON" .-> ReActLoop
    T2 -->|"Vector Cosine Query"| PGVector
    T3 -->|"Execute Adaptive Graph"| LangGraphPipe
    LangGraphPipe --> PGVector
    T3 -. "Telemetry Spans" .-> LangSmithTrace

    %% Styling
    classDef clientNode fill:#0284c7,stroke:#0369a1,color:#ffffff,stroke-width:2px;
    classDef protoNode fill:#f1f5f9,stroke:#64748b,color:#0f172a,stroke-width:1.5px;
    classDef serverNode fill:#ede7f6,stroke:#6d28d9,color:#3b0764,stroke-width:2px;
    classDef toolNode fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px;
    classDef dataNode fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px;

    class AgentCLI,IDE,ReActLoop clientNode;
    class StdioPipe,Handshake,ToolCall protoNode;
    class MCPEngine,TenantGuard serverNode;
    class T1,T2,T3 toolNode;
    class PGVector,LangGraphPipe,LangSmithTrace dataNode;
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

# 3. Live LLM mode: Claude tool-calling loop (AsyncAnthropic) over MCP:
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
        "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/saas_db",
        "MCP_ALLOWED_TENANT_IDS": "<workspace-uuid>,<workspace-uuid>"
      }
    }
  }
}
```

`MCP_ALLOWED_TENANT_IDS` scopes the server to specific workspaces: `list_workspaces` only returns them, and a `tenant_id` outside the list is reported as `TenantNotFound` (indistinguishable from a nonexistent one) before any retrieval runs. Leave it empty only for a local stdio server.

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

**Shared-schema multi-tenancy with PostgreSQL RLS.** Every tenant-owned table
carries a `tenant_id` foreign key. All reads and writes go through `TenantScopedRepository`
(`backend/app/services/base.py`), which applies `WHERE tenant_id = :tenant_id`
and refuses to persist or delete a row belonging to another tenant. On PostgreSQL,
this is backed by database-level Row Level Security (`FORCE ROW LEVEL SECURITY` with
`current_setting('app.tenant_id')::uuid`) across `documents`, `document_chunks`,
`conversations`, and `messages`, ensuring that even an arbitrary SQL injection
(`OR 1=1`) cannot cross tenant boundaries. Retrieval is filtered inside SQL, so the
assistant physically cannot quote another workspace's documents - `tests/test_assistant.py`
and `tests/test_postgres_integration.py` assert exactly that.

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
| `ANTHROPIC_CHAT_MODEL` | `claude-haiku-4-5` | Anthropic Claude model checkpoint |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` (local embeddings) or `fake` |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Local Hugging Face sentence embedding model |
| `EMBEDDING_DIMENSIONS` | `384` | Embedding vector width |
| `HALLUCINATION_PROVIDER` | `deberta` | `deberta` (local DeBERTa-v3 cross-encoder), `llm`, or `fake` |
| `DEBERTA_MODEL_NAME` | `cross-encoder/nli-deberta-v3-small` | Hugging Face cross-encoder model checkpoint |
| `NLI_ENTAILMENT_THRESHOLD` | `0.5` | Minimum entailment probability for factual grounding |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `250` / `40` | Tokens (token-based via `count_tokens`) |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `5` / `0.40` | Retrieval budget and tuned cosine similarity floor |
| `MCP_ALLOWED_TENANT_IDS` | *(empty = all)* | Comma-separated workspace UUIDs the MCP server may expose; disallowed tenants are reported as not found |

Further reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
[`docs/RAG.md`](docs/RAG.md).
