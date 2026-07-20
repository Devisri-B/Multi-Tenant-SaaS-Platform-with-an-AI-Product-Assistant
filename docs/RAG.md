# The product assistant

## Pipeline

```
upload → extract text → chunk → embed → store (tenant_id, embedding)

question → embed → retrieve top-k within tenant → assemble context
        → prompt LLM → answer + citations → persist to conversation
```

### 1. Extraction

`app/rag/ingest.py` decodes Markdown, plain text, CSV and JSON directly, and
uses `pypdf` for PDFs. Anything else is rejected with a `422` rather than
silently indexed as mojibake.

A SHA-256 checksum of the raw bytes is stored per document and uniquely indexed
per tenant, so re-uploading the same file returns `409` instead of duplicating
the index.

### 2. Chunking

`RecursiveCharacterTextSplitter` splits on the separators
`["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]` — headings first, then
paragraphs, then sentences — with a pure-Python fallback if LangChain is
absent. Defaults are 900 characters with 150 of overlap.

The most recent heading seen is carried onto each chunk's metadata, so a chunk
lifted out of the middle of a document still knows which section it came from.

### 3. Embedding

Batched in groups of 64. `EMBEDDING_DIMENSIONS` must match the width in
migration `0002_documents`; changing it requires a migration and a re-index.

Indexing is transactional per document: on failure the document is marked
`failed` with the error message preserved, and its chunks are removed, so a
partially embedded document is never queryable.

### 4. Retrieval

On Postgres:

```sql
SELECT ... , embedding <=> :query AS distance
FROM document_chunks JOIN documents ON ...
WHERE document_chunks.tenant_id = :tenant_id
  AND embedding IS NOT NULL
  AND documents.status = 'indexed'
ORDER BY distance
LIMIT :top_k
```

backed by `CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops)`.

On SQLite the same signature is served by an in-Python cosine scan. Either way
the tenant predicate is inside the query.

Hits below `RAG_MIN_SCORE` (default 0.15 cosine similarity) are discarded. If
nothing survives, the chain returns a fixed "not in your documentation" answer
without calling the LLM at all — cheaper, and it removes the main opportunity
to hallucinate.

### 5. Prompting

The system prompt pins the assistant to the provided context, requires inline
`[n]` citation markers, and forbids outside knowledge. Context is assembled up
to an 8000-character budget; the first chunk is truncated rather than dropped
if it alone exceeds it, so a single long relevant chunk still produces an
answer.

The last four conversation turns are prepended to the question for
follow-up resolution ("and who do I contact?").

### 6. Citations

Every answer returns the chunks it was built from, each with a document title,
chunk ordinal, cosine score and a 320-character excerpt. The prompt-only
`index` key is stripped before serialisation.

## Offline provider

`LLM_PROVIDER=fake` swaps in:

- `FakeEmbeddings` — hashed bag-of-words projected onto the configured
  dimensionality and L2-normalised. Not semantic, but stable and genuinely
  similarity-bearing: shared vocabulary produces higher cosine similarity,
  which is what retrieval tests need to assert.
- `FakeChat` — extracts the context sentences with the highest keyword overlap
  with the question.

This is how CI exercises chunking, embedding, retrieval, ranking, prompt
assembly, citation construction and conversation persistence end to end without
an API key.

## Tuning

| Setting | Raise it when | Lower it when |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | Answers lack surrounding context | Retrieval returns too much noise |
| `RAG_CHUNK_OVERLAP` | Answers get cut mid-procedure | Index size is a concern |
| `RAG_TOP_K` | Questions span several documents | Answers drift off-topic |
| `RAG_MIN_SCORE` | The assistant answers from weak matches | It refuses too often |
