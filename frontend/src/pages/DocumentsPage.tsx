import { useRef, useState, type FormEvent } from 'react'

import { documentApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useAsyncAction, useAsyncData } from '@/hooks/useAsync'
import {
  Banner,
  Button,
  Card,
  EmptyState,
  Spinner,
  StatusBadge,
  formatBytes,
  formatDate,
} from '@/components/ui'
import { roleAtLeast } from '@/types/api'

export function DocumentsPage() {
  const { activeWorkspaceId, activeRole } = useAuth()
  const tenantId = activeWorkspaceId as string
  const canEdit = roleAtLeast(activeRole ?? undefined, 'member')

  const { data, loading, error, reload } = useAsyncData(
    () => documentApi.list(tenantId),
    [tenantId],
  )

  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  const create = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    await documentApi.createFromText(tenantId, title, content)
    setTitle('')
    setContent('')
    reload()
  })

  const upload = useAsyncAction(async (file: File) => {
    await documentApi.upload(tenantId, file)
    if (fileInput.current) fileInput.current.value = ''
    reload()
  })

  const remove = useAsyncAction(async (documentId: string) => {
    await documentApi.remove(tenantId, documentId)
    reload()
  })

  const reindex = useAsyncAction(async (documentId: string) => {
    await documentApi.reindex(tenantId, documentId)
    reload()
  })

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">Documentation</h1>
          <p className="page__subtitle">
            Everything the assistant is allowed to answer from, scoped to this workspace.
          </p>
        </div>
      </header>

      <Banner tone="error">{error ?? create.error ?? upload.error ?? remove.error ?? reindex.error}</Banner>

      {canEdit ? (
        <div className="split">
          <Card title="Add from text">
            <form onSubmit={(event) => void create.run(event)}>
              <label className="field">
                <span>Title</span>
                <input value={title} onChange={(event) => setTitle(event.target.value)} required />
              </label>
              <label className="field">
                <span>Content</span>
                <textarea
                  rows={8}
                  value={content}
                  onChange={(event) => setContent(event.target.value)}
                  placeholder="# Heading&#10;&#10;Paste Markdown here…"
                  required
                />
              </label>
              <Button type="submit" loading={create.pending}>
                Index document
              </Button>
            </form>
          </Card>

          <Card title="Upload a file">
            <p className="muted">Markdown, plain text, CSV, JSON or PDF, up to 10 MB.</p>
            <input
              ref={fileInput}
              type="file"
              accept=".md,.txt,.csv,.json,.pdf"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void upload.run(file)
              }}
            />
            {upload.pending ? <Spinner label="Indexing…" /> : null}
          </Card>
        </div>
      ) : null}

      <Card title={`Indexed documents${data ? ` (${data.total})` : ''}`}>
        {loading ? <Spinner /> : null}
        {data && data.items.length === 0 ? (
          <EmptyState
            title="No documentation yet"
            hint={canEdit ? 'Add a document above to give the assistant something to read.' : undefined}
          />
        ) : null}

        {data && data.items.length > 0 ? (
          <table className="table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Status</th>
                <th>Chunks</th>
                <th>Size</th>
                <th>Added</th>
                {canEdit ? <th aria-label="Actions" /> : null}
              </tr>
            </thead>
            <tbody>
              {data.items.map((document) => (
                <tr key={document.id}>
                  <td>
                    <strong>{document.title}</strong>
                    <div className="muted">{document.source_name}</div>
                    {document.error_message ? (
                      <div className="error-text">{document.error_message}</div>
                    ) : null}
                  </td>
                  <td>
                    <StatusBadge status={document.status} />
                  </td>
                  <td>{document.chunk_count}</td>
                  <td>{formatBytes(document.byte_size)}</td>
                  <td>{formatDate(document.created_at)}</td>
                  {canEdit ? (
                    <td className="row-actions">
                      <Button variant="ghost" onClick={() => void reindex.run(document.id)}>
                        Reindex
                      </Button>
                      <Button variant="danger" onClick={() => void remove.run(document.id)}>
                        Delete
                      </Button>
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Card>
    </div>
  )
}
