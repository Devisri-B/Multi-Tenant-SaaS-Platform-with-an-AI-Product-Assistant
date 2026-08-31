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
import { roleAtLeast, type DocumentDetail, type DocumentRecord } from '@/types/api'

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

  // Document Viewer & Editor Modal State
  const [activeDoc, setActiveDoc] = useState<DocumentDetail | null>(null)
  const [modalLoading, setModalLoading] = useState(false)
  const [modalTab, setModalTab] = useState<'content' | 'chunks' | 'edit'>('content')
  const [editTitle, setEditTitle] = useState('')
  const [editContent, setEditContent] = useState('')
  const [modalNotice, setModalNotice] = useState<string | null>(null)

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
    if (activeDoc?.id === documentId) {
      setActiveDoc(null)
    }
    reload()
  })

  const reindex = useAsyncAction(async (documentId: string) => {
    await documentApi.reindex(tenantId, documentId)
    reload()
  })

  const openDocument = async (doc: DocumentRecord, initialTab: 'content' | 'chunks' | 'edit' = 'content') => {
    setModalLoading(true)
    setModalNotice(null)
    setModalTab(initialTab)
    try {
      const detail = await documentApi.get(tenantId, doc.id)
      setActiveDoc(detail)
      setEditTitle(detail.title)
      setEditContent(detail.content)
    } catch (err) {
      console.error('Failed to load document details:', err)
    } finally {
      setModalLoading(false)
    }
  }

  const saveEdit = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    if (!activeDoc) return
    const updated = await documentApi.update(tenantId, activeDoc.id, {
      title: editTitle,
      content: editContent,
    })
    setActiveDoc(updated)
    setModalNotice('Document updated and re-indexed successfully.')
    setModalTab('content')
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
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {data.items.map((document) => (
                <tr key={document.id}>
                  <td>
                    <strong
                      style={{ cursor: 'pointer', color: 'var(--accent)' }}
                      onClick={() => void openDocument(document, 'content')}
                      title="Click to view document"
                    >
                      {document.title}
                    </strong>
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
                  <td className="row-actions">
                    <Button variant="secondary" onClick={() => void openDocument(document, 'content')}>
                      View
                    </Button>
                    {canEdit ? (
                      <>
                        <Button variant="ghost" onClick={() => void openDocument(document, 'edit')}>
                          Edit
                        </Button>
                        <Button variant="ghost" onClick={() => void reindex.run(document.id)}>
                          Reindex
                        </Button>
                        <Button variant="danger" onClick={() => void remove.run(document.id)}>
                          Delete
                        </Button>
                      </>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Card>

      {/* View & Edit Document Modal */}
      {activeDoc ? (
        <div className="modal-overlay" onClick={() => setActiveDoc(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <header className="modal__header">
              <div className="modal__title-group">
                <h2 className="modal__title">{activeDoc.title}</h2>
                <StatusBadge status={activeDoc.status} />
              </div>
              <button
                type="button"
                className="modal__close"
                onClick={() => setActiveDoc(null)}
                aria-label="Close"
              >
                &times;
              </button>
            </header>

            <div className="modal__body">
              {modalNotice ? <Banner tone="success">{modalNotice}</Banner> : null}
              {saveEdit.error ? <Banner tone="error">{saveEdit.error}</Banner> : null}

              {/* Document Overview Metadata */}
              <div className="modal__meta-grid">
                <div className="modal__meta-item">
                  Source File
                  <strong>{activeDoc.source_name}</strong>
                </div>
                <div className="modal__meta-item">
                  Total Size
                  <strong>{formatBytes(activeDoc.byte_size)}</strong>
                </div>
                <div className="modal__meta-item">
                  Vector Chunks
                  <strong>{activeDoc.chunk_count} chunks</strong>
                </div>
                <div className="modal__meta-item">
                  Date Indexed
                  <strong>{formatDate(activeDoc.created_at)}</strong>
                </div>
              </div>

              {/* Navigation Tabs */}
              <div className="modal__tabs">
                <button
                  type="button"
                  className={`modal__tab ${modalTab === 'content' ? 'modal__tab--active' : ''}`}
                  onClick={() => setModalTab('content')}
                >
                  Document Text
                </button>
                <button
                  type="button"
                  className={`modal__tab ${modalTab === 'chunks' ? 'modal__tab--active' : ''}`}
                  onClick={() => setModalTab('chunks')}
                >
                  Vector Chunks ({activeDoc.chunks?.length ?? 0})
                </button>
                {canEdit ? (
                  <button
                    type="button"
                    className={`modal__tab ${modalTab === 'edit' ? 'modal__tab--active' : ''}`}
                    onClick={() => setModalTab('edit')}
                  >
                    Edit & Reindex
                  </button>
                ) : null}
              </div>

              {/* Tab 1: Full Document Text */}
              {modalTab === 'content' ? (
                <div>
                  <div className="modal__content-box">
                    {activeDoc.content || 'No text content available.'}
                  </div>
                </div>
              ) : null}

              {/* Tab 2: Individual Vector Chunks */}
              {modalTab === 'chunks' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {activeDoc.chunks && activeDoc.chunks.length > 0 ? (
                    activeDoc.chunks.map((chunk) => (
                      <div key={chunk.id} className="chunk-card">
                        <div className="chunk-card__header">
                          <span>Chunk #{chunk.ordinal + 1}</span>
                          <span>~{chunk.token_estimate} tokens</span>
                        </div>
                        <pre className="chunk-card__body">{chunk.content}</pre>
                      </div>
                    ))
                  ) : (
                    <p className="muted">No vector chunks indexed yet.</p>
                  )}
                </div>
              ) : null}

              {/* Tab 3: Edit Document Form */}
              {modalTab === 'edit' && canEdit ? (
                <form id="edit-doc-form" onSubmit={(event) => void saveEdit.run(event)}>
                  <label className="field">
                    <span>Document Title</span>
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(event) => setEditTitle(event.target.value)}
                      required
                    />
                  </label>
                  <label className="field">
                    <span>Document Content (Markdown)</span>
                    <textarea
                      rows={12}
                      value={editContent}
                      onChange={(event) => setEditContent(event.target.value)}
                      required
                    />
                  </label>
                  <div className="muted" style={{ textAlign: 'right', fontSize: '12px' }}>
                    {editContent.length} characters
                  </div>
                </form>
              ) : null}
            </div>

            <footer className="modal__footer">
              {modalTab === 'edit' && canEdit ? (
                <>
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => {
                      setModalTab('content')
                      setEditTitle(activeDoc.title)
                      setEditContent(activeDoc.content)
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    form="edit-doc-form"
                    variant="primary"
                    loading={saveEdit.pending}
                  >
                    Save & Reindex
                  </Button>
                </>
              ) : (
                <>
                  {canEdit ? (
                    <Button type="button" variant="secondary" onClick={() => setModalTab('edit')}>
                      Edit Document
                    </Button>
                  ) : null}
                  <Button type="button" variant="primary" onClick={() => setActiveDoc(null)}>
                    Done
                  </Button>
                </>
              )}
            </footer>
          </div>
        </div>
      ) : null}

      {modalLoading ? (
        <div className="modal-overlay">
          <Spinner label="Loading document details…" />
        </div>
      ) : null}
    </div>
  )
}
