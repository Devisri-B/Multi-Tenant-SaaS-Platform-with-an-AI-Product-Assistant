import { Link } from 'react-router-dom'

import { workspaceApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useAsyncData } from '@/hooks/useAsync'
import { Banner, Card, EmptyState, Spinner } from '@/components/ui'

export function OverviewPage() {
  const { activeWorkspaceId, activeMembership } = useAuth()
  const { data, loading, error } = useAsyncData(
    () => workspaceApi.stats(activeWorkspaceId as string),
    [activeWorkspaceId],
  )

  if (!activeWorkspaceId) {
    return <EmptyState title="No workspace selected" hint="Create one from Settings." />
  }

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">{activeMembership?.tenant_name}</h1>
          <p className="page__subtitle">
            Workspace <code>{activeMembership?.tenant_slug}</code>
          </p>
        </div>
      </header>

      <Banner tone="error">{error}</Banner>
      {loading ? <Spinner label="Loading workspace stats…" /> : null}

      {data ? (
        <div className="stat-grid">
          <StatCard label="Members" value={data.member_count} />
          <StatCard label="Documents" value={data.document_count} />
          <StatCard label="Indexed" value={data.indexed_document_count} />
          <StatCard label="Chunks embedded" value={data.chunk_count} />
          <StatCard label="Conversations" value={data.conversation_count} />
        </div>
      ) : null}

      <Card title="Get started">
        <ol className="steps">
          <li>
            <Link to="/documents">Upload your product documentation</Link> — Markdown, text or PDF.
          </li>
          <li>
            <Link to="/members">Invite your team</Link> and assign roles.
          </li>
          <li>
            <Link to="/assistant">Ask the assistant</Link> a question and check its citations.
          </li>
        </ol>
      </Card>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <span className="stat__value">{value}</span>
      <span className="stat__label">{label}</span>
    </div>
  )
}
