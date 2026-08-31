import { useState, type FormEvent } from 'react'

import { authApi, workspaceApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useTheme } from '@/context/ThemeContext'
import { useAsyncAction } from '@/hooks/useAsync'
import { Banner, Button, Card } from '@/components/ui'
import { roleAtLeast } from '@/types/api'

export function SettingsPage() {
  const { activeWorkspaceId, activeMembership, activeRole, refreshSession, selectWorkspace } =
    useAuth()
  const { theme, setTheme } = useTheme()
  const tenantId = activeWorkspaceId as string

  const [name, setName] = useState(activeMembership?.tenant_name ?? '')
  const [newWorkspace, setNewWorkspace] = useState('')
  const [passwords, setPasswords] = useState({ current: '', next: '' })
  const [notice, setNotice] = useState<string | null>(null)

  const rename = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    await workspaceApi.update(tenantId, { name })
    await refreshSession()
    setNotice('Workspace renamed.')
  })

  const create = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    const workspace = await workspaceApi.create(newWorkspace)
    setNewWorkspace('')
    await refreshSession()
    selectWorkspace(workspace.id)
    setNotice(`Created “${workspace.name}”.`)
  })

  const changePassword = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    await authApi.changePassword(passwords.current, passwords.next)
    setPasswords({ current: '', next: '' })
    setNotice('Password updated.')
  })

  return (
    <div className="page">
      <header className="page__head">
        <h1 className="page__title">Settings</h1>
      </header>

      <Banner tone="error">{rename.error ?? create.error ?? changePassword.error}</Banner>
      <Banner tone="success" onDismiss={() => setNotice(null)}>
        {notice}
      </Banner>

      {roleAtLeast(activeRole ?? undefined, 'admin') ? (
        <Card title="Workspace">
          <form className="inline-form" onSubmit={(event) => void rename.run(event)}>
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={(event) => setName(event.target.value)} required />
            </label>
            <Button type="submit" loading={rename.pending}>
              Save
            </Button>
          </form>
        </Card>
      ) : null}

      <Card title="Create another workspace">
        <p className="muted">
          Each workspace is a separate tenant with its own members, documents and assistant index.
        </p>
        <form className="inline-form" onSubmit={(event) => void create.run(event)}>
          <label className="field">
            <span>Name</span>
            <input
              value={newWorkspace}
              onChange={(event) => setNewWorkspace(event.target.value)}
              placeholder="Orbit CRM"
              required
            />
          </label>
          <Button type="submit" loading={create.pending}>
            Create
          </Button>
        </form>
      </Card>

      <Card title="Appearance">
        <p className="muted">
          Choose your interface theme. Your preference is saved across sessions.
        </p>
        <div style={{ display: 'flex', gap: '10px', marginTop: '14px' }}>
          <Button
            type="button"
            variant={theme === 'light' ? 'primary' : 'secondary'}
            onClick={() => setTheme('light')}
          >
            ☀️ Bright (Light) Mode
          </Button>
          <Button
            type="button"
            variant={theme === 'dark' ? 'primary' : 'secondary'}
            onClick={() => setTheme('dark')}
          >
            🌙 Dark Mode
          </Button>
        </div>
      </Card>

      <Card title="Your password">
        <form className="inline-form" onSubmit={(event) => void changePassword.run(event)}>
          <label className="field">
            <span>Current</span>
            <input
              type="password"
              value={passwords.current}
              onChange={(event) =>
                setPasswords((previous) => ({ ...previous, current: event.target.value }))
              }
              autoComplete="current-password"
              required
            />
          </label>
          <label className="field">
            <span>New</span>
            <input
              type="password"
              value={passwords.next}
              onChange={(event) =>
                setPasswords((previous) => ({ ...previous, next: event.target.value }))
              }
              autoComplete="new-password"
              required
            />
          </label>
          <Button type="submit" loading={changePassword.pending}>
            Update
          </Button>
        </form>
      </Card>
    </div>
  )
}
