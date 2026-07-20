import { useState, type FormEvent } from 'react'

import { memberApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useAsyncAction, useAsyncData } from '@/hooks/useAsync'
import { Banner, Button, Card, RoleBadge, Spinner, formatDate } from '@/components/ui'
import { roleAtLeast, type Role } from '@/types/api'

const ROLES: Role[] = ['viewer', 'member', 'admin', 'owner']

export function MembersPage() {
  const { activeWorkspaceId, activeRole, user } = useAuth()
  const tenantId = activeWorkspaceId as string
  const canManage = roleAtLeast(activeRole ?? undefined, 'admin')

  const { data, loading, error, reload } = useAsyncData(() => memberApi.list(tenantId), [tenantId])

  const [email, setEmail] = useState('')
  const [role, setRole] = useState<Role>('member')
  const [temporaryPassword, setTemporaryPassword] = useState<string | null>(null)

  const invite = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    const result = await memberApi.invite(tenantId, email, role)
    setEmail('')
    setTemporaryPassword(result.temporary_password)
    reload()
  })

  const changeRole = useAsyncAction(async (membershipId: string, nextRole: Role) => {
    await memberApi.changeRole(tenantId, membershipId, nextRole)
    reload()
  })

  const remove = useAsyncAction(async (membershipId: string) => {
    await memberApi.remove(tenantId, membershipId)
    reload()
  })

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <h1 className="page__title">Members</h1>
          <p className="page__subtitle">
            Roles are cumulative: viewer &lt; member &lt; admin &lt; owner.
          </p>
        </div>
      </header>

      <Banner tone="error">{error ?? invite.error ?? changeRole.error ?? remove.error}</Banner>
      {temporaryPassword ? (
        <Banner tone="success" onDismiss={() => setTemporaryPassword(null)}>
          Temporary password for the new account: <code>{temporaryPassword}</code> — share it
          securely, they should change it on first sign-in.
        </Banner>
      ) : null}

      {canManage ? (
        <Card title="Invite someone">
          <form className="inline-form" onSubmit={(event) => void invite.run(event)}>
            <label className="field">
              <span>Email</span>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>Role</span>
              <select
                className="select"
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
              >
                {ROLES.filter((option) => option !== 'owner' || activeRole === 'owner').map(
                  (option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ),
                )}
              </select>
            </label>
            <Button type="submit" loading={invite.pending}>
              Send invite
            </Button>
          </form>
        </Card>
      ) : null}

      <Card title={`Team${data ? ` (${data.total})` : ''}`}>
        {loading ? <Spinner /> : null}
        {data ? (
          <table className="table">
            <thead>
              <tr>
                <th>Member</th>
                <th>Role</th>
                <th>Status</th>
                <th>Joined</th>
                {canManage ? <th aria-label="Actions" /> : null}
              </tr>
            </thead>
            <tbody>
              {data.items.map((member) => {
                const isSelf = member.user_id === user?.id
                return (
                  <tr key={member.id}>
                    <td>
                      <strong>{member.full_name ?? member.email}</strong>
                      <div className="muted">{member.email}</div>
                    </td>
                    <td>
                      {canManage && !isSelf ? (
                        <select
                          className="select select--compact"
                          value={member.role}
                          onChange={(event) =>
                            void changeRole.run(member.id, event.target.value as Role)
                          }
                        >
                          {ROLES.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <RoleBadge role={member.role} />
                      )}
                    </td>
                    <td>{member.status}</td>
                    <td>{formatDate(member.created_at)}</td>
                    {canManage ? (
                      <td className="row-actions">
                        {isSelf ? (
                          <span className="muted">You</span>
                        ) : (
                          <Button variant="danger" onClick={() => void remove.run(member.id)}>
                            Remove
                          </Button>
                        )}
                      </td>
                    ) : null}
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : null}
      </Card>
    </div>
  )
}
