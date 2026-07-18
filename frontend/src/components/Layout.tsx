/** App shell: workspace switcher, primary navigation, account menu. */

import { NavLink, Outlet, useNavigate } from 'react-router-dom'

import { useAuth } from '@/context/AuthContext'
import { RoleBadge } from '@/components/ui'

const NAV_ITEMS = [
  { to: '/', label: 'Overview', end: true },
  { to: '/assistant', label: 'Assistant' },
  { to: '/documents', label: 'Documentation' },
  { to: '/members', label: 'Members' },
  { to: '/settings', label: 'Settings' },
]

export function Layout() {
  const { user, memberships, activeWorkspaceId, activeRole, selectWorkspace, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <span className="sidebar__logo" aria-hidden />
          <span>Nimbus</span>
        </div>

        <label className="sidebar__label" htmlFor="workspace-switcher">
          Workspace
        </label>
        <select
          id="workspace-switcher"
          className="select"
          value={activeWorkspaceId ?? ''}
          onChange={(event) => selectWorkspace(event.target.value)}
        >
          {memberships.map((membership) => (
            <option key={membership.tenant_id} value={membership.tenant_id}>
              {membership.tenant_name}
            </option>
          ))}
        </select>
        {activeRole ? (
          <div className="sidebar__role">
            Your role: <RoleBadge role={activeRole} />
          </div>
        ) : null}

        <nav className="sidebar__nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="sidebar__user">
            <strong>{user?.full_name ?? user?.email}</strong>
            <span>{user?.email}</span>
          </div>
          <button
            className="btn btn--ghost btn--block"
            onClick={() => {
              logout()
              navigate('/login')
            }}
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  )
}
