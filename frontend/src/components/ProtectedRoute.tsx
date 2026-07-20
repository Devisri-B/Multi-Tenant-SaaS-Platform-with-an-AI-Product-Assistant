/** Gates the authenticated area and, optionally, a minimum workspace role. */

import { Navigate, useLocation } from 'react-router-dom'
import type { ReactNode } from 'react'

import { useAuth } from '@/context/AuthContext'
import { Spinner } from '@/components/ui'
import { roleAtLeast, type Role } from '@/types/api'

export function ProtectedRoute({
  children,
  minimumRole,
}: {
  children: ReactNode
  minimumRole?: Role
}) {
  const { user, loading, activeRole } = useAuth()
  const location = useLocation()

  if (loading) return <Spinner label="Checking your session…" />
  if (!user) return <Navigate to="/login" state={{ from: location.pathname }} replace />

  if (minimumRole && !roleAtLeast(activeRole ?? undefined, minimumRole)) {
    return (
      <div className="page">
        <h1 className="page__title">Not available for your role</h1>
        <p className="page__subtitle">
          This section requires the <strong>{minimumRole}</strong> role in the selected workspace.
        </p>
      </div>
    )
  }

  return <>{children}</>
}
