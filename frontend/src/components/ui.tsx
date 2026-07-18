/** Presentational building blocks shared across pages. */

import type { ButtonHTMLAttributes, ReactNode } from 'react'

import type { DocumentStatus, Role } from '@/types/api'

export function Button({
  variant = 'primary',
  loading = false,
  children,
  className = '',
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  loading?: boolean
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={`btn btn--${variant} ${className}`.trim()}
    >
      {loading ? <span className="spinner" aria-hidden /> : null}
      {children}
    </button>
  )
}

export function Card({
  title,
  actions,
  children,
}: {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card__header">
          {title ? <h2 className="card__title">{title}</h2> : <span />}
          {actions}
        </header>
      )}
      <div className="card__body">{children}</div>
    </section>
  )
}

export function Banner({
  tone = 'info',
  children,
  onDismiss,
}: {
  tone?: 'info' | 'error' | 'success'
  children: ReactNode
  onDismiss?: () => void
}) {
  if (!children) return null
  return (
    <div className={`banner banner--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <span>{children}</span>
      {onDismiss ? (
        <button className="banner__close" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      ) : null}
    </div>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <span className="spinner spinner--dark" aria-hidden />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <p className="empty__title">{title}</p>
      {hint ? <p className="empty__hint">{hint}</p> : null}
    </div>
  )
}

const STATUS_TONE: Record<DocumentStatus, string> = {
  pending: 'neutral',
  processing: 'info',
  indexed: 'success',
  failed: 'danger',
}

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return <span className={`badge badge--${STATUS_TONE[status]}`}>{status}</span>
}

export function RoleBadge({ role }: { role: Role }) {
  return <span className={`badge badge--role-${role}`}>{role}</span>
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}
