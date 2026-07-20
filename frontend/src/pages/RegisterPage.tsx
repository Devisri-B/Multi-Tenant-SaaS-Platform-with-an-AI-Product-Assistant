import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'

import { useAuth } from '@/context/AuthContext'
import { useAsyncAction } from '@/hooks/useAsync'
import { Banner, Button } from '@/components/ui'

export function RegisterPage() {
  const { user, register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({
    email: '',
    password: '',
    full_name: '',
    workspace_name: '',
  })
  const { run, pending, error } = useAsyncAction(register)

  if (user) return <Navigate to="/" replace />

  function update(key: keyof typeof form, value: string) {
    setForm((previous) => ({ ...previous, [key]: value }))
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const outcome = await run({
      email: form.email,
      password: form.password,
      full_name: form.full_name || undefined,
      workspace_name: form.workspace_name,
    })
    if (outcome !== undefined) navigate('/')
  }

  return (
    <div className="auth">
      <form className="auth__card" onSubmit={handleSubmit}>
        <h1 className="auth__title">Create your workspace</h1>
        <p className="auth__subtitle">
          One workspace per product. You can add more once you are in.
        </p>

        <Banner tone="error">{error}</Banner>

        <label className="field">
          <span>Workspace name</span>
          <input
            value={form.workspace_name}
            onChange={(event) => update('workspace_name', event.target.value)}
            placeholder="Acme Analytics"
            required
          />
        </label>

        <label className="field">
          <span>Your name</span>
          <input
            value={form.full_name}
            onChange={(event) => update('full_name', event.target.value)}
            autoComplete="name"
          />
        </label>

        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={form.email}
            onChange={(event) => update('email', event.target.value)}
            autoComplete="email"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={form.password}
            onChange={(event) => update('password', event.target.value)}
            autoComplete="new-password"
            required
          />
          <small className="field__hint">
            At least 10 characters, mixed case, and one digit.
          </small>
        </label>

        <Button type="submit" loading={pending} className="btn--block">
          Create workspace
        </Button>

        <p className="auth__foot">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </form>
    </div>
  )
}
