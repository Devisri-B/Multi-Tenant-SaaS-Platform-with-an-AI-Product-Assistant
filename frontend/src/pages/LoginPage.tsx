import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'

import { useAuth } from '@/context/AuthContext'
import { useAsyncAction } from '@/hooks/useAsync'
import { Banner, Button } from '@/components/ui'

export function LoginPage() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const { run, pending, error } = useAsyncAction(login)

  if (user) return <Navigate to="/" replace />

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const outcome = await run(email, password)
    if (outcome !== undefined) navigate('/')
  }

  return (
    <div className="auth">
      <form className="auth__card" onSubmit={handleSubmit}>
        <h1 className="auth__title">Sign in to Nimbus</h1>
        <p className="auth__subtitle">Workspaces and an assistant that knows your docs.</p>

        <Banner tone="error">{error}</Banner>

        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        <Button type="submit" loading={pending} className="btn--block">
          Sign in
        </Button>

        <p className="auth__foot">
          No account yet? <Link to="/register">Create a workspace</Link>
        </p>
      </form>
    </div>
  )
}
