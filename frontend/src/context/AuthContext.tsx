/** Session state: the signed-in user, their workspaces, and the active one. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { tokenStore } from '@/api/client'
import { authApi } from '@/api/endpoints'
import type { MembershipSummary, Role, User } from '@/types/api'

interface AuthState {
  user: User | null
  memberships: MembershipSummary[]
  activeWorkspaceId: string | null
  loading: boolean
  error: string | null
}

interface AuthContextValue extends AuthState {
  activeMembership: MembershipSummary | null
  activeRole: Role | null
  login: (email: string, password: string) => Promise<void>
  register: (payload: {
    email: string
    password: string
    full_name?: string
    workspace_name: string
  }) => Promise<void>
  logout: () => void
  selectWorkspace: (tenantId: string) => void
  refreshSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

const INITIAL: AuthState = {
  user: null,
  memberships: [],
  activeWorkspaceId: null,
  loading: true,
  error: null,
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(INITIAL)

  const loadSession = useCallback(async () => {
    if (!tokenStore.access) {
      setState({ ...INITIAL, loading: false })
      return
    }
    try {
      const session = await authApi.session()
      const stored = tokenStore.workspaceId
      const valid = session.memberships.some((m) => m.tenant_id === stored)
      const active = valid ? stored : (session.memberships[0]?.tenant_id ?? null)
      tokenStore.setWorkspace(active)
      setState({
        user: session.user,
        memberships: session.memberships,
        activeWorkspaceId: active,
        loading: false,
        error: null,
      })
    } catch {
      tokenStore.clear()
      setState({ ...INITIAL, loading: false })
    }
  }, [])

  useEffect(() => {
    void loadSession()
  }, [loadSession])

  const login = useCallback(
    async (email: string, password: string) => {
      const tokens = await authApi.login(email, password)
      tokenStore.save(tokens)
      await loadSession()
    },
    [loadSession],
  )

  const register = useCallback(
    async (payload: {
      email: string
      password: string
      full_name?: string
      workspace_name: string
    }) => {
      const result = await authApi.register(payload)
      tokenStore.save(result.tokens)
      tokenStore.setWorkspace(result.tenant_id)
      await loadSession()
    },
    [loadSession],
  )

  const logout = useCallback(() => {
    authApi.logout()
    setState({ ...INITIAL, loading: false })
  }, [])

  const selectWorkspace = useCallback((tenantId: string) => {
    tokenStore.setWorkspace(tenantId)
    setState((previous) => ({ ...previous, activeWorkspaceId: tenantId }))
  }, [])

  const value = useMemo<AuthContextValue>(() => {
    const activeMembership =
      state.memberships.find((m) => m.tenant_id === state.activeWorkspaceId) ?? null
    return {
      ...state,
      activeMembership,
      activeRole: activeMembership?.role ?? null,
      login,
      register,
      logout,
      selectWorkspace,
      refreshSession: loadSession,
    }
  }, [state, login, register, logout, selectWorkspace, loadSession])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside an <AuthProvider>.')
  return context
}
