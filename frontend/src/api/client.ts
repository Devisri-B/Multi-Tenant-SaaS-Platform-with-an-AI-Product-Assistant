/**
 * Thin fetch wrapper around the backend.
 *
 * Responsibilities kept here so components never touch fetch directly:
 *  - attaches the bearer token and the active workspace header
 *  - transparently refreshes an expired access token exactly once per request
 *  - converts non-2xx responses into a typed `ApiError`
 */

import type { ApiErrorBody, TokenPair } from '@/types/api'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
const API_PREFIX = '/api/v1'

const ACCESS_TOKEN_KEY = 'nimbus.access_token'
const REFRESH_TOKEN_KEY = 'nimbus.refresh_token'
const WORKSPACE_KEY = 'nimbus.workspace_id'

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly detail?: Record<string, unknown>

  constructor(status: number, body: ApiErrorBody) {
    super(body.message)
    this.name = 'ApiError'
    this.status = status
    this.code = body.code
    this.detail = body.detail
  }

  get isAuthError(): boolean {
    return this.status === 401
  }

  get isPermissionError(): boolean {
    return this.status === 403
  }
}

export const tokenStore = {
  get access(): string | null {
    return sessionStorage.getItem(ACCESS_TOKEN_KEY)
  },
  get refresh(): string | null {
    return sessionStorage.getItem(REFRESH_TOKEN_KEY)
  },
  get workspaceId(): string | null {
    return sessionStorage.getItem(WORKSPACE_KEY)
  },
  save(tokens: TokenPair): void {
    sessionStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token)
    sessionStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token)
  },
  setWorkspace(workspaceId: string | null): void {
    if (workspaceId) sessionStorage.setItem(WORKSPACE_KEY, workspaceId)
    else sessionStorage.removeItem(WORKSPACE_KEY)
  },
  clear(): void {
    sessionStorage.removeItem(ACCESS_TOKEN_KEY)
    sessionStorage.removeItem(REFRESH_TOKEN_KEY)
    sessionStorage.removeItem(WORKSPACE_KEY)
  },
}

interface RequestOptions {
  method?: string
  body?: unknown
  workspaceId?: string | null
  formData?: FormData
  signal?: AbortSignal
  /** Internal: prevents an infinite refresh loop. */
  _retried?: boolean
}

async function parseError(response: Response): Promise<ApiError> {
  let body: ApiErrorBody = {
    code: 'unknown_error',
    message: `Request failed with status ${response.status}.`,
  }
  try {
    const parsed = (await response.json()) as Partial<ApiErrorBody>
    if (parsed && typeof parsed.message === 'string') {
      body = { code: parsed.code ?? 'unknown_error', message: parsed.message, detail: parsed.detail }
    }
  } catch {
    // Response had no JSON body — keep the generic message.
  }
  return new ApiError(response.status, body)
}

async function refreshAccessToken(): Promise<boolean> {
  const refresh = tokenStore.refresh
  if (!refresh) return false

  const response = await fetch(`${BASE_URL}${API_PREFIX}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refresh }),
  })
  if (!response.ok) {
    tokenStore.clear()
    return false
  }
  tokenStore.save((await response.json()) as TokenPair)
  return true
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, formData, workspaceId, signal } = options

  const headers: Record<string, string> = {}
  const access = tokenStore.access
  if (access) headers.Authorization = `Bearer ${access}`

  const activeWorkspace = workspaceId ?? tokenStore.workspaceId
  if (activeWorkspace) headers['X-Workspace-Id'] = activeWorkspace
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
    method,
    headers,
    body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
    signal,
  })

  if (response.status === 401 && !options._retried && tokenStore.refresh) {
    if (await refreshAccessToken()) {
      return request<T>(path, { ...options, _retried: true })
    }
  }

  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  get: <T>(path: string, options?: RequestOptions) => request<T>(path, { ...options, method: 'GET' }),
  post: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'POST', body }),
  patch: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'PATCH', body }),
  delete: <T>(path: string, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'DELETE' }),
  upload: <T>(path: string, formData: FormData, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'POST', formData }),
}
