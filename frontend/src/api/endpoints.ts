/** Typed endpoint functions — one per backend route the UI uses. */

import { api, tokenStore } from '@/api/client'
import type {
  AskResponse,
  ConversationDetail,
  ConversationSummary,
  DocumentDetail,
  DocumentRecord,
  Member,
  MemberInviteResult,
  Page,
  RegisterResponse,
  Role,
  SessionResponse,
  TokenPair,
  Workspace,
  WorkspaceStats,
} from '@/types/api'

// -- auth -------------------------------------------------------------------
export const authApi = {
  login: (email: string, password: string) =>
    api.post<TokenPair>('/auth/login', { email, password }),

  register: (payload: {
    email: string
    password: string
    full_name?: string
    workspace_name: string
  }) => api.post<RegisterResponse>('/auth/register', payload),

  session: () => api.get<SessionResponse>('/auth/me'),

  changePassword: (current_password: string, new_password: string) =>
    api.post<{ message: string }>('/auth/password', { current_password, new_password }),

  logout: () => tokenStore.clear(),
}

// -- workspaces -------------------------------------------------------------
export const workspaceApi = {
  list: () => api.get<Workspace[]>('/workspaces'),

  create: (name: string, slug?: string) =>
    api.post<Workspace>('/workspaces', slug ? { name, slug } : { name }),

  get: (tenantId: string) => api.get<Workspace>(`/workspaces/${tenantId}`),

  update: (tenantId: string, payload: { name?: string; settings?: Record<string, unknown> }) =>
    api.patch<Workspace>(`/workspaces/${tenantId}`, payload),

  stats: (tenantId: string) => api.get<WorkspaceStats>(`/workspaces/${tenantId}/stats`),

  archive: (tenantId: string) => api.delete<{ message: string }>(`/workspaces/${tenantId}`),
}

// -- members ----------------------------------------------------------------
export const memberApi = {
  list: (tenantId: string, page = 1, size = 50) =>
    api.get<Page<Member>>(`/workspaces/${tenantId}/members?page=${page}&size=${size}`),

  invite: (tenantId: string, email: string, role: Role, fullName?: string) =>
    api.post<MemberInviteResult>(`/workspaces/${tenantId}/members`, {
      email,
      role,
      full_name: fullName ?? null,
    }),

  changeRole: (tenantId: string, membershipId: string, role: Role) =>
    api.patch<Member>(`/workspaces/${tenantId}/members/${membershipId}`, { role }),

  remove: (tenantId: string, membershipId: string) =>
    api.delete<{ message: string }>(`/workspaces/${tenantId}/members/${membershipId}`),
}

// -- documents --------------------------------------------------------------
export const documentApi = {
  list: (tenantId: string, page = 1, size = 20) =>
    api.get<Page<DocumentRecord>>(`/workspaces/${tenantId}/documents?page=${page}&size=${size}`),

  createFromText: (tenantId: string, title: string, content: string) =>
    api.post<DocumentRecord>(`/workspaces/${tenantId}/documents`, { title, content }),

  upload: (tenantId: string, file: File, title?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (title) formData.append('title', title)
    return api.upload<DocumentRecord>(`/workspaces/${tenantId}/documents/upload`, formData)
  },

  reindex: (tenantId: string, documentId: string) =>
    api.post<{ document_id: string; status: string; chunk_count: number }>(
      `/workspaces/${tenantId}/documents/${documentId}/reindex`,
    ),

  get: (tenantId: string, documentId: string) =>
    api.get<DocumentDetail>(`/workspaces/${tenantId}/documents/${documentId}`),

  update: (tenantId: string, documentId: string, payload: { title?: string; content?: string }) =>
    api.patch<DocumentDetail>(`/workspaces/${tenantId}/documents/${documentId}`, payload),

  remove: (tenantId: string, documentId: string) =>
    api.delete<{ message: string }>(`/workspaces/${tenantId}/documents/${documentId}`),
}

// -- assistant --------------------------------------------------------------
export const assistantApi = {
  ask: (tenantId: string, question: string, conversationId?: string | null) =>
    api.post<AskResponse>(`/workspaces/${tenantId}/assistant/ask`, {
      question,
      conversation_id: conversationId ?? null,
    }),

  conversations: (tenantId: string) =>
    api.get<Page<ConversationSummary>>(`/workspaces/${tenantId}/assistant/conversations`),

  conversation: (tenantId: string, conversationId: string) =>
    api.get<ConversationDetail>(
      `/workspaces/${tenantId}/assistant/conversations/${conversationId}`,
    ),
}
