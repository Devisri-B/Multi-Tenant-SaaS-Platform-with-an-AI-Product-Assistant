/** Types mirroring the FastAPI schemas in `backend/app/schemas`. */

export type Role = 'viewer' | 'member' | 'admin' | 'owner'
export type MembershipStatus = 'invited' | 'active' | 'suspended'
export type DocumentStatus = 'pending' | 'processing' | 'indexed' | 'failed'
export type TenantPlan = 'free' | 'pro' | 'enterprise'

export interface User {
  id: string
  email: string
  full_name: string | null
  is_active: boolean
  is_superuser: boolean
}

export interface TokenPair {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface MembershipSummary {
  tenant_id: string
  tenant_name: string
  tenant_slug: string
  role: Role
}

export interface SessionResponse {
  user: User
  memberships: MembershipSummary[]
}

export interface RegisterResponse {
  user: User
  tenant_id: string
  tokens: TokenPair
}

export interface Workspace {
  id: string
  name: string
  slug: string
  plan: TenantPlan
  is_active: boolean
  seat_limit: number
  document_limit: number
  settings: Record<string, unknown>
  created_at: string
}

export interface WorkspaceStats {
  tenant_id: string
  member_count: number
  document_count: number
  indexed_document_count: number
  chunk_count: number
  conversation_count: number
}

export interface Member {
  id: string
  user_id: string
  email: string
  full_name: string | null
  role: Role
  status: MembershipStatus
  created_at: string
}

export interface MemberInviteResult {
  member: Member
  invited_new_user: boolean
  temporary_password: string | null
}

export interface DocumentRecord {
  id: string
  tenant_id: string
  title: string
  source_name: string
  content_type: string
  status: DocumentStatus
  byte_size: number
  chunk_count: number
  error_message: string | null
  doc_metadata: Record<string, unknown>
  created_at: string
}

export interface DocumentChunk {
  id: string
  document_id: string
  ordinal: number
  content: string
  token_estimate: number
}

export interface DocumentDetail extends DocumentRecord {
  content: string
  chunks: DocumentChunk[]
}

export interface Citation {
  document_id?: string | null
  document_title: string
  chunk_id?: string | null
  ordinal: number
  score: number
  excerpt: string
  url?: string | null
  source_type?: 'document' | 'web'
}

export interface AskResponse {
  conversation_id: string
  message_id: string
  answer: string
  citations: Citation[]
  latency_ms: number
  used_context: boolean
  source_type?: 'workspace_docs' | 'online_search' | 'none'
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  citations: Citation[]
  created_at: string
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[]
}

export interface Page<T> {
  items: T[]
  total: number
  page: number
  size: number
}

export interface ApiErrorBody {
  code: string
  message: string
  detail?: Record<string, unknown>
}

/** Ordering used by the UI to decide which controls to render. */
export const ROLE_RANK: Record<Role, number> = {
  viewer: 10,
  member: 20,
  admin: 30,
  owner: 40,
}

export function roleAtLeast(role: Role | undefined, minimum: Role): boolean {
  if (!role) return false
  return ROLE_RANK[role] >= ROLE_RANK[minimum]
}
