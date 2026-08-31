import { useEffect, useRef, useState, type FormEvent } from 'react'

import { assistantApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useAsyncAction, useAsyncData } from '@/hooks/useAsync'
import { Banner, Button, Card, EmptyState, Spinner } from '@/components/ui'
import type { ChatMessage } from '@/types/api'

export function AssistantPage() {
  const { activeWorkspaceId } = useAuth()
  const tenantId = activeWorkspaceId as string

  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const transcriptEnd = useRef<HTMLDivElement>(null)

  const history = useAsyncData(() => assistantApi.conversations(tenantId), [tenantId])

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const ask = useAsyncAction(async (event: FormEvent) => {
    event.preventDefault()
    const prompt = question.trim()
    if (!prompt) return

    const optimistic: ChatMessage = {
      id: `pending-${Date.now()}`,
      role: 'user',
      content: prompt,
      citations: [],
      created_at: new Date().toISOString(),
    }
    setMessages((previous) => [...previous, optimistic])
    setQuestion('')

    const response = await assistantApi.ask(tenantId, prompt, conversationId)
    setConversationId(response.conversation_id)
    setMessages((previous) => [
      ...previous,
      {
        id: response.message_id,
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        created_at: new Date().toISOString(),
      },
    ])
    history.reload()
  })

  const openConversation = useAsyncAction(async (id: string) => {
    const detail = await assistantApi.conversation(tenantId, id)
    setConversationId(detail.id)
    setMessages(detail.messages)
  })

  function startNew() {
    setConversationId(null)
    setMessages([])
  }

  return (
    <div className="page page--chat">
      <header className="page__head">
        <div>
          <h1 className="page__title">Product assistant</h1>
          <p className="page__subtitle">
            Answers come from this workspace&apos;s documentation, falling back to online search when not found.
          </p>
        </div>
        <Button variant="secondary" onClick={startNew}>
          New conversation
        </Button>
      </header>

      <Banner tone="error">{ask.error ?? openConversation.error}</Banner>

      <div className="chat-layout">
        <aside className="chat-history">
          <h2 className="chat-history__title">Recent</h2>
          {history.loading ? <Spinner label="Loading…" /> : null}
          {history.data?.items.length === 0 ? <p className="muted">No conversations yet.</p> : null}
          <ul className="chat-history__list">
            {history.data?.items.map((conversation) => (
              <li key={conversation.id}>
                <button
                  className={`chat-history__item${
                    conversation.id === conversationId ? ' chat-history__item--active' : ''
                  }`}
                  onClick={() => void openConversation.run(conversation.id)}
                >
                  {conversation.title}
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="chat">
          <div className="chat__transcript">
            {messages.length === 0 ? (
              <EmptyState
                title="Ask about your product"
                hint="For example: “How do refunds work?” or “What are the seat limits on the free plan?”"
              />
            ) : null}

            {messages.map((message) => (
              <article key={message.id} className={`bubble bubble--${message.role}`}>
                <p className="bubble__content">{message.content}</p>
                {message.citations.length > 0 ? (
                  <ol className="citations">
                    {message.citations.map((citation, index) => (
                      <li
                        key={citation.chunk_id || `${citation.document_title}-${index}`}
                        className="citation"
                      >
                        <span className="citation__marker">[{index + 1}]</span>
                        {citation.source_type === 'web' || citation.url ? (
                          <a
                            href={citation.url ?? '#'}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="citation__title citation__link"
                          >
                            {citation.document_title} ↗
                          </a>
                        ) : (
                          <span className="citation__title">{citation.document_title}</span>
                        )}
                        <span className="citation__score">
                          {citation.source_type === 'web'
                            ? 'Online Search'
                            : `${(citation.score * 100).toFixed(0)}% match`}
                        </span>
                        <p className="citation__excerpt">{citation.excerpt}</p>
                      </li>
                    ))}
                  </ol>
                ) : null}
              </article>
            ))}

            {ask.pending ? <Spinner label="Finding answers from docs & online search…" /> : null}
            <div ref={transcriptEnd} />
          </div>

          <form className="chat__composer" onSubmit={(event) => void ask.run(event)}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your product…"
              minLength={3}
              required
            />
            <Button type="submit" loading={ask.pending}>
              Ask
            </Button>
          </form>
        </section>
      </div>

      <Card title="How grounding & online routing work">
        <p className="muted">
          Your question is embedded and matched against chunks of this workspace&apos;s documents.
          Retrieval is filtered by workspace in SQL, guaranteeing multi-tenant isolation.
          If nothing scores above the relevance threshold, LangGraph dynamically routes your question
          to online web search to synthesize a helpful, citation-backed answer.
        </p>
      </Card>
    </div>
  )
}
