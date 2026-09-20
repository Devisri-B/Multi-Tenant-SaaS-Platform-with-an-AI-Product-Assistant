import { useEffect, useRef, useState, type FormEvent } from 'react'

import { assistantApi } from '@/api/endpoints'
import { useAuth } from '@/context/AuthContext'
import { useAsyncAction, useAsyncData } from '@/hooks/useAsync'
import { Banner, Button, Card, EmptyState, Spinner } from '@/components/ui'
import type { ChatMessage, Citation } from '@/types/api'

export function AssistantPage() {
  const { activeWorkspaceId } = useAuth()
  const tenantId = activeWorkspaceId as string

  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const transcriptEnd = useRef<HTMLDivElement>(null)

  const history = useAsyncData(() => assistantApi.conversations(tenantId), [tenantId])
  const metrics = useAsyncData(() => assistantApi.getMetrics(tenantId), [tenantId])

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
        evaluation: response.evaluation,
        ground_truth_score: response.ground_truth_score,
        created_at: new Date().toISOString(),
      },
    ])
    history.reload()
    metrics.reload()
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

  async function handleCopy(message: ChatMessage) {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopiedId(message.id)
      setTimeout(() => setCopiedId(null), 2500)
      await assistantApi.trackTelemetry(tenantId, {
        event_type: 'copy',
        conversation_id: conversationId,
        message_id: message.id,
        event_data: { char_count: message.content.length },
      })
      metrics.reload()
    } catch (err) {
      console.error('Failed to copy text', err)
    }
  }

  async function handleFeedback(message: ChatMessage, rating: 'positive' | 'negative') {
    setMessages((prev) =>
      prev.map((m) => (m.id === message.id ? { ...m, rating } : m)),
    )
    try {
      await assistantApi.trackTelemetry(tenantId, {
        event_type: rating === 'positive' ? 'feedback_positive' : 'feedback_negative',
        conversation_id: conversationId,
        message_id: message.id,
      })
      metrics.reload()
    } catch (err) {
      console.error('Failed to record feedback', err)
    }
  }

  async function handleCitationClick(message: ChatMessage, citation: Citation, index: number) {
    try {
      await assistantApi.trackTelemetry(tenantId, {
        event_type: 'citation_click',
        conversation_id: conversationId,
        message_id: message.id,
        event_data: {
          citation_index: index,
          chunk_id: citation.chunk_id,
          document_id: citation.document_id,
          document_title: citation.document_title,
          source_type: citation.source_type,
        },
      })
      metrics.reload()
    } catch (err) {
      console.error('Failed to track citation click', err)
    }
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

      {/* Production Telemetry & Ground Truth Dashboard */}
      {metrics.data ? (
        <section className="metrics-dashboard" aria-label="RAG Quality & Telemetry Metrics">
          <div className="metrics-dashboard__header">
            <h2 className="metrics-dashboard__title">
              <span>⚡ Production Telemetry & Real-World Ground Truth</span>
            </h2>
            <span className="muted" style={{ fontSize: '12px' }}>
              Evaluated Answers: <strong>{metrics.data.total_evaluations}</strong>
            </span>
          </div>

          <div className="metrics-dashboard__grid">
            <div className="metrics-stat">
              <span className="metrics-stat__label">Numeric Accuracy</span>
              <span className="metrics-stat__value">
                {(metrics.data.numeric_accuracy_avg * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">Deterministic regex</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Entity Grounding</span>
              <span className="metrics-stat__value">
                {(metrics.data.entity_accuracy_avg * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">Token overlap</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Citation Verification</span>
              <span className="metrics-stat__value">
                {(metrics.data.citation_verification_rate_avg * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">Chunk excerpts</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Copy Rate (+1)</span>
              <span className="metrics-stat__value">
                {(metrics.data.copy_rate * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">{metrics.data.copy_count} total copies</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Citation CTR</span>
              <span className="metrics-stat__value">
                {(metrics.data.citation_ctr * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">{metrics.data.citation_clicks_count} clicks</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Friction Rate (0)</span>
              <span className="metrics-stat__value">
                {(metrics.data.friction_rate * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">Re-query friction</span>
            </div>

            <div className="metrics-stat">
              <span className="metrics-stat__label">Ground Truth Score</span>
              <span className="metrics-stat__value" style={{ color: 'var(--accent)' }}>
                {(metrics.data.ground_truth_score * 100).toFixed(0)}%
              </span>
              <span className="metrics-stat__hint">Real satisfaction</span>
            </div>
          </div>
        </section>
      ) : null}

      <Banner tone="error">{ask.error ?? openConversation.error}</Banner>

      <div className="chat-layout">
        <aside className="chat-history">
          <h2 className="chat-history__title">Recent</h2>
          {history.loading ? <Spinner label="Loading..." /> : null}
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
                hint='For example: "How do refunds work?" or "What are the seat limits on the free plan?"'
              />
            ) : null}

            {messages.map((message) => (
              <article key={message.id} className={`bubble bubble--${message.role}`}>
                <p className="bubble__content">{message.content}</p>

                {/* Deterministic Evaluation Badges on Assistant Bubbles */}
                {message.role === 'assistant' && message.evaluation ? (
                  <div className="eval-badge">
                    <span className="eval-badge__pill eval-badge__pill--success">
                      ✓ {(message.evaluation.numeric_accuracy_score * 100).toFixed(0)}% Numeric
                    </span>
                    <span className="eval-badge__pill eval-badge__pill--success">
                      ✓ {(message.evaluation.entity_accuracy_score * 100).toFixed(0)}% Entity
                    </span>
                    <span className="eval-badge__pill eval-badge__pill--info">
                      ✓ {(message.evaluation.citation_verification_rate * 100).toFixed(0)}% Citations
                    </span>
                    {message.evaluation.unsupported_numbers.length > 0 ? (
                      <span className="eval-badge__pill eval-badge__pill--warning">
                        ⚠ Ungrounded: {message.evaluation.unsupported_numbers.join(', ')}
                      </span>
                    ) : null}
                  </div>
                ) : null}

                {/* Telemetry Actions (Copy, Thumbs Up, Thumbs Down) */}
                {message.role === 'assistant' ? (
                  <div className="chat-actions">
                    <button
                      type="button"
                      className={`chat-action-btn ${copiedId === message.id ? 'chat-action-btn--active' : ''}`}
                      onClick={() => void handleCopy(message)}
                      title="Copy answer (+1 high relevance signal)"
                    >
                      {copiedId === message.id ? '✓ Copied!' : '📋 Copy'}
                    </button>

                    <button
                      type="button"
                      className={`chat-action-btn ${message.rating === 'positive' ? 'chat-action-btn--active' : ''}`}
                      onClick={() => void handleFeedback(message, 'positive')}
                      title="Helpful (+1)"
                    >
                      👍 Helpful
                    </button>

                    <button
                      type="button"
                      className={`chat-action-btn ${message.rating === 'negative' ? 'chat-action-btn--active' : ''}`}
                      onClick={() => void handleFeedback(message, 'negative')}
                      title="Not what I asked (0)"
                    >
                      👎 Not helpful
                    </button>
                  </div>
                ) : null}

                {message.citations.length > 0 ? (
                  <ol className="citations">
                    {message.citations.map((citation, index) => (
                      <li
                        key={citation.chunk_id || `${citation.document_title}-${index}`}
                        className="citation"
                        onClick={() => void handleCitationClick(message, citation, index)}
                      >
                        <span className="citation__marker">[{index + 1}]</span>
                        {citation.source_type === 'web' || citation.url ? (
                          <a
                            href={citation.url ?? '#'}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="citation__title citation__link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {citation.document_title}
                          </a>
                        ) : (
                          <span className="citation__title citation__link" style={{ cursor: 'pointer' }}>
                            {citation.document_title}
                          </span>
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

            {ask.pending ? <Spinner label="Finding answers from docs & online search..." /> : null}
            <div ref={transcriptEnd} />
          </div>

          <form className="chat__composer" onSubmit={(event) => void ask.run(event)}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your product..."
              minLength={3}
              required
            />
            <Button type="submit" loading={ask.pending}>
              Ask
            </Button>
          </form>
        </section>
      </div>

      <Card title="How Grounding, Deterministic Accuracy & Telemetry Work">
        <p className="muted">
          Your question is matched against workspace documents using pgvector and LangGraph adaptive routing.
          Every answer is evaluated deterministically in real-time ($0.00 cost):
          <strong> Numeric Accuracy</strong> checks quantities and currency against source passages,
          <strong> Entity Grounding</strong> verifies technical terms, and
          <strong> Citation Verification</strong> authenticates chunk excerpts.
          Real user behavior (Copying answers, clicking citations, or friction re-queries) provides the true production ground truth.
        </p>
      </Card>
    </div>
  )
}
