/** Small async-state helpers used instead of pulling in a data-fetching library. */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from '@/api/client'

export function messageFromError(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return 'Something went wrong.'
}

interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

/** Runs `loader` on mount and whenever a dependency changes. */
export function useAsyncData<T>(
  loader: () => Promise<T>,
  deps: unknown[],
): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null,
  })
  const [nonce, setNonce] = useState(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setState((previous) => ({ ...previous, loading: true, error: null }))
    loader()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null })
      })
      .catch((error: unknown) => {
        if (!cancelled) setState({ data: null, loading: false, error: messageFromError(error) })
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((value) => value + 1), [])
  return { ...state, reload }
}

/** Wraps a one-off action (submit, delete) with pending and error state. */
export function useAsyncAction<Args extends unknown[], T>(
  action: (...args: Args) => Promise<T>,
): {
  run: (...args: Args) => Promise<T | undefined>
  pending: boolean
  error: string | null
  clearError: () => void
} {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(
    async (...args: Args) => {
      setPending(true)
      setError(null)
      try {
        return await action(...args)
      } catch (caught: unknown) {
        setError(messageFromError(caught))
        return undefined
      } finally {
        setPending(false)
      }
    },
    [action],
  )

  return { run, pending, error, clearError: () => setError(null) }
}
