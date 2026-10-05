import { useEffect, useState } from 'react'

export interface Fiber {
  channel: number
  times: number[]
}

export interface Showcase {
  duration: number
  frame: number
  center_freqs: number[]
  waveform: [number, number][]
  cochleagram: number[][]
  raster: Fiber[]
  hair_cell: { input: number[]; output: number[] }
}

export type ShowcaseState =
  | { status: 'loading' }
  | { status: 'ready'; data: Showcase }
  | { status: 'failed'; reason: string }

/** Loads what the model produced for the demo clip (built by tools/build_demo.py). */
export function useShowcase(): ShowcaseState {
  const [state, setState] = useState<ShowcaseState>({ status: 'loading' })
  useEffect(() => {
    let cancelled = false
    fetch('/demo/showcase.json')
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => null)
          throw new Error(body?.detail ?? `The server answered ${response.status}.`)
        }
        return response.json() as Promise<Showcase>
      })
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((error: Error) => {
        if (!cancelled) setState({ status: 'failed', reason: error.message })
      })
    return () => {
      cancelled = true
    }
  }, [])
  return state
}
