import { useCallback, useEffect, useRef, useState } from 'react'
import { assembleShowcase, decodeAudio, encodeWav, snrDb, splitBlocks, type HeardBlock } from '../lib/clip'
import type { Showcase } from './useShowcase'

export interface Limits {
  sample_rate: number
  block_seconds: number
  max_seconds: number
  concurrency: number
}

export interface RegenerationResult {
  source: string
  duration: number
  trimmed: boolean
  maxSeconds: number
  totalFibers: number
  snrDb: number
  spikes: number
  /** Playable files made in the browser from the original and regenerated samples. */
  urls: { original: string; regenerated: string }
}

export type Regeneration =
  | { status: 'idle' }
  | { status: 'preparing' }
  | { status: 'running'; done: number; total: number; elapsed: number; duration: number }
  | { status: 'done'; result: RegenerationResult; showcase: Showcase }
  | { status: 'failed'; reason: string }

interface BlockAnswer extends HeardBlock {
  audio: string
  signal: number
  noise: number
  spikes: number
}

const DEMO = '/demo/original.wav'
const CHANNELS = 32
const NOT_ANSWERING = 'The Resound server is not answering. Try again in a moment.'

async function reason(response: Response): Promise<string> {
  const body = await response.json().catch(() => null)
  return typeof body?.detail === 'string' ? body.detail : `The server answered ${response.status}.`
}

function samplesFromBase64(text: string): Float32Array {
  const bytes = Uint8Array.from(atob(text), (character) => character.charCodeAt(0))
  return new Float32Array(bytes.buffer)
}

/** How many raster rows to ask for per channel: fewer for longer clips, to keep the chart light. */
function rowsFor(seconds: number): number {
  if (seconds <= 15) return 4
  return seconds <= 30 ? 2 : 1
}

export async function fetchLimits(signal?: AbortSignal): Promise<Limits> {
  const response = await fetch('/api/limits', { signal })
  if (!response.ok) throw new Error(await reason(response))
  return response.json()
}

/**
 * Hears a clip through the server one block at a time and rebuilds it here.
 * The server keeps nothing: this hook cuts the clip, sends the blocks a few at
 * a time, and joins the answers.
 */
export function useRegeneration() {
  const [state, setState] = useState<Regeneration>({ status: 'idle' })
  const abort = useRef<AbortController | null>(null)
  const urls = useRef<string[]>([])

  const release = useCallback(() => {
    urls.current.forEach((url) => URL.revokeObjectURL(url))
    urls.current = []
  }, [])

  useEffect(() => () => {
    abort.current?.abort()
    release()
  }, [release])

  const start = useCallback(async (file: File | null, fibers: number, jitter: number) => {
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    const { signal } = controller
    setState({ status: 'preparing' })
    try {
      const limits = await fetchLimits(signal)
      const data = file ? await file.arrayBuffer() : await (await fetch(DEMO, { signal })).arrayBuffer()
      const decoded = await decodeAudio(data, limits.sample_rate)
      const most = Math.round(limits.max_seconds * limits.sample_rate)
      const samples = decoded.subarray(0, most)
      const duration = samples.length / limits.sample_rate
      const blocks = splitBlocks(samples, Math.round(limits.block_seconds * limits.sample_rate))
      const answers: BlockAnswer[] = new Array(blocks.length)
      const began = performance.now()
      const query = new URLSearchParams({ fibers: String(fibers), jitter: String(jitter), shown: String(rowsFor(duration)) })
      let next = 0
      let done = 0
      const report = () => setState({
        status: 'running', done, total: blocks.length, duration, elapsed: (performance.now() - began) / 1000,
      })
      report()
      const ticker = setInterval(report, 1000)

      const worker = async () => {
        while (next < blocks.length) {
          const index = next++
          const response = await fetch(`/api/blocks?${query}`, {
            method: 'POST', signal, headers: { 'content-type': 'application/octet-stream' },
            body: blocks[index].slice().buffer,
          })
          if (!response.ok) throw new Error(await reason(response))
          answers[index] = await response.json()
          done += 1
          report()
        }
      }
      try {
        await Promise.all(Array.from({ length: Math.min(limits.concurrency, blocks.length) }, worker))
      } finally {
        clearInterval(ticker)
      }

      const rebuilt = new Float32Array(samples.length)
      let offset = 0
      for (const answer of answers) {
        const part = samplesFromBase64(answer.audio)
        rebuilt.set(part, offset)
        offset += part.length
      }
      release()
      const made = {
        original: URL.createObjectURL(encodeWav(samples, limits.sample_rate)),
        regenerated: URL.createObjectURL(encodeWav(rebuilt, limits.sample_rate)),
      }
      urls.current = [made.original, made.regenerated]
      setState({
        status: 'done',
        showcase: assembleShowcase(answers),
        result: {
          source: file ? file.name : 'The demo clip',
          duration,
          trimmed: decoded.length > most,
          maxSeconds: limits.max_seconds,
          totalFibers: fibers * CHANNELS,
          snrDb: snrDb(answers.reduce((sum, a) => sum + a.signal, 0), answers.reduce((sum, a) => sum + a.noise, 0)),
          spikes: answers.reduce((sum, a) => sum + a.spikes, 0),
          urls: made,
        },
      })
    } catch (error) {
      if (signal.aborted) return
      controller.abort() // stop the other blocks
      const message = error instanceof TypeError ? NOT_ANSWERING : (error as Error).message
      setState({ status: 'failed', reason: message })
    }
  }, [release])

  return { state, start }
}
