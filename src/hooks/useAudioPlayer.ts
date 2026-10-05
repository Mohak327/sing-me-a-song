import { useCallback, useEffect, useRef, useState } from 'react'

export interface AudioPlayer<T extends string> {
  source: T
  playing: boolean
  /** Current playback position in seconds; read it every animation frame. */
  time: () => number
  toggle: () => void
  /** Switch version, keeping the playback position. */
  select: (name: T) => void
  /** Switch version and start playing. */
  listen: (name: T) => void
  seek: (seconds: number) => void
}

/** Only one player on the page makes sound at a time. */
let silenceOthers: (() => void) | null = null

/** One clip in several versions, sharing a single playback position. */
export function useAudioPlayer<T extends string>(urls: Record<T, string>, initial: T): AudioPlayer<T> {
  const tracks = useRef<Partial<Record<T, HTMLAudioElement>>>({})
  const current = useRef<T>(initial)
  const [source, setSource] = useState<T>(initial)
  const [playing, setPlaying] = useState(false)
  const key = JSON.stringify(urls)

  useEffect(() => {
    const store: Partial<Record<T, HTMLAudioElement>> = {}
    const entries = Object.entries(JSON.parse(key) as Record<T, string>) as [T, string][]
    for (const [name, url] of entries) {
      const audio = new Audio(url)
      audio.preload = 'auto'
      audio.addEventListener('ended', () => {
        if (current.current === name) {
          audio.currentTime = 0
          setPlaying(false)
        }
      })
      store[name] = audio
    }
    tracks.current = store
    current.current = initial
    setSource(initial)
    setPlaying(false)
    return () => {
      for (const [name] of entries) store[name]?.pause()
    }
    // `initial` only matters when the set of clips changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const stop = useCallback(() => {
    tracks.current[current.current]?.pause()
    setPlaying(false)
  }, [])

  const start = useCallback((audio: HTMLAudioElement) => {
    if (silenceOthers && silenceOthers !== stop) silenceOthers()
    silenceOthers = stop
    void audio.play()
    setPlaying(true)
  }, [stop])

  const time = useCallback(() => tracks.current[current.current]?.currentTime ?? 0, [])

  const toggle = useCallback(() => {
    const audio = tracks.current[current.current]
    if (!audio) return
    if (audio.paused) start(audio)
    else stop()
  }, [start, stop])

  const select = useCallback((name: T) => {
    const from = tracks.current[current.current]
    const to = tracks.current[name]
    if (!from || !to || name === current.current) return
    const wasPlaying = !from.paused
    from.pause()
    to.currentTime = from.currentTime
    current.current = name
    setSource(name)
    if (wasPlaying) start(to)
  }, [start])

  const listen = useCallback((name: T) => {
    select(name)
    const audio = tracks.current[name]
    if (audio?.paused) start(audio)
  }, [select, start])

  const seek = useCallback((seconds: number) => {
    const audio = tracks.current[current.current]
    if (audio) audio.currentTime = seconds
  }, [])

  return { source, playing, time, toggle, select, listen, seek }
}
