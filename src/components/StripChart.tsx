import { useEffect, useRef } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import type { Showcase } from '../hooks/useShowcase'
import { spikesInWindow, visibleWindow } from '../lib/chart'

interface Props {
  data: Showcase
  time: () => number
  onSeek: (seconds: number) => void
  /** Legend for the raster, e.g. how many of the fibres are drawn. */
  fiberNote?: string
  /** Draw the chart in from the left when it first appears. */
  reveal?: boolean
}

const CHART_HEIGHT = 330
const LOUPE_HEIGHT = 170
const LOUPE_SPAN = 0.06 // seconds shown in the close-up
const ZONES = { wave: [8, 70], cochlea: [84, 204], raster: [218, 322] } as const

function pens(element: HTMLElement) {
  const style = getComputedStyle(element)
  const read = (name: string) => style.getPropertyValue(name).trim()
  return { ink: read('--ink'), blue: read('--pen-blue'), red: read('--pen-red'), rule: read('--rule') }
}

function fit(canvas: HTMLCanvasElement, height: number) {
  const ratio = window.devicePixelRatio || 1
  const width = canvas.clientWidth
  canvas.width = Math.round(width * ratio)
  canvas.height = Math.round(height * ratio)
  const context = canvas.getContext('2d')!
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  return { context, width }
}

/** The whole clip: sound, cochlea, nerve. Drawn once, then reused every frame. */
function drawRecording(canvas: HTMLCanvasElement, data: Showcase, colors: ReturnType<typeof pens>) {
  const { context, width } = fit(canvas, CHART_HEIGHT)
  context.clearRect(0, 0, width, CHART_HEIGHT)
  const frames = data.waveform.length
  const column = width / frames

  context.strokeStyle = colors.rule
  context.lineWidth = 1
  for (let second = 0; second <= data.duration; second++) {
    const x = Math.round((second / data.duration) * width) + 0.5
    context.beginPath()
    context.moveTo(x, 0)
    context.lineTo(x, CHART_HEIGHT)
    context.stroke()
  }

  const [waveTop, waveBottom] = ZONES.wave
  const middle = (waveTop + waveBottom) / 2
  const half = (waveBottom - waveTop) / 2
  context.strokeStyle = colors.blue
  context.lineWidth = Math.max(column, 1)
  context.beginPath()
  data.waveform.forEach(([low, high], i) => {
    const x = (i + 0.5) * column
    context.moveTo(x, middle - high * half)
    context.lineTo(x, middle - low * half + 0.5)
  })
  context.stroke()

  const [cochleaTop, cochleaBottom] = ZONES.cochlea
  const channels = data.cochleagram.length
  const row = (cochleaBottom - cochleaTop) / channels
  context.fillStyle = colors.ink
  data.cochleagram.forEach((levels, channel) => {
    const y = cochleaBottom - (channel + 1) * row
    levels.forEach((level, i) => {
      if (level < 0.04) return
      context.globalAlpha = Math.pow(level, 1.6)
      context.fillRect(i * column, y, column + 0.5, row + 0.5)
    })
  })

  const [rasterTop, rasterBottom] = ZONES.raster
  const fiberRow = (rasterBottom - rasterTop) / data.raster.length
  context.fillStyle = colors.red
  context.globalAlpha = 0.5
  data.raster.forEach((fiber, index) => {
    const y = rasterBottom - (index + 1) * fiberRow
    for (const spike of fiber.times) {
      context.fillRect((spike / data.duration) * width, y, 1, Math.max(fiberRow, 0.8))
    }
  })
  context.globalAlpha = 1
}

/** A 60 millisecond close-up around the playhead: one tick per spike. */
function drawLoupe(canvas: HTMLCanvasElement, data: Showcase, time: number,
                   colors: ReturnType<typeof pens>) {
  const { context, width } = fit(canvas, LOUPE_HEIGHT)
  const [start, end] = visibleWindow(time, LOUPE_SPAN, data.duration)
  const toX = (seconds: number) => ((seconds - start) / (end - start)) * width
  const top = 6
  const bottom = LOUPE_HEIGHT - 22
  context.clearRect(0, 0, width, LOUPE_HEIGHT)

  context.font = '12px "Figtree Variable", sans-serif'
  context.fillStyle = colors.ink
  context.strokeStyle = colors.rule
  context.lineWidth = 1
  const firstTick = Math.ceil(start / 0.01) * 0.01
  for (let tick = firstTick; tick < end; tick += 0.01) {
    const x = Math.round(toX(tick)) + 0.5
    context.beginPath()
    context.moveTo(x, top)
    context.lineTo(x, bottom)
    context.stroke()
    context.globalAlpha = 0.7
    context.fillText(`${tick.toFixed(2)} s`, x + 4, LOUPE_HEIGHT - 6)
    context.globalAlpha = 1
  }

  const rowHeight = (bottom - top) / data.raster.length
  context.fillStyle = colors.red
  data.raster.forEach((fiber, index) => {
    const [from, to] = spikesInWindow(fiber.times, start, end)
    const y = bottom - (index + 1) * rowHeight
    for (let i = from; i < to; i++) {
      context.fillRect(toX(fiber.times[i]) - 1, y, 2, Math.max(rowHeight - 0.2, 0.9))
    }
  })

  context.strokeStyle = colors.ink
  context.lineWidth = 1.5
  const head = toX(time)
  context.beginPath()
  context.moveTo(head, 0)
  context.lineTo(head, bottom)
  context.stroke()
}

export function StripChart({ data, time, onSeek, fiberNote, reveal = true }: Props) {
  const frame = useRef<HTMLDivElement>(null)
  const recording = useRef<HTMLCanvasElement>(null)
  const loupe = useRef<HTMLCanvasElement>(null)
  const playhead = useRef<HTMLDivElement>(null)
  const slider = useRef<HTMLInputElement>(null)
  const still = useReducedMotion()

  useEffect(() => {
    const host = frame.current!
    const colors = pens(host)
    let last = -1
    let handle = 0
    const redraw = () => {
      drawRecording(recording.current!, data, colors)
      last = -1
    }
    redraw()
    const observer = new ResizeObserver(redraw)
    observer.observe(host)
    document.fonts?.ready.then(() => { last = -1 })

    const tick = () => {
      const now = time()
      if (now !== last) {
        last = now
        const fraction = now / data.duration
        playhead.current!.style.transform = `translateX(${fraction * recording.current!.clientWidth}px)`
        if (document.activeElement !== slider.current) slider.current!.value = String(now)
        drawLoupe(loupe.current!, data, now, colors)
      }
      handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(handle)
      observer.disconnect()
    }
  }, [data, time])

  const seekFromPointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    onSeek(((event.clientX - box.left) / box.width) * data.duration)
  }

  return (
    <div className="chart" ref={frame}>
      <div className="chart-paper">
        <ul className="chart-legend" aria-hidden="true">
          <li style={{ top: ZONES.wave[0] }}>Sound</li>
          <li style={{ top: ZONES.cochlea[0] }}>Cochlea, low to high pitch</li>
          <li style={{ top: ZONES.raster[0] }}>{fiberNote ?? 'Nerve, 128 of 32,768 fibres'}</li>
        </ul>
        <canvas ref={recording} className="chart-canvas" style={{ height: CHART_HEIGHT }}
                onPointerDown={seekFromPointer}
                aria-label="The clip as sound, cochlear activity and nerve spikes. Click to move the playhead." />
        <div className="chart-playhead" ref={playhead} />
        {reveal && !still && (
          <motion.div className="chart-cover" initial={{ scaleX: 1 }} animate={{ scaleX: 0 }}
                      transition={{ duration: 1.8, ease: [0.3, 0, 0.2, 1], delay: 0.25 }} />
        )}
      </div>
      <input ref={slider} className="chart-slider" type="range" min={0} max={data.duration}
             step={0.01} defaultValue={0} aria-label="Position in the clip, in seconds"
             onChange={(event) => onSeek(Number(event.target.value))} />
      <div className="loupe">
        <p className="loupe-title">The same spikes, 60 milliseconds at a time</p>
        <canvas ref={loupe} className="chart-canvas" style={{ height: LOUPE_HEIGHT }}
                aria-label="Close-up of the nerve spikes around the playhead" />
      </div>
    </div>
  )
}
