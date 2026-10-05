/** Pure helpers for the charts. No DOM, so they are tested directly. */

const round6 = (v: number) => Math.round(v * 1e6) / 1e6

export function formatDb(value: number): string {
  return `${Math.round(value)} dB`
}

/** A window of `span` seconds centred on `time`, kept inside the clip. */
export function visibleWindow(time: number, span: number, duration: number): [number, number] {
  const start = Math.min(Math.max(time - span / 2, 0), Math.max(duration - span, 0))
  return [round6(start), round6(start + span)]
}

function lowerBound(times: number[], value: number): number {
  let lo = 0
  let hi = times.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] < value) lo = mid + 1
    else hi = mid
  }
  return lo
}

/** Index range [from, to) of the sorted spike times that fall in [start, end). */
export function spikesInWindow(times: number[], start: number, end: number): [number, number] {
  return [lowerBound(times, start), lowerBound(times, end)]
}

export interface FiberTrace {
  points: [number, number][]
  spikes: number[]
}

/**
 * Membrane voltage of one leaky integrate-and-fire fiber under a constant
 * current: threshold 1, reset 0, the same equations the model uses.
 */
export function fiberTrace(current: number, tau: number, refractory: number,
                           duration: number, samples = 24): FiberTrace {
  const points: [number, number][] = [[0, 0]]
  const spikes: number[] = []
  if (current <= 1) {
    for (let i = 1; i <= samples * 4; i++) {
      const t = (duration * i) / (samples * 4)
      points.push([t, current * (1 - Math.exp(-t / tau))])
    }
    return { points, spikes }
  }
  const rise = tau * Math.log(current / (current - 1))
  let start = 0
  while (start < duration) {
    for (let i = 1; i <= samples; i++) {
      const t = start + (rise * i) / samples
      if (t > duration) return { points, spikes }
      points.push([t, current * (1 - Math.exp(-(t - start) / tau))])
    }
    const spike = start + rise
    spikes.push(spike)
    points.push([spike, 0])
    start = spike + refractory
    if (start < duration) points.push([start, 0])
  }
  return { points, spikes }
}
