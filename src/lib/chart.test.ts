import { describe, expect, it } from 'vitest'
import { fiberTrace, formatDb, spikesInWindow, visibleWindow } from './chart'

describe('formatDb', () => {
  it('rounds to a whole decibel', () => {
    expect(formatDb(174.34)).toBe('174 dB')
    expect(formatDb(0.004)).toBe('0 dB')
  })
})

describe('visibleWindow', () => {
  it('centres the window on the time', () => {
    expect(visibleWindow(5, 0.08, 10)).toEqual([4.96, 5.04])
  })
  it('stays inside the clip at both ends', () => {
    expect(visibleWindow(0, 0.08, 10)).toEqual([0, 0.08])
    expect(visibleWindow(10, 0.08, 10)).toEqual([9.92, 10])
  })
})

describe('spikesInWindow', () => {
  const times = [0.1, 0.2, 0.3, 0.4, 0.5]
  it('returns the index range of spikes inside the window', () => {
    expect(spikesInWindow(times, 0.15, 0.45)).toEqual([1, 4])
  })
  it('returns an empty range when no spike falls inside', () => {
    expect(spikesInWindow(times, 0.21, 0.29)).toEqual([2, 2])
    expect(spikesInWindow([], 0, 1)).toEqual([0, 0])
  })
})

describe('fiberTrace', () => {
  it('fires at the period a leaky integrator predicts', () => {
    const tau = 0.01
    const refractory = 0.001
    const { spikes } = fiberTrace(2, tau, refractory, 0.05)
    const period = tau * Math.log(2 / (2 - 1)) + refractory
    expect(spikes[0]).toBeCloseTo(tau * Math.log(2), 6)
    expect(spikes[1] - spikes[0]).toBeCloseTo(period, 6)
  })
  it('fires sooner when the drive is stronger', () => {
    const weak = fiberTrace(1.5, 0.01, 0.001, 0.05).spikes[0]
    const strong = fiberTrace(1.6, 0.01, 0.001, 0.05).spikes[0]
    expect(strong).toBeLessThan(weak)
  })
  it('never fires below threshold and keeps the trace under it', () => {
    const { spikes, points } = fiberTrace(0.9, 0.01, 0.001, 0.05)
    expect(spikes).toEqual([])
    expect(Math.max(...points.map(([, v]) => v))).toBeLessThan(1)
  })
})
