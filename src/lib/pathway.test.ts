import { describe, expect, it } from 'vitest'
import { pointOnPath, stages } from './pathway'
import { formatDuration, remainingSeconds } from './time'

describe('pointOnPath', () => {
  const path: [number, number, number][] = [[0, 0, 0], [2, 0, 0], [2, 2, 0]]
  it('starts and ends on the first and last points', () => {
    expect(pointOnPath(path, 0)).toEqual([0, 0, 0])
    expect(pointOnPath(path, 1)).toEqual([2, 2, 0])
  })
  it('moves at an even pace along the whole length', () => {
    expect(pointOnPath(path, 0.25)).toEqual([1, 0, 0])
    expect(pointOnPath(path, 0.75)).toEqual([2, 1, 0])
  })
  it('wraps so a pulse can loop', () => {
    expect(pointOnPath(path, 1.25)).toEqual([1, 0, 0])
  })
})

describe('stages', () => {
  it('run from the outside world to the cortex', () => {
    expect(stages.map((stage) => stage.id)).toEqual(['sound', 'outer-ear', 'eardrum', 'bones', 'cochlea',
      'hair-cells', 'nerve', 'brainstem', 'midbrain', 'thalamus', 'cortex'])
  })
  it('each have a place and a size in the anatomy model', () => {
    for (const stage of stages) {
      expect(stage.position).toHaveLength(3)
      expect(stage.size).toBeGreaterThan(0)
    }
  })
  it('go inward from the ear to the brainstem, then up to the cortex', () => {
    const x = (id: string) => stages.find((stage) => stage.id === id)!.position[0]
    const y = (id: string) => stages.find((stage) => stage.id === id)!.position[1]
    expect(x('outer-ear')).toBeLessThan(x('eardrum'))
    expect(x('eardrum')).toBeLessThan(x('cochlea'))
    expect(x('cochlea')).toBeLessThan(x('brainstem'))
    expect(y('brainstem')).toBeLessThan(y('midbrain'))
    expect(y('midbrain')).toBeLessThan(y('cortex'))
  })
  it('mark only the cochlea, hair cells and nerve as modelled', () => {
    expect(stages.filter((stage) => stage.modelled).map((stage) => stage.id))
      .toEqual(['cochlea', 'hair-cells', 'nerve'])
  })
})

describe('time', () => {
  it('formats seconds the way a person would say them', () => {
    expect(formatDuration(8)).toBe('8 seconds')
    expect(formatDuration(1)).toBe('1 second')
    expect(formatDuration(75)).toBe('1 minute')
    expect(formatDuration(150)).toBe('3 minutes')
  })
  it('estimates the time left from the pace so far', () => {
    expect(remainingSeconds(0.25, 30)).toBe(90)
    expect(remainingSeconds(1, 30)).toBe(0)
  })
  it('has no estimate before any progress', () => {
    expect(remainingSeconds(0, 5)).toBeNull()
  })
})
