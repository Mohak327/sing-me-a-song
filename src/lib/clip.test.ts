import { describe, expect, it } from 'vitest'
import { assembleShowcase, encodeWav, snrDb, splitBlocks, type HeardBlock } from './clip'

const block = (seconds: number, level: number, spike: number): HeardBlock => ({
  seconds,
  waveform: Array.from({ length: Math.round(seconds * 100) }, () => [-0.5, 0.5] as [number, number]),
  level_db: [[level, level], [level - 30, level - 90]],
  raster: [{ channel: 0, times: [spike] }, { channel: 1, times: [] }],
  center_freqs: [100, 1000],
})

describe('splitBlocks', () => {
  it('cuts samples into blocks, leaving the remainder in the last one', () => {
    const blocks = splitBlocks(new Float32Array(25), 10)
    expect(blocks.map((part) => part.length)).toEqual([10, 10, 5])
  })
  it('gives nothing for an empty clip', () => {
    expect(splitBlocks(new Float32Array(0), 10)).toEqual([])
  })
})

describe('assembleShowcase', () => {
  const showcase = assembleShowcase([block(0.02, -20, 0.01), block(0.02, -10, 0.005)])
  it('joins the blocks end to end', () => {
    expect(showcase.duration).toBeCloseTo(0.04)
    expect(showcase.waveform).toHaveLength(4)
    expect(showcase.cochleagram[0]).toHaveLength(4)
    expect(showcase.center_freqs).toEqual([100, 1000])
  })
  it('moves each block’s spikes to its place in the clip', () => {
    expect(showcase.raster[0].times).toEqual([0.01, 0.025])
    expect(showcase.raster[1].times).toEqual([])
  })
  it('scales loudness against the loudest moment of the whole clip', () => {
    expect(showcase.cochleagram[0]).toEqual([0.83, 0.83, 1, 1])
    expect(showcase.cochleagram[1][1]).toBe(0)
  })
})

describe('snrDb', () => {
  it('is ten times the log of signal over error', () => {
    expect(snrDb(1, 0.01)).toBeCloseTo(20)
  })
  it('is capped when there is no error at all, and when there is no signal', () => {
    expect(snrDb(1, 0)).toBe(320)
    expect(snrDb(0, 0)).toBe(320)
  })
})

describe('encodeWav', () => {
  it('writes a 16-bit mono WAV of the right length', async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5, 2]), 16000)
    const view = new DataView(await blob.arrayBuffer())
    expect(blob.type).toBe('audio/wav')
    expect(view.byteLength).toBe(44 + 8)
    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))).toBe('RIFF')
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getInt16(44, true)).toBe(0)
    expect(view.getInt16(46, true)).toBe(16384)
    expect(view.getInt16(50, true)).toBe(32767)
  })
})
