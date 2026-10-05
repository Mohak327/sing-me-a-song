/** Cutting a clip into blocks for the server and putting the answers back together. */
import type { Fiber, Showcase } from '../hooks/useShowcase'

/** What the server returns for one block, as far as drawing is concerned. */
export interface HeardBlock {
  seconds: number
  waveform: [number, number][]
  /** Level of each cochlear channel in each frame, in dB, not yet scaled. */
  level_db: number[][]
  /** Spike times measured from the start of the block. */
  raster: Fiber[]
  center_freqs: number[]
}

const FLOOR_DB = -60 // level drawn as empty, relative to the clip's loudest moment
const MAX_SNR_DB = 320 // reported when the copy has no error at all

export function splitBlocks(samples: Float32Array, blockSamples: number): Float32Array[] {
  const blocks: Float32Array[] = []
  for (let start = 0; start < samples.length; start += blockSamples) {
    blocks.push(samples.subarray(start, start + blockSamples))
  }
  return blocks
}

/** Join heard blocks, in order, into what the strip chart draws. */
export function assembleShowcase(blocks: HeardBlock[]): Showcase {
  const channels = blocks[0].level_db.length
  const levels = Array.from({ length: channels }, (_, c) => blocks.flatMap((block) => block.level_db[c]))
  const loudest = Math.max(...levels.map((row) => Math.max(...row)))
  const starts: number[] = []
  let duration = 0
  for (const block of blocks) {
    starts.push(duration)
    duration += block.seconds
  }
  return {
    duration,
    frame: 0.01,
    center_freqs: blocks[0].center_freqs,
    waveform: blocks.flatMap((block) => block.waveform),
    cochleagram: levels.map((row) => row.map((db) => {
      const level = Math.min(Math.max((db - loudest - FLOOR_DB) / -FLOOR_DB, 0), 1)
      return Math.round(level * 100) / 100
    })),
    raster: blocks[0].raster.map((fiber, index) => ({
      channel: fiber.channel,
      times: blocks.flatMap((block, b) => block.raster[index].times.map((t) => Math.round((t + starts[b]) * 1e5) / 1e5)),
    })),
    hair_cell: { input: [], output: [] },
  }
}

/** Signal-to-noise ratio from the energies of the signal and of the error. */
export function snrDb(signal: number, noise: number): number {
  if (noise <= 0 || signal <= 0) return MAX_SNR_DB
  return Math.min(10 * Math.log10(signal / noise), MAX_SNR_DB)
}

/** A 16-bit mono WAV file holding the samples. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2))
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i))
  }
  text(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const clipped = Math.min(Math.max(samples[i], -1), 1)
    view.setInt16(44 + i * 2, Math.round(clipped * 32767), true)
  }
  return new Blob([view], { type: 'audio/wav' })
}

/** Decode any audio the browser can read into mono samples at the given rate, within [-1, 1]. */
export async function decodeAudio(data: ArrayBuffer, sampleRate: number): Promise<Float32Array> {
  const decoder = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await decoder.decodeAudioData(data)
  } catch {
    throw new Error('That file could not be read as audio. Try a WAV, MP3, FLAC or OGG file.')
  } finally {
    void decoder.close()
  }
  const frames = Math.max(Math.round(decoded.duration * sampleRate), 1)
  const mixer = new OfflineAudioContext(1, frames, sampleRate)
  const source = mixer.createBufferSource()
  source.buffer = decoded
  source.connect(mixer.destination)
  source.start()
  const samples = (await mixer.startRendering()).getChannelData(0)
  let peak = 0
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample))
  return peak > 1 ? samples.map((sample) => sample / peak) : samples
}
