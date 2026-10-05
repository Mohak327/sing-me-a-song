/**
 * Measured results. Sources: `recover.py --seconds 10` on sound_db/test2.mp3
 * (paths), and hear()/regenerate() on a 0.1 s test signal with 8 channels
 * (jitter and fiber sweeps). See AGENTS.md, "Expected results".
 */

export type SourceName = 'original' | 'regenerated' | 'rate-code' | 'envelope'

export interface PathResult {
  name: string
  detail: string
  snrDb: number
  listen?: SourceName
}

export const paths: PathResult[] = [
  {
    name: 'Loudness only',
    detail: 'How loud each pitch band is, with the fine wave thrown away. Roughly what a cochlear implant passes on.',
    snrDb: 0.01,
    listen: 'envelope',
  },
  {
    name: 'Firing rate',
    detail: 'How often each group of fibres fires, the usual way to read a nerve.',
    snrDb: 0.0,
    listen: 'rate-code',
  },
  {
    name: 'Spike timing, 512 fibres',
    detail: 'Exact spike times, with the hair cells left out.',
    snrDb: 205.02,
  },
  {
    name: 'The whole ear, 32,768 fibres',
    detail: 'Cochlea, hair cells and nerve, then each one undone in turn.',
    snrDb: 174.34,
    listen: 'regenerated',
  },
]

/** Signal-to-noise ratio of 16-bit audio, for scale. */
export const cdQualityDb = 96

export interface SweepPoint {
  label: string
  snrDb: number
}

export const jitterSweep: SweepPoint[] = [
  { label: 'none', snrDb: 209.4 },
  { label: '1 nanosecond', snrDb: 72.7 },
  { label: '100 nanoseconds', snrDb: 32.2 },
  { label: '10 microseconds', snrDb: -6.2 },
]

export const fiberSweep: SweepPoint[] = [
  { label: '1,024', snrDb: 209.4 },
  { label: '512', snrDb: 202.3 },
  { label: '256', snrDb: 44.0 },
  { label: '64', snrDb: 10.6 },
  { label: '16', snrDb: 5.5 },
]

export const clip = { seconds: 10, spikes: 43_207_837, fibers: 32_768, ratePerFiber: 131.5 }

export const sourceLabels: Record<SourceName, string> = {
  original: 'Original',
  regenerated: 'Rebuilt from spikes',
  'rate-code': 'Firing rate only',
  envelope: 'Loudness only',
}
