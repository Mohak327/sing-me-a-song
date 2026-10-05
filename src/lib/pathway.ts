/** The route a sound takes from the air to the auditory cortex, as shown in the 3D model. */
import anatomy from '../data/anatomy.json'

export type Point = [number, number, number]

export interface Stage {
  id: string
  name: string
  /** Where the stage sits in the 3D model, and how big it is there (scene units). */
  position: Point
  size: number
  /** True for the stages the model actually computes. */
  modelled: boolean
  what: string
}

const described: Omit<Stage, 'position' | 'size'>[] = [
  {
    id: 'sound', name: 'Sound in the air', modelled: false,
    what: 'A sound is a wave of pressure moving through the air. Nothing about it is a nerve signal yet.',
  },
  {
    id: 'outer-ear', name: 'Outer ear', modelled: false,
    what: 'The outer ear gathers the wave and funnels it down the ear canal. Its folds colour the sound slightly, which helps tell where it came from.',
  },
  {
    id: 'eardrum', name: 'Eardrum', modelled: false,
    what: 'A thin membrane at the end of the canal. The pressure wave pushes it in and out, turning sound into movement.',
  },
  {
    id: 'bones', name: 'Middle-ear bones', modelled: false,
    what: 'Three tiny bones carry the eardrum’s movement to the cochlea and concentrate its force, so it can move fluid instead of air.',
  },
  {
    id: 'cochlea', name: 'Cochlea', modelled: true,
    what: 'A fluid-filled spiral. Each place along it moves most for one pitch: high notes near the entrance, low notes at the centre. The model’s 32 filters are this.',
  },
  {
    id: 'hair-cells', name: 'Hair cells', modelled: true,
    what: 'Rows of cells along the inside of the spiral turn its movement into an electrical signal. They respond more one way than the other, squash loud sounds, and adapt. They are far too small to see at this scale.',
  },
  {
    id: 'nerve', name: 'Auditory nerve', modelled: true,
    what: 'About 30,000 fibres carry the hair cells’ signal away as spikes. The model stops here: everything it rebuilds, it rebuilds from these spikes. The scan shows the hearing and balance nerves together.',
  },
  {
    id: 'brainstem', name: 'Brainstem', modelled: false,
    what: 'The nerve enters the side of the brainstem. In its first relay stations, signals from both ears meet and are compared, which is how the brain starts working out where a sound is.',
  },
  {
    id: 'midbrain', name: 'Midbrain', modelled: false,
    what: 'The inferior colliculus, a small bump on the back of the midbrain, gathers nearly everything coming up from below and combines pitch, timing and direction.',
  },
  {
    id: 'thalamus', name: 'Thalamus', modelled: false,
    what: 'The medial geniculate body, the last relay. It passes the signal on to the cortex and is shaped by what you are paying attention to.',
  },
  {
    id: 'cortex', name: 'Auditory cortex', modelled: false,
    what: 'The upper fold of the temporal lobe, just above the ear. This is where the signal becomes something heard: a voice, a note, a word. Most of the signal crosses to the opposite side of the brain; the same side is shown here to keep the route in one view.',
  },
]

const stops = anatomy.stops as Record<string, { position: number[]; size: number }>

export const stages: Stage[] = described.map((stage) => ({
  ...stage,
  position: stops[stage.id].position as Point,
  size: stops[stage.id].size,
}))

export function stageById(id: string): Stage {
  const stage = stages.find((one) => one.id === id)
  if (!stage) throw new Error(`No stage called ${id}`)
  return stage
}

function distance(a: Point, b: Point): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
}

const round = (v: number) => Math.round(v * 1e6) / 1e6

/** The point a fraction `t` of the way along a polyline, at even speed. `t` wraps past 1. */
export function pointOnPath(points: Point[], t: number): Point {
  const lengths = points.slice(1).map((point, i) => distance(points[i], point))
  const total = lengths.reduce((sum, length) => sum + length, 0)
  const wrapped = t === 1 ? 1 : ((t % 1) + 1) % 1
  let remaining = wrapped * total
  for (let i = 0; i < lengths.length; i++) {
    if (remaining <= lengths[i] || i === lengths.length - 1) {
      const f = lengths[i] === 0 ? 0 : Math.min(remaining / lengths[i], 1)
      const a = points[i]
      const b = points[i + 1]
      return [round(a[0] + (b[0] - a[0]) * f), round(a[1] + (b[1] - a[1]) * f), round(a[2] + (b[2] - a[2]) * f)]
    }
    remaining -= lengths[i]
  }
  return points[points.length - 1]
}
