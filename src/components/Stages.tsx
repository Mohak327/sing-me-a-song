import { useMemo, useState } from 'react'
import type { Showcase } from '../hooks/useShowcase'
import { fiberTrace } from '../lib/chart'

const W = 360
const H = 200

/** The 32 overlapping band filters, at the model's real centre frequencies. */
function CochleaFilters({ centerFreqs }: { centerFreqs: number[] }) {
  const low = Math.log(40)
  const high = Math.log(8000)
  const toX = (hz: number) => ((Math.log(hz) - low) / (high - low)) * W
  const paths = centerFreqs.map((fc) => {
    const bandwidth = 24.7 * (4.37 * fc / 1000 + 1) * 1.019
    const points: string[] = []
    for (let i = 0; i <= 40; i++) {
      const hz = Math.exp(low + ((high - low) * i) / 40)
      const gain = Math.pow(1 + ((hz - fc) / bandwidth) ** 2, -2)
      points.push(`${toX(hz).toFixed(1)},${(H - 24 - gain * (H - 44)).toFixed(1)}`)
    }
    return points.join(' ')
  })
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="figure" role="img"
         aria-label="Thirty-two overlapping filters from low pitch to high pitch">
      {paths.map((points, i) => (
        <polyline key={i} points={points} fill="none" stroke="var(--pen-blue)" strokeWidth="1.1"
                  opacity={0.35 + 0.65 * (i / paths.length)} />
      ))}
      <line x1="0" y1={H - 24} x2={W} y2={H - 24} stroke="var(--ink)" />
      {[100, 1000, 8000].map((hz) => (
        <text key={hz} x={Math.min(toX(hz), W - 4)} y={H - 6} className="figure-label"
              textAnchor={hz === 8000 ? 'end' : 'middle'}>
          {hz >= 1000 ? `${hz / 1000} kHz` : `${hz} Hz`}
        </text>
      ))}
    </svg>
  )
}

/** The hair cell's response curve, from the model, with a push you can move. */
function HairCellCurve({ curve }: { curve: Showcase['hair_cell'] }) {
  const [push, setPush] = useState(0.5)
  const top = Math.max(...curve.output)
  const bottom = Math.min(...curve.output)
  const toX = (v: number) => ((v + 1) / 2) * W
  const toY = (v: number) => 12 + ((top - v) / (top - bottom)) * (H - 36)
  const points = curve.input.map((v, i) => `${toX(v).toFixed(1)},${toY(curve.output[i]).toFixed(1)}`).join(' ')
  const at = (value: number) => curve.output[Math.round((value + 1) * 100)]
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="figure" role="img"
           aria-label="Hair cell response against how far it is pushed">
        <line x1={toX(0)} y1="8" x2={toX(0)} y2={H - 20} stroke="var(--rule)" />
        <line x1="0" y1={toY(0)} x2={W} y2={toY(0)} stroke="var(--rule)" />
        <polyline points={points} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <circle cx={toX(-push)} cy={toY(at(-push))} r="5" fill="none" stroke="var(--pen-red)" strokeWidth="2" />
        <circle cx={toX(push)} cy={toY(at(push))} r="6" fill="var(--pen-red)" />
        <text x="2" y={H - 4} className="figure-label">pushed back</text>
        <text x={W - 2} y={H - 4} className="figure-label" textAnchor="end">pushed forward</text>
      </svg>
      <label className="control">
        <span>How hard the sound pushes</span>
        <input type="range" min={0.02} max={1} step={0.01} value={push}
               onChange={(event) => setPush(Number(event.target.value))} />
      </label>
      <p className="readout">
        Forward gives {at(push).toFixed(3)}. The same push backward gives only {at(-push).toFixed(3)}.
        A push ten times weaker would still give {at(push / 10).toFixed(3)}.
      </p>
    </div>
  )
}

/** One fibre charging to threshold and firing, using the model's equations. */
function FiberDemo() {
  const [sound, setSound] = useState(0.1)
  const tau = 0.01
  const refractory = 0.001
  const span = 0.04
  const quiet = useMemo(() => fiberTrace(2, tau, refractory, span), [])
  const driven = useMemo(() => fiberTrace(2 + sound, tau, refractory, span), [sound])
  const toX = (t: number) => (t / span) * W
  const toY = (v: number) => H - 30 - v * (H - 70)
  const shift = (quiet.spikes[0] - driven.spikes[0]) * 1000
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="figure" role="img"
           aria-label="A nerve fibre charging up and firing">
        <line x1="0" y1={toY(1)} x2={W} y2={toY(1)} stroke="var(--rule)" strokeDasharray="4 4" />
        <text x={W - 2} y={toY(1) - 6} className="figure-label" textAnchor="end">fires here</text>
        {quiet.spikes.map((t) => (
          <line key={`q${t}`} x1={toX(t)} y1="10" x2={toX(t)} y2="26" stroke="var(--ink)" opacity="0.35" strokeWidth="2" />
        ))}
        {driven.spikes.map((t) => (
          <line key={`d${t}`} x1={toX(t)} y1="10" x2={toX(t)} y2="26" stroke="var(--pen-red)" strokeWidth="2" />
        ))}
        <polyline points={driven.points.map(([t, v]) => `${toX(t).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')}
                  fill="none" stroke="var(--pen-red)" strokeWidth="2" strokeLinejoin="round" />
        <text x="2" y={H - 6} className="figure-label">0</text>
        <text x={W - 2} y={H - 6} className="figure-label" textAnchor="end">40 ms</text>
      </svg>
      <label className="control">
        <span>Sound reaching this fibre</span>
        <input type="range" min={-0.14} max={0.14} step={0.005} value={sound}
               onChange={(event) => setSound(Number(event.target.value))} />
      </label>
      <p className="readout">
        {Math.abs(shift) < 0.005
          ? 'In silence the fibre keeps its own steady beat (grey ticks).'
          : `The first spike lands ${Math.abs(shift).toFixed(2)} ms ${shift > 0 ? 'earlier' : 'later'} than in silence (grey ticks).`}
      </p>
    </div>
  )
}

export function Stages({ data }: { data: Showcase }) {
  return (
    <section className="section" id="how">
      <h2>From sound to spikes, in three steps</h2>
      <ol className="stages">
        <li>
          <CochleaFilters centerFreqs={data.center_freqs} />
          <div>
            <h3>The cochlea sorts the sound by pitch</h3>
            <p>
              Thirty-two overlapping filters split the sound into bands, low pitch to high, the way
              the basilar membrane does. They are tuned so that adding the bands back together
              returns the sound exactly.
            </p>
          </div>
        </li>
        <li>
          <HairCellCurve curve={data.hair_cell} />
          <div>
            <h3>Hair cells squash and tilt each band</h3>
            <p>
              A hair cell responds more to a push one way than the other, and loud sounds are
              squeezed far more than quiet ones. It also adapts: a steady sound fades to half its
              first response. Each of these can be undone exactly.
            </p>
          </div>
        </li>
        <li>
          <FiberDemo />
          <div>
            <h3>Nerve fibres turn it into spike times</h3>
            <p>
              Each fibre charges up, fires, rests for a millisecond and starts again. Sound makes it
              charge a little faster or slower, which moves every spike. With 1,024 fibres on each
              band, there are about eight spikes for every sample of audio.
            </p>
          </div>
        </li>
      </ol>
      <div className="backwards">
        <h3>Then backwards, from the spikes alone</h3>
        <p>
          Each gap between two spikes of a fibre is one equation: what the fibre received in that
          gap added up to exactly its threshold. Solving all of them gives back what the hair cells
          sent. Undoing the hair cells gives the bands, and adding the bands gives the sound.
        </p>
      </div>
    </section>
  )
}
