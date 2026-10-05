import { lazy, Suspense, useState } from 'react'
import { useReducedMotion } from 'motion/react'
import { stages } from '../lib/pathway'

/** A link can open the model on one stop: add ?stop=cochlea (or any stop's id) to the address. */
function stopInAddress(): string | null {
  const id = new URLSearchParams(window.location.search).get('stop')
  return stages.some((stage) => stage.id === id) ? id : null
}

// three.js is large; load it only when this section is on the page.
const EarScene = lazy(() => import('./EarScene'))

/** The whole route of hearing in 3D, with the part the model covers marked. */
export function Pathway() {
  const [linked] = useState(stopInAddress)
  const [selected, setSelected] = useState(linked ?? 'cochlea')
  const still = useReducedMotion()
  const stage = stages.find((one) => one.id === selected)!
  return (
    <section className="section" id="path">
      <h2>The whole route, from the air to where sound is heard</h2>
      <p className="lede">
        Follow a sound from the air to the part of the brain that hears it. Resound models the
        middle of the journey. Drag to turn, scroll to zoom, or pick a stop.
      </p>
      <div className="pathway">
        <div className="pathway-scene">
          <Suspense fallback={<p className="pathway-wait">Loading the anatomy.</p>}>
            <EarScene selected={selected} moving={!still} flyOnOpen={linked !== null} />
          </Suspense>
        </div>
        <div className="pathway-side">
          <ol className="pathway-stops">
            {stages.map((one) => (
              <li key={one.id}>
                <button type="button" aria-pressed={one.id === selected}
                        className={one.id === selected ? 'stop chosen' : 'stop'}
                        onClick={() => setSelected(one.id)}>
                  {one.name}
                  {one.modelled && <span className="stop-mark">in the model</span>}
                </button>
              </li>
            ))}
          </ol>
          <div className="pathway-detail" aria-live="polite">
            <h3>{stage.name}</h3>
            <p>{stage.what}</p>
            <p className="pathway-status">
              {stage.modelled ? 'Resound computes this stage.' : 'Resound does not model this stage.'}
            </p>
          </div>
        </div>
      </div>
      <p className="credit">
        Brain and outer ear: BodyParts3D, © The Database Center for Life Science, licensed CC BY-SA
        2.1 Japan. Ear canal, eardrum, bones, cochlea and nerve: the OpenEar library (Sieber and
        others, 2019), licensed CC BY 4.0, enlarged by 6% to fit this head. The skull is left out.
      </p>
    </section>
  )
}
