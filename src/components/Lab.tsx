import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { cdQualityDb, fiberSweep, jitterSweep } from '../data/results'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { fetchLimits, useRegeneration, type RegenerationResult } from '../hooks/useRegeneration'
import type { Showcase } from '../hooks/useShowcase'
import { formatDb } from '../lib/chart'
import { formatDuration, remainingSeconds } from '../lib/time'
import { Choices } from './Choices'
import { PlayButton } from './PlayButton'
import { StripChart } from './StripChart'

const JITTER_SECONDS = [0, 1e-9, 1e-7, 1e-5]
const FIBERS = [1024, 512, 256, 64, 16]
type Version = 'original' | 'regenerated'
const VERSION_LABELS: Record<Version, string> = { original: 'Original', regenerated: 'Rebuilt from spikes' }

function verdict(decibels: number): string {
  if (decibels >= cdQualityDb) return 'The rebuilt sound is identical to the original.'
  if (decibels > 40) return 'Close to the original, with faint noise.'
  if (decibels > 20) return 'Recognisable, but clearly noisy.'
  return 'Too little survives to rebuild the sound.'
}

function Result({ result, showcase }: { result: RegenerationResult; showcase: Showcase }) {
  const player = useAudioPlayer<Version>(result.urls, 'regenerated')
  return (
    <div className="lab-result">
      <p className="lab-verdict" aria-live="polite">
        <strong>{formatDb(result.snrDb)}.</strong> {verdict(result.snrDb)}
      </p>
      <p className="lab-facts">
        {result.source}, {formatDuration(result.duration)}
        {result.trimmed ? ` (cut to the first ${result.maxSeconds} seconds)` : ''}. Heard by{' '}
        {result.totalFibers.toLocaleString('en-US')} fibres as {result.spikes.toLocaleString('en-US')} spikes.
        A CD is {cdQualityDb} dB.
      </p>
      <div className="transport">
        <PlayButton playing={player.playing} onClick={player.toggle} />
        <div className="sources" role="radiogroup" aria-label="Version to play">
          {(Object.keys(VERSION_LABELS) as Version[]).map((name) => (
            <button key={name} type="button" role="radio" aria-checked={player.source === name}
                    className={player.source === name ? 'source chosen' : 'source'}
                    onClick={() => player.select(name)}>
              {VERSION_LABELS[name]}
            </button>
          ))}
        </div>
      </div>
      <StripChart data={showcase} time={player.time} onSeek={player.seek} reveal={false}
                  fiberNote={`Nerve, ${showcase.raster.length} of ${result.totalFibers.toLocaleString('en-US')} fibres`} />
    </div>
  )
}

/** Regenerate the demo clip or an uploaded one with chosen fibre count and timing blur. */
export function Lab() {
  const { state, start } = useRegeneration()
  const [file, setFile] = useState<File | null>(null)
  const [jitter, setJitter] = useState(0)
  const [fibers, setFibers] = useState(0)
  const [maxSeconds, setMaxSeconds] = useState<number | null>(null)
  const picker = useRef<HTMLInputElement>(null)
  const busy = state.status === 'preparing' || state.status === 'running'
  const left = state.status === 'running' ? remainingSeconds(state.done / state.total, state.elapsed) : null

  useEffect(() => {
    const controller = new AbortController()
    fetchLimits(controller.signal).then((limits) => setMaxSeconds(limits.max_seconds)).catch(() => {})
    return () => controller.abort()
  }, [])

  return (
    <section className="section lab" id="try">
      <h2>Run a sound through the ear yourself</h2>
      <p className="lede">
        Pick a sound, decide how good the nerve is, and the model will hear it and rebuild it from
        the spikes. Your audio is not kept: it is heard a second at a time and forgotten.
      </p>

      <div className="lab-controls">
        <div className="lab-step">
          <h3>The sound</h3>
          <input ref={picker} type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a" hidden
                 onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          <div className="lab-file">
            <button type="button" className="outline-button" disabled={busy}
                    onClick={() => picker.current?.click()}>
              {file ? 'Choose another file' : 'Upload your own audio'}
            </button>
            {file && (
              <button type="button" className="link-button" disabled={busy}
                      onClick={() => { setFile(null); if (picker.current) picker.current.value = '' }}>
                Use the demo clip instead
              </button>
            )}
          </div>
          <p className="lab-note">
            {file ? `Using ${file.name}.` : 'Using the demo clip.'}
            {maxSeconds ? ` Up to ${maxSeconds} seconds; anything longer is cut.` : ''} It is mixed to
            mono at 16,000 samples a second.
          </p>
        </div>

        <div className="lab-step">
          <h3>Blur the spike times</h3>
          <Choices question="Random error added to every spike time" disabled={busy}
                   labels={jitterSweep.map((point) => point.label)} chosen={jitter} onChoose={setJitter} />
          <p className="lab-note">On a short test sound this gave {formatDb(jitterSweep[jitter].snrDb)}.</p>
        </div>

        <div className="lab-step">
          <h3>Remove fibres</h3>
          <Choices question="Fibres on each of the cochlea's 32 bands" disabled={busy}
                   labels={fiberSweep.map((point) => point.label)} chosen={fibers} onChoose={setFibers} />
          <p className="lab-note">On a short test sound this gave {formatDb(fiberSweep[fibers].snrDb)}.</p>
        </div>
      </div>

      <div className="lab-run">
        <button type="button" className="play" disabled={busy}
                onClick={() => start(file, FIBERS[fibers], JITTER_SECONDS[jitter])}>
          {busy ? 'Hearing it' : 'Hear it and rebuild it'}
        </button>
        <p className="lab-note">
          This runs the real model, which is slow: with 1,024 fibres, one to two minutes of computing
          for each second of sound. Fewer fibres are faster.
        </p>
      </div>

      {state.status === 'running' && (
        <div className="lab-progress" role="status">
          <div className="meter">
            <motion.div className="meter-fill exact"
                        animate={{ width: `${Math.max(state.done / state.total, 0.01) * 100}%` }}
                        transition={{ ease: 'linear', duration: 0.6 }} />
          </div>
          <p>
            {state.done} of {state.total} seconds heard.
            {left !== null && left > 0 ? ` About ${formatDuration(left)} left.` : ''}
            {state.done === 0 && state.elapsed > 3 ? ' The first second takes the longest to come back.' : ''}
          </p>
        </div>
      )}
      {state.status === 'preparing' && <p className="lab-progress" role="status">Reading the sound.</p>}
      {state.status === 'failed' && <p className="notice" role="alert">{state.reason}</p>}
      {state.status === 'done' && <Result result={state.result} showcase={state.showcase} />}
    </section>
  )
}
