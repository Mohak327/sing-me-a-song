import { motion, useReducedMotion } from 'motion/react'
import { Lab } from './components/Lab'
import { Limits } from './components/Limits'
import { Pathway } from './components/Pathway'
import { PlayButton } from './components/PlayButton'
import { Scoreboard } from './components/Scoreboard'
import { Stages } from './components/Stages'
import { StripChart } from './components/StripChart'
import { clip, sourceLabels, type SourceName } from './data/results'
import { useAudioPlayer } from './hooks/useAudioPlayer'
import { useShowcase } from './hooks/useShowcase'

const HERO_SOURCES: SourceName[] = ['original', 'regenerated', 'rate-code']
const DEMO_URLS: Record<SourceName, string> = {
  original: '/demo/original.wav',
  regenerated: '/demo/regenerated.wav',
  'rate-code': '/demo/rate-code.wav',
  envelope: '/demo/envelope.wav',
}

export function App() {
  const showcase = useShowcase()
  const player = useAudioPlayer<SourceName>(DEMO_URLS, 'regenerated')
  const still = useReducedMotion()
  const rise = (delay: number) => still
    ? {}
    : { initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.7, delay } }

  return (
    <>
      <header className="masthead">
        <a className="wordmark" href="#top">Resound</a>
        <nav aria-label="Sections">
          <a href="#try">Try it</a>
          <a href="#path">The route</a>
          <a href="#how">How it works</a>
          <a href="#score">Results</a>
          <a href="#limits">Limits</a>
        </nav>
      </header>

      <main id="top">
        <section className="hero">
          <motion.h1 {...rise(0)}>
            What your ear tells your brain, turned back into the song.
          </motion.h1>
          <motion.p className="hero-lede" {...rise(0.12)}>
            Resound is a model of the inner ear. It turns music into the spikes of{' '}
            {clip.fibers.toLocaleString('en-US')} nerve fibres ({(clip.spikes / 1e6).toFixed(0)} million of them
            for the clip below), then rebuilds the music from the spikes alone. The copy matches the
            original sample for sample.
          </motion.p>

          <motion.div className="transport" {...rise(0.24)}>
            <PlayButton playing={player.playing} onClick={player.toggle} />
            <div className="sources" role="radiogroup" aria-label="Version to play">
              {HERO_SOURCES.map((name) => (
                <button key={name} type="button" role="radio" aria-checked={player.source === name}
                        className={player.source === name ? 'source chosen' : 'source'}
                        onClick={() => player.select(name)}>
                  {sourceLabels[name]}
                </button>
              ))}
            </div>
          </motion.div>

          {showcase.status === 'ready' && (
            <StripChart data={showcase.data} time={player.time} onSeek={player.seek} />
          )}
          {showcase.status === 'loading' && <p className="notice">Loading the nerve recording.</p>}
          {showcase.status === 'failed' && (
            <p className="notice">The nerve recording did not load. {showcase.reason}</p>
          )}

          <p className="caption">
            Each red dot is one spike. The rows look evenly filled because sound barely changes how
            often a fibre fires. The song is in exactly when each spike lands: switch between the
            versions while it plays, and try the one that keeps only how often.
          </p>
        </section>

        <Lab />
        <Pathway />
        {showcase.status === 'ready' && <Stages data={showcase.data} />}
        <Scoreboard player={player} />
        <Limits />
      </main>

      <footer className="footer">
        <p>
          Resound is the website for <strong>sing-me-a-song</strong>, a study of how human hearing
          encodes sound, as groundwork for machine perception. Nothing here is trained: every stage
          is fixed mathematics with an exact inverse.
        </p>
        <pre><code>{`python recover.py --seconds 10    # in sing-me-a-song: run the model on a clip
npm run serve                     # in resound: the server that hears uploaded sounds`}</code></pre>
      </footer>
    </>
  )
}
