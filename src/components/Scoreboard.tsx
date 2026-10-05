import { motion, useReducedMotion } from 'motion/react'
import type { AudioPlayer } from '../hooks/useAudioPlayer'
import { cdQualityDb, paths, type SourceName } from '../data/results'
import { formatDb } from '../lib/chart'

const SCALE_DB = 220

export function Scoreboard({ player }: { player: AudioPlayer<SourceName> }) {
  const still = useReducedMotion()
  return (
    <section className="section" id="score">
      <h2>What each way of reading the nerve gets back</h2>
      <p className="lede">
        The score is signal-to-noise ratio against the original: every 20 dB is ten times less
        error. A CD is {cdQualityDb} dB. Above that, the copy and the original are the same file.
      </p>
      <div className="score" style={{ ['--cd' as string]: `${(cdQualityDb / SCALE_DB) * 100}%` }}>
        <span className="score-mark">CD quality</span>
        {paths.map((path) => {
          const width = `${Math.max(path.snrDb / SCALE_DB, 0.004) * 100}%`
          return (
            <div className="score-row" key={path.name}>
              <div className="score-text">
                <h3>{path.name}</h3>
                <p>{path.detail}</p>
                {path.listen && (
                  <button type="button" className="link-button"
                          onClick={() => player.listen(path.listen!)}>
                    Listen to it
                  </button>
                )}
              </div>
              <div className="score-track">
                <motion.div className={path.snrDb > cdQualityDb ? 'score-bar exact' : 'score-bar'}
                            initial={still ? false : { width: 0 }} whileInView={{ width }}
                            viewport={{ once: true, amount: 0.6 }}
                            transition={{ duration: 1.1, ease: [0.2, 0.7, 0.2, 1] }}
                            style={still ? { width } : undefined} />
                <span className="score-value">{formatDb(path.snrDb)}</span>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
