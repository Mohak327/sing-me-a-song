export function Limits() {
  return (
    <section className="section" id="limits">
      <h2>Where it stops working</h2>
      <p className="lede">
        The copy is exact because the model's fibres are perfectly regular and their spike times
        are known exactly. Real nerves are neither. Blur the spike times by a millionth of a second,
        or take away most of the fibres, and the sound falls apart: you can try both{' '}
        <a href="#try">near the top of the page</a>.
      </p>
      <div className="missing">
        <h3>What the model leaves out</h3>
        <ul>
          <li>Real fibres fire at random moments. These fire like clockwork.</li>
          <li>The outer and middle ear, and everything after the nerve.</li>
          <li>The way the cochlea retunes itself for loud and quiet sounds.</li>
          <li>Sound above 8 kHz. The model runs at 16,000 samples a second.</li>
        </ul>
      </div>
    </section>
  )
}
