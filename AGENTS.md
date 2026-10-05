# sing-me-a-song

Project description for coding agents. Read this first.

## What this is

A model of the human auditory periphery (cochlea, inner hair cells, auditory
nerve) built to regenerate a sound from the nerve's spike times alone. A clip
goes in, spike times come out, and every stage is then inverted in turn to get
the clip back. On real music the result is bit-identical to the input at
16-bit.

This is not a machine-learning project. Nothing is trained. Every stage is a
fixed model with an exact inverse, and decoding is linear algebra.

It is the sibling of `biovision` (the same idea for the eye). Both exist to
learn how human perception encodes the world, as groundwork for robot
perception.

## Documents

- Plan for the invertible path, with the measurements behind it:
  `docs/superpowers/plans/2026-10-04-perfect-regeneration.md`
- `README.md`: what each reconstruction path is, how to run the model and the
  site, and the licences of the 3D anatomy.

## Commands

```
python recover.py                          # 4 s of test2.mp3, about 4 minutes
python recover.py <file> --seconds N       # another clip from sound_db/
python -m pytest -q                        # model and API tests, about 90 seconds
npm test                                   # page logic tests
npm run serve                              # API on :8010
npm run dev                                # page on :5173
npm run build
python tools/build_demo.py                 # refresh public/demo from the last recover.py run
python tools/build_anatomy.py              # rebuild the 3D anatomy
python tools/sync_model.py                 # refresh api/vendor after changing the model
vercel deploy --prod                       # https://resoundlab.vercel.app
```

Run Python with the repo's `.venv` (`.venv\Scripts\python.exe`); the `python`
on PATH is a different interpreter. Install `requirements-dev.txt`.

## Architecture

```
sound
  -> cochlea/gammatone_frame.analyze           bands (channels, samples)
  -> haircell/inner_hair_cell.transduce        nerve drive
  -> neuron_models/spike_timing.encode_spike_times   spike times
  -> reconstruction/decode_spike_times.decode_drive  nerve drive
  -> haircell/inner_hair_cell.inverse_transduce      bands
  -> cochlea/gammatone_frame.synthesize        sound
```

- `auditory_periphery.hear()` runs the first three steps; `regenerate()` runs
  the last three. They are the entry points. `recover.py` is a driver that
  scores this path (H) against the older ones and contains no model code.
- **Path N** skips the hair cells and decodes all channels jointly
  (`decode_spike_times`, LSQR). 512 neurons are enough.
- **Path H** is the whole ear. It decodes each channel's drive separately
  (`decode_drive`, sparse normal equations), so it needs several spike
  intervals per sample per channel: about 1,024 fibers per channel at 16 kHz.
- Paths B, C and C+ are the older envelope and firing-rate pipeline
  (`cochlea/filterbank.py`, `haircell/transduction.py`,
  `neuron_models/neuron_population.py`, `reconstruction/vocoder.py`). They are
  kept as the baseline that shows what a rate code loses. Do not "fix" them.
- **Resound**, the website, lives in this repository: the page in `src/` (React, Vite),
  the API in `api/server/` (FastAPI) and its static files in `public/`. The server
  imports the model; the model imports nothing from the server or the page.
  The server keeps nothing between requests: the page cuts a clip into
  one-second blocks, posts each to `/api/blocks`, and joins the answers. That
  is what lets it run as a serverless function. Do not add server-side state.
- `api/vendor/` is a copy of the five model files the API imports
  (`auditory_periphery.py` and one file each from `cochlea/`, `haircell/`,
  `neuron_models/`, `reconstruction/`). Vercel builds the API from `api/`
  alone. Never edit the copy: edit the original and run
  `python tools/sync_model.py`. A test fails while they differ.

## Rules

1. Every forward stage on the invertible path has an exact inverse, and a test
   that round-trips it. Do not add a stage without both.
2. `regenerate()` may use spike times and fixed anatomy only. Nothing that
   depends on the sound, other than its length, may pass from `hear()` to it.
3. Fibers on the invertible path are deterministic. Noise is added only through
   the explicit `jitter` argument.
4. A fiber that falls silent stops reporting. Keep the hair-cell-to-fiber gain
   low enough that no full-scale sound pushes a fiber below threshold.
5. New numerical code uses float64. `load_audio` returns float32.
6. Arrays are `(channels, samples)`. Spikes are two parallel arrays:
   `spike_neuron` and `spike_time` in seconds.
7. The "perfect" claim is scored with `recover.raw_snr_db`, which fits neither
   gain nor delay.
8. `api/requirements.txt` is what the hosted API installs: NumPy, SciPy and
   FastAPI. Keep it that small; nothing the API imports may need more.
   Everything for local work is in `requirements-dev.txt`.
9. Each new test file starts with the `sys.path.insert` header the others use.

## Engineering guidelines

These apply to every change. They combine Andrej Karpathy's agent guidelines
(github.com/forrestchang/andrej-karpathy-skills), Matt Pocock's agent skills
(github.com/mattpocock/skills), and the SRP, KISS and DRY principles.

### How to work

1. **Think before coding.** State assumptions. If a request has two readings,
   say so instead of picking one silently. If a simpler approach exists, say so.
2. **Simplicity first.** The minimum code that solves the problem. No feature,
   option or abstraction that was not asked for. If 200 lines could be 50,
   rewrite it.
3. **Surgical changes.** Touch only what the task needs. Match the existing
   style. Remove only the orphans your own change created; mention other dead
   code, do not delete it.
4. **Goal-driven.** Turn the task into checks that can be run (a test, an SNR,
   a screenshot) and loop until they pass.
5. **Small steps, red then green.** Write the failing test, watch it fail,
   make it pass, then tidy. "The rate of feedback is your speed limit."
6. **Use the domain's words.** Band, drive, receptor potential, fiber, spike
   time, inter-spike interval, refractory period, tail. Code, tests and docs
   share one vocabulary.

### How to design

- **Deep modules.** A module offers a small interface over a lot of behaviour.
  `hear()` and `regenerate()` are the model: one call each. Do not widen an
  interface to expose something only one caller needs.
- **Single responsibility.** One module, one reason to change. Filters in
  `cochlea/`, hair cells in `haircell/`, fibers in `neuron_models/`, decoding
  in `reconstruction/`, orchestration in `auditory_periphery.py`, scoring in
  `recover.py`, the API in `api/server/`, presentation in `src/`.
- **Do not repeat yourself.** Each fact lives in one place: a constant, a
  default argument, or one function others call.
- **Classes where there is state with behaviour, or more than one
  implementation of one interface.** Otherwise a function. Most of this
  codebase is stateless array mathematics, so most of it is functions. The
  patterns in use, to follow and extend:

  | Pattern | Where | Use it when |
  |---|---|---|
  | Inverse pair | `analyze`/`synthesize`, `transduce`/`inverse_transduce`, `adapt`/`inverse_adapt`, `hear`/`regenerate` | Adding a stage to the ear: write the forward function and its exact inverse together |
  | Facade | `hear()`, `regenerate()` | One entry point over many parts |
  | Strategy by argument | `vocoder_reconstruct(method=...)`, `extract_envelope(method=...)` | Two or three interchangeable computations; move to classes only if they grow state |
  | Plain record | The `population` and `ear` dicts | Passing fixed parameters between stages; never mutate one after it is built |

- **A pattern must earn its place.** Introduce one when it removes real
  duplication or isolates a responsibility that already exists twice. A
  pattern added for a single use is complexity, not design.
- **Scale by composition.** A new stage is one inverse pair inserted into
  `hear()` and `regenerate()`. It should not require editing the decoders, the
  server or the page.

## Decisions that were measured, not guessed

Each of these came out of a prototype. Do not undo one without re-measuring.

- **Tight gammatone frame.** The original log-spaced bank summed to a response
  30 to 55 dB down across most of the band and could not be inverted (3.4 dB).
  ERB spacing normalised to a Parseval frame round-trips at about 300 dB.
- **Gain grows with channel frequency on path N.** With equal gain the joint
  decode stalled at 44.5 dB after 2000 iterations; with
  `sqrt(1 + (2*pi*fc*tau)^2)` it reaches 180 dB or more in about 500.
- **About four spike intervals per sample.** Path N at 2x reached 30.7 dB, at
  4.3x 180 dB.
- **Staged inverse for path H.** A joint Gauss-Newton decode through the hair
  cell stalled near 24 dB. Solving each channel's drive and then inverting the
  hair cell is exact, at the cost of 64 times more fibers.
- **Smooth hair cell.** A soft rectifier, asinh compression and linear
  adaptation are each exactly invertible. A hard rectifier or a clip is not.
- **Hair-cell-to-fiber gain of 1.** At 20, a loud pure tone silenced the fibers
  of its band and the decode returned samples as large as 1e17.
- **Smoothness prior, weight 1e-15.** It decides only what the spikes cannot:
  a sample no fiber observed is interpolated, not left arbitrary.

## Expected results

`test2.mp3`, 16 kHz, 32 channels:

| Path | 4 s | 10 s |
|---|---|---|
| A, cochlea only | 302.75 dB | 303.15 dB |
| N, 512 neurons | 181.11 dB | 205.02 dB |
| H, whole ear, 32,768 fibers | 181.18 dB | 174.34 dB |
| C, old rate code | 0.01 dB | 0.00 dB |

Path H on a short test signal: 73 dB with 1 ns of spike-time jitter, 32 dB at
100 ns, nothing useful at 10 microseconds. 512 fibers per channel gives 202 dB,
256 gives 44 dB, 64 gives 11 dB.
