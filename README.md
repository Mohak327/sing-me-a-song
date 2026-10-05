# Sing Me A Song

A model of the human auditory periphery, built to answer one question: can a
sound be regenerated from the auditory nerve's spikes alone?

It can. A clip goes through a cochlea, inner hair cells and 32,768 nerve
fibres; every stage is then inverted in turn, using only the spike times. On
real music the result matches the input sample for sample.

The site that shows it, **Resound**, is live at https://resoundlab.vercel.app.

Nothing here is trained. Every stage is fixed mathematics with an exact inverse.

## Run the model

Python 3.13, in a virtual environment:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.venv\Scripts\python.exe recover.py [sound_file] [--seconds N]
```

`recover.py` pushes one clip from `sound_db/` through seven reconstruction paths
and scores each against the original. Audio for every path is written to
`output/`. Four seconds takes about four minutes.

| Path | What it keeps | Result |
|---|---|---|
| T | STFT magnitude + phase | perfect (reference) |
| A | tight gammatone frame, all bands | perfect |
| B | band envelopes only, noise carrier | cochlear-implant quality |
| C | spike firing rates -> envelopes | below B |
| C+ | C with fine structure borrowed from analysis | diagnostic only |
| N | exact spike times of 512 deterministic neurons, no hair cells | perfect |
| H | the whole ear: cochlea, hair cells, 32,768 nerve fibres | perfect |

Path H is the full model:

```python
from auditory_periphery import hear, regenerate
spike_neuron, spike_time, ear = hear(audio, fs)
audio_again, info = regenerate(spike_neuron, spike_time, ear)
```

`hear()` turns sound into spike times through a gammatone cochlea, inner hair
cells (soft half-wave rectification, logarithmic compression, adaptation) and
leaky integrate-and-fire nerve fibres. `regenerate()` gets only the spike times
and inverts each stage in turn: nerve, hair cells, cochlea.

Paths N and H are exact only for noise-free spike times and deterministic
fibres. On a short test signal, path H gives about 73 dB with one nanosecond of
timing noise, 32 dB at 100 nanoseconds and nothing useful at 10 microseconds.
It also needs several spike intervals per audio sample in every channel: 512
fibres per channel still gave 202 dB, 256 gave 44 dB. Audio passed to `hear()`
must be mono and within [-1, 1].

## Run the site

```
npm install

npm run serve      # the API on http://127.0.0.1:8010 (run inside the activated .venv)
npm run dev        # the page on http://localhost:5173, passing /api to the API
```

The page shows a demo clip as sound, cochlear activity and nerve spikes, lets
you upload a sound and regenerate it with fewer fibres or blurred spike times,
and walks the route of hearing through a 3D model of real anatomy.

To regenerate a sound, the page decodes it in the browser, cuts it into
one-second blocks and posts each to `POST /api/blocks`. The server hears and
regenerates that block and keeps nothing. This is what lets the API run as a
serverless function. Limits come from `GET /api/limits`: 60 seconds and three
blocks at a time locally; 10 seconds and one block at a time when hosted.

One second of sound at 1,024 fibres per band takes about 58 s on the machine
this was built on and about 105 s on Vercel. Fewer fibres are faster.

## Tests

```
.venv\Scripts\python.exe -m pytest -q     # model and API
npm test                                  # page logic
```

## Layout

```
auditory_periphery.py   hear() and regenerate(): the ear as one invertible system
cochlea/                gammatone_frame.py (invertible filterbank); filterbank.py and
                        envelope_extract.py (the older envelope pipeline)
haircell/               inner_hair_cell.py (invertible); transduction.py (older)
neuron_models/          spike_timing.py (spike-time encoder); neuron_population.py (older)
reconstruction/         decode_spike_times.py (decoders); vocoder.py, decode_spikes.py (older)
recover.py              runs and scores every path
api/server/             the API (FastAPI)
api/vendor/             a copy of the five model files the hosted API runs
src/                    the page (React, Vite)
public/demo/            the demo clip in four versions and its chart data
public/models/          the 3D anatomy
tools/                  build_demo.py, build_anatomy.py and sync_model.py
tests/, api/server/tests/   Python tests
docs/                   the plan the invertible path was built from
```

The older envelope and firing-rate pipeline is kept on purpose: it is paths B,
C and C+, the baseline that shows what a rate code loses.

`AGENTS.md` holds the architecture, the rules and the measurements behind each
design decision.

## Deployment

Hosted on Vercel as project `resoundlab`. `vercel.json` defines two services:
`web` (this folder, Vite) and `api` (the `api/` folder, FastAPI, 300 s limit).

Vercel builds the API from `api/` alone, so that folder carries its own
`requirements.txt` and, in `api/vendor`, a copy of the five model files the API
imports. Building the API from the repository root instead does not fit
Vercel's 500 MB function limit once the page's dependencies are installed
there. After changing one of those model files, run `python tools/sync_model.py`;
a test fails until you do.

```
vercel deploy --prod
```

After a new `recover.py --seconds 10` run, `python tools/build_demo.py`
refreshes the demo the page opens with.

## The 3D anatomy and its licences

`public/models/hearing.glb` is built by `tools/build_anatomy.py` from two open
datasets of real anatomy, and `src/data/anatomy.json` holds where each stop of
the route sits in it.

- Brain and outer ear: BodyParts3D 4.0, © The Database Center for Life
  Science, licensed CC BY-SA 2.1 Japan.
- Ear canal, eardrum, ossicles, cochlea and nerve: the OpenEar library (Sieber
  et al., Scientific Data 2019, doi:10.5281/zenodo.1473724), specimen ZETA,
  licensed CC BY 4.0.

Because it contains BodyParts3D meshes, `hearing.glb` is itself under CC BY-SA
2.1 Japan, and both credits must stay visible wherever it is shown. The ear
specimen was scanned on its own; the script places it in the BodyParts3D head
(canal opening at the outer ear, nerve end at the side of the pons, ossicles
above the eardrum) and enlarges it by 6% to span that distance. The placement
is a fit, not a registration against the skull, which is not shown.

Add `?stop=cochlea` (or another stop's id) to the page's address to open the 3D
view at that stop.
