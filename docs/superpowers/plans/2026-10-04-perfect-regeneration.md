# Perfect Audio Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the input waveform from auditory-nerve spike times alone, to numerical precision (> 120 dB SNR), through a cochlea → neuron → decoder path.

**Architecture:** Replace the non-invertible gammatone bank with a tight (Parseval) gammatone frame whose adjoint is its exact inverse. Drive a deterministic LIF population with the full band signal (envelope and fine structure) and record exact sub-sample spike times. Decode by treating every inter-spike interval as one exact linear equation in the audio samples and solving the stacked system with LSQR.

**Tech Stack:** Python 3.13 (repo `.venv`), NumPy, SciPy (`scipy.signal.lfilter`, `scipy.sparse`, `scipy.sparse.linalg.lsqr`), pytest. No new dependencies.

**Spec:** None written. Requirements come from the 2026-10-04 conversation: fix the three measured losses (filterbank not invertible; spike code carries envelope only; decoder inverts only compression) with the goal of perfect regeneration.

## Background the implementer needs

Measured on 4 s of `sound_db/test2.mp3` before this work (`.\.venv\Scripts\python.exe recover.py`):

| Path | SNR dB |
|---|---|
| T: STFT magnitude + phase | 140.47 |
| A: summed gammatone bands + Wiener inversion | 3.39 |
| B: envelope + noise vocoder | 0.02 |
| C: spikes → rate → vocoder | 0.01 |

Why each is lost, and what fixes it:

1. **Path A.** `cochlea/filterbank.py` spaces channels with `np.logspace` and normalizes each kernel by `sum(|h|)`. The summed response peaks at 79 Hz and sits 30–55 dB lower across most of the band, so 84% of in-band bins fall under the Wiener regularizer in `recover.py`. Fix: ERB-rate spacing, unit peak gain, then divide every channel by `sqrt(sum_k |H_k(f)|^2)`. That makes the bank a Parseval frame: `synthesize(analyze(x)) == x`.
2. **Path C.** Neurons are driven by the Hilbert envelope, the spikes are binned to the sample grid, and the decoder reads a 5 ms rate. Fine structure never enters the code. Fix: drive neurons with the band signal itself and keep exact spike times.
3. **Decoder.** A LIF neuron that integrates from time `a` (end of refractory period) and fires at time `b` satisfies exactly
   `v_a·e^(-(b-a)/τ) + ∫_a^b (bias + gain·u(s))·e^(-(b-s)/τ) ds/τ = θ`,
   which is linear in the band signal `u`, and `u` is linear in the audio `x`. Stacking one equation per inter-spike interval gives `A·x = q`; with roughly 4× more intervals than samples, least squares recovers `x`.

The code in this plan was prototyped and run before the plan was written. Verified results with the exact code below (32 channels × 16 neurons, τ = 10 ms, 136 Hz per neuron):

| Clip | Frame round trip | Spike-time decode | Decode time |
|---|---|---|---|
| 1 s of `test2.mp3` | 303.6 dB | 184.6 dB | 11 s |
| 4 s of `test2.mp3` | 303.2 dB | 181.1 dB (675 LSQR iterations) | 79 s |

Two findings from the prototype that shaped the design:

- **Gain must grow with frequency.** With equal gain in every channel, LSQR stalls at 44.5 dB after 2000 iterations because the 10 ms membrane low-passes high channels. Scaling gain by `sqrt(1 + (2π·fc·τ)^2)` gives 184.6 dB in 502 iterations.
- **Spike density matters.** 2.0× Nyquist (62 Hz/neuron, 512 neurons) reached only 30.7 dB in 1000 iterations; 3.9× reached 89.7 dB; 4.3× reaches 180+ dB.

### What "perfect" means here, and what it does not

Perfect regeneration holds for a **deterministic** population with **exact** spike times. Spike-time noise breaks it quickly (1 s clip, 512 neurons):

| Spike-time jitter (std) | SNR |
|---|---|
| 0 | 184.6 dB |
| 1 µs | 39.0 dB |
| 10 µs | 18.9 dB |
| 100 µs | 3.1 dB (11.4 dB with 2048 neurons) |

Real auditory-nerve fibers are stochastic, so this plan builds the invertible idealization and exposes jitter as a knob. Hair-cell compression, adaptation and half-wave rectification are deliberately left out of the new path: each is a further loss to add back and measure later, not part of this plan.

## Global Constraints

- Run everything with the repo venv: `.\.venv\Scripts\python.exe`. The `python` on PATH is a different interpreter without the project's dependencies.
- No new dependencies. `requirements.txt` is unchanged.
- All new numerical code uses float64. `load_audio` returns float32; convert with `.astype(np.float64)` before the new path.
- Do not modify `cochlea/filterbank.py`, `neuron_models/neuron_population.py`, `neuron_models/lif_neuron.py`, `reconstruction/vocoder.py` or `reconstruction/decode_spikes.py`. Paths T, B, C and C+ in `recover.py` keep their current behavior so the old and new numbers stay comparable.
- Every new test file starts with the `sys.path.insert` header shown in the tasks. Without it, pytest resolves `audio_io` to `tests/audio_io` instead of the project package.
- `refractory_period` and `theta` must be the same value in the encoder and decoder calls. Defaults are 0.001 s and 1.0 in both.
- Match the existing code style: module docstring, Google-style `Args:`/`Returns:` docstrings, no type hints.

## Review Focus

Inputs the requirements imply but that are easy to leave untested. Each has a test in the task named.

1. **Silent input.** All-zero audio must still produce spikes (bias-driven spontaneous firing) and decode to silence, not NaN. Tasks 2 and 3.
2. **Too few spikes.** A small population gives fewer equations than samples; the decoder must warn that the result is under-determined rather than return a plausible-looking wrong waveform silently. Task 3.
3. **No spikes at all.** Empty spike arrays must return silence with a warning, not crash on empty-array indexing. Task 3.
4. **Odd-length and very short clips.** `rfft`/`irfft` round trips differ for odd `n`, and a clip shorter than the 0.25 s kernel must still reconstruct. Task 1.
5. **Spikes delivered in arbitrary order, or perturbed.** The decoder must sort by neuron and time itself, and jittered times that produce non-positive or out-of-range intervals must be dropped, not corrupt the solve. Task 3.

---

### Task 1: Tight gammatone frame

**Files:**
- Create: `cochlea/gammatone_frame.py`
- Test: `tests/test_gammatone_frame.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `erb_space(low_freq, high_freq, num_channels) -> np.ndarray` of shape `(num_channels,)`
  - `gammatone_frame(n, fs, num_channels=64, low_freq=50.0, high_freq=7600.0, order=4) -> (H, center_freqs)`; `H` is complex with shape `(num_channels, n // 2 + 1)`
  - `analyze(x, H) -> bands` with shape `(num_channels, n)`
  - `synthesize(bands, H) -> x` with shape `(n,)`

- [ ] **Step 1: Create a working branch and commit the existing work-in-progress as a baseline**

The tree has uncommitted work (`recover.py`, `tests/test_recover.py` and edits to three modules). Confirm with the repo owner before committing it; later tasks modify `recover.py`.

```powershell
git checkout -b perfect-regeneration
git add recover.py tests/test_recover.py audio_io/load_audio.py neuron_models/neuron_population.py reconstruction/vocoder.py
git commit -m "wip: reconstruction ladder baseline (paths T, A, B, C, C+)"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_gammatone_frame.py`:

```python
"""
Tests for the perfect-reconstruction gammatone frame.

Run:  python -m pytest tests/test_gammatone_frame.py -v
"""
import os
import sys
# Pin the project root first so top-level packages resolve here, not under tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from cochlea.gammatone_frame import erb_space, gammatone_frame, analyze, synthesize


def _snr_db(reference, estimate):
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum((reference - estimate) ** 2) + 1e-300))


def _make_signal(fs=16000, dur=0.25, seed=0):
    """Tones + sweep + broadband noise: energy from 100 Hz up to Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
         + 0.3 * np.sin(2 * np.pi * (300 + 12000 * t) * t) + 0.1 * rng.standard_normal(len(t)))
    return x / np.max(np.abs(x)), fs


def test_erb_space_is_ascending_and_hits_endpoints():
    cfs = erb_space(50.0, 7600.0, 32)
    assert len(cfs) == 32
    assert np.all(np.diff(cfs) > 0)
    assert abs(cfs[0] - 50.0) < 1e-6 and abs(cfs[-1] - 7600.0) < 1e-6


def test_frame_is_perfect_reconstruction():
    x, fs = _make_signal()
    H, _ = gammatone_frame(len(x), fs, num_channels=32)
    assert _snr_db(x, synthesize(analyze(x, H), H)) > 200


def test_frame_is_perfect_for_odd_length_and_tiny_signals():
    x, fs = _make_signal()
    for n in (3999, 37):
        H, _ = gammatone_frame(n, fs, num_channels=8)
        assert _snr_db(x[:n], synthesize(analyze(x[:n], H), H)) > 200


def test_frame_keeps_dc_and_nyquist():
    fs, n = 16000, 4000
    H, _ = gammatone_frame(n, fs, num_channels=32)
    for x in (np.ones(n), np.cos(np.pi * np.arange(n))):
        assert _snr_db(x, synthesize(analyze(x, H), H)) > 150


def test_analyze_rejects_wrong_length():
    H, _ = gammatone_frame(4000, 16000, num_channels=8)
    with pytest.raises(ValueError):
        analyze(np.zeros(3000), H)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gammatone_frame.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'cochlea.gammatone_frame'`

- [ ] **Step 4: Write the implementation**

Create `cochlea/gammatone_frame.py`:

```python
"""
Perfect-reconstruction gammatone filterbank (a Parseval tight frame).

analyze() splits a signal into ERB-spaced gammatone bands; synthesize() is its
exact inverse. Each band keeps envelope AND temporal fine structure.
"""

import numpy as np


def erb_space(low_freq, high_freq, num_channels):
    """
    Center frequencies equally spaced on the ERB-rate scale (Glasberg & Moore).

    Args:
        low_freq (float): Lowest center frequency (Hz)
        high_freq (float): Highest center frequency (Hz)
        num_channels (int): Number of channels

    Returns:
        center_freqs (np.ndarray): Shape (num_channels,), ascending (Hz)
    """
    erb_rate = np.linspace(21.4 * np.log10(4.37e-3 * low_freq + 1),
                           21.4 * np.log10(4.37e-3 * high_freq + 1),
                           num_channels)
    return (10 ** (erb_rate / 21.4) - 1) / 4.37e-3


def gammatone_frame(n, fs, num_channels=64, low_freq=50.0, high_freq=7600.0, order=4):
    """
    Build the frequency responses of a tight gammatone frame for length-n signals.

    Every channel is a 4th-order gammatone with bandwidth 1.019 * ERB. Dividing all
    channels by sqrt(sum_k |H_k(f)|^2) makes the bank a Parseval frame, so the
    adjoint (synthesize) inverts the analysis exactly.

    Args:
        n (int): Signal length in samples
        fs (int): Sampling rate (Hz)
        num_channels (int): Number of frequency channels
        low_freq (float): Lowest center frequency (Hz)
        high_freq (float): Highest center frequency (Hz)
        order (int): Gammatone order

    Returns:
        H (np.ndarray): Complex, shape (num_channels, n // 2 + 1)
        center_freqs (np.ndarray): Shape (num_channels,)
    """
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2")
    center_freqs = erb_space(low_freq, high_freq, num_channels)
    erb = 24.7 * (4.37 * center_freqs / 1000 + 1)
    t = np.arange(min(n, int(0.25 * fs))) / fs
    kernels = ((t ** (order - 1))
               * np.exp(-2 * np.pi * 1.019 * erb[:, None] * t)
               * np.cos(2 * np.pi * center_freqs[:, None] * t))
    H = np.fft.rfft(kernels, n=n, axis=1)
    H /= np.max(np.abs(H), axis=1, keepdims=True)
    H /= np.sqrt(np.sum(np.abs(H) ** 2, axis=0))
    return H, center_freqs


def analyze(x, H):
    """
    Split a signal into gammatone bands.

    Args:
        x (np.ndarray): Signal, shape (n,)
        H (np.ndarray): Frame from gammatone_frame(len(x), ...)

    Returns:
        bands (np.ndarray): Shape (num_channels, n)
    """
    if H.shape[1] != len(x) // 2 + 1:
        raise ValueError("frame was built for a different signal length")
    return np.fft.irfft(H * np.fft.rfft(x), n=len(x), axis=1)


def synthesize(bands, H):
    """
    Exact inverse of analyze(): recombine bands into the signal.

    Args:
        bands (np.ndarray): Shape (num_channels, n)
        H (np.ndarray): The frame used for analysis

    Returns:
        x (np.ndarray): Shape (n,)
    """
    n = bands.shape[1]
    return np.fft.irfft(np.sum(np.conj(H) * np.fft.rfft(bands, axis=1), axis=0), n=n)
```

Note for the implementer: the length check in `analyze` compares bin counts, so it cannot tell `n` from `n + 1` when `n` is even. Always build the frame with `gammatone_frame(len(x), ...)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gammatone_frame.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```powershell
git add cochlea/gammatone_frame.py tests/test_gammatone_frame.py
git commit -m "feat: perfect-reconstruction tight gammatone frame"
```

---

### Task 2: Exact-spike-time LIF population

**Files:**
- Create: `neuron_models/spike_timing.py`
- Test: `tests/test_spike_timing.py`

**Interfaces:**
- Consumes from Task 1: `gammatone_frame(n, fs, num_channels) -> (H, center_freqs)`, `analyze(x, H) -> bands` of shape `(num_channels, n)`.
- Produces:
  - `make_population(center_freqs, neurons_per_channel=16, tau=0.010, base_gain=5.0, bias_range=(1.2, 3.0), seed=0) -> dict` with keys `'channel'` (int array), `'bias'`, `'gain'`, `'v0'` (float arrays, length `num_channels * neurons_per_channel`) and `'tau'` (float)
  - `encode_spike_times(bands, fs, population, refractory_period=0.001, theta=1.0, jitter=0.0, seed=0) -> (spike_neuron, spike_time)`; int array and float-seconds array of equal length

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spike_timing.py`:

```python
"""
Tests for the deterministic exact-spike-time LIF population.

Run:  python -m pytest tests/test_spike_timing.py -v
"""
import os
import sys
# Pin the project root first so top-level packages resolve here, not under tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from cochlea.gammatone_frame import gammatone_frame, analyze
from neuron_models.spike_timing import make_population, encode_spike_times


def _make_signal(fs=16000, dur=0.25, seed=0):
    """Tones + sweep + broadband noise: energy from 100 Hz up to Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
         + 0.3 * np.sin(2 * np.pi * (300 + 12000 * t) * t) + 0.1 * rng.standard_normal(len(t)))
    return x / np.max(np.abs(x)), fs


def _encode(x, fs, num_channels=32, neurons_per_channel=16, **kwargs):
    H, cfs = gammatone_frame(len(x), fs, num_channels=num_channels)
    pop = make_population(cfs, neurons_per_channel=neurons_per_channel)
    neuron, time = encode_spike_times(analyze(x, H), fs, pop, **kwargs)
    return H, pop, neuron, time


def test_population_has_one_entry_per_neuron():
    _, cfs = gammatone_frame(4000, 16000, num_channels=8)
    pop = make_population(cfs, neurons_per_channel=3)
    for key in ('channel', 'bias', 'gain', 'v0'):
        assert len(pop[key]) == 24
    assert np.all(np.diff(pop['gain'][::3]) > 0), "gain must grow with center frequency"


def test_spikes_respect_refractory_period():
    x, fs = _make_signal()
    _, pop, neuron, time = _encode(x, fs)
    assert len(time) > 0
    for j in range(len(pop['bias'])):
        t = np.sort(time[neuron == j])
        if len(t) > 1:
            assert np.min(np.diff(t)) >= 0.001 - 1e-12


def test_spike_times_are_sub_sample():
    x, fs = _make_signal()
    _, _, _, time = _encode(x, fs)
    frac = (time * fs) % 1.0
    assert np.std(frac) > 0.1, "spike times look quantized to the sample grid"


def test_every_neuron_fires_in_silence():
    fs = 16000
    _, pop, neuron, _ = _encode(np.zeros(4000), fs)
    assert len(np.unique(neuron)) == len(pop['bias'])


def test_encoder_is_deterministic():
    x, fs = _make_signal()
    _, _, n1, t1 = _encode(x, fs)
    _, _, n2, t2 = _encode(x, fs)
    assert np.array_equal(n1, n2) and np.array_equal(t1, t2)


def test_encoder_rejects_short_refractory():
    x, fs = _make_signal()
    with pytest.raises(ValueError):
        _encode(x, fs, refractory_period=0.5 / fs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_spike_timing.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'neuron_models.spike_timing'`

- [ ] **Step 3: Write the implementation**

Create `neuron_models/spike_timing.py`:

```python
"""
Deterministic LIF population with exact (sub-sample) spike times.

Each neuron is driven by the full band signal of its channel (envelope and fine
structure), so spike timing phase-locks to the waveform. Between samples the input
is held constant, which makes the membrane trajectory an exact exponential and the
threshold-crossing time solvable in closed form.

Units are normalized: reset = 0, threshold = theta, tau * dv/dt = -v + I.
"""

import numpy as np


def make_population(center_freqs, neurons_per_channel=16, tau=0.010, base_gain=5.0,
                    bias_range=(1.2, 3.0), seed=0):
    """
    Create the fixed parameters of the neuron population.

    Gain grows with center frequency as sqrt(1 + (2*pi*fc*tau)^2), cancelling the
    membrane low-pass so every channel is measured equally well.

    Args:
        center_freqs (np.ndarray): Channel center frequencies (Hz)
        neurons_per_channel (int): Neurons per frequency channel
        tau (float): Membrane time constant (seconds)
        base_gain (float): Input gain of the lowest channel
        bias_range (tuple): (low, high) constant drive; must exceed the threshold
            so every neuron fires spontaneously
        seed (int): Seed for per-neuron bias and initial voltage

    Returns:
        population (dict): 'channel', 'bias', 'gain', 'v0' arrays of length
            num_channels * neurons_per_channel, plus scalar 'tau'
    """
    rng = np.random.default_rng(seed)
    num_channels = len(center_freqs)
    total = num_channels * neurons_per_channel
    lowpass = np.sqrt(1 + (2 * np.pi * np.asarray(center_freqs) * tau) ** 2)
    return {
        'channel': np.repeat(np.arange(num_channels), neurons_per_channel),
        'bias': rng.uniform(bias_range[0], bias_range[1], total),
        'gain': np.repeat(base_gain * lowpass / lowpass[0], neurons_per_channel),
        'v0': rng.uniform(0.0, 1.0, total),
        'tau': tau,
    }


def encode_spike_times(bands, fs, population, refractory_period=0.001, theta=1.0,
                       jitter=0.0, seed=0):
    """
    Encode band signals as spike times.

    Args:
        bands (np.ndarray): Shape (num_channels, n), output of cochlea analyze()
        fs (int): Sampling rate (Hz)
        population (dict): From make_population()
        refractory_period (float): Absolute refractory period (seconds), >= 1/fs
        theta (float): Spike threshold
        jitter (float): Std-dev of Gaussian noise added to spike times (seconds);
            0 keeps the code exact
        seed (int): Seed for the jitter

    Returns:
        spike_neuron (np.ndarray): int, neuron index of each spike
        spike_time (np.ndarray): float seconds, same length
    """
    dt = 1.0 / fs
    if refractory_period < dt:
        raise ValueError("refractory_period must be at least one sample (1/fs)")
    if np.min(population['bias']) <= theta:
        raise ValueError("every bias must exceed theta so every neuron fires")
    channel, bias, gain, tau = (population['channel'], population['bias'],
                                population['gain'], population['tau'])
    n = bands.shape[1]
    v = population['v0'].astype(float).copy()
    resume = np.zeros(len(v))  # time each neuron leaves its refractory period
    neurons, times = [], []

    for i in range(n):
        t0 = i * dt
        t1 = t0 + dt
        current = bias + gain * bands[channel, i]
        start = np.clip(resume, t0, t1)
        span = t1 - start
        can_fire = (current > theta) & (span > 0)
        ratio = np.where(can_fire, (current - v) / np.where(can_fire, current - theta, 1.0), 1.0)
        time_to_threshold = tau * np.log(ratio)
        fired = can_fire & (time_to_threshold <= span)
        v = np.where(fired, 0.0, current + (v - current) * np.exp(-span / tau))
        if fired.any():
            idx = np.flatnonzero(fired)
            t_spike = start[idx] + time_to_threshold[idx]
            neurons.append(idx)
            times.append(t_spike)
            resume[idx] = t_spike + refractory_period

    spike_neuron = np.concatenate(neurons) if neurons else np.zeros(0, dtype=int)
    spike_time = np.concatenate(times) if times else np.zeros(0)
    if jitter > 0:
        spike_time = spike_time + np.random.default_rng(seed).normal(0.0, jitter, len(spike_time))
    return spike_neuron, spike_time
```

How the loop works, for a reviewer: within sample `i` the input `current` is constant, so `v(t) = current + (v - current)·e^(-(t - start)/τ)`. Solving `v(t) = θ` gives `time_to_threshold`. A neuron still refractory at `t0` starts integrating at `resume` instead; `refractory_period >= dt` guarantees at most one spike per neuron per sample.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_spike_timing.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```powershell
git add neuron_models/spike_timing.py tests/test_spike_timing.py
git commit -m "feat: deterministic LIF population with exact spike times"
```

---

### Task 3: Spike-time decoder

**Files:**
- Create: `reconstruction/decode_spike_times.py`
- Test: `tests/test_decode_spike_times.py`

**Interfaces:**
- Consumes from Task 1: `gammatone_frame`, `analyze(x, H)`, `synthesize(bands, H)`.
- Consumes from Task 2: `make_population(...) -> dict` (keys `'channel'`, `'bias'`, `'gain'`, `'v0'`, `'tau'`), `encode_spike_times(bands, fs, population, refractory_period=0.001, theta=1.0, jitter=0.0, seed=0) -> (spike_neuron, spike_time)`.
- Produces: `decode_spike_times(spike_neuron, spike_time, n, fs, H, population, refractory_period=0.001, theta=1.0, iter_lim=2000, tol=1e-12) -> (x, info)`; `x` has shape `(n,)`, `info` is a dict with keys `'measurements'` (int), `'oversampling'` (float, measurements / n), `'iterations'` (int).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decode_spike_times.py`:

```python
"""
Tests for reconstructing audio from spike times.

Run:  python -m pytest tests/test_decode_spike_times.py -v
"""
import os
import sys
# Pin the project root first so top-level packages resolve here, not under tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pytest

from cochlea.gammatone_frame import gammatone_frame, analyze
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_spike_times


def _snr_db(reference, estimate):
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum((reference - estimate) ** 2) + 1e-300))


def _make_signal(fs=16000, dur=0.25, seed=0):
    """Tones + sweep + broadband noise: energy from 100 Hz up to Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
         + 0.3 * np.sin(2 * np.pi * (300 + 12000 * t) * t) + 0.1 * rng.standard_normal(len(t)))
    return x / np.max(np.abs(x)), fs


def _encode(x, fs, num_channels=32, neurons_per_channel=16, **kwargs):
    H, cfs = gammatone_frame(len(x), fs, num_channels=num_channels)
    pop = make_population(cfs, neurons_per_channel=neurons_per_channel)
    neuron, time = encode_spike_times(analyze(x, H), fs, pop, **kwargs)
    return H, pop, neuron, time


def test_decode_is_near_perfect():
    x, fs = _make_signal()
    H, pop, neuron, time = _encode(x, fs)
    y, info = decode_spike_times(neuron, time, len(x), fs, H, pop)
    assert info['oversampling'] > 3
    assert _snr_db(x, y) > 120, f"got {_snr_db(x, y):.1f} dB"


def test_decode_silence_returns_silence():
    fs, n = 16000, 4000
    H, pop, neuron, time = _encode(np.zeros(n), fs)
    y, _ = decode_spike_times(neuron, time, n, fs, H, pop)
    assert np.max(np.abs(y)) < 1e-9


def test_decode_ignores_spike_order():
    x, fs = _make_signal()
    H, pop, neuron, time = _encode(x, fs)
    perm = np.random.default_rng(1).permutation(len(time))
    y, _ = decode_spike_times(neuron[perm], time[perm], len(x), fs, H, pop)
    assert _snr_db(x, y) > 120


def test_decode_with_no_spikes_warns_and_returns_silence():
    fs, n = 16000, 4000
    H, cfs = gammatone_frame(n, fs, num_channels=8)
    pop = make_population(cfs, neurons_per_channel=2)
    with pytest.warns(UserWarning):
        y, info = decode_spike_times(np.zeros(0, dtype=int), np.zeros(0), n, fs, H, pop)
    assert info['measurements'] == 0 and np.all(y == 0)


def test_decode_warns_when_under_determined():
    x, fs = _make_signal()
    H, pop, neuron, time = _encode(x, fs, num_channels=8, neurons_per_channel=2)
    with pytest.warns(UserWarning, match="under-determined"):
        y, _ = decode_spike_times(neuron, time, len(x), fs, H, pop, iter_lim=50)
    assert np.all(np.isfinite(y))


def test_jitter_degrades_gracefully():
    x, fs = _make_signal()
    H, pop, neuron, time = _encode(x, fs, jitter=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = decode_spike_times(neuron, time, len(x), fs, H, pop)
    assert np.all(np.isfinite(y))
    assert 3 < _snr_db(x, y) < 80
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_decode_spike_times.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'reconstruction.decode_spike_times'`

- [ ] **Step 3: Write the implementation**

Create `reconstruction/decode_spike_times.py`:

```python
"""
Reconstruct audio from spike times alone.

Every inter-spike interval of a LIF neuron is one exact linear equation in the
audio samples: the leaky integral of the neuron's input from the end of its
refractory period to its next spike equals the threshold. Stacking the equations
of all neurons and solving them by least squares recovers the waveform.
"""

import warnings
import numpy as np
from scipy.signal import lfilter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsqr

from cochlea.gammatone_frame import analyze, synthesize


def decode_spike_times(spike_neuron, spike_time, n, fs, H, population,
                       refractory_period=0.001, theta=1.0, iter_lim=2000, tol=1e-12):
    """
    Decode spike times back to audio.

    Args:
        spike_neuron (np.ndarray): Neuron index of each spike
        spike_time (np.ndarray): Time of each spike (seconds)
        n (int): Number of audio samples to reconstruct
        fs (int): Sampling rate (Hz)
        H (np.ndarray): Gammatone frame used by the encoder
        population (dict): The encoder's make_population() dict
        refractory_period (float): Must match the encoder
        theta (float): Must match the encoder
        iter_lim (int): Maximum LSQR iterations
        tol (float): LSQR atol/btol

    Returns:
        x (np.ndarray): Reconstructed audio, shape (n,)
        info (dict): 'measurements', 'oversampling' (measurements / n), 'iterations'
    """
    num_channels = H.shape[0]
    dt = 1.0 / fs
    tau = population['tau']

    # One measurement per inter-spike interval: integration runs from a to b.
    order = np.lexsort((spike_time, spike_neuron))
    j = np.asarray(spike_neuron)[order]
    b = np.asarray(spike_time, dtype=float)[order]
    first = np.r_[True, j[1:] != j[:-1]] if len(j) else np.zeros(0, dtype=bool)
    a = np.where(first, 0.0, np.r_[0.0, b[:-1]] + refractory_period)
    v_a = np.where(first, population['v0'][j], 0.0)
    keep = (b > a) & (a >= 0) & (b < n * dt)
    j, a, b, v_a = j[keep], a[keep], b[keep], v_a[keep]
    m = len(j)
    info = {'measurements': m, 'oversampling': m / n, 'iterations': 0}
    if m == 0:
        warnings.warn("no usable spikes; returning silence")
        return np.zeros(n), info
    if m < n:
        warnings.warn(f"only {m} spike intervals for {n} samples; "
                      "reconstruction is under-determined")

    decay = np.exp(-(b - a) / tau)
    gain = population['gain'][j]
    rhs = theta - v_a * decay - population['bias'][j] * (1 - decay)

    # Leaky integral up to time t = E[i] * e + U[i] * (1 - e), i = sample holding t.
    def weights(t):
        i = np.minimum(np.floor(t / dt).astype(int), n - 1)
        e = np.exp(-(t - i * dt) / tau)
        return i, e, 1 - e

    i_b, e_b, u_b = weights(b)
    i_a, e_a, u_a = weights(a)
    col_b = population['channel'][j] * n + i_b
    col_a = population['channel'][j] * n + i_a
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[col_b, col_a]
    shape = (m, num_channels * n)
    pick_e = csr_matrix((np.r_[gain * e_b, -gain * decay * e_a], (rows, cols)), shape=shape)
    pick_u = csr_matrix((np.r_[gain * u_b, -gain * decay * u_a], (rows, cols)), shape=shape)

    alpha = np.exp(-dt / tau)
    num, den = [0.0, 1 - alpha], [1.0, -alpha]  # E[i+1] = alpha*E[i] + (1-alpha)*U[i]

    def forward(x):
        bands = analyze(x, H)
        leaky = lfilter(num, den, bands, axis=1)
        return pick_e @ leaky.ravel() + pick_u @ bands.ravel()

    def adjoint(r):
        back_e = (pick_e.T @ r).reshape(num_channels, n)
        back_u = (pick_u.T @ r).reshape(num_channels, n)
        back_u = back_u + lfilter(num, den, back_e[:, ::-1], axis=1)[:, ::-1]
        return synthesize(back_u, H)

    operator = LinearOperator((m, n), matvec=forward, rmatvec=adjoint, dtype=float)
    solution = lsqr(operator, rhs, atol=tol, btol=tol, iter_lim=iter_lim)
    info['iterations'] = int(solution[2])
    return solution[0], info
```

How the operator works, for a reviewer: `E` is the causal leaky integral of a band sampled on the grid (`E[i+1] = α·E[i] + (1-α)·U[i]`). The leaky integral up to any time `t` inside sample `i` is `e·E[i] + (1-e)·U[i]` with `e = exp(-(t - i·dt)/τ)`. The integral over `[a, b]` is that value at `b` minus `decay` times that value at `a`. `pick_e` and `pick_u` gather those four terms per equation; `adjoint` is the exact transpose (scatter, time-reversed filter, frame synthesis).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_decode_spike_times.py -v`
Expected: 6 passed (about 10 seconds)

- [ ] **Step 5: Commit**

```powershell
git add reconstruction/decode_spike_times.py tests/test_decode_spike_times.py
git commit -m "feat: least-squares audio reconstruction from spike times"
```

---

### Task 4: Wire the new paths into `recover.py`

**Files:**
- Modify: `recover.py` (docstring, imports, argument parser, Path A block, new Path N block, outputs, report rows, interpretation text)
- Modify: `tests/test_recover.py` (one new end-to-end test, and the `__main__` list)
- Modify: `README.md` (replace the planned-pipeline text with the real entry point and paths)

**Interfaces:**
- Consumes from Task 1: `gammatone_frame(n, fs, num_channels) -> (H, center_freqs)`, `analyze(x, H)`, `synthesize(bands, H)`.
- Consumes from Task 2: `make_population(center_freqs, neurons_per_channel) -> dict`, `encode_spike_times(bands, fs, population, jitter=0.0) -> (spike_neuron, spike_time)`.
- Consumes from Task 3: `decode_spike_times(spike_neuron, spike_time, n, fs, H, population) -> (x, info)`.
- Produces: `output/pathA_coherent_tfs.wav` (now from the tight frame), `output/pathN_spike_timing.wav`, and an `N` row in the printed table. New flags `--timing-channels` (default 32), `--timing-neurons` (default 16), `--jitter` (seconds, default 0.0).

- [ ] **Step 1: Write the failing end-to-end test**

In `tests/test_recover.py`, add these imports below the existing `from reconstruction.vocoder import ...` line:

```python
from cochlea.gammatone_frame import gammatone_frame, analyze, synthesize
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_spike_times
import recover
```

Add this test above the `if __name__ == "__main__":` block:

```python
def test_spike_timing_path_is_scored_as_near_perfect():
    """Path N must beat 100 dB under recover.py's own SNR metric (alignment + gain fit)."""
    x, fs = _make_signal(dur=0.25)
    H, cfs = gammatone_frame(len(x), fs, num_channels=32)
    bands = analyze(x, H)
    assert recover.snr_db(x, synthesize(bands, H)) > 200
    population = make_population(cfs, neurons_per_channel=16)
    spike_neuron, spike_time = encode_spike_times(bands, fs, population)
    y_n, _ = decode_spike_times(spike_neuron, spike_time, len(x), fs, H, population)
    assert recover.snr_db(x, y_n) > 100
    assert recover.PATH_N_NAME == 'pathN_spike_timing'
```

Add `test_spike_timing_path_is_scored_as_near_perfect` to the list of functions in the `__main__` block.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_recover.py::test_spike_timing_path_is_scored_as_near_perfect -v`
Expected: FAIL with `AttributeError: module 'recover' has no attribute 'PATH_N_NAME'`

- [ ] **Step 3: Update `recover.py` imports, docstring and arguments**

Replace the module docstring's path list (the lines from `Runs one input clip through three reconstruction paths` through `The B->C gap = information lost in the stochastic spike rate-code.`) with:

```python
Runs one input clip through six reconstruction paths and measures how faithfully
each recovers the original waveform:

  PATH T  Transparent STFT             magnitude + phase; reference ceiling.
  PATH A  Coherent (TFS-preserving)    tight gammatone frame, analysis then synthesis.
                                       What the cochlea MECHANICALLY encodes.
  PATH B  Envelope + noise vocoder     cochlear-implant style; TFS discarded.
  PATH C  Neural rate code             envelopes -> hair cell -> LIF spikes ->
                                       decoded rate -> vocoder.
  PATH C+ Rate code + borrowed TFS     diagnostic only; carrier is not from spikes.
  PATH N  Neural spike-timing code     band signal -> deterministic LIF spike times
                                       -> least-squares decode. Spikes only.

The A->B gap = information carried by temporal fine structure.
The B->C gap = information lost in the stochastic spike rate-code.
The C->N gap = what exact spike timing carries that firing rate does not.
```

Replace the `from reconstruction.vocoder import ...` statement and the line after it with:

```python
from reconstruction.vocoder import (transparent_reconstruct, tfs_vocoder,
                                    vocoder_reconstruct)
from reconstruction.decode_spikes import envelope_expansion
from cochlea.gammatone_frame import gammatone_frame, analyze, synthesize
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_spike_times

PATH_N_NAME = 'pathN_spike_timing'
```

In `main()`, add after the `--high` argument:

```python
    ap.add_argument('--timing-channels', type=int, default=32,
                    help='gammatone frame channels for paths A and N')
    ap.add_argument('--timing-neurons', type=int, default=16,
                    help='neurons per channel for path N')
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='spike-time jitter std in seconds for path N (0 = exact)')
```

- [ ] **Step 4: Replace the Path A block**

Delete everything from the line `print("Path A: coherent TFS reconstruction ...")` through the line `y_a = y_a * (orig_rms / (np.sqrt(np.mean(y_a ** 2)) + 1e-12))` and put this in its place:

```python
    print(f"Path A: tight gammatone frame, {args.timing_channels} channels ...")
    x64 = x.astype(np.float64)
    H, frame_cfs = gammatone_frame(len(x64), fs, num_channels=args.timing_channels)
    bands = analyze(x64, H)
    y_a = synthesize(bands, H)
```

- [ ] **Step 5: Add the Path N block**

Insert directly after the line `y_cp = tfs_vocoder(decoded_env, fine_structure, normalize=True, target_rms=orig_rms * 0.9)`:

```python

    # =======================================================================
    # PATH N -- neural spike-timing code (spikes only, no borrowed carrier)
    # =======================================================================
    population = make_population(frame_cfs, neurons_per_channel=args.timing_neurons)
    print(f"Path N: {len(population['bias'])} deterministic LIF neurons -> spike times ...")
    spike_neuron, spike_time = encode_spike_times(bands, fs, population, jitter=args.jitter)
    print(f"  {len(spike_time)} spikes, mean rate "
          f"{len(spike_time) / (len(population['bias']) * dur):.1f} Hz/neuron; decoding ...")
    y_n, info = decode_spike_times(spike_neuron, spike_time, len(x64), fs, H, population)
    print(f"  {info['measurements']} equations for {len(x64)} samples "
          f"({info['oversampling']:.1f}x), {info['iterations']} LSQR iterations")
```

- [ ] **Step 6: Add Path N to the outputs, the table and the interpretation**

In the `outputs` dict, add after the `'pathC+_neural_tfs': y_cp,` entry:

```python
        PATH_N_NAME: y_n,
```

In the `rows` list, add after the `report_row('C+ neural + TFS carrier', x, y_cp, fs),` entry:

```python
        report_row('N  spike timing code', x, y_n, fs),
```

Replace the interpretation `print` lines (from `print("  T ~ perfect -> ...` through the line containing `natural hearing keeps ~T.")`) with:

```python
    print("  T ~ perfect -> keeping magnitude AND phase is losslessly invertible.")
    print("  A ~ perfect -> a tight gammatone frame is invertible too.")
    print("  B << A      -> the A->B drop IS the phase / temporal fine structure.")
    print("  C <= B      -> the stochastic spike rate-code adds further, real loss.")
    print("  C+          -> diagnostic: its carrier is borrowed, not decoded from spikes.")
    print("  N ~ perfect -> exact spike TIMES carry the whole waveform; rerun with")
    print("                 --jitter 1e-5 to see how timing noise erodes it.")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests -v`
Expected: all pass, including the 5 pre-existing tests in `tests/test_recover.py` and the new one. `test_coherent_beats_vocoder_on_waveform` still imports `coherent_reconstruct` from `reconstruction.vocoder`, which is unchanged.

- [ ] **Step 8: Run the driver on the real clip and check the numbers**

Run: `.\.venv\Scripts\python.exe recover.py`
Expected (about 2.5 minutes; Path N decode is roughly 80 seconds): rows T, B, C and C+ unchanged from the Background table, and

```
A  coherent (TFS kept)        ~303      1.000    1.000      0.00
N  spike timing code          ~181      1.000    1.000      0.00
```

The run prints about 278,000 spikes at 136 Hz/neuron, 4.3x equations per sample and about 675 LSQR iterations. If Path N is below 120 dB, check that `refractory_period` and `theta` defaults match between encoder and decoder and that `bands` came from the same `H` passed to the decoder.

Then run: `.\.venv\Scripts\python.exe recover.py --jitter 1e-5`
Expected: Path N drops to roughly 15–25 dB; all other rows unchanged.

- [ ] **Step 9: Update `README.md`**

Replace lines 3–25 (the numbered seven-step list ending with the `7. Audio Reconstruction` item) with:

```markdown
A model of the human auditory periphery, built to answer one question: how much
of a sound survives each stage of hearing, and can the sound be regenerated from
the nerve signal alone?

Run it:

    .\.venv\Scripts\python.exe recover.py [sound_file] [--seconds N]

`recover.py` pushes one clip through six reconstruction paths and scores each
against the original (SNR, waveform correlation, envelope correlation, log
spectral distance). Audio for every path is written to `output/`.

| Path | What it keeps | Result |
|---|---|---|
| T | STFT magnitude + phase | perfect (reference) |
| A | tight gammatone frame, all bands | perfect |
| B | band envelopes only, noise carrier | cochlear-implant quality |
| C | spike firing rates -> envelopes | below B |
| C+ | C with fine structure borrowed from analysis | diagnostic only |
| N | exact spike times of a deterministic LIF population | perfect |

Path N is exact only for noise-free spike times. `--jitter 1e-5` adds 10
microseconds of timing noise and shows how quickly fidelity falls.

Modules: `cochlea/gammatone_frame.py` (invertible filterbank),
`neuron_models/spike_timing.py` (spike-time encoder),
`reconstruction/decode_spike_times.py` (least-squares decoder). The older
envelope/rate pipeline lives in `cochlea/filterbank.py`, `haircell/`,
`neuron_models/neuron_population.py` and `reconstruction/vocoder.py`.
```

In the directory tree further down the README, add `recover.py` under `config.py` with the comment `# End-to-end driver: runs and scores all paths`.

- [ ] **Step 10: Refresh the knowledge graph**

The project `CLAUDE.md` requires this after code changes. graphify is installed in the interpreter recorded in `.graphify_python`, not in the venv.

Run: `& (Get-Content .graphify_python) -m graphify update .`
Expected: a message that the graph was rebuilt, with `graphify-out/graph.json` and `graphify-out/GRAPH_REPORT.md` updated.

- [ ] **Step 11: Commit**

```powershell
git add recover.py tests/test_recover.py README.md
git commit -m "feat: perfect paths A and N in the reconstruction driver"
```

---

## Out of scope (next plans, in order)

1. **Hair-cell stage on the timing path.** Add compression, then adaptation, then half-wave rectification in front of the neurons one at a time, each with its own inverse where one exists, and record the SNR each costs.
2. **Stochastic fibers.** Replace the fixed bias with noisy drive and study how many fibers are needed per dB, toward the ~30,000 fibers of a real nerve.
3. **Decode speed.** The 80 s decode for 4 s of audio is LSQR on one core; block-wise decoding with overlap would make it linear in clip length and streamable.
