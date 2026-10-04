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


def test_decode_drive_recovers_each_channels_neuron_input():
    from reconstruction.decode_spike_times import decode_drive
    fs, n, num_channels = 16000, 2000, 4
    rng = np.random.default_rng(0)
    drive = 0.01 * rng.standard_normal((num_channels, n))
    drive[:, -400:] = 0.0  # quiet tail so the last samples are still followed by spikes
    pop = make_population(np.linspace(100, 4000, num_channels), neurons_per_channel=1024,
                          base_gain=20.0, frequency_gain=False)
    neuron, time = encode_spike_times(drive, fs, pop)
    decoded, info = decode_drive(neuron, time, n, fs, num_channels, pop)
    assert decoded.shape == drive.shape
    assert info['oversampling'] > 4
    assert _snr_db(drive[:, :-400], decoded[:, :-400]) > 120


def test_decode_drive_warns_when_samples_are_unobserved():
    """A drive strong enough to silence every fiber for a whole sample leaves gaps."""
    from reconstruction.decode_spike_times import decode_drive
    fs, n, num_channels = 16000, 2000, 2
    drive = 0.05 * np.random.default_rng(0).standard_normal((num_channels, n))
    pop = make_population(np.linspace(100, 4000, num_channels), neurons_per_channel=1024,
                          base_gain=20.0, frequency_gain=False)
    neuron, time = encode_spike_times(drive, fs, pop)
    with pytest.warns(UserWarning, match="unobserved"):
        decoded, info = decode_drive(neuron, time, n, fs, num_channels, pop)
    assert info['unobserved_samples'] > 0
    assert np.all(np.isfinite(decoded))
