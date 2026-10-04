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
