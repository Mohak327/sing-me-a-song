"""
End-to-end tests: sound -> ear model -> auditory nerve spikes -> regenerated sound.

Run:  python -m pytest tests/test_auditory_periphery.py -v
"""
import os
import sys
# Pin the project root first so top-level packages resolve here, not under tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from auditory_periphery import hear, regenerate


def _snr_db(reference, estimate):
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum((reference - estimate) ** 2) + 1e-300))


def _make_signal(fs=16000, dur=0.1, seed=0):
    """Tones + sweep + broadband noise: energy from 100 Hz up to Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
         + 0.3 * np.sin(2 * np.pi * (300 + 12000 * t) * t) + 0.1 * rng.standard_normal(len(t)))
    return x / np.max(np.abs(x)), fs


def test_regenerates_audio_from_spikes_alone():
    x, fs = _make_signal()
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=8, fibers_per_channel=1024)
    y, info = regenerate(spike_neuron, spike_time, ear)
    assert len(y) == len(x)
    assert info['oversampling'] > 4
    assert _snr_db(x, y) > 120, f"got {_snr_db(x, y):.1f} dB"


def test_regenerates_quiet_and_loud_audio():
    x, fs = _make_signal()
    for level in (0.001, 1.0):
        spike_neuron, spike_time, ear = hear(level * x, fs, num_channels=8, fibers_per_channel=1024)
        y, _ = regenerate(spike_neuron, spike_time, ear)
        assert _snr_db(level * x, y) > 100, f"level {level}: {_snr_db(level * x, y):.1f} dB"


def test_silence_regenerates_as_silence():
    fs = 16000
    spike_neuron, spike_time, ear = hear(np.zeros(1600), fs, num_channels=8, fibers_per_channel=1024)
    assert len(spike_time) > 0, "fibers must fire spontaneously in silence"
    y, _ = regenerate(spike_neuron, spike_time, ear)
    assert np.max(np.abs(y)) < 1e-9


def test_too_few_fibers_warns():
    x, fs = _make_signal()
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=8, fibers_per_channel=16)
    with pytest.warns(UserWarning, match="under-determined"):
        y, _ = regenerate(spike_neuron, spike_time, ear)
    assert np.all(np.isfinite(y))


def test_float32_input_is_accepted():
    x, fs = _make_signal()
    x32 = x.astype(np.float32)
    spike_neuron, spike_time, ear = hear(x32, fs, num_channels=8, fibers_per_channel=1024)
    y, _ = regenerate(spike_neuron, spike_time, ear)
    assert _snr_db(x32.astype(np.float64), y) > 120
