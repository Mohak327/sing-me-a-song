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
