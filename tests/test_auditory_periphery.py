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
    assert info['misfit'] < 1e-9, "exact spike times must satisfy the spike equations"
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
    assert np.max(np.abs(y)) < 2.0


def test_float32_input_is_accepted():
    x, fs = _make_signal()
    x32 = x.astype(np.float32)
    spike_neuron, spike_time, ear = hear(x32, fs, num_channels=8, fibers_per_channel=1024)
    y, _ = regenerate(spike_neuron, spike_time, ear)
    assert _snr_db(x32.astype(np.float64), y) > 120


def _tone(freq, amplitude, fs=16000, dur=0.1):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(fs * dur)) / fs)


def _square(freq, fs=16000, dur=0.1):
    return np.sign(np.sin(2 * np.pi * freq * np.arange(int(fs * dur)) / fs) + 1e-12)


@pytest.mark.parametrize("name,x", [
    ("1 kHz full scale", _tone(1000, 1.0)),
    ("1 kHz half scale", _tone(1000, 0.5)),
    ("100 Hz full scale", _tone(100, 1.0)),
    ("60 Hz square", _square(60)),
])
def test_regenerates_loud_narrowband_sounds(name, x):
    """Sustained energy in one band must not silence the fibers of that band."""
    spike_neuron, spike_time, ear = hear(x, 16000, num_channels=8, fibers_per_channel=1024)
    y, info = regenerate(spike_neuron, spike_time, ear)
    assert info['unobserved_samples'] == 0, name
    assert _snr_db(x, y) > 100, f"{name}: {_snr_db(x, y):.1f} dB"


def test_regenerates_loud_tone_with_default_channels():
    x = _tone(1000, 1.0, dur=0.05)
    spike_neuron, spike_time, ear = hear(x, 16000)
    y, _ = regenerate(spike_neuron, spike_time, ear)
    assert _snr_db(x, y) > 100, f"got {_snr_db(x, y):.1f} dB"


@pytest.mark.parametrize("fs,fibers", [(8000, 1024), (44100, 3072)])
def test_regenerates_at_other_sample_rates(fs, fibers):
    x, _ = _make_signal(fs=fs, dur=0.05)
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=8, fibers_per_channel=fibers)
    y, _ = regenerate(spike_neuron, spike_time, ear)
    assert _snr_db(x, y) > 100, f"fs={fs}: {_snr_db(x, y):.1f} dB"


@pytest.mark.parametrize("fibers", [256, 64])
def test_too_few_fibers_degrades_without_blowing_up(fibers):
    """Missing information must cost accuracy, never produce huge samples."""
    x, fs = _make_signal()
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=8, fibers_per_channel=fibers)
    with pytest.warns(UserWarning):
        y, _ = regenerate(spike_neuron, spike_time, ear)
    assert np.max(np.abs(y)) < 2.0, f"{fibers} fibers: peak {np.max(np.abs(y)):.3g}"


def test_spike_time_jitter_degrades_without_blowing_up():
    x, fs = _make_signal()
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=8, fibers_per_channel=1024, jitter=1e-5)
    with pytest.warns(UserWarning, match="inconsistent"):
        y, info = regenerate(spike_neuron, spike_time, ear)
    assert info['misfit'] > 1e-6
    assert np.max(np.abs(y)) <= 1.0
    assert _snr_db(x, y) < 100, "10 microseconds of jitter cannot leave the result exact"


def test_hear_rejects_input_outside_full_scale():
    x, fs = _make_signal()
    with pytest.raises(ValueError, match="full scale"):
        hear(2.0 * x, fs, num_channels=8, fibers_per_channel=16)


def test_hear_rejects_stereo_and_non_finite_input():
    x, fs = _make_signal()
    with pytest.raises(ValueError, match="mono"):
        hear(np.stack([x, x]), fs, num_channels=8, fibers_per_channel=16)
    bad = x.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        hear(bad, fs, num_channels=8, fibers_per_channel=16)


def test_hear_rejects_a_tail_too_short_to_follow_the_last_samples():
    x, fs = _make_signal()
    with pytest.raises(ValueError, match="tail"):
        hear(x, fs, num_channels=8, fibers_per_channel=16, tail=0.0)
