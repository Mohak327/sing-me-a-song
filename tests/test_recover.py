"""
Integration tests for the reconstruction pipeline (recover.py building blocks).

Run:  python -m pytest tests/test_recover.py -v
or:   python tests/test_recover.py
"""
import os
import sys
# Pin the project root first so top-level `audio_io` resolves here, not the
# shadowing `tests/audio_io` package under pytest's path insertion.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cochlea.filterbank import apply_filterbank
from cochlea.envelope_extract import extract_envelopes_from_filterbank
from haircell.transduction import apply_transduction
from neuron_models.neuron_population import simulate_population_vectorized
from reconstruction.vocoder import (transparent_reconstruct, coherent_reconstruct,
                                    vocoder_reconstruct, tfs_vocoder)
from cochlea.gammatone_frame import gammatone_frame, analyze, synthesize
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_spike_times
import recover
from audio_io.load_audio import load_audio
import config


def _make_signal(fs=16000, dur=1.0):
    """A two-tone + sweep test signal: has both pitch and broadband content."""
    t = np.arange(int(fs * dur)) / fs
    tone = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    sweep = 0.3 * np.sin(2 * np.pi * (300 + 1500 * t) * t)
    x = tone + sweep
    return x / np.max(np.abs(x)), fs


def test_load_audio_silence_no_nan():
    """The zero-guard fix: silent input must not produce NaNs."""
    import soundfile as sf, tempfile, os
    fs = 16000
    silent = np.zeros(fs, dtype=np.float32)
    path = os.path.join(tempfile.gettempdir(), "_silent_test.wav")
    sf.write(path, silent, fs)
    sig, sr = load_audio(path, target_sr=fs, normalize=True)
    os.remove(path)
    assert not np.any(np.isnan(sig)), "silent input produced NaNs"


def test_vectorized_lif_respects_refractory():
    """No neuron may spike twice within its refractory window."""
    fs = 16000
    dt = 1.0 / fs
    receptor = np.ones((4, fs))  # strong constant drive -> high firing
    spikes, rates = simulate_population_vectorized(
        receptor, dt=dt, neurons_per_channel=5,
        refractory_period=config.LIF_REFRACTORY_PERIOD, input_scale=100.0, seed=1)
    assert spikes.sum() > 0, "no spikes produced under strong drive"
    refrac_steps = int(config.LIF_REFRACTORY_PERIOD / dt)
    for row in spikes:
        idx = np.flatnonzero(row)
        if len(idx) > 1:
            assert np.min(np.diff(idx)) >= refrac_steps, "spikes violate refractory period"
    assert np.all(np.isfinite(rates))


def test_transparent_is_near_perfect():
    """The PR filterbank (mag+phase) must reconstruct x to high SNR."""
    x, fs = _make_signal()
    y = transparent_reconstruct(x, fs)
    n = min(len(x), len(y))
    a, b = x[:n], y[:n]
    noise = a - b * (np.dot(a, b) / np.dot(b, b))
    snr = 10 * np.log10(np.dot(a, a) / (np.dot(noise, noise) + 1e-20))
    assert snr > 40, f"transparent reconstruction not transparent: {snr:.1f} dB"


def test_coherent_beats_vocoder_on_waveform():
    """Path A (TFS preserved) must track the waveform far better than Path B (envelope only)."""
    x, fs = _make_signal()
    filtered, cfs = apply_filterbank(x, fs, num_channels=64,
                                     low_freq=config.LOW_FREQ, high_freq=config.HIGH_FREQ)
    envelopes = extract_envelopes_from_filterbank(filtered, method='hilbert')

    y_a = coherent_reconstruct(filtered, normalize=True)
    np.random.seed(0)
    y_b = vocoder_reconstruct(envelopes, cfs, fs, method='noise', normalize=True)

    def corr(a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        # align by best lag (filterbank group delay)
        from scipy.signal import correlate
        xc = correlate(b, a, mode='full', method='fft')
        lag = np.arange(-len(a) + 1, len(b))[np.argmax(np.abs(xc))]
        if lag > 0:
            b = b[lag:]
        elif lag < 0:
            a = a[-lag:]
        n = min(len(a), len(b))
        return abs(np.corrcoef(a[:n], b[:n])[0, 1])

    ca, cb = corr(x, y_a), corr(x, y_b)
    # Primary claim: TFS-preserving reconstruction dominates the envelope vocoder.
    assert ca > 2 * cb, f"coherent ({ca:.3f}) should clearly beat vocoder ({cb:.3f})"
    # Bare band-summation is spectrally colored (~0.4); the driver's matched-
    # synthesis deconvolution lifts it to ~0.7. Bound the bare version loosely.
    assert ca > 0.35, f"coherent reconstruction too weak: {ca:.3f}"


def test_outputs_are_finite_and_bounded():
    """Every reconstruction path must produce finite, in-range audio."""
    x, fs = _make_signal()
    filtered, cfs = apply_filterbank(x, fs, num_channels=48,
                                     low_freq=config.LOW_FREQ, high_freq=config.HIGH_FREQ)
    env = extract_envelopes_from_filterbank(filtered, method='hilbert')
    fine = filtered / (env + 1e-10)
    receptor = apply_transduction(env, fs)
    spikes, rates = simulate_population_vectorized(receptor, dt=1.0 / fs,
                                                   neurons_per_channel=6, seed=0)
    decoded = rates / (np.max(rates) + 1e-12)
    for name, y in [
        ("A", coherent_reconstruct(filtered)),
        ("B", vocoder_reconstruct(env, cfs, fs, method='noise')),
        ("C+", tfs_vocoder(decoded, fine)),
    ]:
        assert np.all(np.isfinite(y)), f"path {name} has non-finite samples"
        assert np.max(np.abs(y)) <= 1.0 + 1e-6, f"path {name} out of [-1,1]"


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


if __name__ == "__main__":
    for fn in [test_load_audio_silence_no_nan,
               test_vectorized_lif_respects_refractory,
               test_coherent_beats_vocoder_on_waveform,
               test_outputs_are_finite_and_bounded,
               test_spike_timing_path_is_scored_as_near_perfect]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("All reconstruction tests passed.")
