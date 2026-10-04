"""
recover.py -- end-to-end auditory reconstruction driver.

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

Usage:
    python recover.py [sound_file] [--seconds N] [--channels K] [--neurons M]
"""
import argparse
import os
import numpy as np
from scipy import signal as sps

import config
from audio_io.load_audio import load_audio
from audio_io.save_audio import save_audio
from cochlea.filterbank import apply_filterbank
from cochlea.envelope_extract import extract_envelopes_from_filterbank
from haircell.transduction import apply_transduction
from neuron_models.neuron_population import simulate_population_vectorized
from reconstruction.vocoder import (transparent_reconstruct, tfs_vocoder,
                                    vocoder_reconstruct)
from reconstruction.decode_spikes import envelope_expansion
from cochlea.gammatone_frame import gammatone_frame, analyze, synthesize
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_spike_times

PATH_N_NAME = 'pathN_spike_timing'


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _best_lag(reference, estimate, max_lag=3200):
    """Find the integer lag (samples) that maximizes cross-correlation.

    Filterbank group delay is latency, not distortion, so we compensate it before
    scoring. max_lag default 3200 = 200 ms at 16 kHz (covers the 0.2 s kernel).
    """
    xc = sps.correlate(estimate, reference, mode='full', method='fft')
    lags = np.arange(-len(reference) + 1, len(estimate))
    mask = np.abs(lags) <= max_lag
    lag = lags[mask][np.argmax(xc[mask])]
    return int(lag)


def _align_and_trim(a, b):
    """Align b to a by cross-correlation lag, then trim to common length."""
    lag = _best_lag(a, b)
    if lag > 0:
        b = b[lag:]
    elif lag < 0:
        a = a[-lag:]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def snr_db(reference, estimate):
    """Signal-to-noise ratio after optimal scaling of the estimate (dB)."""
    reference, estimate = _align_and_trim(reference, estimate)
    # optimal gain that minimizes ||reference - g*estimate||
    denom = np.dot(estimate, estimate)
    g = np.dot(reference, estimate) / denom if denom > 0 else 0.0
    est = g * estimate
    noise = reference - est
    sig_p = np.dot(reference, reference)
    noise_p = np.dot(noise, noise)
    if noise_p <= 0:
        return float('inf')
    return 10 * np.log10(sig_p / noise_p)


def waveform_corr(reference, estimate):
    reference, estimate = _align_and_trim(reference, estimate)
    if np.std(reference) == 0 or np.std(estimate) == 0:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


def envelope_corr(reference, estimate, fs):
    """Correlation of broadband Hilbert envelopes -- a proxy for intelligibility."""
    reference, estimate = _align_and_trim(reference, estimate)
    er = np.abs(sps.hilbert(reference))
    ee = np.abs(sps.hilbert(estimate))
    if np.std(er) == 0 or np.std(ee) == 0:
        return 0.0
    return float(np.corrcoef(er, ee)[0, 1])


def log_spectral_distance(reference, estimate, fs, nfft=512):
    """Mean log-magnitude spectral distance (dB). Lower = closer timbre."""
    reference, estimate = _align_and_trim(reference, estimate)
    fr, _, Sr = sps.stft(reference, fs=fs, nperseg=nfft)
    fe, _, Se = sps.stft(estimate, fs=fs, nperseg=nfft)
    m = min(Sr.shape[1], Se.shape[1])
    Lr = 20 * np.log10(np.abs(Sr[:, :m]) + 1e-8)
    Le = 20 * np.log10(np.abs(Se[:, :m]) + 1e-8)
    return float(np.mean(np.abs(Lr - Le)))


def report_row(name, ref, est, fs):
    return {
        'path': name,
        'snr_db': snr_db(ref, est),
        'wave_corr': waveform_corr(ref, est),
        'env_corr': envelope_corr(ref, est, fs),
        'lsd_db': log_spectral_distance(ref, est, fs),
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sound_file', nargs='?', default='test2.mp3')
    ap.add_argument('--seconds', type=float, default=4.0)
    ap.add_argument('--channels', type=int, default=200)
    ap.add_argument('--neurons', type=int, default=10)
    ap.add_argument('--low', type=float, default=80.0, help='lowest channel freq (Hz)')
    ap.add_argument('--high', type=float, default=7800.0, help='highest channel freq (Hz)')
    ap.add_argument('--timing-channels', type=int, default=32,
                    help='gammatone frame channels for paths A and N')
    ap.add_argument('--timing-neurons', type=int, default=16,
                    help='neurons per channel for path N')
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='spike-time jitter std in seconds for path N (0 = exact)')
    args = ap.parse_args()

    fs = config.TARGET_SAMPLE_RATE
    path = config.get_sound_file(args.sound_file)
    print(f"Loading {path} @ {fs} Hz ...")
    x, fs = load_audio(path, target_sr=fs, normalize=True)
    if args.seconds:
        x = x[:int(args.seconds * fs)]
    dur = len(x) / fs
    orig_rms = np.sqrt(np.mean(x ** 2))
    print(f"  {dur:.2f}s, {len(x)} samples, RMS {orig_rms:.4f}")

    # --- Cochlear analysis --------------------------------------------------
    print(f"Filterbank: {args.channels} gammatone channels "
          f"({args.low:.0f}-{args.high:.0f} Hz) ...")
    filtered, cfs = apply_filterbank(x, fs, num_channels=args.channels,
                                     low_freq=args.low, high_freq=args.high)
    envelopes = extract_envelopes_from_filterbank(filtered, method='hilbert')
    # Temporal fine structure: the carrier inside each band (env removed).
    fine_structure = filtered / (envelopes + 1e-10)

    # =======================================================================
    # PATH T -- transparent perfect-reconstruction filterbank (true ceiling)
    # =======================================================================
    print("Path T: transparent PR filterbank (STFT, mag+phase) ...")
    y_t = transparent_reconstruct(x, fs)

    # =======================================================================
    # PATH A -- coherent, TFS preserved (cochlear mechanical upper bound)
    # =======================================================================
    print(f"Path A: tight gammatone frame, {args.timing_channels} channels ...")
    x64 = x.astype(np.float64)
    H, frame_cfs = gammatone_frame(len(x64), fs, num_channels=args.timing_channels)
    bands = analyze(x64, H)
    y_a = synthesize(bands, H)

    # =======================================================================
    # PATH B -- envelope-only noise vocoder (cochlear-implant percept)
    # =======================================================================
    print("Path B: envelope + noise vocoder (TFS discarded) ...")
    np.random.seed(0)  # reproducible carriers
    y_b = vocoder_reconstruct(envelopes, cfs, fs, method='noise',
                              normalize=True, target_rms=orig_rms * 0.9)

    # =======================================================================
    # PATH C -- full neural spike code
    # =======================================================================
    print(f"Path C: hair cell -> {args.channels * args.neurons} LIF neurons -> spikes ...")
    receptor = apply_transduction(
        envelopes, fs,
        compression_exp=config.COMPRESSION_EXPONENT,
        compression_thresh=config.COMPRESSION_THRESHOLD,
        adaptation_tau=config.ADAPTATION_TIME_CONSTANT,
        adaptation_strength=config.ADAPTATION_STRENGTH,
        apply_saturation=True)

    # Neurons run at the audio rate so decoded envelopes line up sample-for-sample.
    spikes, rates = simulate_population_vectorized(
        receptor, dt=1.0 / fs, neurons_per_channel=args.neurons,
        tau_m=config.LIF_TAU_M, v_threshold=config.LIF_V_THRESHOLD,
        v_reset=config.LIF_V_RESET, v_rest=config.LIF_V_REST,
        refractory_period=config.LIF_REFRACTORY_PERIOD,
        spontaneous_rate=config.SPONTANEOUS_RATE_MID,
        input_scale=config.INPUT_CURRENT_SCALE, seed=0)
    total_spikes = int(spikes.sum())
    print(f"  {total_spikes} spikes, mean rate "
          f"{total_spikes / (spikes.shape[0] * dur):.1f} Hz/neuron")

    # Decode rate -> envelope, invert the hair-cell compression (exp 0.3 -> ^1/0.3).
    decoded_env = rates / (np.max(rates) + 1e-12)
    decoded_env = envelope_expansion(decoded_env, expansion_factor=1.0 / config.COMPRESSION_EXPONENT)
    np.random.seed(0)
    y_c = vocoder_reconstruct(decoded_env, cfs, fs, method='noise',
                              normalize=True, target_rms=orig_rms * 0.9)

    # PATH C+ -- best honest neural decode: re-use preserved TFS as carrier.
    # (Shows the ceiling if a future decoder could also recover fine structure.)
    print("Path C+: neural envelope + preserved TFS carrier ...")
    y_cp = tfs_vocoder(decoded_env, fine_structure, normalize=True, target_rms=orig_rms * 0.9)

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

    # --- Save + score -------------------------------------------------------
    config.ensure_output_dir()
    outputs = {
        'original': x,
        'pathT_transparent': y_t,
        'pathA_coherent_tfs': y_a,
        'pathB_envelope_vocoder': y_b,
        'pathC_neural_spikes': y_c,
        'pathC+_neural_tfs': y_cp,
        PATH_N_NAME: y_n,
    }
    def listenable(sig, level=0.95):
        """Peak-normalize for a healthy, consistent playback volume."""
        sig = np.asarray(sig, dtype=np.float32)
        peak = np.max(np.abs(sig))
        return sig * (level / peak) if peak > 0 else sig

    for name, sig in outputs.items():
        save_audio(listenable(sig), fs,
                   os.path.join(config.OUTPUT_PATH, f"{name}.wav"))

    rows = [
        report_row('T  transparent (PR bank)', x, y_t, fs),
        report_row('A  coherent (TFS kept)', x, y_a, fs),
        report_row('B  envelope vocoder', x, y_b, fs),
        report_row('C  neural spikes', x, y_c, fs),
        report_row('C+ neural + TFS carrier', x, y_cp, fs),
        report_row('N  spike timing code', x, y_n, fs),
    ]

    print("\n" + "=" * 74)
    print(f"{'path':<26}{'SNR dB':>9}{'wave r':>9}{'env r':>9}{'LSD dB':>10}")
    print("-" * 74)
    for r in rows:
        snr = f"{r['snr_db']:.2f}" if np.isfinite(r['snr_db']) else "inf"
        print(f"{r['path']:<26}{snr:>9}{r['wave_corr']:>9.3f}"
              f"{r['env_corr']:>9.3f}{r['lsd_db']:>10.2f}")
    print("=" * 74)
    print(f"Audio written to {config.OUTPUT_PATH}\\  ({len(outputs)} wav files)")
    print("\nInterpretation:")
    print("  T ~ perfect -> keeping magnitude AND phase is losslessly invertible.")
    print("  A ~ perfect -> a tight gammatone frame is invertible too.")
    print("  B << A      -> the A->B drop IS the phase / temporal fine structure.")
    print("  C <= B      -> the stochastic spike rate-code adds further, real loss.")
    print("  C+          -> diagnostic: its carrier is borrowed, not decoded from spikes.")
    print("  N ~ perfect -> exact spike TIMES carry the whole waveform; rerun with")
    print("                 --jitter 1e-5 to see how timing noise erodes it.")


if __name__ == '__main__':
    main()
