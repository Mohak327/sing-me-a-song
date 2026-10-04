"""
recover.py -- end-to-end auditory reconstruction driver.

Runs one input clip through three reconstruction paths and measures how faithfully
each recovers the original waveform:

  PATH A  Coherent (TFS-preserving)   sum the gammatone bands directly.
                                       Upper bound: what the cochlea MECHANICALLY encodes.
  PATH B  Envelope + noise vocoder     cochlear-implant style; TFS discarded.
  PATH C  Neural spike code            full biology: envelopes -> hair cell ->
                                       LIF spikes -> decoded rate -> vocoder.
                                       What the AUDITORY NERVE actually transmits.

The A->B gap = information carried by temporal fine structure.
The B->C gap = information lost in the stochastic spike rate-code.

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
from reconstruction.vocoder import (transparent_reconstruct, coherent_reconstruct,
                                    tfs_vocoder, vocoder_reconstruct)
from reconstruction.decode_spikes import envelope_expansion


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
    print("Path A: coherent TFS reconstruction ...")
    y_a_raw = coherent_reconstruct(filtered, normalize=False)
    # Matched synthesis: the summed filterbank acts as one linear filter h_sum.
    # Measure it with a centered impulse and invert it (regularized / Wiener),
    # proving the cochlear front-end is invertible within its passband.
    delta = np.zeros(len(x)); delta[len(x) // 2] = 1.0
    imp, _ = apply_filterbank(delta, fs, num_channels=args.channels,
                              low_freq=args.low, high_freq=args.high)
    h_sum = np.sum(imp, axis=0)
    H = np.fft.rfft(np.fft.ifftshift(h_sum))
    Y = np.fft.rfft(y_a_raw)
    lam = 1e-3 * np.max(np.abs(H) ** 2)
    y_a = np.fft.irfft(Y * np.conj(H) / (np.abs(H) ** 2 + lam), n=len(x))
    y_a = y_a * (orig_rms / (np.sqrt(np.mean(y_a ** 2)) + 1e-12))

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

    # --- Save + score -------------------------------------------------------
    config.ensure_output_dir()
    outputs = {
        'original': x,
        'pathT_transparent': y_t,
        'pathA_coherent_tfs': y_a,
        'pathB_envelope_vocoder': y_b,
        'pathC_neural_spikes': y_c,
        'pathC+_neural_tfs': y_cp,
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
    print("  A high      -> even a biological gammatone bank is nearly invertible.")
    print("  B << T      -> the T->B drop IS the phase / temporal fine structure.")
    print("  C <= B      -> the stochastic spike rate-code adds further, real loss.")
    print("  C+ recovers -> most of the gap is the CARRIER, not the envelope:")
    print("                 a cochlear implant transmits ~B; natural hearing keeps ~T.")


if __name__ == '__main__':
    main()
