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
