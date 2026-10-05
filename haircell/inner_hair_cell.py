"""
Invertible inner hair cell stage.

Turns basilar membrane motion (band signals) into the receptor drive of the
auditory nerve: a smooth half-wave rectifier, logarithmic compression, then
adaptation. Every step is strictly monotonic or a minimum-phase filter, so the
whole stage has an exact inverse.
"""

import numpy as np
from scipy.signal import lfilter


def _rectify(b, knee, asymmetry):
    """Smooth half-wave rectifier and its slope. Slope runs from asymmetry to 1."""
    width = knee / 10
    z = b / width
    rectified = asymmetry * b + (1 - asymmetry) * width * (np.logaddexp(0.0, z) - np.logaddexp(0.0, 0.0))
    slope = asymmetry + (1 - asymmetry) / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return rectified, slope


def receptor_nonlinearity(bands, knee=0.02, asymmetry=0.3):
    """
    Hair cell transfer function: asymmetric and compressive.

    Deflection in the excitatory direction is passed with slope 1, the opposite
    direction with slope `asymmetry` (a soft half-wave rectifier). The result is
    compressed logarithmically above `knee`.

    Args:
        bands (np.ndarray): Basilar membrane motion
        knee (float): Level where compression sets in
        asymmetry (float): Response to inhibitory deflection relative to excitatory (0-1)

    Returns:
        potential (np.ndarray): Receptor potential, same shape
    """
    rectified, _ = _rectify(np.asarray(bands, dtype=float), knee, asymmetry)
    return knee * np.arcsinh(rectified / knee)


def inverse_receptor_nonlinearity(potential, knee=0.02, asymmetry=0.3, iterations=60):
    """
    Exact inverse of receptor_nonlinearity() (Newton's method on the rectifier).

    Args:
        potential (np.ndarray): Receptor potential
        knee (float): Must match the forward call
        asymmetry (float): Must match the forward call
        iterations (int): Newton iterations

    Returns:
        bands (np.ndarray): Basilar membrane motion, same shape
    """
    target = knee * np.sinh(np.asarray(potential, dtype=float) / knee)
    b = target.copy()
    for _ in range(iterations):
        rectified, slope = _rectify(b, knee, asymmetry)
        b = b - (rectified - target) / slope
    return b


def adapt(potential, fs, tau=0.010, strength=0.5):
    """
    Adaptation: subtract a slow running average, so onsets pass and sustained
    input settles at (1 - strength) of its level.

    Args:
        potential (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate (Hz)
        tau (float): Adaptation time constant (seconds)
        strength (float): 0 = none, must be below 1

    Returns:
        adapted (np.ndarray): Same shape
    """
    k = (1.0 / fs) / tau
    return potential - strength * lfilter([k], [1.0, -(1 - k)], potential, axis=1)


def inverse_adapt(adapted, fs, tau=0.010, strength=0.5):
    """
    Exact inverse of adapt().

    Args:
        adapted (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate (Hz)
        tau (float): Must match the forward call
        strength (float): Must match the forward call

    Returns:
        potential (np.ndarray): Same shape
    """
    k = (1.0 / fs) / tau
    return lfilter([1.0, -(1 - k)], [1 - strength * k, -(1 - k)], adapted, axis=1)


def transduce(bands, fs, knee=0.02, asymmetry=0.3, adaptation_tau=0.010,
              adaptation_strength=0.5):
    """
    Full inner hair cell stage: basilar membrane motion -> nerve drive.

    Args:
        bands (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate (Hz)
        knee (float): Compression knee
        asymmetry (float): Rectifier asymmetry
        adaptation_tau (float): Adaptation time constant (seconds)
        adaptation_strength (float): Adaptation strength (0-1)

    Returns:
        drive (np.ndarray): Same shape
    """
    potential = receptor_nonlinearity(bands, knee, asymmetry)
    return adapt(potential, fs, adaptation_tau, adaptation_strength)


def inverse_transduce(drive, fs, knee=0.02, asymmetry=0.3, adaptation_tau=0.010,
                      adaptation_strength=0.5):
    """
    Exact inverse of transduce(): nerve drive -> basilar membrane motion.

    Args:
        drive (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate (Hz)
        knee, asymmetry, adaptation_tau, adaptation_strength: Must match transduce()

    Returns:
        bands (np.ndarray): Same shape
    """
    potential = inverse_adapt(drive, fs, adaptation_tau, adaptation_strength)
    return inverse_receptor_nonlinearity(potential, knee, asymmetry)
