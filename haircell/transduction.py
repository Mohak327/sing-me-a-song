"""
Nonlinear hair cell transformation routines.
Simulates inner hair cell transduction from mechanical motion to receptor potential.
"""

import numpy as np
from scipy import signal


def nonlinear_compression(envelope, exponent=0.3, threshold=0.01):
    """
    Apply nonlinear compression mimicking outer hair cell function.
    Uses power-law compression to simulate cochlear amplification.
    
    Args:
        envelope (np.ndarray): Input envelope signal
        exponent (float): Compression exponent (0.2-0.4 typical for hearing)
        threshold (float): Threshold below which no compression applied
    
    Returns:
        compressed (np.ndarray): Compressed signal
    """
    # Apply threshold
    compressed = np.copy(envelope)
    above_threshold = envelope > threshold
    
    # Power-law compression for signals above threshold
    compressed[above_threshold] = threshold + (
        (envelope[above_threshold] - threshold) ** exponent
    )
    
    return compressed


def apply_adaptation(signal_data, fs, tau=0.01, strength=0.5):
    """
    Apply adaptation dynamics to simulate hair cell adaptation.
    Hair cells adapt to sustained stimuli, reducing their response over time.
    
    Args:
        signal_data (np.ndarray): Input signal (can be 1D or 2D)
        fs (int): Sampling rate (Hz)
        tau (float): Adaptation time constant (seconds)
        strength (float): Adaptation strength (0=none, 1=full)
    
    Returns:
        adapted_signal (np.ndarray): Signal with adaptation applied
    """
    if signal_data.ndim == 1:
        # Single channel
        return _adapt_single_channel(signal_data, fs, tau, strength)
    else:
        # Multiple channels
        adapted = np.zeros_like(signal_data)
        for i in range(signal_data.shape[0]):
            adapted[i, :] = _adapt_single_channel(signal_data[i, :], fs, tau, strength)
        return adapted


def _adapt_single_channel(signal_data, fs, tau, strength):
    """Apply adaptation to a single channel."""
    # Create exponential adaptation kernel
    dt = 1.0 / fs
    adaptation_state = 0.0
    adapted = np.zeros_like(signal_data)
    
    for i, val in enumerate(signal_data):
        # Update adaptation state (low-pass filtered version of input)
        adaptation_state += (val - adaptation_state) * (dt / tau)
        
        # Subtract adapted component
        adapted[i] = val - strength * adaptation_state
    
    # Ensure non-negative
    adapted = np.maximum(adapted, 0)
    
    return adapted


def saturation_nonlinearity(signal_data, max_response=1.0):
    """
    Apply saturation to simulate hair cell response limits.
    
    Args:
        signal_data (np.ndarray): Input signal
        max_response (float): Maximum response level
    
    Returns:
        saturated (np.ndarray): Saturated signal
    """
    # Sigmoid saturation
    saturated = max_response * (2 / (1 + np.exp(-2 * signal_data / max_response)) - 1)
    return saturated


def apply_transduction(envelopes, fs, compression_exp=0.3, 
                       compression_thresh=0.01, adaptation_tau=0.01,
                       adaptation_strength=0.5, apply_saturation=True):
    """
    Complete hair cell transduction pipeline.
    Converts basilar membrane motion (envelopes) to receptor potentials.
    
    Args:
        envelopes (np.ndarray): Envelope signals, shape (num_channels, signal_length)
        fs (int): Sampling rate (Hz)
        compression_exp (float): Nonlinear compression exponent
        compression_thresh (float): Compression threshold
        adaptation_tau (float): Adaptation time constant (seconds)
        adaptation_strength (float): Adaptation strength (0-1)
        apply_saturation (bool): Whether to apply saturation nonlinearity
    
    Returns:
        receptor_potentials (np.ndarray): Receptor potential signals
    """
    # Step 1: Nonlinear compression (outer hair cell effect)
    compressed = np.zeros_like(envelopes)
    for i in range(envelopes.shape[0]):
        compressed[i, :] = nonlinear_compression(
            envelopes[i, :], 
            exponent=compression_exp,
            threshold=compression_thresh
        )
    
    # Step 2: Adaptation (inner hair cell dynamics)
    adapted = apply_adaptation(
        compressed, 
        fs, 
        tau=adaptation_tau,
        strength=adaptation_strength
    )
    
    # Step 3: Optional saturation
    if apply_saturation:
        receptor_potentials = saturation_nonlinearity(adapted, max_response=1.0)
    else:
        receptor_potentials = adapted
    
    # Normalize to reasonable range
    receptor_potentials = receptor_potentials / (np.max(np.abs(receptor_potentials)) + 1e-10)
    
    return receptor_potentials


def half_wave_rectification(signal_data):
    """
    Apply half-wave rectification (remove negative values).
    Models the unidirectional nature of hair cell transduction.
    
    Args:
        signal_data (np.ndarray): Input signal
    
    Returns:
        rectified (np.ndarray): Rectified signal (only positive values)
    """
    return np.maximum(signal_data, 0)
