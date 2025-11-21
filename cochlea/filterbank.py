"""
Gammatone filterbank implementation for cochlear frequency decomposition.
"""

import numpy as np
from scipy import signal


def design_gammatone_filterbank(fs, num_channels=32, low_freq=100, high_freq=8000):
    """
    Design a gammatone filterbank mimicking cochlear frequency selectivity.
    
    Args:
        fs (int): Sampling rate (Hz)
        num_channels (int): Number of frequency channels
        low_freq (float): Lowest center frequency (Hz)
        high_freq (float): Highest center frequency (Hz)
    
    Returns:
        center_freqs (np.ndarray): Center frequencies for each channel (Hz)
        bandwidths (np.ndarray): Bandwidth for each channel (Hz)
    """
    # ERB (Equivalent Rectangular Bandwidth) scale for human hearing
    # Greenwood function approximation for frequency-to-place mapping
    center_freqs = np.logspace(
        np.log10(low_freq), 
        np.log10(high_freq), 
        num_channels
    )
    
    # ERB bandwidth approximation: ERB = 24.7 * (4.37 * f/1000 + 1)
    bandwidths = 24.7 * (4.37 * center_freqs / 1000 + 1)
    
    return center_freqs, bandwidths


def gammatone_impulse_response(t, fc, bandwidth, n=4, phi=0):
    """
    Generate gammatone impulse response.
    
    Args:
        t (np.ndarray): Time array (seconds)
        fc (float): Center frequency (Hz)
        bandwidth (float): Bandwidth (Hz)
        n (int): Filter order (typically 4 for gammatone)
        phi (float): Phase offset (radians)
    
    Returns:
        h (np.ndarray): Impulse response
    """
    # Gammatone function: a * t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*fc*t + phi)
    b = bandwidth
    a = 1.0  # Normalization constant
    
    h = a * (t ** (n - 1)) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t + phi)
    h[t < 0] = 0  # Causal filter
    
    return h


def apply_filterbank(signal_data, fs, num_channels=32, low_freq=100, high_freq=8000):
    """
    Apply gammatone filterbank to an audio signal.
    
    Args:
        signal_data (np.ndarray): Input audio signal
        fs (int): Sampling rate (Hz)
        num_channels (int): Number of frequency channels
        low_freq (float): Lowest frequency (Hz)
        high_freq (float): Highest frequency (Hz)
    
    Returns:
        filtered_signals (np.ndarray): Shape (num_channels, signal_length)
        center_freqs (np.ndarray): Center frequency for each channel
    """
    center_freqs, bandwidths = design_gammatone_filterbank(
        fs, num_channels, low_freq, high_freq
    )
    
    signal_length = len(signal_data)
    filtered_signals = np.zeros((num_channels, signal_length))
    
    # Design and apply each gammatone filter
    for i, (fc, bw) in enumerate(zip(center_freqs, bandwidths)):
        # Create impulse response (200ms duration should be sufficient)
        duration = 0.2
        t = np.arange(0, duration, 1/fs)
        h = gammatone_impulse_response(t, fc, bw)
        
        # Normalize
        h = h / np.sum(np.abs(h))
        
        # Apply filter via convolution
        filtered_signals[i, :] = signal.convolve(signal_data, h, mode='same')
    
    return filtered_signals, center_freqs


def apply_butterworth_filterbank(signal_data, fs, num_channels=32, low_freq=100, high_freq=8000, order=4):
    """
    Alternative: Apply Butterworth bandpass filterbank (simpler, faster).
    
    Args:
        signal_data (np.ndarray): Input audio signal
        fs (int): Sampling rate (Hz)
        num_channels (int): Number of frequency channels
        low_freq (float): Lowest frequency (Hz)
        high_freq (float): Highest frequency (Hz)
        order (int): Filter order
    
    Returns:
        filtered_signals (np.ndarray): Shape (num_channels, signal_length)
        center_freqs (np.ndarray): Center frequency for each channel
    """
    center_freqs, bandwidths = design_gammatone_filterbank(
        fs, num_channels, low_freq, high_freq
    )
    
    signal_length = len(signal_data)
    filtered_signals = np.zeros((num_channels, signal_length))
    
    for i, (fc, bw) in enumerate(zip(center_freqs, bandwidths)):
        # Define bandpass region
        low_cutoff = max(fc - bw/2, 20)  # Don't go below 20 Hz
        high_cutoff = min(fc + bw/2, fs/2 - 1)  # Nyquist limit
        
        # Design Butterworth bandpass filter
        sos = signal.butter(
            order, 
            [low_cutoff, high_cutoff], 
            btype='bandpass', 
            fs=fs, 
            output='sos'
        )
        
        # Apply filter
        filtered_signals[i, :] = signal.sosfilt(sos, signal_data)
    
    return filtered_signals, center_freqs
