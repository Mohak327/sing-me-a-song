"""
Spike-based envelope decoding and synthesis.
"""

import numpy as np
from scipy import signal


def smooth_firing_rates(firing_rates, fs, window_size=0.02):
    """
    Smooth firing rates using a moving average filter.
    
    Args:
        firing_rates (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate (Hz)
        window_size (float): Window size in seconds
    
    Returns:
        smoothed_rates (np.ndarray): Smoothed firing rates
    """
    window_samples = int(window_size * fs)
    
    if window_samples < 3:
        return firing_rates
    
    # Apply moving average to each channel
    smoothed = np.zeros_like(firing_rates)
    window = np.ones(window_samples) / window_samples
    
    for i in range(firing_rates.shape[0]):
        smoothed[i, :] = np.convolve(firing_rates[i, :], window, mode='same')
    
    return smoothed


def decode_to_envelopes(spike_trains, neurons_per_channel, dt, 
                        smooth_window=0.02):
    """
    Decode spike trains to envelope estimates per channel.
    
    Args:
        spike_trains (np.ndarray): Shape (total_neurons, time_steps)
        neurons_per_channel (int): Number of neurons per channel
        dt (float): Time step (seconds)
        smooth_window (float): Smoothing window (seconds)
    
    Returns:
        envelopes (np.ndarray): Decoded envelopes, shape (num_channels, time_steps)
    """
    total_neurons, n_steps = spike_trains.shape
    num_channels = total_neurons // neurons_per_channel
    
    # Sum spikes within each channel
    envelopes = np.zeros((num_channels, n_steps))
    
    for ch in range(num_channels):
        start_idx = ch * neurons_per_channel
        end_idx = start_idx + neurons_per_channel
        
        # Sum spikes across neurons in this channel
        channel_spikes = np.sum(spike_trains[start_idx:end_idx, :], axis=0)
        envelopes[ch, :] = channel_spikes
    
    # Smooth to get continuous envelopes
    if smooth_window > 0:
        envelopes = smooth_firing_rates(envelopes, 1/dt, smooth_window)
    
    # Normalize
    max_val = np.max(envelopes)
    if max_val > 0:
        envelopes = envelopes / max_val
    
    return envelopes


def decode_from_firing_rates(firing_rates, normalize=True):
    """
    Convert firing rates directly to envelopes.
    
    Args:
        firing_rates (np.ndarray): Shape (num_channels, time_steps)
        normalize (bool): Whether to normalize
    
    Returns:
        envelopes (np.ndarray): Envelope estimates
    """
    envelopes = np.copy(firing_rates)
    
    if normalize:
        max_val = np.max(envelopes)
        if max_val > 0:
            envelopes = envelopes / max_val
    
    return envelopes


def upsample_envelopes(envelopes, current_fs, target_fs):
    """
    Upsample envelopes to match target sampling rate.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        current_fs (int): Current sampling rate
        target_fs (int): Target sampling rate
    
    Returns:
        upsampled_envelopes (np.ndarray): Upsampled envelopes
    """
    if current_fs >= target_fs:
        return envelopes
    
    upsample_factor = int(target_fs / current_fs)
    
    num_channels = envelopes.shape[0]
    upsampled = []
    
    for ch in range(num_channels):
        # Use scipy resample
        upsampled_ch = signal.resample(
            envelopes[ch, :], 
            len(envelopes[ch, :]) * upsample_factor
        )
        upsampled.append(upsampled_ch)
    
    return np.array(upsampled)


def apply_gain_control(envelopes, gain=1.0):
    """
    Apply gain control to envelopes.
    
    Args:
        envelopes (np.ndarray): Input envelopes
        gain (float): Gain factor
    
    Returns:
        adjusted_envelopes (np.ndarray): Gain-adjusted envelopes
    """
    return envelopes * gain


def envelope_expansion(envelopes, expansion_factor=2.0):
    """
    Apply expansion (opposite of compression) to envelopes.
    Can help restore dynamics lost during hair cell compression.
    
    Args:
        envelopes (np.ndarray): Input envelopes
        expansion_factor (float): Expansion exponent
    
    Returns:
        expanded_envelopes (np.ndarray): Expanded envelopes
    """
    # Power-law expansion
    expanded = np.sign(envelopes) * (np.abs(envelopes) ** expansion_factor)
    return expanded
