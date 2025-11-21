"""
Envelope extraction and phase analysis from filtered cochlear signals.
"""

import numpy as np
from scipy import signal


def extract_envelope(filtered_signal, method='hilbert'):
    """
    Extract the amplitude envelope from a filtered signal.
    This represents the mechanical motion amplitude of the basilar membrane.
    
    Args:
        filtered_signal (np.ndarray): Single channel filtered signal
        method (str): Method to use ('hilbert' or 'rectify')
    
    Returns:
        envelope (np.ndarray): Amplitude envelope of the signal
    """
    if method == 'hilbert':
        # Hilbert transform gives analytic signal
        analytic_signal = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
    elif method == 'rectify':
        # Half-wave rectification + low-pass filtering
        rectified = np.maximum(filtered_signal, 0)
        # Low-pass filter to smooth (cutoff ~400 Hz for envelope)
        sos = signal.butter(4, 400, btype='low', fs=16000, output='sos')
        envelope = signal.sosfilt(sos, rectified)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hilbert' or 'rectify'.")
    
    return envelope


def extract_instantaneous_phase(filtered_signal):
    """
    Extract instantaneous phase from filtered signal using Hilbert transform.
    
    Args:
        filtered_signal (np.ndarray): Single channel filtered signal
    
    Returns:
        inst_phase (np.ndarray): Instantaneous phase (radians)
        inst_freq (np.ndarray): Instantaneous frequency (Hz)
    """
    # Hilbert transform
    analytic_signal = signal.hilbert(filtered_signal)
    
    # Instantaneous phase
    inst_phase = np.angle(analytic_signal)
    
    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(np.unwrap(inst_phase))
    inst_freq = np.concatenate(([inst_freq[0]], inst_freq))  # Pad to original length
    
    return inst_phase, inst_freq


def extract_envelopes_from_filterbank(filtered_signals, method='hilbert'):
    """
    Extract envelopes from all channels of a filterbank output.
    
    Args:
        filtered_signals (np.ndarray): Shape (num_channels, signal_length)
        method (str): Envelope extraction method
    
    Returns:
        envelopes (np.ndarray): Shape (num_channels, signal_length)
    """
    num_channels = filtered_signals.shape[0]
    signal_length = filtered_signals.shape[1]
    envelopes = np.zeros((num_channels, signal_length))
    
    for i in range(num_channels):
        envelopes[i, :] = extract_envelope(filtered_signals[i, :], method=method)
    
    return envelopes


def temporal_fine_structure(filtered_signal, envelope):
    """
    Extract temporal fine structure (TFS) by dividing out the envelope.
    TFS represents the fast oscillations within each frequency channel.
    
    Args:
        filtered_signal (np.ndarray): Filtered signal
        envelope (np.ndarray): Amplitude envelope
    
    Returns:
        tfs (np.ndarray): Temporal fine structure
    """
    # Avoid division by zero
    epsilon = 1e-10
    tfs = filtered_signal / (envelope + epsilon)
    
    return tfs


def downsample_envelopes(envelopes, original_fs, target_fs):
    """
    Downsample envelope signals for efficiency (envelopes vary slowly).
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, signal_length)
        original_fs (int): Original sampling rate
        target_fs (int): Target sampling rate (typically 1000-2000 Hz)
    
    Returns:
        downsampled_envelopes (np.ndarray): Downsampled envelopes
        target_fs (int): Actual downsampling rate
    """
    if target_fs >= original_fs:
        return envelopes, original_fs
    
    num_channels = envelopes.shape[0]
    
    # Calculate decimation factor
    decimation_factor = int(original_fs / target_fs)
    
    # Downsample each channel
    downsampled = []
    for i in range(num_channels):
        downsampled_channel = signal.decimate(
            envelopes[i, :], 
            decimation_factor, 
            ftype='fir'
        )
        downsampled.append(downsampled_channel)
    
    downsampled_envelopes = np.array(downsampled)
    actual_fs = original_fs / decimation_factor
    
    return downsampled_envelopes, int(actual_fs)
