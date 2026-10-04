"""
Recombine/envelope-sum for final audio using vocoder techniques.
"""

import numpy as np
from scipy import signal


def noise_vocoder(envelopes, center_freqs, fs, duration):
    """
    Reconstruct audio using noise-band vocoder.
    Each channel uses bandpass-filtered noise modulated by the envelope.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        center_freqs (np.ndarray): Center frequency for each channel
        fs (int): Sampling rate
        duration (float): Duration in seconds
    
    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio signal
    """
    num_channels, n_steps = envelopes.shape
    
    # Generate output
    reconstructed = np.zeros(n_steps)
    
    for ch in range(num_channels):
        # Generate bandpass-filtered noise
        noise = np.random.randn(n_steps)
        
        # Design bandpass filter for this channel
        # Approximate bandwidth
        if ch == 0:
            bw = center_freqs[1] - center_freqs[0]
        elif ch == num_channels - 1:
            bw = center_freqs[-1] - center_freqs[-2]
        else:
            bw = (center_freqs[ch + 1] - center_freqs[ch - 1]) / 2
        
        low_cutoff = max(center_freqs[ch] - bw/2, 20)
        high_cutoff = min(center_freqs[ch] + bw/2, fs/2 - 1)
        
        # Apply bandpass filter
        sos = signal.butter(4, [low_cutoff, high_cutoff], 
                          btype='bandpass', fs=fs, output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        
        # Modulate with envelope
        modulated = filtered_noise * envelopes[ch, :]
        
        # Add to output
        reconstructed += modulated
    
    return reconstructed


def sine_vocoder(envelopes, center_freqs, fs, duration):
    """
    Reconstruct audio using sine-wave vocoder.
    Each channel uses a sine wave at the center frequency modulated by envelope.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        center_freqs (np.ndarray): Center frequency for each channel
        fs (int): Sampling rate
        duration (float): Duration in seconds
    
    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio signal
    """
    num_channels, n_steps = envelopes.shape
    
    # Time array
    t = np.arange(n_steps) / fs
    
    # Generate output
    reconstructed = np.zeros(n_steps)
    
    for ch in range(num_channels):
        # Generate sine wave at center frequency
        carrier = np.sin(2 * np.pi * center_freqs[ch] * t)
        
        # Modulate with envelope
        modulated = carrier * envelopes[ch, :]
        
        # Add to output
        reconstructed += modulated
    
    return reconstructed


def vocoder_reconstruct(envelopes, center_freqs, fs, method='noise', 
                       normalize=True, gain=1.0, target_rms=None):
    """
    Reconstruct audio from envelopes using vocoder method.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        center_freqs (np.ndarray): Center frequencies
        fs (int): Sampling rate
        method (str): 'noise' or 'sine'
        normalize (bool): Whether to normalize output to [-1, 1]
        gain (float): Output gain multiplier
        target_rms (float): If provided, scale output to match this RMS level
    
    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio
    """
    duration = envelopes.shape[1] / fs
    
    if method == 'noise':
        reconstructed = noise_vocoder(envelopes, center_freqs, fs, duration)
    elif method == 'sine':
        reconstructed = sine_vocoder(envelopes, center_freqs, fs, duration)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'noise' or 'sine'.")
    
    # Normalize to [-1, 1] first to prevent clipping
    max_val = np.max(np.abs(reconstructed))
    if max_val > 0:
        reconstructed = reconstructed / max_val
    
    # Match RMS level if target provided
    if target_rms is not None:
        current_rms = np.sqrt(np.mean(reconstructed**2))
        if current_rms > 0:
            reconstructed = reconstructed * (target_rms / current_rms)
    else:
        # Just apply gain
        reconstructed = reconstructed * gain
    
    # Final clipping protection
    reconstructed = np.clip(reconstructed, -1.0, 1.0)
    
    return reconstructed


def transparent_reconstruct(x, fs, nperseg=512):
    """
    Perfect-reconstruction filterbank (STFT analysis / iSTFT synthesis).

    Each STFT bin is a bandpass channel; its complex value carries BOTH the
    amplitude envelope (magnitude) AND the temporal fine structure (phase).
    Keeping both and inverting is exact to numerical precision, so this is the
    true transparent upper bound: a frequency decomposition discards nothing as
    long as phase is retained. The distance from here to the envelope-only
    vocoder (Path B) is exactly the information carried by phase / fine structure.

    Args:
        x (np.ndarray): Input audio.
        fs (int): Sampling rate (Hz).
        nperseg (int): STFT window length.

    Returns:
        reconstructed_audio (np.ndarray): Near bit-exact reconstruction of x.
    """
    from scipy.signal import stft, istft
    noverlap = nperseg * 3 // 4  # 75% overlap, Hann window -> COLA satisfied
    _, _, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    env = np.abs(Z)              # per-channel envelope
    phase = np.angle(Z)          # per-channel temporal fine structure
    Z_rec = env * np.exp(1j * phase)
    _, y = istft(Z_rec, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.real(y)


def coherent_reconstruct(filtered_signals, normalize=True):
    """
    Near-perfect reconstruction by summing the analysis filterbank channels.

    Each gammatone channel already contains BOTH the amplitude envelope AND the
    temporal fine structure (the carrier phase). Summing the bands inverts the
    analysis filterbank up to its frequency coloration, so this preserves pitch,
    harmonics and phase. This is the "what the cochlea mechanically encodes"
    upper bound -- it deliberately bypasses the neural spike code.

    Args:
        filtered_signals (np.ndarray): Shape (num_channels, time_steps), the raw
            (real) output of the gammatone filterbank -- NOT the envelopes.
        normalize (bool): Scale peak to 1.0.

    Returns:
        reconstructed_audio (np.ndarray): TFS-preserving reconstruction.
    """
    reconstructed = np.sum(filtered_signals, axis=0)
    if normalize:
        peak = np.max(np.abs(reconstructed))
        if peak > 0:
            reconstructed = reconstructed / peak
    return reconstructed


def tfs_vocoder(envelopes, fine_structure, normalize=True, target_rms=None):
    """
    Re-modulate preserved temporal fine structure with (possibly decoded) envelopes.

    Unlike the noise/sine vocoders, the carrier here is the real per-channel fine
    structure recovered at analysis time (filtered_signal / envelope). Driving it
    with a *decoded* envelope shows how much fidelity returns when only the slow
    amplitude information has passed through the lossy stage but the fast carrier
    is retained -- the dividing line between cochlear-implant audio and natural audio.

    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps). Drive envelopes.
        fine_structure (np.ndarray): Shape (num_channels, time_steps). Carrier per channel.
        normalize (bool): Scale peak to 1.0 before RMS matching.
        target_rms (float): If given, rescale to this RMS.

    Returns:
        reconstructed_audio (np.ndarray)
    """
    reconstructed = np.sum(envelopes * fine_structure, axis=0)
    peak = np.max(np.abs(reconstructed))
    if normalize and peak > 0:
        reconstructed = reconstructed / peak
    if target_rms is not None:
        cur = np.sqrt(np.mean(reconstructed ** 2))
        if cur > 0:
            reconstructed = reconstructed * (target_rms / cur)
    return np.clip(reconstructed, -1.0, 1.0)


def simple_sum_reconstruct(envelopes, normalize=True):
    """
    Simple reconstruction by summing all envelope channels.
    Not as realistic but fast.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        normalize (bool): Whether to normalize
    
    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio
    """
    # Sum across channels
    reconstructed = np.sum(envelopes, axis=0)
    
    # Normalize
    if normalize:
        max_val = np.max(np.abs(reconstructed))
        if max_val > 0:
            reconstructed = reconstructed / max_val
    
    return reconstructed


def filter_and_sum_reconstruct(envelopes, center_freqs, fs, use_original_filters=True):
    """
    Reconstruct by filtering envelopes and summing.
    Uses the same filterbank as the forward pass.
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        center_freqs (np.ndarray): Center frequencies
        fs (int): Sampling rate
        use_original_filters (bool): Whether to use original gammatone filters
    
    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio
    """
    num_channels, n_steps = envelopes.shape
    reconstructed = np.zeros(n_steps)
    
    for ch in range(num_channels):
        # Create impulse train from envelope
        impulse_train = envelopes[ch, :]
        
        # Design bandpass filter
        if ch == 0:
            bw = center_freqs[1] - center_freqs[0]
        elif ch == num_channels - 1:
            bw = center_freqs[-1] - center_freqs[-2]
        else:
            bw = (center_freqs[ch + 1] - center_freqs[ch - 1]) / 2
        
        low_cutoff = max(center_freqs[ch] - bw/2, 20)
        high_cutoff = min(center_freqs[ch] + bw/2, fs/2 - 1)
        
        sos = signal.butter(4, [low_cutoff, high_cutoff],
                          btype='bandpass', fs=fs, output='sos')
        
        # Filter the envelope (acts as carrier)
        filtered = signal.sosfilt(sos, impulse_train)
        
        reconstructed += filtered
    
    # Normalize
    max_val = np.max(np.abs(reconstructed))
    if max_val > 0:
        reconstructed = reconstructed / max_val
    
    return reconstructed
