"""
Input and filtered spectrogram visualizer.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform(signal, fs, title='Audio Waveform', figsize=(12, 4)):
    """
    Plot audio waveform.
    
    Args:
        signal (np.ndarray): Audio signal
        fs (int): Sampling rate
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    time_axis = np.arange(len(signal)) / fs
    ax.plot(time_axis, signal, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_spectrogram(signal, fs, title='Spectrogram', nfft=512, 
                     hop_length=256, cmap='viridis', figsize=(12, 5)):
    """
    Plot spectrogram of audio signal.
    
    Args:
        signal (np.ndarray): Audio signal
        fs (int): Sampling rate
        title (str): Plot title
        nfft (int): FFT size
        hop_length (int): Hop length for STFT
        cmap (str): Colormap name
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal, n_fft=nfft, hop_length=hop_length)), 
        ref=np.max
    )
    
    img = librosa.display.specshow(
        D, sr=fs, x_axis='time', y_axis='hz', 
        cmap=cmap, ax=ax, hop_length=hop_length
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax


def plot_mel_spectrogram(signal, fs, title='Mel Spectrogram', 
                         n_mels=128, cmap='viridis', figsize=(12, 5)):
    """
    Plot mel-scaled spectrogram.
    
    Args:
        signal (np.ndarray): Audio signal
        fs (int): Sampling rate
        title (str): Plot title
        n_mels (int): Number of mel bands
        cmap (str): Colormap
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(
        S_db, sr=fs, x_axis='time', y_axis='mel',
        cmap=cmap, ax=ax
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax


def plot_comparison(original_signal, reconstructed_signal, fs, figsize=(12, 8)):
    """
    Plot comparison between original and reconstructed audio.
    
    Args:
        original_signal (np.ndarray): Original audio
        reconstructed_signal (np.ndarray): Reconstructed audio
        fs (int): Sampling rate
        figsize (tuple): Figure size
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    time_axis_orig = np.arange(len(original_signal)) / fs
    time_axis_recon = np.arange(len(reconstructed_signal)) / fs
    
    # Waveforms
    axes[0, 0].plot(time_axis_orig, original_signal, linewidth=0.5)
    axes[0, 0].set_title('Original Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_axis_recon, reconstructed_signal, linewidth=0.5, color='red')
    axes[0, 1].set_title('Reconstructed Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectrograms
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_signal)), ref=np.max)
    librosa.display.specshow(D_orig, sr=fs, x_axis='time', y_axis='hz', 
                            cmap='viridis', ax=axes[1, 0])
    axes[1, 0].set_title('Original Spectrogram')
    
    D_recon = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed_signal)), ref=np.max)
    librosa.display.specshow(D_recon, sr=fs, x_axis='time', y_axis='hz',
                            cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Reconstructed Spectrogram')
    
    plt.tight_layout()
    
    return fig, axes
