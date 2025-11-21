import librosa
import numpy as np

def load_audio(file_path, target_sr=16000, normalize=True):
    """
    Load an audio file, downsample, and normalize.
    Args:
        file_path (str): Path to WAV file.
        target_sr (int): Target sample rate (Hz).
        normalize (bool): If True, normalize to [-1, 1].
    Returns:
        signal (np.ndarray): Mono audio array.
        sr (int): Sample rate.
    """
    signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if normalize:
        signal = signal / np.max(np.abs(signal))
    return signal, sr