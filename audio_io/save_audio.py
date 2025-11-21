import soundfile as sf
import numpy as np
import os

def save_audio(signal, sr, file_path):
    """
    Save audio signal to a WAV file.
    Args:
        signal (np.ndarray): Audio signal.
        sr (int): Sample rate.
        file_path (str): Output path.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Ensure floating point type and [-1, 1] range
    signal = signal.astype(np.float32)
    sf.write(file_path, signal, sr)
