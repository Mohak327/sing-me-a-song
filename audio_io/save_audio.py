import soundfile as sf
import numpy as np

def save_audio(signal, file_path, sr=16000):
    """
    Save audio signal to a WAV file.
    Args:
        signal (np.ndarray): Audio signal.
        file_path (str): Output path.
        sr (int): Sample rate.
    """
    # Ensure floating point type and [-1, 1] range
    signal = signal.astype(np.float32)
    sf.write(file_path, signal, sr)
