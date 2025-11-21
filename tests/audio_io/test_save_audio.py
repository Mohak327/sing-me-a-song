# python -m  tests.audio_io.test_save_audio
import os
import numpy as np
from audio_io.save_audio import save_audio

def test_save_audio():
    sr = 16000
    duration = 0.1  # seconds
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    output_path = "sound_db/test_save.wav"
    save_audio(signal, output_path, sr)
    assert os.path.exists(output_path), "Output WAV file not created"

    # Check that the file is non-empty
    assert os.path.getsize(output_path) > 100, "Saved WAV file is empty or too small"
    print("Passed: save_audio writes non-empty WAV file.")

if __name__ == "__main__":
    test_save_audio()

