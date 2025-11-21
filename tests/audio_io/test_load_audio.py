from audio_io.load_audio import load_audio


def test_valid_audio_load():
    signal, sr = load_audio("sound_db/example_voice.wav", target_sr=16000)
    assert signal.ndim == 1 and sr == 16000, "Loaded audio should be mono and sr=16000"
    print("Passed: valid audio load.")

def test_invalid_path_load():
    try:
        load_audio("sound_db/nonexistent.wav")
        print("Error: nonexistent file did not raise exception.")
    except Exception:
        print("Passed: invalid path raised exception.")

def test_invalid_file_type_load():
    invalid_file = "sound_db/invalid.txt"
    with open(invalid_file, "w") as f:
        f.write("hello world")
    try:
        load_audio(invalid_file)
        print("Error: invalid file type did not raise exception.")
    except Exception:
        print("Passed: invalid file type raised exception.")

def test_argument_type_load():
    try:
        load_audio(123)
        print("Error: non-str file_path did not raise exception.")
    except Exception:
        print("Passed: non-str file_path raised exception.")

def run_load_tests():
    test_valid_audio_load()
    test_invalid_path_load()
    test_invalid_file_type_load()
    test_argument_type_load()
    print("All Audio loading tests finished!")

if __name__ == "__main__":
    run_load_tests()
