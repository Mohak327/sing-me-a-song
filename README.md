**Sing Me A Song**

1. Load and preprocess the audio input:
    - Use librosa or scipy to load WAV files, downsample, normalize.

2. Simulate cochlear filtering:
    - Design a bank of bandpass filters to extract multiple frequency channels.
    - For each channel, generate cochlear basilar membrane motion analog.

3. Model inner hair cell transduction:
    - Apply nonlinear transfer functions simulating receptor potentials.

4. Simulate auditory nerve spiking:
    - Drive a population of LIF or Hodgkin-Huxley neurons with processed signals.
    - Monitor spike timings representing neuronal "voice encoding."

5. Reconstruct or visualize neuronal output:
    - Sum neural firing rates as neurograms or cochleograms.
    - Optionally, perform reverse-transform to audio using vocoder principles.

6. Visualize intermediate signals:
    - Plot spectrograms, cochlear responses, spike rasters for understanding.

7. Audio Reconstruction
    - Sum/envelope spike rates per channel, reconstruct with amplitude modulation per channel, superpose to reconstruct audio.

-----

/sing-me-a-song/

├── README.md
├── requirements.txt
├── main_pipeline.ipynb           # Master Jupyter notebook: runs each step, 
├── config.py                     # Central config (params, file paths, etc)
├── audio_io/
│   ├── __init__.py
│   ├── load_audio.py             # Load, downsample, normalize audio files
│   └── save_audio.py             # Write reconstructed audio to WAV
├── cochlea/
│   ├── __init__.py
│   ├── filterbank.py             # Gammatone/Butterworth filterbank functions
│   └── envelope_extract.py       # Envelope, phase extraction from filtered signals
├── haircell/
│   ├── __init__.py
│   └── transduction.py           # Nonlinear hair cell transformation routines
├── neuron_models/
│   ├── __init__.py
│   ├── lif_neuron.py             # Leaky Integrate & Fire neuron implementation
│   ├── hh_neuron.py              # Hodgkin-Huxley neuron implementation
│   └── neuron_population.py      # Simulation and population management
├── visualization/
│   ├── __init__.py
│   ├── spectrograms.py           # Input and filtered spectrogram visualizer
│   ├── cochlear_response.py      # Plots of filterbank output
│   └── spike_raster.py           # Raster plots of neuronal output
│   └── neurogram.py              # Heatmap of firing rates
├── reconstruction/
│   ├── __init__.py
│   ├── decode_spikes.py          # Spike-based envelope decoding, synthesis
│   └── vocoder.py                # Recombine/envelope-sum for final audio
├── tests/
│   ├── test_io.py
│   ├── test_cochlea.py
│   ├── test_haircell.py
│   ├── test_neurons.py
│   ├── test_visualization.py
│   └── test_reconstruction.py
└── sound_db/
    ├── example_voice.wav
    └── ...
