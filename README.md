**Sing Me A Song**

A model of the human auditory periphery, built to answer one question: how much
of a sound survives each stage of hearing, and can the sound be regenerated from
the nerve signal alone?

Run it:

    .\.venv\Scripts\python.exe recover.py [sound_file] [--seconds N]

`recover.py` pushes one clip through six reconstruction paths and scores each
against the original (SNR, waveform correlation, envelope correlation, log
spectral distance). Audio for every path is written to `output/`.

| Path | What it keeps | Result |
|---|---|---|
| T | STFT magnitude + phase | perfect (reference) |
| A | tight gammatone frame, all bands | perfect |
| B | band envelopes only, noise carrier | cochlear-implant quality |
| C | spike firing rates -> envelopes | below B |
| C+ | C with fine structure borrowed from analysis | diagnostic only |
| N | exact spike times of a deterministic LIF population | perfect |

Path N is exact only for noise-free spike times. `--jitter 1e-5` adds 10
microseconds of timing noise and shows how quickly fidelity falls.

Modules: `cochlea/gammatone_frame.py` (invertible filterbank),
`neuron_models/spike_timing.py` (spike-time encoder),
`reconstruction/decode_spike_times.py` (least-squares decoder). The older
envelope/rate pipeline lives in `cochlea/filterbank.py`, `haircell/`,
`neuron_models/neuron_population.py` and `reconstruction/vocoder.py`.

-----

/sing-me-a-song/  
  
├── README.md  
├── requirements.txt  
├── main_pipeline.ipynb           # Master Jupyter notebook: runs each step,  
├── config.py                     # Central config (params, file paths, etc)  
├── recover.py                    # End-to-end driver: runs and scores all paths  
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
