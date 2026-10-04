**Sing Me A Song**

A model of the human auditory periphery, built to answer one question: how much
of a sound survives each stage of hearing, and can the sound be regenerated from
the nerve signal alone?

Run it:

    .\.venv\Scripts\python.exe recover.py [sound_file] [--seconds N]

`recover.py` pushes one clip through seven reconstruction paths and scores each
against the original (SNR, waveform correlation, envelope correlation, log
spectral distance). Audio for every path is written to `output/`.

| Path | What it keeps | Result |
|---|---|---|
| T | STFT magnitude + phase | perfect (reference) |
| A | tight gammatone frame, all bands | perfect |
| B | band envelopes only, noise carrier | cochlear-implant quality |
| C | spike firing rates -> envelopes | below B |
| C+ | C with fine structure borrowed from analysis | diagnostic only |
| N | exact spike times of 512 deterministic LIF neurons, no hair cells | perfect |
| H | the whole ear: cochlea, hair cells, 32,768 nerve fibers | perfect |

Path H is the full model. `auditory_periphery.hear()` turns sound into spike
times through a gammatone cochlea, inner hair cells (soft half-wave
rectification, logarithmic compression, adaptation) and a population of leaky
integrate-and-fire nerve fibers. `regenerate()` gets only the spike times and
inverts each stage in turn: nerve, hair cells, cochlea.

    from auditory_periphery import hear, regenerate
    spike_neuron, spike_time, ear = hear(audio, fs)
    audio_again, info = regenerate(spike_neuron, spike_time, ear)

Both N and H are exact only for noise-free spike times and deterministic
fibers. `--jitter 1e-9` adds one nanosecond of timing noise to both and shows how
quickly fidelity falls; path H is the more fragile (about 73 dB at 1 ns, 32 dB at
100 ns, nothing useful at 10 microseconds on a short test signal). When spike
times do not fit noise-free fibers, `regenerate()` warns and reports the misfit.

Path H needs several spike intervals per audio sample in every channel: 512
fibers per channel still gave 202 dB on a test signal, 256 gave 44 dB, and fewer
degrade further with a warning. Audio passed to `hear()` must be mono and within
[-1, 1]. The sample rate sets how many fibers are needed; at 44.1 kHz use about
3,000 per channel.

`recover.py` prints two scores for the exact paths: the table's SNR, which fits a
gain and a delay first, and a raw SNR with no fitting at all.

Modules: `auditory_periphery.py` (the ear as one invertible system),
`cochlea/gammatone_frame.py` (invertible filterbank),
`haircell/inner_hair_cell.py` (invertible hair cell stage),
`neuron_models/spike_timing.py` (spike-time encoder),
`reconstruction/decode_spike_times.py` (decoders). The older
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
