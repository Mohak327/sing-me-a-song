"""
Central configuration for the Sing Me A Song project.
Contains all parameters and paths used throughout the pipeline.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOUND_DB_PATH = os.path.join(PROJECT_ROOT, 'sound_db')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output')

# ============================================================================
# AUDIO I/O PARAMETERS
# ============================================================================
TARGET_SAMPLE_RATE = 16000  # Hz - downsampled rate for processing
NORMALIZE_AUDIO = True

# ============================================================================
# COCHLEAR FILTERBANK PARAMETERS
# ============================================================================
NUM_CHANNELS = 192  # Number of frequency channels (cochlear filters) - more = better freq resolution
LOW_FREQ = 80  # Hz - lowest center frequency
HIGH_FREQ = 8000  # Hz - highest center frequency
FILTER_TYPE = 'gammatone'  # 'gammatone' or 'butterworth'
FILTER_ORDER = 4  # For butterworth filters

# ============================================================================
# ENVELOPE EXTRACTION PARAMETERS
# ============================================================================
ENVELOPE_METHOD = 'hilbert'  # 'hilbert' or 'rectify'
ENVELOPE_DOWNSAMPLE_RATE = 8000  # Hz - downsample envelopes (higher = better temporal detail)

# ============================================================================
# HAIR CELL TRANSDUCTION PARAMETERS
# ============================================================================
# Nonlinear compression parameters (simulating outer hair cell function)
COMPRESSION_EXPONENT = 0.3  # Power-law exponent (0.2-0.4 typical)
COMPRESSION_THRESHOLD = 0.01  # Threshold below which no compression

# Adaptation parameters
ADAPTATION_TIME_CONSTANT = 0.01  # seconds
ADAPTATION_STRENGTH = 0.5  # 0 = no adaptation, 1 = full adaptation

# ============================================================================
# NEURON MODEL PARAMETERS
# ============================================================================
# Leaky Integrate-and-Fire (LIF) parameters
LIF_TAU_M = 0.010  # seconds - membrane time constant
LIF_V_THRESHOLD = -50  # mV - spike threshold
LIF_V_RESET = -70  # mV - reset potential after spike
LIF_V_REST = -65  # mV - resting potential
LIF_REFRACTORY_PERIOD = 0.002  # seconds - absolute refractory period

# Neuron population parameters
NEURONS_PER_CHANNEL = 20  # Number of neurons per frequency channel (more = smoother firing rates)
TOTAL_NEURONS = NUM_CHANNELS * NEURONS_PER_CHANNEL

# Spontaneous firing rate parameters
SPONTANEOUS_RATE_LOW = 1.0  # Hz
SPONTANEOUS_RATE_MID = 20.0  # Hz
SPONTANEOUS_RATE_HIGH = 80.0  # Hz

# Current injection scaling
INPUT_CURRENT_SCALE = 100.0  # Scales envelope to input current (pA)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
SIMULATION_DT = 0.0003  # seconds - time step for neuron simulation (0.3 ms)

# ============================================================================
# RECONSTRUCTION PARAMETERS
# ============================================================================
RECONSTRUCTION_METHOD = 'vocoder'  # 'vocoder' or 'simple_sum'
VOCODER_CARRIER_TYPE = 'noise'  # 'noise' or 'sine'

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
PLOT_DPI = 100
FIGURE_SIZE = (12, 8)
SPECTROGRAM_NFFT = 512
SPECTROGRAM_HOP_LENGTH = 256

# Color maps
CMAP_SPECTROGRAM = 'viridis'
CMAP_NEUROGRAM = 'hot'
CMAP_COCHLEA = 'plasma'

# ============================================================================
# LOGGING & DEBUG
# ============================================================================
VERBOSE = True
DEBUG = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_sound_file(filename):
    """Get full path to a sound file in sound_db."""
    return os.path.join(SOUND_DB_PATH, filename)
