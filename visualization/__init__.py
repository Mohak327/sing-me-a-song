"""
Visualization module: Plotting functions for pipeline stages.
"""

from .spectrograms import plot_spectrogram, plot_waveform
from .cochlear_response import plot_cochleogram, plot_channel_responses
from .spike_raster import plot_raster, plot_raster_by_channel
from .neurogram import plot_neurogram, plot_firing_rates

__all__ = [
    'plot_spectrogram',
    'plot_waveform',
    'plot_cochleogram',
    'plot_channel_responses',
    'plot_raster',
    'plot_raster_by_channel',
    'plot_neurogram',
    'plot_firing_rates'
]
