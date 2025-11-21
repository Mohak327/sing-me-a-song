"""
Reconstruction module: Decode spikes back to audio.
"""

from .decode_spikes import decode_to_envelopes, smooth_firing_rates
from .vocoder import vocoder_reconstruct, noise_vocoder, sine_vocoder

__all__ = [
    'decode_to_envelopes',
    'smooth_firing_rates',
    'vocoder_reconstruct',
    'noise_vocoder',
    'sine_vocoder'
]
