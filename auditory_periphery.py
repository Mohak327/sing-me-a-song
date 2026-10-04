"""
The auditory periphery as one invertible system.

    hear():        sound -> cochlea -> inner hair cells -> auditory nerve spikes
    regenerate():  spikes -> nerve drive -> hair cell inverse -> cochlea inverse -> sound

regenerate() sees only the spike times and the fixed anatomy returned by hear()
(filter shapes and fiber parameters); it never sees the sound.
"""

import numpy as np

from cochlea.gammatone_frame import gammatone_frame, analyze, synthesize
from haircell.inner_hair_cell import transduce, inverse_transduce
from neuron_models.spike_timing import make_population, encode_spike_times
from reconstruction.decode_spike_times import decode_drive


def hear(x, fs, num_channels=32, fibers_per_channel=1024, knee=0.02, asymmetry=0.3,
         adaptation_tau=0.010, adaptation_strength=0.5, base_gain=20.0, tail=0.03, seed=0):
    """
    Encode a sound as auditory nerve spike times.

    The defaults give 32 * 1024 = 32,768 fibers, about the number in a human
    auditory nerve.

    Args:
        x (np.ndarray): Audio samples
        fs (int): Sampling rate (Hz)
        num_channels (int): Cochlear frequency channels
        fibers_per_channel (int): Nerve fibers per channel
        knee (float): Hair cell compression knee
        asymmetry (float): Hair cell rectifier asymmetry
        adaptation_tau (float): Hair cell adaptation time constant (seconds)
        adaptation_strength (float): Hair cell adaptation strength (0-1)
        base_gain (float): Synaptic gain from hair cell to fiber
        tail (float): Seconds of silence heard after the sound, so its last
            samples are still followed by spikes
        seed (int): Seed for the fibers' thresholds and initial state

    Returns:
        spike_neuron (np.ndarray): Fiber index of each spike
        spike_time (np.ndarray): Time of each spike (seconds)
        ear (dict): The fixed anatomy needed to invert the code
    """
    x = np.asarray(x, dtype=np.float64)
    padded = np.r_[x, np.zeros(int(round(tail * fs)))]
    H, center_freqs = gammatone_frame(len(padded), fs, num_channels=num_channels)
    hair_cell = {'knee': knee, 'asymmetry': asymmetry, 'adaptation_tau': adaptation_tau,
                 'adaptation_strength': adaptation_strength}
    drive = transduce(analyze(padded, H), fs, **hair_cell)
    population = make_population(center_freqs, neurons_per_channel=fibers_per_channel,
                                 base_gain=base_gain, seed=seed, frequency_gain=False)
    spike_neuron, spike_time = encode_spike_times(drive, fs, population)
    ear = {'fs': fs, 'num_samples': len(x), 'padded_samples': len(padded), 'H': H,
           'center_freqs': center_freqs, 'hair_cell': hair_cell, 'population': population}
    return spike_neuron, spike_time, ear


def regenerate(spike_neuron, spike_time, ear):
    """
    Regenerate the sound from nerve spikes by inverting each stage in turn.

    Args:
        spike_neuron (np.ndarray): Fiber index of each spike
        spike_time (np.ndarray): Time of each spike (seconds)
        ear (dict): From hear()

    Returns:
        x (np.ndarray): Regenerated audio, same length as the original
        info (dict): 'measurements', 'oversampling' from decode_drive()
    """
    fs = ear['fs']
    drive, info = decode_drive(spike_neuron, spike_time, ear['padded_samples'], fs,
                               ear['H'].shape[0], ear['population'])
    bands = inverse_transduce(drive, fs, **ear['hair_cell'])
    return synthesize(bands, ear['H'])[:ear['num_samples']], info
