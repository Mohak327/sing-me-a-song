"""
Deterministic LIF population with exact (sub-sample) spike times.

Each neuron is driven by the full band signal of its channel (envelope and fine
structure), so spike timing phase-locks to the waveform. Between samples the input
is held constant, which makes the membrane trajectory an exact exponential and the
threshold-crossing time solvable in closed form.

Units are normalized: reset = 0, threshold = theta, tau * dv/dt = -v + I.
"""

import numpy as np


def make_population(center_freqs, neurons_per_channel=16, tau=0.010, base_gain=5.0,
                    bias_range=(1.2, 3.0), seed=0):
    """
    Create the fixed parameters of the neuron population.

    Gain grows with center frequency as sqrt(1 + (2*pi*fc*tau)^2), cancelling the
    membrane low-pass so every channel is measured equally well.

    Args:
        center_freqs (np.ndarray): Channel center frequencies (Hz)
        neurons_per_channel (int): Neurons per frequency channel
        tau (float): Membrane time constant (seconds)
        base_gain (float): Input gain of the lowest channel
        bias_range (tuple): (low, high) constant drive; must exceed the threshold
            so every neuron fires spontaneously
        seed (int): Seed for per-neuron bias and initial voltage

    Returns:
        population (dict): 'channel', 'bias', 'gain', 'v0' arrays of length
            num_channels * neurons_per_channel, plus scalar 'tau'
    """
    rng = np.random.default_rng(seed)
    num_channels = len(center_freqs)
    total = num_channels * neurons_per_channel
    lowpass = np.sqrt(1 + (2 * np.pi * np.asarray(center_freqs) * tau) ** 2)
    return {
        'channel': np.repeat(np.arange(num_channels), neurons_per_channel),
        'bias': rng.uniform(bias_range[0], bias_range[1], total),
        'gain': np.repeat(base_gain * lowpass / lowpass[0], neurons_per_channel),
        'v0': rng.uniform(0.0, 1.0, total),
        'tau': tau,
    }


def encode_spike_times(bands, fs, population, refractory_period=0.001, theta=1.0,
                       jitter=0.0, seed=0):
    """
    Encode band signals as spike times.

    Args:
        bands (np.ndarray): Shape (num_channels, n), output of cochlea analyze()
        fs (int): Sampling rate (Hz)
        population (dict): From make_population()
        refractory_period (float): Absolute refractory period (seconds), >= 1/fs
        theta (float): Spike threshold
        jitter (float): Std-dev of Gaussian noise added to spike times (seconds);
            0 keeps the code exact
        seed (int): Seed for the jitter

    Returns:
        spike_neuron (np.ndarray): int, neuron index of each spike
        spike_time (np.ndarray): float seconds, same length
    """
    dt = 1.0 / fs
    if refractory_period < dt:
        raise ValueError("refractory_period must be at least one sample (1/fs)")
    if np.min(population['bias']) <= theta:
        raise ValueError("every bias must exceed theta so every neuron fires")
    channel, bias, gain, tau = (population['channel'], population['bias'],
                                population['gain'], population['tau'])
    n = bands.shape[1]
    v = population['v0'].astype(float).copy()
    resume = np.zeros(len(v))  # time each neuron leaves its refractory period
    neurons, times = [], []

    for i in range(n):
        t0 = i * dt
        t1 = t0 + dt
        current = bias + gain * bands[channel, i]
        start = np.clip(resume, t0, t1)
        span = t1 - start
        can_fire = (current > theta) & (span > 0)
        ratio = np.where(can_fire, (current - v) / np.where(can_fire, current - theta, 1.0), 1.0)
        time_to_threshold = tau * np.log(ratio)
        fired = can_fire & (time_to_threshold <= span)
        v = np.where(fired, 0.0, current + (v - current) * np.exp(-span / tau))
        if fired.any():
            idx = np.flatnonzero(fired)
            t_spike = start[idx] + time_to_threshold[idx]
            neurons.append(idx)
            times.append(t_spike)
            resume[idx] = t_spike + refractory_period

    spike_neuron = np.concatenate(neurons) if neurons else np.zeros(0, dtype=int)
    spike_time = np.concatenate(times) if times else np.zeros(0)
    if jitter > 0:
        spike_time = spike_time + np.random.default_rng(seed).normal(0.0, jitter, len(spike_time))
    return spike_neuron, spike_time
