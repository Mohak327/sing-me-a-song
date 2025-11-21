"""
Leaky Integrate-and-Fire (LIF) neuron implementation.
"""

import numpy as np


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.
    """
    
    def __init__(self, tau_m=0.010, v_threshold=-50, v_reset=-70, 
                 v_rest=-65, refractory_period=0.002):
        """
        Initialize LIF neuron parameters.
        
        Args:
            tau_m (float): Membrane time constant (seconds)
            v_threshold (float): Spike threshold (mV)
            v_reset (float): Reset potential after spike (mV)
            v_rest (float): Resting potential (mV)
            refractory_period (float): Absolute refractory period (seconds)
        """
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory_period = refractory_period
        
        # State variables
        self.v = v_rest
        self.last_spike_time = -np.inf
        self.spike_times = []
    
    def reset(self):
        """Reset neuron state."""
        self.v = self.v_rest
        self.last_spike_time = -np.inf
        self.spike_times = []
    
    def step(self, current, dt, current_time):
        """
        Simulate one time step.
        
        Args:
            current (float): Input current (arbitrary units)
            dt (float): Time step (seconds)
            current_time (float): Current simulation time (seconds)
        
        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Check if in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        # Update membrane potential using Euler method
        # dV/dt = (V_rest - V + R*I) / tau_m
        dv = ((self.v_rest - self.v) + current) / self.tau_m
        self.v += dv * dt
        
        # Check for spike
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            return True
        
        return False


def simulate_lif_neuron(input_current, dt, tau_m=0.010, v_threshold=-50,
                        v_reset=-70, v_rest=-65, refractory_period=0.002,
                        spontaneous_rate=0.0):
    """
    Simulate a single LIF neuron for a given input current time series.
    
    Args:
        input_current (np.ndarray): Input current over time
        dt (float): Time step (seconds)
        tau_m (float): Membrane time constant (seconds)
        v_threshold (float): Spike threshold (mV)
        v_reset (float): Reset potential (mV)
        v_rest (float): Resting potential (mV)
        refractory_period (float): Refractory period (seconds)
        spontaneous_rate (float): Spontaneous firing rate (Hz)
    
    Returns:
        spike_train (np.ndarray): Binary spike train (1=spike, 0=no spike)
        spike_times (np.ndarray): Times of spikes (seconds)
        voltage_trace (np.ndarray): Membrane voltage over time
    """
    neuron = LIFNeuron(tau_m, v_threshold, v_reset, v_rest, refractory_period)
    
    n_steps = len(input_current)
    spike_train = np.zeros(n_steps, dtype=int)
    voltage_trace = np.zeros(n_steps)
    
    # Add spontaneous activity as Poisson noise
    if spontaneous_rate > 0:
        spontaneous_spikes = np.random.poisson(spontaneous_rate * dt, n_steps)
        spontaneous_current = spontaneous_spikes * (v_threshold - v_rest) * 0.5
    else:
        spontaneous_current = np.zeros(n_steps)
    
    for i in range(n_steps):
        current_time = i * dt
        total_current = input_current[i] + spontaneous_current[i]
        
        spiked = neuron.step(total_current, dt, current_time)
        spike_train[i] = int(spiked)
        voltage_trace[i] = neuron.v
    
    spike_times = np.array(neuron.spike_times)
    
    return spike_train, spike_times, voltage_trace


def calculate_firing_rate(spike_train, dt, window_size=0.05):
    """
    Calculate instantaneous firing rate from spike train.
    
    Args:
        spike_train (np.ndarray): Binary spike train
        dt (float): Time step (seconds)
        window_size (float): Window size for smoothing (seconds)
    
    Returns:
        firing_rate (np.ndarray): Firing rate in Hz
    """
    # Convert window size to samples
    window_samples = int(window_size / dt)
    
    # Convolve with rectangular window
    window = np.ones(window_samples) / (window_samples * dt)
    firing_rate = np.convolve(spike_train, window, mode='same')
    
    return firing_rate


def poisson_spike_train(rate, duration, dt):
    """
    Generate a Poisson spike train with given rate.
    
    Args:
        rate (float): Firing rate (Hz)
        duration (float): Duration (seconds)
        dt (float): Time step (seconds)
    
    Returns:
        spike_train (np.ndarray): Binary spike train
    """
    n_steps = int(duration / dt)
    spike_probability = rate * dt
    spike_train = (np.random.random(n_steps) < spike_probability).astype(int)
    
    return spike_train
