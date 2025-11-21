"""
Hodgkin-Huxley neuron implementation.
"""

import numpy as np


class HHNeuron:
    """
    Hodgkin-Huxley neuron model (simplified).
    More biophysically realistic than LIF but computationally expensive.
    """
    
    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.4):
        """
        Initialize Hodgkin-Huxley neuron parameters.
        
        Args:
            C_m (float): Membrane capacitance (μF/cm²)
            g_Na (float): Sodium conductance (mS/cm²)
            g_K (float): Potassium conductance (mS/cm²)
            g_L (float): Leak conductance (mS/cm²)
            E_Na (float): Sodium reversal potential (mV)
            E_K (float): Potassium reversal potential (mV)
            E_L (float): Leak reversal potential (mV)
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        
        # State variables
        self.V = -65.0  # Membrane potential
        self.m = 0.05   # Sodium activation
        self.h = 0.6    # Sodium inactivation
        self.n = 0.32   # Potassium activation
        
        self.spike_times = []
        self.last_v = self.V
    
    def alpha_m(self, V):
        """Sodium activation rate."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Sodium activation rate."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Sodium inactivation rate."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Sodium inactivation rate."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """Potassium activation rate."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """Potassium activation rate."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def step(self, I_ext, dt, current_time):
        """
        Simulate one time step using Euler method.
        
        Args:
            I_ext (float): External current (μA/cm²)
            dt (float): Time step (ms)
            current_time (float): Current time (ms)
        
        Returns:
            bool: True if spike detected
        """
        V = self.V
        m = self.m
        h = self.h
        n = self.n
        
        # Calculate currents
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        # Update voltage
        dV = (I_ext - I_Na - I_K - I_L) / self.C_m
        self.V += dV * dt
        
        # Update gating variables
        self.m += (self.alpha_m(V) * (1 - m) - self.beta_m(V) * m) * dt
        self.h += (self.alpha_h(V) * (1 - h) - self.beta_h(V) * h) * dt
        self.n += (self.alpha_n(V) * (1 - n) - self.beta_n(V) * n) * dt
        
        # Detect spike (crossing threshold upward)
        spiked = False
        if self.last_v < 0 and self.V >= 0:
            self.spike_times.append(current_time)
            spiked = True
        
        self.last_v = self.V
        
        return spiked


def simulate_hh_neuron(input_current, dt):
    """
    Simulate Hodgkin-Huxley neuron.
    
    Args:
        input_current (np.ndarray): Input current over time (μA/cm²)
        dt (float): Time step (ms)
    
    Returns:
        spike_train (np.ndarray): Binary spike train
        spike_times (np.ndarray): Spike times
        voltage_trace (np.ndarray): Membrane voltage
    """
    neuron = HHNeuron()
    
    n_steps = len(input_current)
    spike_train = np.zeros(n_steps, dtype=int)
    voltage_trace = np.zeros(n_steps)
    
    for i in range(n_steps):
        current_time = i * dt
        spiked = neuron.step(input_current[i], dt, current_time)
        spike_train[i] = int(spiked)
        voltage_trace[i] = neuron.V
    
    spike_times = np.array(neuron.spike_times)
    
    return spike_train, spike_times, voltage_trace
