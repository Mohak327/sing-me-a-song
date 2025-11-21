"""
Neuron population management and simulation.
"""

import numpy as np
from .lif_neuron import simulate_lif_neuron, calculate_firing_rate


class NeuronPopulation:
    """
    Manages a population of neurons across multiple frequency channels.
    """
    
    def __init__(self, num_channels, neurons_per_channel, neuron_params=None):
        """
        Initialize neuron population.
        
        Args:
            num_channels (int): Number of frequency channels
            neurons_per_channel (int): Number of neurons per channel
            neuron_params (dict): Dictionary of neuron parameters
        """
        self.num_channels = num_channels
        self.neurons_per_channel = neurons_per_channel
        self.total_neurons = num_channels * neurons_per_channel
        
        # Default parameters
        self.params = {
            'tau_m': 0.010,
            'v_threshold': -50,
            'v_reset': -70,
            'v_rest': -65,
            'refractory_period': 0.002,
            'spontaneous_rate': 10.0,
            'input_scale': 100.0
        }
        
        if neuron_params:
            self.params.update(neuron_params)
        
        # Storage for spike data
        self.spike_trains = None
        self.firing_rates = None
    
    def simulate(self, receptor_potentials, dt):
        """
        Simulate the entire neuron population.
        
        Args:
            receptor_potentials (np.ndarray): Shape (num_channels, time_steps)
            dt (float): Time step (seconds)
        
        Returns:
            spike_trains (np.ndarray): Shape (total_neurons, time_steps)
            firing_rates (np.ndarray): Shape (num_channels, time_steps)
        """
        num_channels, n_steps = receptor_potentials.shape
        
        # Initialize storage
        all_spike_trains = []
        channel_firing_rates = np.zeros((num_channels, n_steps))
        
        print(f"Simulating {self.total_neurons} neurons...")
        
        for ch in range(num_channels):
            # Get input for this channel
            channel_input = receptor_potentials[ch, :] * self.params['input_scale']
            
            # Simulate multiple neurons for this channel
            channel_spikes = []
            for neuron_idx in range(self.neurons_per_channel):
                # Add variability: different spontaneous rates
                spont_rate = self.params['spontaneous_rate'] * (0.5 + np.random.random())
                
                spike_train, _, _ = simulate_lif_neuron(
                    channel_input,
                    dt,
                    tau_m=self.params['tau_m'],
                    v_threshold=self.params['v_threshold'],
                    v_reset=self.params['v_reset'],
                    v_rest=self.params['v_rest'],
                    refractory_period=self.params['refractory_period'],
                    spontaneous_rate=spont_rate
                )
                
                channel_spikes.append(spike_train)
                all_spike_trains.append(spike_train)
            
            # Calculate average firing rate for this channel
            channel_spikes_array = np.array(channel_spikes)
            summed_spikes = np.sum(channel_spikes_array, axis=0)
            channel_firing_rates[ch, :] = calculate_firing_rate(summed_spikes, dt)
            
            if (ch + 1) % 8 == 0:
                print(f"  Completed {ch + 1}/{num_channels} channels")
        
        self.spike_trains = np.array(all_spike_trains)
        self.firing_rates = channel_firing_rates
        
        print(f"✓ Simulation complete: {self.spike_trains.shape}")
        
        return self.spike_trains, self.firing_rates
    
    def get_channel_spikes(self, channel_idx):
        """
        Get spike trains for a specific channel.
        
        Args:
            channel_idx (int): Channel index
        
        Returns:
            spikes (np.ndarray): Shape (neurons_per_channel, time_steps)
        """
        start_idx = channel_idx * self.neurons_per_channel
        end_idx = start_idx + self.neurons_per_channel
        return self.spike_trains[start_idx:end_idx, :]
    
    def get_spike_times_by_neuron(self, dt):
        """
        Convert spike trains to spike times for each neuron.
        
        Args:
            dt (float): Time step
        
        Returns:
            spike_times_list (list): List of arrays, one per neuron
        """
        spike_times_list = []
        
        for neuron_idx in range(self.total_neurons):
            spike_indices = np.where(self.spike_trains[neuron_idx, :] == 1)[0]
            spike_times = spike_indices * dt
            spike_times_list.append(spike_times)
        
        return spike_times_list


def simulate_population(receptor_potentials, dt, num_channels, 
                       neurons_per_channel=10, neuron_params=None):
    """
    Convenience function to simulate a neuron population.
    
    Args:
        receptor_potentials (np.ndarray): Shape (num_channels, time_steps)
        dt (float): Time step (seconds)
        num_channels (int): Number of frequency channels
        neurons_per_channel (int): Neurons per channel
        neuron_params (dict): Optional neuron parameters
    
    Returns:
        spike_trains (np.ndarray): Binary spike trains
        firing_rates (np.ndarray): Firing rates per channel
        population (NeuronPopulation): Population object
    """
    population = NeuronPopulation(num_channels, neurons_per_channel, neuron_params)
    spike_trains, firing_rates = population.simulate(receptor_potentials, dt)
    
    return spike_trains, firing_rates, population
