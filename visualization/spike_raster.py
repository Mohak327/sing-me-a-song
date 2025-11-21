"""
Raster plots of neuronal output.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_raster(spike_trains, dt, max_neurons=None, figsize=(12, 6),
                title='Spike Raster Plot'):
    """
    Plot spike raster for all neurons.
    
    Args:
        spike_trains (np.ndarray): Binary spike trains, shape (num_neurons, time_steps)
        dt (float): Time step (seconds)
        max_neurons (int): Maximum number of neurons to plot (None = all)
        figsize (tuple): Figure size
        title (str): Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_neurons, n_steps = spike_trains.shape
    duration = n_steps * dt
    
    if max_neurons and num_neurons > max_neurons:
        # Sample neurons to plot
        neuron_indices = np.linspace(0, num_neurons - 1, max_neurons, dtype=int)
        plot_spike_trains = spike_trains[neuron_indices, :]
    else:
        neuron_indices = np.arange(num_neurons)
        plot_spike_trains = spike_trains
    
    # Plot spikes
    for i, neuron_idx in enumerate(neuron_indices):
        spike_times = np.where(plot_spike_trains[i, :] == 1)[0] * dt
        ax.plot(spike_times, np.ones_like(spike_times) * neuron_idx, 
               'k|', markersize=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title(title)
    ax.set_xlim([0, duration])
    ax.set_ylim([-1, num_neurons])
    plt.tight_layout()
    
    return fig, ax


def plot_raster_by_channel(spike_trains, neurons_per_channel, dt,
                           channels_to_plot=None, figsize=(12, 8)):
    """
    Plot spike raster organized by frequency channel.
    
    Args:
        spike_trains (np.ndarray): Shape (total_neurons, time_steps)
        neurons_per_channel (int): Number of neurons per channel
        dt (float): Time step (seconds)
        channels_to_plot (list): List of channel indices to plot (None = all)
        figsize (tuple): Figure size
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    total_neurons, n_steps = spike_trains.shape
    num_channels = total_neurons // neurons_per_channel
    duration = n_steps * dt
    
    if channels_to_plot is None:
        # Plot subset of channels
        channels_to_plot = np.linspace(0, num_channels - 1, 
                                      min(8, num_channels), dtype=int)
    
    n_plots = len(channels_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, ch in enumerate(channels_to_plot):
        # Get neurons for this channel
        start_neuron = ch * neurons_per_channel
        end_neuron = start_neuron + neurons_per_channel
        channel_spikes = spike_trains[start_neuron:end_neuron, :]
        
        # Plot spikes for this channel
        for neuron_idx in range(neurons_per_channel):
            spike_times = np.where(channel_spikes[neuron_idx, :] == 1)[0] * dt
            axes[idx].plot(spike_times, np.ones_like(spike_times) * neuron_idx,
                          'k|', markersize=4)
        
        axes[idx].set_ylabel(f'Ch {ch}')
        axes[idx].set_ylim([-0.5, neurons_per_channel - 0.5])
        axes[idx].set_xlim([0, duration])
        
        if idx < n_plots - 1:
            axes[idx].set_xticklabels([])
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Spike Raster by Channel')
    plt.tight_layout()
    
    return fig, axes


def plot_spike_histogram(spike_trains, dt, bin_size=0.010, figsize=(12, 4)):
    """
    Plot histogram of spike counts over time (PSTH - Peri-Stimulus Time Histogram).
    
    Args:
        spike_trains (np.ndarray): Shape (num_neurons, time_steps)
        dt (float): Time step (seconds)
        bin_size (float): Bin size for histogram (seconds)
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sum spikes across all neurons
    total_spikes = np.sum(spike_trains, axis=0)
    
    # Bin the spikes
    n_steps = len(total_spikes)
    duration = n_steps * dt
    n_bins = int(duration / bin_size)
    
    time_bins = np.linspace(0, duration, n_bins + 1)
    spike_counts, _ = np.histogram(
        np.repeat(np.arange(n_steps) * dt, total_spikes.astype(int)),
        bins=time_bins
    )
    
    # Convert to rate (spikes/sec)
    spike_rate = spike_counts / (bin_size * spike_trains.shape[0])
    
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    ax.bar(bin_centers, spike_rate, width=bin_size, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population Firing Rate (Hz)')
    ax.set_title('Population Spike Histogram (PSTH)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig, ax
