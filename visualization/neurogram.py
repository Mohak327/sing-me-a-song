"""
Heatmap of firing rates (neurogram).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_neurogram(firing_rates, duration, center_freqs=None,
                   cmap='hot', figsize=(12, 6), title='Neurogram'):
    """
    Plot neurogram (heatmap of firing rates over time and frequency).
    
    Args:
        firing_rates (np.ndarray): Shape (num_channels, time_steps)
        duration (float): Duration in seconds
        center_freqs (np.ndarray): Center frequencies for channels
        cmap (str): Colormap
        figsize (tuple): Figure size
        title (str): Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_channels = firing_rates.shape[0]
    extent = [0, duration, 0, num_channels]
    
    im = ax.imshow(
        firing_rates,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        interpolation='bilinear'
    )
    
    cbar = fig.colorbar(im, ax=ax, label='Firing Rate (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel Number')
    ax.set_title(title)
    
    # Add frequency labels
    if center_freqs is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(0, num_channels)
        freq_ticks = np.arange(0, num_channels, max(1, num_channels // 8))
        freq_labels = [f"{center_freqs[int(i)]:.0f}" for i in freq_ticks if i < len(center_freqs)]
        ax2.set_yticks(freq_ticks[:len(freq_labels)])
        ax2.set_yticklabels(freq_labels)
        ax2.set_ylabel('Center Frequency (Hz)')
    
    plt.tight_layout()
    
    return fig, ax


def plot_firing_rates(firing_rates, dt, channels_to_plot=None,
                     center_freqs=None, figsize=(12, 8)):
    """
    Plot firing rates for individual channels over time.
    
    Args:
        firing_rates (np.ndarray): Shape (num_channels, time_steps)
        dt (float): Time step (seconds)
        channels_to_plot (list): Channel indices to plot
        center_freqs (np.ndarray): Center frequencies
        figsize (tuple): Figure size
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    num_channels, n_steps = firing_rates.shape
    
    if channels_to_plot is None:
        # Select subset of channels
        channels_to_plot = np.linspace(0, num_channels - 1, 
                                      min(6, num_channels), dtype=int)
    
    n_plots = len(channels_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    time_axis = np.arange(n_steps) * dt
    
    for idx, ch in enumerate(channels_to_plot):
        axes[idx].plot(time_axis, firing_rates[ch, :], linewidth=1)
        axes[idx].fill_between(time_axis, firing_rates[ch, :], alpha=0.3)
        
        if center_freqs is not None:
            axes[idx].set_ylabel(f'Ch {ch}\n{center_freqs[ch]:.0f} Hz\n(Hz)')
        else:
            axes[idx].set_ylabel(f'Ch {ch}\n(Hz)')
        
        axes[idx].grid(True, alpha=0.3)
        
        if idx < n_plots - 1:
            axes[idx].set_xticklabels([])
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Firing Rates by Channel')
    plt.tight_layout()
    
    return fig, axes


def plot_average_firing_rate(firing_rates, center_freqs=None, figsize=(10, 5)):
    """
    Plot average firing rate across channels (frequency tuning).
    
    Args:
        firing_rates (np.ndarray): Shape (num_channels, time_steps)
        center_freqs (np.ndarray): Center frequencies
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Average over time
    avg_rates = np.mean(firing_rates, axis=1)
    std_rates = np.std(firing_rates, axis=1)
    
    channel_indices = np.arange(len(avg_rates))
    
    ax.plot(channel_indices, avg_rates, 'o-', linewidth=2, markersize=6)
    ax.fill_between(channel_indices, 
                    avg_rates - std_rates,
                    avg_rates + std_rates,
                    alpha=0.3)
    
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Average Firing Rate (Hz)')
    ax.set_title('Average Firing Rate Across Channels')
    ax.grid(True, alpha=0.3)
    
    # Add frequency labels if available
    if center_freqs is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        freq_ticks = np.linspace(0, len(center_freqs) - 1, 8, dtype=int)
        freq_labels = [f"{center_freqs[i]:.0f}" for i in freq_ticks]
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels(freq_labels)
        ax2.set_xlabel('Center Frequency (Hz)')
    
    plt.tight_layout()
    
    return fig, ax


def compare_envelopes_and_firing_rates(envelopes, firing_rates, dt,
                                       channels_to_plot=[0, 8, 16, 24],
                                       figsize=(12, 10)):
    """
    Compare input envelopes with output firing rates.
    
    Args:
        envelopes (np.ndarray): Input envelopes, shape (num_channels, time_steps)
        firing_rates (np.ndarray): Output firing rates, shape (num_channels, time_steps)
        dt (float): Time step (seconds)
        channels_to_plot (list): Channel indices to compare
        figsize (tuple): Figure size
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    n_plots = len(channels_to_plot)
    fig, axes = plt.subplots(n_plots, 2, figsize=figsize, sharex='col')
    
    duration = envelopes.shape[1] * dt
    time_axis = np.arange(envelopes.shape[1]) * dt
    
    for idx, ch in enumerate(channels_to_plot):
        # Plot envelope
        axes[idx, 0].plot(time_axis, envelopes[ch, :], linewidth=1, color='blue')
        axes[idx, 0].set_ylabel(f'Ch {ch}')
        axes[idx, 0].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 0].set_title('Input Envelope')
        if idx < n_plots - 1:
            axes[idx, 0].set_xticklabels([])
        
        # Plot firing rate
        axes[idx, 1].plot(time_axis, firing_rates[ch, :], linewidth=1, color='red')
        axes[idx, 1].set_ylabel(f'Ch {ch}')
        axes[idx, 1].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 1].set_title('Output Firing Rate')
        if idx < n_plots - 1:
            axes[idx, 1].set_xticklabels([])
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Envelope vs Firing Rate Comparison')
    plt.tight_layout()
    
    return fig, axes
