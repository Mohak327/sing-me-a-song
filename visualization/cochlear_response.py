"""
Plots of filterbank output and cochlear response.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_cochleogram(filtered_signals, duration, center_freqs=None, 
                     cmap='plasma', figsize=(12, 6), title='Cochleogram'):
    """
    Plot cochleogram (time-frequency representation from filterbank).
    
    Args:
        filtered_signals (np.ndarray): Shape (num_channels, time_steps)
        duration (float): Duration in seconds
        center_freqs (np.ndarray): Center frequencies for each channel
        cmap (str): Colormap
        figsize (tuple): Figure size
        title (str): Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_channels = filtered_signals.shape[0]
    extent = [0, duration, 0, num_channels]
    
    im = ax.imshow(
        np.abs(filtered_signals),
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        interpolation='bilinear'
    )
    
    fig.colorbar(im, ax=ax, label='Amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel Number')
    ax.set_title(title)
    
    # Add frequency labels if provided
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


def plot_channel_responses(filtered_signals, fs, channels_to_plot, 
                           center_freqs=None, time_window=None, figsize=(12, 8)):
    """
    Plot individual channel responses.
    
    Args:
        filtered_signals (np.ndarray): Shape (num_channels, time_steps)
        fs (int): Sampling rate
        channels_to_plot (list): List of channel indices to plot
        center_freqs (np.ndarray): Center frequencies
        time_window (tuple): (start_time, end_time) in seconds, or None for all
        figsize (tuple): Figure size
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    n_plots = len(channels_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Determine time window
    if time_window:
        start_idx = int(time_window[0] * fs)
        end_idx = int(time_window[1] * fs)
        time_slice = slice(start_idx, end_idx)
    else:
        time_slice = slice(None)
    
    time_axis = np.arange(filtered_signals.shape[1])[time_slice] / fs
    
    for idx, ch in enumerate(channels_to_plot):
        axes[idx].plot(time_axis, filtered_signals[ch, time_slice], linewidth=0.5)
        
        if center_freqs is not None:
            axes[idx].set_ylabel(f'Ch {ch}\n{center_freqs[ch]:.0f} Hz')
        else:
            axes[idx].set_ylabel(f'Channel {ch}')
        
        axes[idx].grid(True, alpha=0.3)
        
        if idx < n_plots - 1:
            axes[idx].set_xticklabels([])
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Filtered Channel Responses')
    plt.tight_layout()
    
    return fig, axes


def plot_envelopes(envelopes, duration, center_freqs=None,
                   cmap='plasma', figsize=(12, 6)):
    """
    Plot envelope extraction (similar to cochleogram but for envelopes).
    
    Args:
        envelopes (np.ndarray): Shape (num_channels, time_steps)
        duration (float): Duration in seconds
        center_freqs (np.ndarray): Center frequencies
        cmap (str): Colormap
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    return plot_cochleogram(
        envelopes, 
        duration, 
        center_freqs, 
        cmap, 
        figsize, 
        title='Envelope Extraction (Basilar Membrane Motion)'
    )


def plot_single_channel_with_envelope(filtered_signal, envelope, fs, 
                                     channel_idx, center_freq=None,
                                     time_window=(0, 0.1), figsize=(12, 4)):
    """
    Plot filtered signal and its envelope for a single channel.
    
    Args:
        filtered_signal (np.ndarray): Single channel filtered signal
        envelope (np.ndarray): Envelope of the signal
        fs (int): Sampling rate
        channel_idx (int): Channel number
        center_freq (float): Center frequency (Hz)
        time_window (tuple): (start, end) in seconds
        figsize (tuple): Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    start_idx = int(time_window[0] * fs)
    end_idx = int(time_window[1] * fs)
    time_slice = slice(start_idx, end_idx)
    
    t = np.arange(len(filtered_signal[time_slice])) / fs + time_window[0]
    
    ax.plot(t, filtered_signal[time_slice], alpha=0.6, linewidth=0.5, label='Filtered signal')
    ax.plot(t, envelope[time_slice], color='red', linewidth=2, label='Envelope')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    if center_freq:
        title = f'Channel {channel_idx} ({center_freq:.0f} Hz): Signal and Envelope'
    else:
        title = f'Channel {channel_idx}: Signal and Envelope'
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax
