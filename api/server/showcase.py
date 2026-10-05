"""
Everything the page draws for one clip: waveform, cochleagram, a sample of
nerve fibers' spike times and the hair cell curve, computed by the real model.

A long clip is heard in blocks; block_features() describes one block and
assemble() joins the blocks into what the page receives.
"""
import numpy as np

from auditory_periphery import hear
from cochlea.gammatone_frame import analyze
from haircell.inner_hair_cell import receptor_nonlinearity

FRAME = 0.01       # seconds per drawn column
FLOOR_DB = -60.0   # cochleagram level drawn as empty


def block_features(x, fs, spike_neuron, spike_time, ear, fibers_shown_per_channel):
    """
    Describe one block of audio and the spikes hear() produced for it.

    Args:
        x (np.ndarray): The block's audio, as passed to hear()
        fs (int): Sampling rate (Hz)
        spike_neuron, spike_time (np.ndarray): From hear()
        ear (dict): From hear()
        fibers_shown_per_channel (int): Fibers per channel kept for the raster

    Returns:
        features (dict): 'seconds' drawn, 'waveform' (frames, 2), 'level_db'
            (channels, frames), 'raster' ([(channel, times)], times from block start)
    """
    x = np.asarray(x, dtype=np.float64)
    hop = int(round(FRAME * fs))
    frames = len(x) // hop
    seconds = frames * hop / fs
    columns = x[:frames * hop].reshape(frames, hop)
    waveform = np.stack([columns.min(axis=1), columns.max(axis=1)], axis=1) if frames else np.zeros((0, 2))

    num_channels = ear['H'].shape[0]
    padded = np.r_[x, np.zeros(ear['padded_samples'] - len(x))]
    bands = analyze(padded, ear['H'])[:, :frames * hop].reshape(num_channels, frames, hop)
    level_db = 10 * np.log10(np.mean(bands ** 2, axis=2) + 1e-12)

    per_channel = len(ear['population']['channel']) // num_channels
    shown = min(fibers_shown_per_channel, per_channel)
    raster = []
    for channel in range(num_channels):
        for fiber in range(channel * per_channel, channel * per_channel + shown):
            times = np.sort(spike_time[spike_neuron == fiber])
            raster.append((channel, times[(times >= 0) & (times < seconds)]))
    return {'seconds': seconds, 'waveform': waveform, 'level_db': level_db, 'raster': raster}


def assemble(blocks, starts, center_freqs):
    """
    Join block features into the showcase the page receives.

    Args:
        blocks (list): block_features() results, in order
        starts (list): Start time of each block in the clip (seconds)
        center_freqs (np.ndarray): Channel center frequencies (Hz)

    Returns:
        showcase (dict): JSON-serialisable; 'duration', 'frame', 'center_freqs',
            'waveform' ([low, high] per frame), 'cochleagram' (channels x frames,
            0-1), 'raster' ([{channel, times}]), 'hair_cell' ({input, output})
    """
    waveform = np.concatenate([block['waveform'] for block in blocks], axis=0)
    level_db = np.concatenate([block['level_db'] for block in blocks], axis=1)
    level = np.clip((level_db - level_db.max() - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0)
    raster = []
    for index, (channel, _) in enumerate(blocks[0]['raster']):
        times = np.concatenate([block['raster'][index][1] + start
                                for block, start in zip(blocks, starts)])
        raster.append({'channel': int(channel), 'times': np.round(times, 5).tolist()})
    deflection = np.linspace(-1.0, 1.0, 201)
    return {
        'duration': float(sum(block['seconds'] for block in blocks)),
        'frame': FRAME,
        'center_freqs': np.round(center_freqs, 1).tolist(),
        'waveform': np.round(waveform, 3).tolist(),
        'cochleagram': np.round(level, 2).tolist(),
        'raster': raster,
        'hair_cell': {'input': np.round(deflection, 3).tolist(),
                      'output': np.round(receptor_nonlinearity(deflection), 5).tolist()},
    }


def build_showcase(x, fs, num_channels=32, fibers_shown_per_channel=4):
    """
    Compute what the page draws for one short clip, heard in a single block by
    a small sample of fibers drawn the same way as the full nerve's.

    Args:
        x (np.ndarray): Mono audio within [-1, 1]
        fs (int): Sampling rate (Hz)
        num_channels (int): Cochlear channels
        fibers_shown_per_channel (int): Fibers per channel in the raster

    Returns:
        showcase (dict): See assemble()
    """
    spike_neuron, spike_time, ear = hear(x, fs, num_channels=num_channels,
                                         fibers_per_channel=fibers_shown_per_channel)
    block = block_features(x, fs, spike_neuron, spike_time, ear, fibers_shown_per_channel)
    return assemble([block], [0.0], ear['center_freqs'])
