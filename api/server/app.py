"""
The Resound API.

    GET  /api/limits   how the page should cut and send a clip
    POST /api/blocks   hear one block of audio and regenerate it from the spikes

The server keeps nothing between requests: the page cuts a clip into blocks,
sends each one here, and puts the answers back together. That is what lets it
run on serverless hosting, where no request may take long.

Run from the repository root:

    python -m uvicorn server.app:app --app-dir api --port 8010 --workers 3
"""
import base64
import os
import warnings

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query

import server  # noqa: F401  (puts the model on the import path)
from auditory_periphery import hear, regenerate
from server.showcase import block_features

FS = 16000
ALLOWED_FIBERS = [1024, 512, 256, 64, 16]
ALLOWED_JITTER = [0.0, 1e-9, 1e-7, 1e-5]


def hear_block(x, fibers, jitter, num_channels, shown):
    """
    Hear one block and regenerate it from its spikes.

    Args:
        x (np.ndarray): float64 samples within [-1, 1]
        fibers (int): Nerve fibers per cochlear channel
        jitter (float): Std-dev of noise added to spike times (seconds)
        num_channels (int): Cochlear channels
        shown (int): Fibers per channel returned for the raster

    Returns:
        block (dict): JSON-serialisable; the regenerated audio (base64 float32),
            the energies of the signal and of the error, and what the page draws
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # reported through the numbers instead
        spike_neuron, spike_time, ear = hear(x, FS, num_channels=num_channels,
                                             fibers_per_channel=fibers, jitter=jitter)
        audio, info = regenerate(spike_neuron, spike_time, ear)
    features = block_features(x, FS, spike_neuron, spike_time, ear, shown)
    return {
        'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode('ascii'),
        'signal': float(np.sum(x ** 2)),
        'noise': float(np.sum((x - audio) ** 2)),
        'spikes': int(len(spike_time)),
        'unobserved_samples': int(info['unobserved_samples']),
        'seconds': features['seconds'],
        'waveform': np.round(features['waveform'], 3).tolist(),
        'level_db': np.round(features['level_db'], 1).tolist(),
        'raster': [{'channel': int(channel), 'times': np.round(times, 5).tolist()}
                   for channel, times in features['raster']],
        'center_freqs': np.round(ear['center_freqs'], 1).tolist(),
    }


def create_app(num_channels=32, block_seconds=1.0, hosted=False):
    """
    Build the server.

    Args:
        num_channels (int): Cochlear channels
        block_seconds (float): Longest block one request may carry
        hosted (bool): True on serverless hosting, where clips are kept short
            and fewer blocks are sent at once

    Returns:
        app (FastAPI)
    """
    app = FastAPI(title='Resound')
    block_samples = int(round(block_seconds * FS))

    @app.get('/api/limits')
    def limits():
        return {
            'sample_rate': FS,
            'block_seconds': block_seconds,
            'max_seconds': 10 if hosted else 60,
            # Hosted, two blocks at once share one processor and each takes twice as
            # long (191 s measured, against a 300 s limit), so they go one at a time.
            'concurrency': 1 if hosted else 3,
            'fibers': ALLOWED_FIBERS,
            'jitter': ALLOWED_JITTER,
        }

    @app.post('/api/blocks')
    def blocks(body: bytes = Body(..., media_type='application/octet-stream'),
               fibers: int = Query(...), jitter: float = Query(...),
               shown: int = Query(4, ge=1, le=8)):
        if fibers not in ALLOWED_FIBERS:
            raise HTTPException(status_code=422, detail=f'fibers must be one of {ALLOWED_FIBERS}.')
        if not any(np.isclose(jitter, allowed, rtol=1e-6, atol=0) or jitter == allowed
                   for allowed in ALLOWED_JITTER):
            raise HTTPException(status_code=422, detail=f'jitter must be one of {ALLOWED_JITTER}.')
        if len(body) == 0 or len(body) % 4:
            raise HTTPException(status_code=400, detail='The body must be 32-bit float samples.')
        if len(body) > block_samples * 4:
            raise HTTPException(status_code=413,
                                detail=f'One block may hold at most {block_seconds} seconds of audio.')
        x = np.frombuffer(body, dtype='<f4').astype(np.float64)
        if not np.all(np.isfinite(x)):
            raise HTTPException(status_code=400, detail='Every sample must be a finite number.')
        return hear_block(np.clip(x, -1.0, 1.0), fibers, jitter, num_channels, shown)

    return app


app = create_app(hosted=bool(os.environ.get('VERCEL')))
