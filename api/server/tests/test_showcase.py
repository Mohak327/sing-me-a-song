"""
Tests for the data the page draws.

Run:  python -m pytest server -q
"""
import json

import numpy as np

from server.showcase import build_showcase


def _clip(fs=16000, dur=0.2):
    t = np.arange(int(fs * dur)) / fs
    return 0.8 * np.sin(2 * np.pi * 440 * t) * (t > 0.05), fs


def test_showcase_has_one_frame_per_ten_milliseconds():
    x, fs = _clip()
    data = build_showcase(x, fs, num_channels=8, fibers_shown_per_channel=2)
    assert data['duration'] == 0.2
    assert len(data['waveform']) == 20
    assert len(data['cochleagram']) == 8
    assert all(len(row) == 20 for row in data['cochleagram'])


def test_cochleagram_is_normalised_and_loudest_where_the_tone_is():
    x, fs = _clip()
    data = build_showcase(x, fs, num_channels=8, fibers_shown_per_channel=2)
    levels = np.array(data['cochleagram'])
    assert levels.min() >= 0.0 and levels.max() == 1.0
    loudest = int(np.argmax(levels.max(axis=1)))
    freqs = data['center_freqs']
    assert freqs[max(loudest - 1, 0)] <= 440 <= freqs[min(loudest + 1, 7)]
    assert levels[:, :4].max() < levels[:, 10:].max(), "silence before the tone must be quieter"


def test_raster_lists_spike_times_inside_the_clip_for_each_shown_fiber():
    x, fs = _clip()
    data = build_showcase(x, fs, num_channels=8, fibers_shown_per_channel=2)
    raster = data['raster']
    assert len(raster) == 16
    for fiber in raster:
        times = fiber['times']
        assert 0 <= fiber['channel'] < 8
        assert len(times) > 0, "every fiber fires spontaneously"
        assert times == sorted(times)
        assert 0 <= times[0] and times[-1] < 0.2


def test_showcase_is_json_serialisable():
    x, fs = _clip()
    json.dumps(build_showcase(x, fs, num_channels=8, fibers_shown_per_channel=2))
