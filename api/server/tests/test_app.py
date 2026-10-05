"""
Tests for the Resound API.

Run:  python -m pytest server -q
"""
import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app

FS = 16000


def _tone(seconds, freq=440.0, level=0.5):
    t = np.arange(int(round(FS * seconds))) / FS
    return (level * np.sin(2 * np.pi * freq * t)).astype('<f4')


@pytest.fixture
def client():
    return TestClient(create_app(num_channels=8, block_seconds=0.1))


def _hear(client, samples, fibers=1024, jitter=0, **more):
    params = {'fibers': fibers, 'jitter': jitter, **more}
    return client.post('/api/blocks', params=params, content=samples.tobytes(),
                       headers={'content-type': 'application/octet-stream'})


def test_a_block_comes_back_regenerated_with_its_error_measured(client):
    x = _tone(0.1)
    response = _hear(client, x)
    assert response.status_code == 200, response.text
    body = response.json()
    audio = np.frombuffer(base64.b64decode(body['audio']), dtype='<f4')
    assert len(audio) == len(x)
    assert np.max(np.abs(audio - x)) < 1e-6
    assert body['signal'] > 0
    assert 10 * np.log10(body['signal'] / max(body['noise'], 1e-300)) > 100
    assert body['spikes'] > 0 and body['unobserved_samples'] == 0


def test_a_block_carries_what_the_page_draws_for_it(client):
    body = _hear(client, _tone(0.1)).json()
    assert body['seconds'] == pytest.approx(0.1)
    assert len(body['waveform']) == 10
    assert len(body['level_db']) == 8 and len(body['level_db'][0]) == 10
    assert len(body['center_freqs']) == 8
    assert len(body['raster']) == 8 * 4
    assert all(0 <= t < 0.1 for fiber in body['raster'] for t in fiber['times'])


def test_fewer_raster_rows_can_be_asked_for(client):
    body = _hear(client, _tone(0.1), shown=1).json()
    assert len(body['raster']) == 8


def test_fewer_fibers_and_jitter_cost_fidelity_without_breaking(client):
    x = _tone(0.1)
    for kwargs in ({'fibers': 64}, {'jitter': 1e-5}):
        body = _hear(client, x, **kwargs).json()
        audio = np.frombuffer(base64.b64decode(body['audio']), dtype='<f4')
        assert 10 * np.log10(body['signal'] / body['noise']) < 80, kwargs
        assert np.max(np.abs(audio)) <= 1.0


def test_silence_is_regenerated_as_silence(client):
    body = _hear(client, np.zeros(1600, dtype='<f4')).json()
    assert body['signal'] == 0 and body['noise'] < 1e-18


def test_samples_slightly_over_full_scale_are_clipped_not_refused(client):
    x = _tone(0.1, level=1.02)
    assert _hear(client, x).status_code == 200


def test_settings_outside_the_offered_choices_are_refused(client):
    x = _tone(0.1)
    assert _hear(client, x, fibers=2000).status_code == 422
    assert _hear(client, x, jitter=0.5).status_code == 422
    assert _hear(client, x, shown=0).status_code == 422


def test_a_block_longer_than_the_block_length_is_refused(client):
    response = _hear(client, _tone(0.2))
    assert response.status_code == 413
    assert 'block' in response.json()['detail']


def test_a_body_that_is_not_samples_is_refused_with_a_reason(client):
    headers = {'content-type': 'application/octet-stream'}
    params = {'fibers': 16, 'jitter': 0}
    assert client.post('/api/blocks', params=params, content=b'', headers=headers).status_code in (400, 422)
    assert client.post('/api/blocks', params=params, content=b'abc', headers=headers).status_code == 400
    bad = _tone(0.1).copy()
    bad[5] = np.nan
    response = _hear(client, bad, fibers=16)
    assert response.status_code == 400 and 'finite' in response.json()['detail']


def test_limits_tell_the_page_how_to_cut_and_send_a_clip():
    local = TestClient(create_app()).get('/api/limits').json()
    hosted = TestClient(create_app(hosted=True)).get('/api/limits').json()
    assert local['sample_rate'] == 16000 and local['block_seconds'] == 1.0
    assert local['max_seconds'] == 60 and hosted['max_seconds'] == 10
    assert hosted['concurrency'] <= local['concurrency']
    assert local['fibers'] == [1024, 512, 256, 64, 16]


def test_the_copied_model_matches_the_repository():
    """api/vendor is the copy the hosted API runs; tools/sync_model.py refreshes it."""
    api = Path(__file__).resolve().parents[2]
    copies = [copy for copy in (api / 'vendor').rglob('*.py') if copy.name != '__init__.py']
    assert len(copies) == 5
    for copy in copies:
        original = api.parent / copy.relative_to(api / 'vendor')
        assert copy.read_bytes() == original.read_bytes(), f'{copy.name} is out of date; run tools/sync_model.py'
