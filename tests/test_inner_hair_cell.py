"""
Tests for the invertible inner hair cell stage.

Run:  python -m pytest tests/test_inner_hair_cell.py -v
"""
import os
import sys
# Pin the project root first so top-level packages resolve here, not under tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from haircell.inner_hair_cell import (receptor_nonlinearity, inverse_receptor_nonlinearity,
                                      adapt, inverse_adapt, transduce, inverse_transduce)


def _snr_db(reference, estimate):
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum((reference - estimate) ** 2) + 1e-300))


def test_nonlinearity_is_strictly_increasing_and_zero_at_rest():
    b = np.linspace(-1.0, 1.0, 20001)
    v = receptor_nonlinearity(b)
    assert np.all(np.diff(v) > 0)
    assert receptor_nonlinearity(np.zeros(1))[0] == 0.0


def test_nonlinearity_is_compressive():
    quiet, loud = receptor_nonlinearity(np.array([0.01, 0.5]))
    assert loud / 0.5 < 0.5 * (quiet / 0.01), "loud input should get less gain than quiet input"


def test_nonlinearity_is_asymmetric_like_a_rectifier():
    up, down = receptor_nonlinearity(np.array([0.05, -0.05]))
    assert up > 0 > down
    assert abs(down) < 0.6 * up, "negative deflection should produce a smaller response"


def test_nonlinearity_inverse_round_trip():
    b = np.linspace(-1.0, 1.0, 20001)
    assert np.max(np.abs(inverse_receptor_nonlinearity(receptor_nonlinearity(b)) - b)) < 1e-10


def test_adaptation_reduces_sustained_response():
    fs = 16000
    step = np.ones((1, fs // 4))
    out = adapt(step, fs, tau=0.010, strength=0.5)
    assert out[0, 0] > 0.95, "onset should pass almost unchanged"
    assert abs(out[0, -1] - 0.5) < 1e-3, "sustained response should settle at 1 - strength"


def test_adaptation_inverse_round_trip():
    fs = 16000
    v = np.random.default_rng(0).standard_normal((3, 4000))
    assert _snr_db(v, inverse_adapt(adapt(v, fs), fs)) > 200


def test_transduce_round_trip():
    fs = 16000
    bands = 0.2 * np.random.default_rng(1).standard_normal((4, 4000))
    assert _snr_db(bands, inverse_transduce(transduce(bands, fs), fs)) > 200


def test_transduce_silence_is_silence():
    out = transduce(np.zeros((2, 100)), 16000)
    assert np.all(out == 0.0)
