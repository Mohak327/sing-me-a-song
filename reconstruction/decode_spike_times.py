"""
Reconstruct audio from spike times alone.

Every inter-spike interval of a LIF neuron is one exact linear equation in the
audio samples: the leaky integral of the neuron's input from the end of its
refractory period to its next spike equals the threshold. Stacking the equations
of all neurons and solving them by least squares recovers the waveform.
"""

import warnings
import numpy as np
from scipy.signal import lfilter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsqr

from cochlea.gammatone_frame import analyze, synthesize


def decode_spike_times(spike_neuron, spike_time, n, fs, H, population,
                       refractory_period=0.001, theta=1.0, iter_lim=2000, tol=1e-12):
    """
    Decode spike times back to audio.

    Args:
        spike_neuron (np.ndarray): Neuron index of each spike
        spike_time (np.ndarray): Time of each spike (seconds)
        n (int): Number of audio samples to reconstruct
        fs (int): Sampling rate (Hz)
        H (np.ndarray): Gammatone frame used by the encoder
        population (dict): The encoder's make_population() dict
        refractory_period (float): Must match the encoder
        theta (float): Must match the encoder
        iter_lim (int): Maximum LSQR iterations
        tol (float): LSQR atol/btol

    Returns:
        x (np.ndarray): Reconstructed audio, shape (n,)
        info (dict): 'measurements', 'oversampling' (measurements / n), 'iterations'
    """
    num_channels = H.shape[0]
    dt = 1.0 / fs
    tau = population['tau']

    # One measurement per inter-spike interval: integration runs from a to b.
    order = np.lexsort((spike_time, spike_neuron))
    j = np.asarray(spike_neuron)[order]
    b = np.asarray(spike_time, dtype=float)[order]
    first = np.r_[True, j[1:] != j[:-1]] if len(j) else np.zeros(0, dtype=bool)
    a = np.where(first, 0.0, np.r_[0.0, b[:-1]] + refractory_period)
    v_a = np.where(first, population['v0'][j], 0.0)
    keep = (b > a) & (a >= 0) & (b < n * dt)
    j, a, b, v_a = j[keep], a[keep], b[keep], v_a[keep]
    m = len(j)
    info = {'measurements': m, 'oversampling': m / n, 'iterations': 0}
    if m == 0:
        warnings.warn("no usable spikes; returning silence")
        return np.zeros(n), info
    if m < n:
        warnings.warn(f"only {m} spike intervals for {n} samples; "
                      "reconstruction is under-determined")

    decay = np.exp(-(b - a) / tau)
    gain = population['gain'][j]
    rhs = theta - v_a * decay - population['bias'][j] * (1 - decay)

    # Leaky integral up to time t = E[i] * e + U[i] * (1 - e), i = sample holding t.
    def weights(t):
        i = np.minimum(np.floor(t / dt).astype(int), n - 1)
        e = np.exp(-(t - i * dt) / tau)
        return i, e, 1 - e

    i_b, e_b, u_b = weights(b)
    i_a, e_a, u_a = weights(a)
    col_b = population['channel'][j] * n + i_b
    col_a = population['channel'][j] * n + i_a
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[col_b, col_a]
    shape = (m, num_channels * n)
    pick_e = csr_matrix((np.r_[gain * e_b, -gain * decay * e_a], (rows, cols)), shape=shape)
    pick_u = csr_matrix((np.r_[gain * u_b, -gain * decay * u_a], (rows, cols)), shape=shape)

    alpha = np.exp(-dt / tau)
    num, den = [0.0, 1 - alpha], [1.0, -alpha]  # E[i+1] = alpha*E[i] + (1-alpha)*U[i]

    def forward(x):
        bands = analyze(x, H)
        leaky = lfilter(num, den, bands, axis=1)
        return pick_e @ leaky.ravel() + pick_u @ bands.ravel()

    def adjoint(r):
        back_e = (pick_e.T @ r).reshape(num_channels, n)
        back_u = (pick_u.T @ r).reshape(num_channels, n)
        back_u = back_u + lfilter(num, den, back_e[:, ::-1], axis=1)[:, ::-1]
        return synthesize(back_u, H)

    operator = LinearOperator((m, n), matvec=forward, rmatvec=adjoint, dtype=float)
    solution = lsqr(operator, rhs, atol=tol, btol=tol, iter_lim=iter_lim)
    info['iterations'] = int(solution[2])
    return solution[0], info
