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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, lsqr, spsolve

from cochlea.gammatone_frame import analyze, synthesize


def _spike_intervals(spike_neuron, spike_time, n, dt, population, refractory_period, theta):
    """
    Turn spikes into one measurement per inter-spike interval.

    Returns:
        j (np.ndarray): Neuron of each interval
        a, b (np.ndarray): Integration start and end (the spike) in seconds
        decay (np.ndarray): exp(-(b - a) / tau)
        rhs (np.ndarray): What gain * (leaky integral of the drive over [a, b]) must equal
    """
    order = np.lexsort((spike_time, spike_neuron))
    j = np.asarray(spike_neuron)[order]
    b = np.asarray(spike_time, dtype=float)[order]
    first = np.r_[True, j[1:] != j[:-1]] if len(j) else np.zeros(0, dtype=bool)
    a = np.where(first, 0.0, np.r_[0.0, b[:-1]] + refractory_period)
    v_a = np.where(first, population['v0'][j], 0.0)
    keep = (b > a) & (a >= 0) & (b < n * dt)
    j, a, b, v_a = j[keep], a[keep], b[keep], v_a[keep]
    decay = np.exp(-(b - a) / population['tau'])
    rhs = theta - v_a * decay - population['bias'][j] * (1 - decay)
    return j, a, b, decay, rhs


def decode_drive(spike_neuron, spike_time, n, fs, num_channels, population,
                 refractory_period=0.001, theta=1.0):
    """
    Recover every channel's neuron drive from spike times, one channel at a time.

    The unknown is the channel's leaky integral E on the sample grid; each
    inter-spike interval is a sparse equation in four of its values. The drive
    follows from E by E[i+1] = alpha*E[i] + (1-alpha)*drive[i]. Unlike
    decode_spike_times() this assumes nothing about the drive, so it works behind
    a nonlinear hair cell, but each channel needs several intervals per sample.

    Args:
        spike_neuron (np.ndarray): Neuron index of each spike
        spike_time (np.ndarray): Time of each spike (seconds)
        n (int): Number of samples per channel
        fs (int): Sampling rate (Hz)
        num_channels (int): Number of frequency channels
        population (dict): The encoder's make_population() dict
        refractory_period (float): Must match the encoder
        theta (float): Must match the encoder

    Returns:
        drive (np.ndarray): Shape (num_channels, n)
        info (dict): 'measurements', 'oversampling' (mean intervals per sample per
            channel), 'unobserved_samples', 'misfit' (worst relative residual of
            the spike equations; below 1e-8 when spike times are exact)
    """
    dt = 1.0 / fs
    alpha = np.exp(-dt / population['tau'])
    j, a, b, decay, rhs = _spike_intervals(spike_neuron, spike_time, n, dt, population,
                                           refractory_period, theta)
    rhs = rhs / population['gain'][j]
    channel = population['channel'][j]
    info = {'measurements': len(j), 'oversampling': len(j) / (n * num_channels)}

    # Leaky integral at time t inside sample i = c0 * E[i] + c1 * E[i+1].
    def interpolate(t):
        i = np.minimum(np.floor(t / dt).astype(int), n - 1)
        e = np.exp(-(t - i * dt) / population['tau'])
        c1 = (1 - e) / (1 - alpha)
        return i, e - alpha * c1, c1

    # Smoothness prior on the drive, weighted far below the data. It only decides
    # what data cannot: where no fiber observed a sample, the drive is interpolated
    # from its neighbours instead of collapsing to an arbitrary value.
    to_drive = diags([np.full(n, 1 / (1 - alpha)), np.full(n - 1, -alpha / (1 - alpha))],
                     [0, -1], format='csr')
    roughness = (diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) @ to_drive)
    prior = 1e-15 * (roughness.T @ roughness)

    drive = np.zeros((num_channels, n))
    starved = 0
    unobserved = 0
    misfit = 0.0
    for k in range(num_channels):
        sel = channel == k
        m = int(np.sum(sel))
        if m < n:
            starved += 1
        if m == 0:
            continue
        i_b, b0, b1 = interpolate(b[sel])
        i_a, a0, a1 = interpolate(a[sel])
        d = decay[sel]
        rows = np.tile(np.arange(m), 4)
        cols = np.r_[i_b, i_b + 1, i_a, i_a + 1]  # column c holds E[c], c = 0..n
        vals = np.r_[b0, b1, -d * a0, -d * a1]
        # A grid point no interval starts or ends next to is seen only through later
        # integrals; up to the last spike that leaves its sample undetermined.
        touched = np.bincount(cols, minlength=n + 1) > 0
        unobserved += int(np.sum(~touched[1:int(np.max(i_b)) + 1]))
        system = csr_matrix((vals, (rows, cols)), shape=(m, n + 1))[:, 1:]  # E[0] = 0
        normal = (system.T @ system + prior).tocsc()
        solution = spsolve(normal, system.T @ rhs[sel])
        misfit = max(misfit, float(np.linalg.norm(system @ solution - rhs[sel])
                                   / (np.linalg.norm(rhs[sel]) + 1e-300)))
        leaky = np.r_[0.0, solution]
        drive[k] = (leaky[1:] - alpha * leaky[:-1]) / (1 - alpha)
    if starved:
        warnings.warn(f"{starved} of {num_channels} channels have fewer spike intervals "
                      "than samples; reconstruction is under-determined")
    info['unobserved_samples'] = unobserved
    info['misfit'] = misfit
    if misfit > 1e-6:
        warnings.warn(f"spike times are inconsistent with noise-free fibers (relative misfit "
                      f"{misfit:.1e}); the reconstruction is approximate")
    if unobserved:
        warnings.warn(f"{unobserved} samples were unobserved (no fiber spiked or left its "
                      "refractory period near them); they were interpolated, not recovered")
    return drive, info


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

    j, a, b, decay, rhs = _spike_intervals(spike_neuron, spike_time, n, dt, population,
                                           refractory_period, theta)
    m = len(j)
    info = {'measurements': m, 'oversampling': m / n, 'iterations': 0}
    if m == 0:
        warnings.warn("no usable spikes; returning silence")
        return np.zeros(n), info
    if m < n:
        warnings.warn(f"only {m} spike intervals for {n} samples; "
                      "reconstruction is under-determined")

    gain = population['gain'][j]

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
