"""Toy irregularly-sampled multi-harmonic periodic series.

Extends toy_sine.py's pure sine generator with 1-3 harmonics of random,
decaying amplitude and phase, so training data ranges from near-pure-sine
(1 harmonic) to genuinely non-sinusoidal periodic shapes (2-3 harmonics),
while the fundamental `period` stays a well-defined ground-truth label.

Motivation: mTAND's learned time embedding is itself built from sin/cos
basis functions. Validating periodicity recovery only on pure sine input
risked a "sine-on-sine" confound -- good performance might partly reflect
that resonance rather than genuine ability to capture periodic structure
in general. This generator is the empirical test of that concern. Also,
a non-sinusoidal periodic signal has real energy at harmonics (2/period,
3/period, ...), so it aliases *more* readily than a pure tone at the same
fundamental period -- the resolvable/aliased boundary found on pure sine
data is probably optimistic for this harder case.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .toy_sine import clustered_times


def generate_complex_periodic_series(rng, t_min=0.0, t_max=20.0, num_context_range=(15, 35),
                                      n_clusters_range=(5, 6), cluster_width_range=(1, 2),
                                      noise_std=0.15, max_harmonics=3):
    """One irregularly-sampled series with a fundamental `period` plus
    1..max_harmonics harmonics of decaying, randomized amplitude and phase.
    n_harmonics=1 reduces to a (phase-shifted) pure sine, matching the
    Phase 1 dataset's marginal case.
    """
    num_context = rng.integers(*num_context_range)
    x = clustered_times(npoints=num_context, t_min=t_min, t_max=t_max,
                         n_clusters_range=n_clusters_range,
                         cluster_width_range=cluster_width_range, rng=rng)
    period = rng.gamma(1, 1)

    n_harmonics = int(rng.integers(1, max_harmonics + 1))
    y = np.zeros_like(x)
    total_weight = 0.0
    for k in range(1, n_harmonics + 1):
        # decaying amplitude so higher harmonics contribute less, like a
        # real band-limited periodic signal rather than arbitrary noise
        amp_k = rng.uniform(0.3, 1.0) / k
        phase_k = rng.uniform(0, 2 * np.pi)
        y = y + amp_k * np.sin(k * x / period + phase_k)
        total_weight += amp_k
    y = y / total_weight  # keeps peak amplitude ~O(1), comparable to the pure-sine dataset
    y = y + rng.normal(0, noise_std, size=x.shape)

    keep = (x != t_min) & (x != t_max)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    return x[order], y[order], period, n_harmonics


class ToyComplexPeriodicDataset(Dataset):
    def __init__(self, n_series=5000, seed=0, t_max=20.0, **gen_kwargs):
        rng = np.random.default_rng(seed)
        self.t_max = t_max
        self.series = [generate_complex_periodic_series(rng, t_max=t_max, **gen_kwargs)
                       for _ in range(n_series)]

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x, y, period, n_harmonics = self.series[idx]
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                torch.tensor(period, dtype=torch.float32),
                torch.tensor(n_harmonics, dtype=torch.float32))


def collate_complex_periodic(batch, t_max=20.0):
    """Same padding/masking scheme as collate_toy_sine, plus n_harmonics."""
    max_len = max(x.shape[0] for x, _, _, _ in batch)
    bsz = len(batch)

    time_steps = torch.zeros(bsz, max_len)
    values = torch.zeros(bsz, max_len)
    mask = torch.zeros(bsz, max_len)
    periods = torch.zeros(bsz)
    n_harmonics = torch.zeros(bsz)

    for i, (x, y, period, nh) in enumerate(batch):
        n = x.shape[0]
        time_steps[i, :n] = x / t_max
        values[i, :n] = y
        mask[i, :n] = 1.0
        periods[i] = period
        n_harmonics[i] = nh

    x_full = torch.stack([values, mask], dim=-1)
    return x_full, time_steps, periods, n_harmonics
