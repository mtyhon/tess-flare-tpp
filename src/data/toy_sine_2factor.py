"""Toy irregularly-sampled sine dataset with two deliberately correlated
generative factors: `period` and `amplitude`.

This is Phase 2's synthetic validation set. DIOSC's Independence-of-Support
loss is specifically built to disentangle factors that are *statistically
correlated* in the training data (unlike beta-VAE/FactorVAE-style methods,
which implicitly assume the true factors are marginally independent). To
actually test that, the synthetic ground truth needs a real, deliberate
correlation between factors -- not just more factors. If we only validated
on independent factors we couldn't tell whether the mTAND+DIOSC composition
solves the hard problem it's designed for, or would silently fail the
moment a real correlation shows up (which is the situation on real data).

`period` keeps its Phase 1 marginal (`Gamma(1, 1)`). `amplitude` is
generated from a Gaussian-copula-style construction in log-space: partly
driven by `log(period)` (the correlated component) and partly by
independent noise, so the two factors remain distinct generative
mechanisms -- just statistically entangled in their joint distribution,
with a tunable, verifiable correlation strength (`corr_alpha`).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .toy_sine import clustered_times

# E[log(X)] and Var[log(X)] for X ~ Gamma(1, 1) = Exponential(1): these are
# exact (Euler-Mascheroni constant and pi^2/6), used to keep the amplitude
# marginal well-behaved regardless of corr_alpha.
LOG_PERIOD_MEAN = -0.5772156649
LOG_PERIOD_VAR = np.pi ** 2 / 6


def generate_toy_series_2factor(rng, t_min=0.0, t_max=20.0, num_context_range=(15, 35),
                                 n_clusters_range=(5, 6), cluster_width_range=(1, 2),
                                 noise_std=0.15, corr_alpha=0.5, amplitude_noise_std=0.4):
    """One irregularly-sampled series with two correlated factors:
    `period` (unchanged from Phase 1) and `amplitude`, correlated with
    `period` via `corr_alpha` (0 = independent, larger = more correlated).
    """
    num_context = rng.integers(*num_context_range)
    x = clustered_times(npoints=num_context, t_min=t_min, t_max=t_max,
                         n_clusters_range=n_clusters_range,
                         cluster_width_range=cluster_width_range, rng=rng)
    period = rng.gamma(1, 1)

    log_amplitude = corr_alpha * (np.log(period) - LOG_PERIOD_MEAN) + rng.normal(0, amplitude_noise_std)
    amplitude = np.exp(log_amplitude)

    y = amplitude * np.sin(x / period) + rng.normal(0, noise_std, size=x.shape)

    keep = (x != t_min) & (x != t_max)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    return x[order], y[order], period, amplitude


class ToySine2FactorDataset(Dataset):
    def __init__(self, n_series=5000, seed=0, t_max=20.0, **gen_kwargs):
        rng = np.random.default_rng(seed)
        self.t_max = t_max
        self.series = [generate_toy_series_2factor(rng, t_max=t_max, **gen_kwargs)
                       for _ in range(n_series)]

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x, y, period, amplitude = self.series[idx]
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                torch.tensor(period, dtype=torch.float32),
                torch.tensor(amplitude, dtype=torch.float32))


def collate_toy_sine_2factor(batch, t_max=20.0):
    """Same padding/masking scheme as collate_toy_sine, plus amplitude."""
    max_len = max(x.shape[0] for x, _, _, _ in batch)
    bsz = len(batch)

    time_steps = torch.zeros(bsz, max_len)
    values = torch.zeros(bsz, max_len)
    mask = torch.zeros(bsz, max_len)
    periods = torch.zeros(bsz)
    amplitudes = torch.zeros(bsz)

    for i, (x, y, period, amplitude) in enumerate(batch):
        n = x.shape[0]
        time_steps[i, :n] = x / t_max
        values[i, :n] = y
        mask[i, :n] = 1.0
        periods[i] = period
        amplitudes[i] = amplitude

    x_full = torch.stack([values, mask], dim=-1)
    return x_full, time_steps, periods, amplitudes
