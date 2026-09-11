"""Toy irregularly-sampled sine dataset (ported from the gist).

Each series has a single known generative factor, `period ~ Gamma(1, 1)`,
observed at Poisson-cluster timestamps with Gaussian observation noise.
This is the Phase 1 sanity dataset: no disentanglement yet, just enough
structure (one labeled factor) to confirm mTAND produces a representation
that carries real signal and attends to plausible timestamps.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def clustered_times(npoints=200,
                     t_min=0.0,
                     t_max=10.0,
                     n_clusters_range=(2, 6),
                     cluster_width_range=(0.2, 1.0),
                     background_frac=0.1,
                     rng=None):
    """Returns sorted timestamps with stochastic clusters + gaps."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng

    n_clusters = rng.integers(*n_clusters_range)
    centers = rng.uniform(t_min, t_max, size=n_clusters)
    cluster_weights = rng.dirichlet(alpha=np.ones(n_clusters))
    n_cluster_points = int((1 - background_frac) * npoints)
    counts = rng.multinomial(n_cluster_points, cluster_weights)

    xs = []
    for c, k in zip(centers, counts):
        if k == 0:
            continue
        width = rng.uniform(*cluster_width_range)
        xs.append(rng.normal(loc=c, scale=width, size=k))

    n_bg = npoints - sum(counts)
    if n_bg > 0:
        xs.append(rng.uniform(t_min, t_max, size=n_bg))

    x = np.concatenate(xs)
    x = np.clip(x, t_min, t_max)
    return x


def generate_toy_series(rng, t_min=0.0, t_max=20.0, num_context_range=(15, 35),
                         n_clusters_range=(5, 6), cluster_width_range=(1, 2),
                         noise_std=0.15):
    """One irregularly-sampled sine series with a random period factor."""
    num_context = rng.integers(*num_context_range)
    x = clustered_times(npoints=num_context, t_min=t_min, t_max=t_max,
                         n_clusters_range=n_clusters_range,
                         cluster_width_range=cluster_width_range, rng=rng)
    period = rng.gamma(1, 1)
    y = np.sin(x / period) + rng.normal(0, noise_std, size=x.shape)

    keep = (x != t_min) & (x != t_max)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    return x[order], y[order], period


class ToySineDataset(Dataset):
    def __init__(self, n_series=5000, seed=0, t_max=20.0, **gen_kwargs):
        rng = np.random.default_rng(seed)
        self.t_max = t_max
        self.series = [generate_toy_series(rng, t_max=t_max, **gen_kwargs)
                       for _ in range(n_series)]

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x, y, period = self.series[idx]
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                torch.tensor(period, dtype=torch.float32))


def collate_toy_sine(batch, t_max=20.0):
    """Pads to the batch's max length; normalizes time to [0, 1] for mTAND."""
    max_len = max(x.shape[0] for x, _, _ in batch)
    bsz = len(batch)

    time_steps = torch.zeros(bsz, max_len)
    values = torch.zeros(bsz, max_len)
    mask = torch.zeros(bsz, max_len)
    periods = torch.zeros(bsz)

    for i, (x, y, period) in enumerate(batch):
        n = x.shape[0]
        time_steps[i, :n] = x / t_max
        values[i, :n] = y
        mask[i, :n] = 1.0
        periods[i] = period

    # mTAND expects x as concat([values, mask], dim=-1)
    x_full = torch.stack([values, mask], dim=-1)
    return x_full, time_steps, periods
