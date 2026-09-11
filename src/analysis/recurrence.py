"""Within-series recurrence analysis: does a series revisit similar states
over its own trajectory? Two ways to build the recurrence matrix:

1. `recurrence_matrix_twed`: on the raw irregular (t, v) observations
   directly. Naive point-to-point |v_i - v_j| ignores the irregular
   timestamps entirely, so instead we compare small local windows of
   neighboring (t, v) points around each observation using TWED, which is
   time-aware -- two points can look "close" in value but sit in very
   differently-shaped local neighborhoods once real elapsed time is
   considered.
2. `recurrence_matrix_euclidean`: on mTAND's per-reference-point latent
   sequence, which already sits on a uniform grid (no TWED needed there).

`rqa_metrics` extracts scalar recurrence-quantification-analysis (RQA)
features (recurrence rate, determinism) from a recurrence matrix -- these
per-series summaries are what let us rank a whole training set for
"unusual" examples, even though each recurrence matrix itself is a
within-series object.

Both recurrence-matrix builders here use overlapping windows / share
values between neighboring indices, so nearby indices are trivially
close regardless of periodicity -- a classic recurrence-plot pitfall
(the same reason real RQA excludes a "Theiler window" around the main
diagonal). `rqa_metrics` takes an explicit `exclude` mask so this trivial
near-diagonal closeness never gets counted as genuine recurrence.
"""
import numpy as np

from .twed import twed


def recurrence_matrix_twed(t, v, window=1, nu=1.0, lam=1.0):
    """Recurrence matrix over the observations of ONE series, using TWED
    between small local (t, v) windows around each index rather than raw
    point-to-point value differences.
    """
    n = len(v)
    windows = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        windows.append((t[lo:hi], v[lo:hi]))

    D = np.zeros((n, n))
    for i in range(n):
        ti, vi = windows[i]
        for j in range(i + 1, n):
            tj, vj = windows[j]
            d = twed(ti, vi, tj, vj, nu=nu, lam=lam)
            D[i, j] = D[j, i] = d
    return D


def recurrence_matrix_euclidean(Z):
    """Z: (num_ref, latent_dim) array of per-reference-point latent
    vectors, already on a uniform grid. Pairwise Euclidean distance."""
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def theiler_mask_from_time(t, min_gap):
    """Exclude pairs whose timestamps are within `min_gap` of each other
    -- for the TWED recurrence matrix, where windows already make
    neighboring-in-time points trivially similar."""
    diff = np.abs(np.asarray(t)[:, None] - np.asarray(t)[None, :])
    return diff < min_gap


def theiler_mask_from_index(n, band):
    """Same idea, but for the uniformly-spaced latent recurrence matrix:
    exclude pairs within `band` reference-point indices of each other."""
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return diff < band


def rqa_metrics(D, percentile=10, l_min=2, exclude=None):
    """Threshold D at the given percentile of its *valid* (non-diagonal,
    non-excluded) entries to get a binary recurrence matrix, then compute:
    - recurrence_rate: fraction of valid points that are recurrent.
    - determinism: fraction of recurrent points that lie on diagonal
      lines of length >= l_min, i.e. genuine repeated trajectory
      segments rather than isolated coincidental closeness.
    """
    n = D.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    if exclude is None:
        exclude = np.zeros((n, n), dtype=bool)
    valid = ~diag_mask & ~exclude

    off_diag = D[valid]
    if len(off_diag) == 0:
        return {'recurrence_rate': np.nan, 'determinism': np.nan}
    eps = np.percentile(off_diag, percentile)

    R = (D <= eps) & valid
    recurrence_rate = R.sum() / valid.sum()

    total_recurrent = R.sum()
    on_diag_lines = 0
    for offset in range(-(n - 1), n):
        if offset == 0:
            continue
        diag_R = np.diagonal(R, offset=offset)
        diag_valid = np.diagonal(valid, offset=offset)
        run = 0
        for val, ok in zip(diag_R, diag_valid):
            if ok and val:
                run += 1
            else:
                if run >= l_min:
                    on_diag_lines += run
                run = 0
        if run >= l_min:
            on_diag_lines += run

    determinism = on_diag_lines / total_recurrent if total_recurrent > 0 else 0.0
    return {'recurrence_rate': float(recurrence_rate), 'determinism': float(determinism)}
