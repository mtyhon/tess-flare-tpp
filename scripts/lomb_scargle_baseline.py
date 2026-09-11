"""Classical Lomb-Scargle periodogram baseline for period recovery.

Decisive diagnostic proposed earlier and finally run here: is short-period
recovery failing because of genuine information limits (aliasing), or
because mTAND specifically isn't extracting information that's actually
there? Uses scipy's trusted, tested implementation (not a homemade
reimplementation) on the exact same validation series and the same
context-only subset mTAND's encoder saw (same split_context_target seed),
for a fair, apples-to-apples comparison -- no training involved.
"""
import sys
import os

import numpy as np
from scipy.signal import lombscargle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset  # noqa: E402
import torch  # noqa: E402
import scripts.train_masked_reconstruction as m  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

T_MAX = 20.0
SEED = 0
PERIOD_MIN, PERIOD_MAX, N_GRID = 0.01, 20.0, 5000


def estimate_period(t, y, period_grid, angular_freqs):
    y = y - y.mean()
    power = lombscargle(t, y, angular_freqs, normalize=True)
    return period_grid[np.argmax(power)]


def main():
    full = ToySineDataset(n_series=5000, seed=SEED, t_max=T_MAX)
    train, val = torch.utils.data.random_split(
        full, [4500, 500], generator=torch.Generator().manual_seed(SEED))

    period_grid = np.logspace(np.log10(PERIOD_MIN), np.log10(PERIOD_MAX), N_GRID)
    angular_freqs = 2 * np.pi / period_grid

    mask_rng = np.random.default_rng(m.SEED)
    true_periods, ls_estimates = [], []
    for i in range(len(val)):
        x, y, period = val.dataset[val.indices[i]]
        x_np, y_np = x.numpy(), y.numpy()
        n = len(x_np)
        mask = torch.ones(1, n)
        context_mask, target_mask = m.split_context_target(mask, m.MASK_RATIO, mask_rng)
        ctx_bool = context_mask[0].numpy().astype(bool)
        t_ctx, y_ctx = x_np[ctx_bool], y_np[ctx_bool]

        est = estimate_period(t_ctx, y_ctx, period_grid, angular_freqs)
        true_periods.append(period.item())
        ls_estimates.append(est)

    true_periods = np.array(true_periods)
    ls_estimates = np.array(ls_estimates)
    np.savez(os.path.join(OUTDIR, 'lomb_scargle_baseline.npz'),
              true_periods=true_periods, ls_estimates=ls_estimates)

    corr = np.corrcoef(ls_estimates, true_periods)[0, 1]
    log_corr = np.corrcoef(np.log(ls_estimates), np.log(true_periods))[0, 1]
    ss_res = ((true_periods - ls_estimates) ** 2).sum()
    ss_tot = ((true_periods - true_periods.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    resolvable = true_periods > 1.0
    print(f'N series: {len(true_periods)}  N resolvable: {resolvable.sum()}  N aliased: {(~resolvable).sum()}')
    print(f'LS overall: corr={corr:.3f}  log-corr={log_corr:.3f}  R2={r2:.3f}')

    for label, sel in [('resolvable (period>1.0)', resolvable), ('aliased (period<=1.0)', ~resolvable)]:
        if sel.sum() > 5:
            c = np.corrcoef(ls_estimates[sel], true_periods[sel])[0, 1]
            mse = np.mean((ls_estimates[sel] - true_periods[sel]) ** 2)
            rel_err = np.median(np.abs(ls_estimates[sel] - true_periods[sel]) / true_periods[sel])
            print(f'  {label}: corr={c:.3f}  MSE={mse:.4f}  median relative error={rel_err:.3f}  n={sel.sum()}')

    with open(os.path.join(OUTDIR, 'lomb_scargle_baseline_metrics.txt'), 'w') as f:
        f.write(f'corr={corr:.4f}\nlog_corr={log_corr:.4f}\nR2={r2:.4f}\n')


if __name__ == '__main__':
    main()
