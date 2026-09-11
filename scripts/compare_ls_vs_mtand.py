"""Fair head-to-head: mTAND's linear probe vs. the Lomb-Scargle baseline,
on the identical held-out subset of validation series (same seed, same
train/test split used throughout this project's probe evaluations).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def main():
    data = np.load(os.path.join(OUTDIR, 'masked_recon_600ep_latents.npz'))
    latents, periods = data['latents'], data['periods']
    n = len(periods)
    idx = np.random.default_rng(0).permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]
    reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
    mtand_pred = reg.predict(latents[test_idx])
    true = periods[test_idx]

    ls_data = np.load(os.path.join(OUTDIR, 'lomb_scargle_baseline.npz'))
    ls_true, ls_est_all = ls_data['true_periods'], ls_data['ls_estimates']
    assert np.allclose(ls_true, periods), 'series ordering mismatch between LS and mTAND evaluations'
    ls_pred = ls_est_all[test_idx]

    resolvable = true > 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, name in [(axes[0], mtand_pred, 'mTAND (linear probe)'), (axes[1], ls_pred, 'Lomb-Scargle')]:
        ax.scatter(true[resolvable], pred[resolvable], s=12, alpha=0.6, label='resolvable (period>1.0)')
        ax.scatter(true[~resolvable], pred[~resolvable], s=12, alpha=0.6, label='aliased (period<=1.0)')
        lims = [0, min(20, max(true.max(), np.nanpercentile(pred, 95)))]
        ax.plot(lims, lims, 'k--', linewidth=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('true period'); ax.set_ylabel('predicted period')
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.suptitle('mTAND vs. Lomb-Scargle, identical held-out series (n=250)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'ls_vs_mtand_comparison.png'), dpi=120)

    print(f'N in comparison: {len(test_idx)}')
    results = {}
    for name, pred in [('mTAND', mtand_pred), ('Lomb-Scargle', ls_pred)]:
        corr = np.corrcoef(pred, true)[0, 1]
        mse = np.mean((pred - true) ** 2)
        ss_res = ((true - pred) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        row = {'corr': corr, 'mse': mse, 'r2': r2}
        print(f'{name}: corr={corr:.3f}  MSE={mse:.4f}  R2={r2:.3f}')
        for label, sel in [('resolvable', resolvable), ('aliased', ~resolvable)]:
            c = np.corrcoef(pred[sel], true[sel])[0, 1]
            m = np.mean((pred[sel] - true[sel]) ** 2)
            row[f'{label}_corr'] = c
            row[f'{label}_mse'] = m
            print(f'    {label} (n={sel.sum()}): corr={c:.3f}  MSE={m:.4f}')
        results[name] = row

    with open(os.path.join(OUTDIR, 'ls_vs_mtand_comparison.txt'), 'w') as f:
        f.write(str(results))


if __name__ == '__main__':
    main()
