"""Three-way comparison on the identical held-out series: plain mTAND
(embed_time=128 baseline, R^2=0.835), Lomb-Scargle, and the new
LS-conditioned-attention model (MTANReconEncoderLS/DecoderLS).

Tests whether letting attention condition on the LS hint (vs. just
concatenating an LS feature, which was never actually built) transfers
the LS periodogram's strength on the aliased regime into the learned
representation -- or whether, since both encoder and decoder receive
ls_omega directly, the network can shortcut reconstruction through the
LS-conditioned attention pathway without needing to encode period into z
at all (in which case reconstruction quality would improve while the
probe gets *worse*, not better).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def probe(latents, periods, test_idx, fit_idx):
    reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
    return reg.predict(latents[test_idx])


def main():
    base = np.load(os.path.join(OUTDIR, 'masked_recon_600ep_latents.npz'))
    base_latents, periods = base['latents'], base['periods']
    n = len(periods)
    idx = np.random.default_rng(0).permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]
    true = periods[test_idx]

    ls_data = np.load(os.path.join(OUTDIR, 'lomb_scargle_baseline.npz'))
    ls_true, ls_est_all = ls_data['true_periods'], ls_data['ls_estimates']
    assert np.allclose(ls_true, periods), 'series ordering mismatch (LS baseline)'
    ls_pred = ls_est_all[test_idx]

    lscond = np.load(os.path.join(OUTDIR, 'masked_recon_ls_latents.npz'))
    lscond_latents, lscond_periods = lscond['latents'], lscond['periods']
    assert np.allclose(lscond_periods, periods), 'series ordering mismatch (LS-conditioned)'

    mtand_pred = probe(base_latents, periods, test_idx, fit_idx)
    lscond_pred = probe(lscond_latents, periods, test_idx, fit_idx)

    resolvable = true > 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, pred, name in [(axes[0], mtand_pred, 'mTAND (plain, R2=0.835)'),
                             (axes[1], ls_pred, 'Lomb-Scargle'),
                             (axes[2], lscond_pred, 'mTAND (LS-conditioned attn)')]:
        ax.scatter(true[resolvable], pred[resolvable], s=12, alpha=0.6, label='resolvable (period>1.0)')
        ax.scatter(true[~resolvable], pred[~resolvable], s=12, alpha=0.6, label='aliased (period<=1.0)')
        lims = [0, min(20, max(true.max(), np.nanpercentile(pred, 95)))]
        ax.plot(lims, lims, 'k--', linewidth=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('true period'); ax.set_ylabel('predicted period')
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.suptitle('mTAND vs. Lomb-Scargle vs. LS-conditioned attention, identical held-out series (n=250)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'ls_conditioned_comparison.png'), dpi=120)

    print(f'N in comparison: {len(test_idx)}')
    results = {}
    for name, pred in [('mTAND (plain)', mtand_pred), ('Lomb-Scargle', ls_pred),
                        ('mTAND (LS-conditioned attn)', lscond_pred)]:
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

    with open(os.path.join(OUTDIR, 'ls_conditioned_comparison.txt'), 'w') as f:
        f.write(str(results))


if __name__ == '__main__':
    main()
