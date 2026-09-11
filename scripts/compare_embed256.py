"""Plain mTAND, embed_time=128 baseline vs. embed_time=256, identical
held-out series, with the resolvable/aliased breakdown. Direct test of
whether more time-embedding frequency capacity alone (no LS conditioning)
helps short-period (aliased) recovery, following the finding that
LS-conditioned attention improved reconstruction loss but collapsed the
period probe.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def main():
    base = np.load(os.path.join(OUTDIR, 'masked_recon_600ep_latents.npz'))
    base_latents, periods = base['latents'], base['periods']
    n = len(periods)
    idx = np.random.default_rng(0).permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]
    true = periods[test_idx]
    resolvable = true > 1.0

    e256 = np.load(os.path.join(OUTDIR, 'masked_recon_embed256_latents.npz'))
    assert np.allclose(e256['periods'], periods), 'series ordering mismatch'
    e256_latents = e256['latents']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    results = {}
    for ax, latents, name in [(axes[0], base_latents, 'embed_time=128 (baseline)'),
                                (axes[1], e256_latents, 'embed_time=256')]:
        reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
        pred = reg.predict(latents[test_idx])
        corr = np.corrcoef(pred, true)[0, 1]
        ss_res = ((true - pred) ** 2).sum()
        ss_tot = ((true - true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        row = {'corr': corr, 'r2': r2}
        print(f'{name}: overall corr={corr:.3f}  R2={r2:.3f}')
        for label, sel in [('resolvable', resolvable), ('aliased', ~resolvable)]:
            c = np.corrcoef(pred[sel], true[sel])[0, 1]
            mse = np.mean((pred[sel] - true[sel]) ** 2)
            row[f'{label}_corr'] = c
            row[f'{label}_mse'] = mse
            print(f'    {label} (n={sel.sum()}): corr={c:.3f}  MSE={mse:.4f}')
        results[name] = row

        ax.scatter(true[resolvable], pred[resolvable], s=12, alpha=0.6, label='resolvable (period>1.0)')
        ax.scatter(true[~resolvable], pred[~resolvable], s=12, alpha=0.6, label='aliased (period<=1.0)')
        lims = [0, min(20, max(true.max(), np.nanpercentile(pred, 95)))]
        ax.plot(lims, lims, 'k--', linewidth=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('true period'); ax.set_ylabel('predicted period')
        ax.set_title(f'{name}\ncorr={corr:.3f} R2={r2:.3f}')
        ax.legend(fontsize=7)

    fig.suptitle('embed_time=128 vs. embed_time=256, identical held-out series (n=250)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'embed256_comparison.png'), dpi=120)

    with open(os.path.join(OUTDIR, 'embed256_comparison.txt'), 'w') as f:
        f.write(str(results))


if __name__ == '__main__':
    main()
