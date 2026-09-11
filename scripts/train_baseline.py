"""Phase 1: baseline mTAND encoder, no disentanglement yet.

Trains MTANEncoder + a regression head to recover the single known
generative factor (`period`) of the toy irregular-sine dataset from the raw
(value, mask, time) observations. This isolates backbone problems (can
mTAND represent this data at all?) from disentanglement problems, which
come in Phase 2.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset, collate_toy_sine  # noqa: E402
from src.mtand.model import MTANEncoder, MTANRegressor  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 32
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
SEED = 0


def make_loaders():
    full = ToySineDataset(n_series=5000, seed=SEED, t_max=T_MAX)
    train, val = torch.utils.data.random_split(
        full, [4500, 500], generator=torch.Generator().manual_seed(SEED))

    def collate(batch):
        return collate_toy_sine(batch, t_max=T_MAX)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def build_model():
    query = torch.linspace(0, 1, NUM_REF)
    encoder = MTANEncoder(input_dim=1, query=query, nhidden=NHIDDEN,
                           embed_time=EMBED_TIME, num_heads=1, learn_emb=True)
    return MTANRegressor(encoder, nhidden=NHIDDEN)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, t, period in loader:
            pred, _ = model(x, t)
            preds.append(pred)
            targets.append(period)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    corr = float(np.corrcoef(preds, targets)[0, 1])
    return mse, corr, preds, targets


def _series_attention(model, x, y, period):
    """Run one series through the model, return per-reference-point peak
    time, sharpness (max attention weight), and whether the peak equals the
    series' own latest observation."""
    xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
    with torch.no_grad():
        _, attn = model(xb, tb)
    # attn: (batch=1, heads=1, num_ref, seq_len, dim=2); channels are
    # identical duplicated masks, so channel 0 is representative.
    attn_map = attn[0, 0, :, :, 0].numpy()  # (num_ref, seq_len)
    obs_times = x.numpy()
    peak_idx = attn_map.argmax(axis=1)
    peak_times = obs_times[np.clip(peak_idx, 0, len(obs_times) - 1)]
    sharpness = attn_map.max(axis=1)
    return peak_times, sharpness, obs_times


def attention_sanity_check(model, val_ds, n_examples=4):
    """For a handful of series, inspect what each reference point's
    attention actually locks onto.

    Empirically (see period_vs_fixation_analysis), reference points early
    in the domain are often extremely sharp (near one-hot) and lock onto
    the single *latest* observation in the series rather than the nearest
    one in time -- this is not a locality-seeking mechanism. For the period
    regression task, a late observation constrains `period` more tightly
    than an early one, since accumulated phase x/period grows with x, so
    this behavior is plausibly task-driven rather than a bug. We plot
    attended-time and sharpness separately rather than compare attended-time
    to reference-time on a shared diagonal -- there's no reason to expect
    them to match, and a previous version of this plot implied there was.
    """
    model.eval()
    fig, axes = plt.subplots(n_examples, 2, figsize=(11, 2.8 * n_examples),
                              gridspec_kw={'width_ratios': [2, 1]})
    ref_times = np.linspace(0, T_MAX, NUM_REF)
    report = []
    for i in range(n_examples):
        x, y, period = val_ds[i]
        peak_times, sharpness, obs_times = _series_attention(model, x, y, period)
        obs_vals = y.numpy()
        within_range = np.mean((peak_times >= obs_times.min()) & (peak_times <= obs_times.max()))
        matches_latest = np.mean(np.isclose(peak_times, obs_times.max()))
        report.append({'within_range': within_range, 'frac_matches_latest': matches_latest})

        ax = axes[i, 0]
        ax.scatter(obs_times, obs_vals, c='k', s=15, label='observations', zorder=3)
        sc = ax.scatter(peak_times, np.full_like(peak_times, obs_vals.min() - 0.3),
                         c=sharpness, cmap='viridis', vmin=0, vmax=1, s=20, marker='|',
                         label='peak-attended timestamp\n(per ref. point, color=sharpness)')
        ax.set_title(f'series {i}: period={period.item():.2f}  '
                     f'{within_range * 100:.0f}% peaks in range, '
                     f'{matches_latest * 100:.0f}% match latest obs.')
        ax.set_xlabel('time')
        ax.set_ylabel('y (obs)')
        ax.legend(loc='upper right', fontsize=6)

        ax2 = axes[i, 1]
        ax2.plot(ref_times, sharpness, color='tab:orange')
        ax2.axhline(1.0 / len(obs_times), color='gray', linestyle='--', linewidth=0.8,
                    label='uniform (1/n_obs)')
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('reference time')
        ax2.set_ylabel('attention sharpness\n(max weight)')
        ax2.legend(fontsize=6)
    fig.colorbar(sc, ax=axes[:, 0], label='sharpness (max attn. weight)', shrink=0.6)
    fig.savefig(os.path.join(OUTDIR, 'attention_sanity_check.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    return report


def period_vs_fixation_analysis(model, val_ds):
    """Does fixation on the latest observation get stronger for
    short-period series, as the naive d/d(period) sin(x/period) ~ x/period^2
    argument would suggest? Check across the whole validation set."""
    from scipy.stats import spearmanr, pointbiserialr

    periods, sharpness0, matches = [], [], []
    for i in range(len(val_ds)):
        x, y, period = val_ds[i]
        peak_times, sharpness, obs_times = _series_attention(model, x, y, period)
        periods.append(period.item())
        sharpness0.append(sharpness[0])  # ref_time = 0
        matches.append(np.isclose(peak_times[0], obs_times.max()))
    periods = np.array(periods)
    sharpness0 = np.array(sharpness0)
    matches = np.array(matches, dtype=int)

    pearson_r = float(np.corrcoef(periods, sharpness0)[0, 1])
    spearman_r, spearman_p = spearmanr(periods, sharpness0)
    pb_r, pb_p = pointbiserialr(matches, periods)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(periods, sharpness0, c=matches, cmap='coolwarm', s=12, alpha=0.7)
    ax.set_xlabel('true period')
    ax.set_ylabel('ref_time=0 attention sharpness (max weight)')
    ax.set_title(f'Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f} (p={spearman_p:.2f})\n'
                 f'color = whether peak matches series\' latest observation')
    fig.colorbar(sc, ax=ax, label='matches latest obs. (1=yes)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'period_vs_attention_fixation.png'), dpi=120)
    plt.close(fig)

    return {
        'pearson_r_period_sharpness': pearson_r,
        'spearman_r_period_sharpness': float(spearman_r),
        'spearman_p_period_sharpness': float(spearman_p),
        'pointbiserial_r_match_period': float(pb_r),
        'pointbiserial_p_match_period': float(pb_p),
        'overall_match_latest_rate': float(matches.mean()),
    }


def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x, t, period in train_loader:
            opt.zero_grad()
            pred, _ = model(x, t)
            loss = loss_fn(pred, period)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_mse = epoch_loss / n
        val_mse, val_corr, _, _ = evaluate(model, val_loader)
        history.append((epoch, train_mse, val_mse, val_corr))
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f'epoch {epoch:3d}  train_mse={train_mse:.4f}  '
                  f'val_mse={val_mse:.4f}  val_corr={val_corr:.4f}')

    val_mse, val_corr, preds, targets = evaluate(model, val_loader)
    print(f'\nFinal: val_mse={val_mse:.4f}  val_corr(pred,true period)={val_corr:.4f}')
    print(f'Target variance (baseline for comparison): {np.var(targets):.4f}')

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, s=8, alpha=0.5)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('true period')
    plt.ylabel('predicted period')
    plt.title(f'val corr={val_corr:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'period_pred_vs_true.png'), dpi=120)
    plt.close()

    report = attention_sanity_check(model, val_loader.dataset)
    print(f'Attention sanity check (per example series): {report}')

    fixation_stats = period_vs_fixation_analysis(model, val_loader.dataset)
    print('Period vs. attention-fixation correlation (full val set):', fixation_stats)

    torch.save(model.state_dict(), os.path.join(OUTDIR, 'mtand_phase1_baseline.pt'))
    with open(os.path.join(OUTDIR, 'phase1_metrics.txt'), 'w') as f:
        f.write(f'val_mse={val_mse:.4f}\nval_corr={val_corr:.4f}\n'
                f'target_var={np.var(targets):.4f}\n'
                f'attn_sanity_report={report}\n'
                f'period_vs_fixation={fixation_stats}\n')


if __name__ == '__main__':
    main()
