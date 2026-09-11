"""Combines the two remaining asks in one run: reconstruction snapshots
during training, and Neural-Process-style predictive uncertainty.

MTANReconEncoderVAE (proven in train_reconstruction_vae.py to improve
period-separability: probe R^2 0.44 -> 0.49) + MTANReconDecoderProbabilistic
(mean, log-variance per query time), trained end-to-end with a proper
heteroscedastic-likelihood ELBO (reconstruction Gaussian NLL using the
decoder's own learned variance, not a fixed noise_std, + KL-to-standard-
normal on the latent). Captures (mean, std) reconstruction snapshots for a
couple of fixed validation series at several epochs during the actual
training run.
"""
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset, collate_toy_sine  # noqa: E402
from src.mtand.model import MTANReconEncoderVAE, MTANReconDecoderProbabilistic  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 64
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 75
LR = 1e-3
SEED = 0
KL_WARMUP_EPOCHS = 10
SNAPSHOT_EPOCHS = [0, 2, 5, 10, 20, 40, 74]
SNAPSHOT_SERIES_IDX = [3, 7]


def make_loaders():
    full = ToySineDataset(n_series=5000, seed=SEED, t_max=T_MAX)
    train, val = torch.utils.data.random_split(
        full, [4500, 500], generator=torch.Generator().manual_seed(SEED))

    def collate(batch):
        return collate_toy_sine(batch, t_max=T_MAX)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, train, val


def build_model():
    query = torch.linspace(0, 1, NUM_REF)
    encoder = MTANReconEncoderVAE(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                   nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                   learn_emb=True)
    decoder = MTANReconDecoderProbabilistic(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                             nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                             learn_emb=True)
    return encoder, decoder


def elbo_loss(mean, logvar_obs, y, mask, z_mean, z_logvar, kl_coef):
    nll = 0.5 * (logvar_obs + (y - mean) ** 2 / torch.exp(logvar_obs) + math.log(2 * math.pi))
    nll = (nll * mask).sum() / mask.sum()
    kl = (-0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())).sum(-1).mean()
    return nll + kl_coef * kl, nll, kl


def evaluate(encoder, decoder, loader):
    encoder.eval(); decoder.eval()
    total_sq_err, total_mask = 0.0, 0.0
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            z_mean, z_logvar, _ = encoder(x, t)
            mean, logvar_obs, _ = decoder(z_mean, t)
            total_sq_err += ((mean - y) ** 2 * mask).sum().item()
            total_mask += mask.sum().item()
    return total_sq_err / total_mask


def linear_probe_period(encoder, val_ds):
    from sklearn.linear_model import LinearRegression
    encoder.eval()
    latents, periods = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            z_mean, z_logvar, _ = encoder(xb, tb)
            latents.append(z_mean[0].mean(dim=0).numpy())
            periods.append(period.item())
    latents = np.stack(latents); periods = np.array(periods)
    n = len(periods)
    idx = np.random.default_rng(SEED).permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]
    reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
    pred = reg.predict(latents[test_idx])
    corr = float(np.corrcoef(pred, periods[test_idx])[0, 1])
    r2 = float(reg.score(latents[test_idx], periods[test_idx]))
    return latents, periods, {'probe_corr': corr, 'probe_r2': r2}


def uncertainty_vs_distance_check(encoder, decoder, val_ds, n_series=200):
    encoder.eval(); decoder.eval()
    n_series = min(n_series, len(val_ds))
    dense_t = torch.linspace(0, 1, 100).unsqueeze(0)
    dists, stds = [], []
    with torch.no_grad():
        for i in range(n_series):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            z_mean, _, _ = encoder(xb, tb)
            dmean, dlogvar, _ = decoder(z_mean, dense_t)
            std = torch.exp(0.5 * dlogvar)[0].numpy()
            obs_t = x.numpy()
            query_t = dense_t[0].numpy() * T_MAX
            nearest_dist = np.min(np.abs(query_t[:, None] - obs_t[None, :]), axis=1)
            dists.append(nearest_dist)
            stds.append(std)
    dists = np.concatenate(dists); stds = np.concatenate(stds)
    return dists, stds, float(np.corrcoef(dists, stds)[0, 1])


def take_snapshot(encoder, decoder, val_dataset, dense_t):
    encoder.eval(); decoder.eval()
    snaps = {}
    with torch.no_grad():
        for idx in SNAPSHOT_SERIES_IDX:
            x, y, period = val_dataset[idx]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            z_mean, _, _ = encoder(xb, tb)
            dmean, dlogvar, _ = decoder(z_mean, dense_t)
            dstd = torch.exp(0.5 * dlogvar)[0].numpy()
            snaps[idx] = (dmean[0].numpy().copy(), dstd.copy())
    return snaps


def main():
    torch.manual_seed(SEED)
    train_loader, val_loader, train_ds, val_ds = make_loaders()
    encoder, decoder = build_model()
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    dense_t = torch.linspace(0, 1, 300).unsqueeze(0)
    snapshot_series = {}
    for idx in SNAPSHOT_SERIES_IDX:
        x_t, y_t, period_t = val_ds.dataset[val_ds.indices[idx]]
        snapshot_series[idx] = (x_t.numpy(), y_t.numpy(), float(period_t))
    all_snapshots = {-1: take_snapshot(encoder, decoder, val_ds, dense_t)}

    t_start = time.time()
    for epoch in range(EPOCHS):
        encoder.train(); decoder.train()
        kl_coef = min(1.0, epoch / KL_WARMUP_EPOCHS)
        epoch_loss, epoch_nll, epoch_kl, n = 0.0, 0.0, 0.0, 0
        for x, t, period in train_loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            opt.zero_grad()
            z_mean, z_logvar, _ = encoder(x, t)
            z = encoder.reparameterize(z_mean, z_logvar)
            mean, logvar_obs, _ = decoder(z, t)
            loss, nll, kl = elbo_loss(mean, logvar_obs, y, mask, z_mean, z_logvar, kl_coef)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_nll += nll.item() * x.size(0)
            epoch_kl += kl.item() * x.size(0)
            n += x.size(0)
        if epoch in SNAPSHOT_EPOCHS:
            all_snapshots[epoch] = take_snapshot(encoder, decoder, val_ds, dense_t)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            val_mse = evaluate(encoder, decoder, val_loader)
            print(f'epoch {epoch:3d}  kl_coef={kl_coef:.2f}  '
                  f'nll={epoch_nll/n:.4f}  kl={epoch_kl/n:.4f}  '
                  f'val_mse={val_mse:.4f}  elapsed={time.time()-t_start:.0f}s', flush=True)

    val_mse = evaluate(encoder, decoder, val_loader)
    latents, periods, probe = linear_probe_period(encoder, val_loader.dataset)
    dists, stds, dist_std_corr = uncertainty_vs_distance_check(encoder, decoder, val_loader.dataset)
    print(f'\nFinal val_mse={val_mse:.4f}')
    print('Linear probe (VAE posterior mean -> period):', probe)
    print(f'corr(distance to nearest observation, predicted std) = {dist_std_corr:.3f}')

    # -- plot: reconstruction snapshots over training --
    fig, axes = plt.subplots(len(SNAPSHOT_SERIES_IDX), 1, figsize=(8, 4.5 * len(SNAPSHOT_SERIES_IDX)))
    cmap = plt.cm.viridis
    all_ep_keys = sorted(all_snapshots.keys())
    for row, idx in enumerate(SNAPSHOT_SERIES_IDX):
        ax = axes[row] if len(SNAPSHOT_SERIES_IDX) > 1 else axes
        x_np, y_np, period_val = snapshot_series[idx]
        ax.scatter(x_np, y_np, c='k', s=25, zorder=5, label='observed')
        for i, ep in enumerate(all_ep_keys):
            mean_snap, std_snap = all_snapshots[ep][idx]
            color = cmap(i / (len(all_ep_keys) - 1))
            label = 'init' if ep == -1 else f'epoch {ep}'
            ax.plot(dense_t[0].numpy() * T_MAX, mean_snap, color=color, alpha=0.85,
                    linewidth=1.2, label=label)
        ax.set_title(f'series (val idx {idx}): period={period_val:.2f}')
        ax.set_xlabel('time')
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle('Reconstruction mean over training (color = training progress)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'training_snapshots.png'), dpi=120)
    plt.close(fig)

    # -- plot: final reconstruction with uncertainty band --
    fig, axes = plt.subplots(len(SNAPSHOT_SERIES_IDX), 1, figsize=(8, 3 * len(SNAPSHOT_SERIES_IDX)))
    for row, idx in enumerate(SNAPSHOT_SERIES_IDX):
        ax = axes[row] if len(SNAPSHOT_SERIES_IDX) > 1 else axes
        x_np, y_np, period_val = snapshot_series[idx]
        mean_final, std_final = all_snapshots[EPOCHS - 1][idx]
        dense_x = dense_t[0].numpy() * T_MAX
        ax.scatter(x_np, y_np, c='k', s=25, zorder=5, label='observed')
        ax.plot(dense_x, mean_final, color='tab:blue', linewidth=1.5, label='predicted mean')
        ax.fill_between(dense_x, mean_final - 2 * std_final, mean_final + 2 * std_final,
                         color='tab:blue', alpha=0.2, label='mean $\\pm$ 2 std')
        true_curve_t = np.linspace(x_np.min(), x_np.max(), 300)
        ax.plot(true_curve_t, np.sin(true_curve_t / period_val), color='gray', linestyle='--',
                linewidth=1, alpha=0.7, label='true noiseless sin(x/period)')
        ax.set_title(f'series (val idx {idx}): period={period_val:.2f}')
        ax.set_xlabel('time')
        ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'final_uncertainty.png'), dpi=120)
    plt.close(fig)

    # -- plot: uncertainty vs distance to nearest observation --
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dists, stds, s=3, alpha=0.15)
    ax.set_xlabel('distance to nearest real observation')
    ax.set_ylabel('predicted std')
    ax.set_title(f'corr={dist_std_corr:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'uncertainty_vs_distance.png'), dpi=120)
    plt.close(fig)

    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
               os.path.join(OUTDIR, 'mtand_vae_uncertainty.pt'))
    np.savez(os.path.join(OUTDIR, 'vae_uncertainty_latents.npz'), latents=latents, periods=periods)
    with open(os.path.join(OUTDIR, 'vae_uncertainty_metrics.txt'), 'w') as f:
        f.write(f'val_mse={val_mse:.4f}\nprobe={probe}\ndist_std_corr={dist_std_corr:.4f}\n')


if __name__ == '__main__':
    main()
