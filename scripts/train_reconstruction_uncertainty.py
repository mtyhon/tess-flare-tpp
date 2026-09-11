"""Phase 1 (unsupervised, with uncertainty): mTAND autoencoder trained
with a heteroscedastic Gaussian decoder instead of a point-estimate MSE
decoder, for Neural-Process-style predictive intervals.

Plain mTAND (reml-lab/mTAN) doesn't do this: its VAE variant only puts a
distribution over the encoder's latent `z`, not over decoder predictions.
Here the decoder emits (mean, log-variance) per query time, trained with
Gaussian NLL, so we can plot confidence bands and check whether predicted
uncertainty rises in the gaps -- which it should, if calibrated.
"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset, collate_toy_sine  # noqa: E402
from src.mtand.model import (MTANReconEncoder, MTANReconDecoderProbabilistic,  # noqa: E402
                              MTANAutoencoderProbabilistic)

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 32
LATENT_DIM = 8
BATCH_SIZE = 64
EPOCHS = 60
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
    encoder = MTANReconEncoder(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                learn_emb=True)
    decoder = MTANReconDecoderProbabilistic(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                             nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                             learn_emb=True)
    return MTANAutoencoderProbabilistic(encoder, decoder)


def masked_gaussian_nll(mean, logvar, y, mask):
    nll = 0.5 * (logvar + (y - mean) ** 2 / torch.exp(logvar) + math.log(2 * math.pi))
    return (nll * mask).sum() / mask.sum()


def evaluate(model, loader):
    model.eval()
    total_nll, total_sq_err, total_mask = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            mean, logvar, z, enc_attn, dec_attn = model(x, t)
            total_nll += masked_gaussian_nll(mean, logvar, y, mask).item() * mask.sum().item()
            total_sq_err += ((mean - y) ** 2 * mask).sum().item()
            total_mask += mask.sum().item()
    return total_nll / total_mask, total_sq_err / total_mask


def uncertainty_vs_distance_check(model, val_ds, n_series=200):
    """For dense query points, does predicted std rise with distance to the
    nearest real observation? Well-calibrated predictive uncertainty
    should increase in the gaps."""
    model.eval()
    dense_t = torch.linspace(0, 1, 100).unsqueeze(0)
    dists, stds = [], []
    with torch.no_grad():
        for i in range(n_series):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            z, _ = model.encoder(xb, tb)
            dmean, dlogvar, _ = model.decoder(z, dense_t)
            std = torch.exp(0.5 * dlogvar)[0].numpy()

            obs_t = x.numpy()
            query_t = dense_t[0].numpy() * T_MAX
            nearest_dist = np.min(np.abs(query_t[:, None] - obs_t[None, :]), axis=1)
            dists.append(nearest_dist)
            stds.append(std)
    dists = np.concatenate(dists)
    stds = np.concatenate(stds)
    corr = float(np.corrcoef(dists, stds)[0, 1])
    return dists, stds, corr


def uncertainty_plots(model, val_ds, n_examples=4):
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(7, 2.8 * n_examples))
    dense_t = torch.linspace(0, 1, 300).unsqueeze(0)
    with torch.no_grad():
        for i in range(n_examples):
            x, y, period = val_ds[i]
            z, _ = model.encoder(*collate_toy_sine([(x, y, period)], t_max=T_MAX)[:2])
            dmean, dlogvar, _ = model.decoder(z, dense_t)
            dstd = torch.exp(0.5 * dlogvar)[0].numpy()
            dmean = dmean[0].numpy()
            dense_x = dense_t[0].numpy() * T_MAX

            ax = axes[i]
            ax.scatter(x.numpy(), y.numpy(), c='k', s=20, zorder=3, label='observed (noisy)')
            ax.plot(dense_x, dmean, color='tab:blue', linewidth=1.5, label='predicted mean')
            ax.fill_between(dense_x, dmean - 2 * dstd, dmean + 2 * dstd, color='tab:blue',
                             alpha=0.2, label='mean $\\pm$ 2 std')
            true_curve_t = np.linspace(x.numpy().min(), x.numpy().max(), 300)
            ax.plot(true_curve_t, np.sin(true_curve_t / period.item()), color='gray',
                    linestyle='--', linewidth=1, alpha=0.7, label='true noiseless sin(x/period)')
            ax.set_title(f'series {i}: period={period.item():.2f}')
            ax.set_xlabel('time')
            ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'uncertainty_examples.png'), dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, n = 0.0, 0
        for x, t, period in train_loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            opt.zero_grad()
            mean, logvar, z, enc_attn, dec_attn = model(x, t)
            loss = masked_gaussian_nll(mean, logvar, y, mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_nll = epoch_loss / n
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_nll, val_mse = evaluate(model, val_loader)
            print(f'epoch {epoch:3d}  train_nll={train_nll:.4f}  '
                  f'val_nll={val_nll:.4f}  val_mse(mean)={val_mse:.4f}')

    val_nll, val_mse = evaluate(model, val_loader)
    print(f'\nFinal: val_nll={val_nll:.4f}  val_mse(mean)={val_mse:.4f}')

    dists, stds, corr = uncertainty_vs_distance_check(model, val_loader.dataset)
    print(f'corr(distance to nearest observation, predicted std) = {corr:.3f}')

    uncertainty_plots(model, val_loader.dataset)

    torch.save(model.state_dict(), os.path.join(OUTDIR, 'mtand_autoencoder_uncertainty.pt'))
    with open(os.path.join(OUTDIR, 'uncertainty_metrics.txt'), 'w') as f:
        f.write(f'val_nll={val_nll:.4f}\nval_mse={val_mse:.4f}\n'
                f'corr_dist_std={corr:.4f}\n')


if __name__ == '__main__':
    main()
