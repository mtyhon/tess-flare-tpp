"""Phase 1 (unsupervised variant): mTAND autoencoder, reconstruction loss.

Same toy irregular-sine data and backbone family as train_baseline.py, but
no labels anywhere in the training loop -- the encoder is trained purely to
support reconstructing the series' own observed values through the
decoder's time-conditioned cross-attention. `period` is only used
afterwards, as a post-hoc linear probe, to check whether an unsupervised
objective happens to make the known factor linearly recoverable -- which
matters for Phase 2, since DIOSC's contrastive loss needs a representation
that already carries usable factor information to disentangle.
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
from src.mtand.model import MTANReconEncoder, MTANReconDecoder, MTANAutoencoder  # noqa: E402

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
    decoder = MTANReconDecoder(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                learn_emb=True)
    return MTANAutoencoder(encoder, decoder)


def masked_mse(recon, y, mask):
    return ((recon - y) ** 2 * mask).sum() / mask.sum()


def evaluate(model, loader):
    model.eval()
    total_loss, total_mask = 0.0, 0.0
    recon_all, target_all = [], []
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            recon, z, enc_attn, dec_attn = model(x, t)
            total_loss += ((recon - y) ** 2 * mask).sum().item()
            total_mask += mask.sum().item()
            recon_all.append(recon[mask.bool()])
            target_all.append(y[mask.bool()])
    mse = total_loss / total_mask
    recon_all = torch.cat(recon_all).numpy()
    target_all = torch.cat(target_all).numpy()
    corr = float(np.corrcoef(recon_all, target_all)[0, 1])
    return mse, corr


def reconstruction_plots(model, val_ds, n_examples=4):
    """Show reconstruction at the observed points plus a dense query curve
    (querying timestamps the encoder never saw), against the true
    noiseless sin(x/period) for reference."""
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(7, 2.8 * n_examples))
    dense_t = torch.linspace(0, 1, 300).unsqueeze(0)
    with torch.no_grad():
        for i in range(n_examples):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            recon, z, enc_attn, dec_attn = model(xb, tb)
            dense_recon, _ = model.decoder(z, dense_t)

            ax = axes[i]
            ax.scatter(x.numpy(), y.numpy(), c='k', s=20, zorder=3, label='observed (noisy)')
            ax.scatter(x.numpy(), recon[0].numpy(), c='tab:red', s=20, marker='x', zorder=4,
                       label='reconstructed at observed times')
            ax.plot(dense_t[0].numpy() * T_MAX, dense_recon[0].numpy(), color='tab:red',
                    alpha=0.6, linewidth=1.5, label='decoded curve (dense query, unseen times)')
            true_curve_t = np.linspace(x.numpy().min(), x.numpy().max(), 300)
            ax.plot(true_curve_t, np.sin(true_curve_t / period.item()), color='gray',
                    linestyle='--', linewidth=1, alpha=0.7, label='true noiseless sin(x/period)')
            ax.set_title(f'series {i}: period={period.item():.2f}')
            ax.set_xlabel('time')
            ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'reconstruction_examples.png'), dpi=120)
    plt.close(fig)


def linear_probe_period(model, val_ds):
    """Does an unsupervised representation happen to make `period`
    linearly recoverable? Pool the (num_ref, latent_dim) latent sequence to
    a single vector per series, fit linear regression probe->period on
    half the validation set, evaluate on the other half. period is never
    used for training -- only for this diagnostic."""
    from sklearn.linear_model import LinearRegression

    model.eval()
    latents, periods = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            z, _ = model.encoder(xb, tb)
            latents.append(z[0].mean(dim=0).numpy())  # mean-pool over reference points
            periods.append(period.item())
    latents = np.stack(latents)
    periods = np.array(periods)

    n = len(periods)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]

    reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
    pred = reg.predict(latents[test_idx])
    corr = float(np.corrcoef(pred, periods[test_idx])[0, 1])
    r2 = float(reg.score(latents[test_idx], periods[test_idx]))
    return {'probe_corr': corr, 'probe_r2': r2, 'n_fit': len(fit_idx), 'n_test': len(test_idx)}


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
            recon, z, enc_attn, dec_attn = model(x, t)
            loss = masked_mse(recon, y, mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_mse = epoch_loss / n
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_mse, val_corr = evaluate(model, val_loader)
            print(f'epoch {epoch:3d}  train_mse={train_mse:.4f}  '
                  f'val_mse={val_mse:.4f}  val_corr={val_corr:.4f}')

    val_mse, val_corr = evaluate(model, val_loader)
    print(f'\nFinal: val_mse={val_mse:.4f}  val_corr(recon,true)={val_corr:.4f}')

    reconstruction_plots(model, val_loader.dataset)
    probe = linear_probe_period(model, val_loader.dataset)
    print('Linear probe (unsupervised latent -> period, never trained on period):', probe)

    torch.save(model.state_dict(), os.path.join(OUTDIR, 'mtand_autoencoder.pt'))
    with open(os.path.join(OUTDIR, 'reconstruction_metrics.txt'), 'w') as f:
        f.write(f'val_mse={val_mse:.4f}\nval_corr={val_corr:.4f}\n'
                f'linear_probe={probe}\n')


if __name__ == '__main__':
    main()
