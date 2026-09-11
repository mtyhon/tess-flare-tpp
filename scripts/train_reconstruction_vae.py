"""Phase 1 (unsupervised, VAE variant): mTAND encoder-decoder trained as
an actual VAE (reconstruction likelihood + KL-to-standard-normal), rather
than the plain deterministic autoencoder in train_reconstruction.py.

Motivated by a comparison notebook the user shared, which used the
official reml-lab/mTAN enc_mtan_rnn/dec_mtan_rnn as a proper VAE
(latent_dim=64, embed_time=128, KL regularization, ~5000 training
iterations) and got a visibly more separable latent space by `period` than
our plain deterministic AE (latent_dim=8, no KL). This script tests
whether the VAE's KL regularization (which pushes the aggregate posterior
toward an isotropic Gaussian -- known to produce smoother, more
separable latent geometry than an unregularized bottleneck) is the
active ingredient, independent of matching their exact scale/iteration
count.
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
from src.mtand.model import MTANReconEncoderVAE, MTANReconDecoder  # noqa: E402

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
NOISE_STD = 0.15  # matches the true data-generating noise, unlike the
                   # shared notebook's std=0.01 (much lower observation
                   # noise, which itself makes period recovery easier)
KL_WARMUP_EPOCHS = 10


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
    encoder = MTANReconEncoderVAE(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                   nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                   learn_emb=True)
    decoder = MTANReconDecoder(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                learn_emb=True)
    return encoder, decoder


def elbo_loss(recon, y, mask, mean, logvar, kl_coef, noise_std=NOISE_STD):
    noise_var = noise_std ** 2
    nll = 0.5 * (math.log(2 * math.pi * noise_var) + (y - recon) ** 2 / noise_var)
    nll = (nll * mask).sum() / mask.sum()
    kl = (-0.5 * (1 + logvar - mean.pow(2) - logvar.exp())).sum(-1).mean()
    return nll + kl_coef * kl, nll, kl


def evaluate(encoder, decoder, loader):
    encoder.eval(); decoder.eval()
    total_sq_err, total_mask = 0.0, 0.0
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            mean, logvar, _ = encoder(x, t)
            recon, _ = decoder(mean, t)  # eval: use posterior mean, no sampling
            total_sq_err += ((recon - y) ** 2 * mask).sum().item()
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
            mean, logvar, _ = encoder(xb, tb)
            latents.append(mean[0].mean(dim=0).numpy())
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


def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()
    encoder, decoder = build_model()
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    t_start = time.time()
    for epoch in range(EPOCHS):
        encoder.train(); decoder.train()
        kl_coef = min(1.0, epoch / KL_WARMUP_EPOCHS)
        epoch_loss, epoch_nll, epoch_kl, n = 0.0, 0.0, 0.0, 0
        for x, t, period in train_loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            opt.zero_grad()
            mean, logvar, _ = encoder(x, t)
            z = encoder.reparameterize(mean, logvar)
            recon, _ = decoder(z, t)
            loss, nll, kl = elbo_loss(recon, y, mask, mean, logvar, kl_coef)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_nll += nll.item() * x.size(0)
            epoch_kl += kl.item() * x.size(0)
            n += x.size(0)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            val_mse = evaluate(encoder, decoder, val_loader)
            print(f'epoch {epoch:3d}  kl_coef={kl_coef:.2f}  '
                  f'nll={epoch_nll/n:.4f}  kl={epoch_kl/n:.4f}  '
                  f'val_mse={val_mse:.4f}  elapsed={time.time()-t_start:.0f}s')

    val_mse = evaluate(encoder, decoder, val_loader)
    latents, periods, probe = linear_probe_period(encoder, val_loader.dataset)
    print(f'\nFinal val_mse={val_mse:.4f}')
    print('Linear probe (VAE posterior mean -> period):', probe)

    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
               os.path.join(OUTDIR, 'mtand_vae.pt'))
    np.savez(os.path.join(OUTDIR, 'vae_latents.npz'), latents=latents, periods=periods)
    with open(os.path.join(OUTDIR, 'vae_metrics.txt'), 'w') as f:
        f.write(f'val_mse={val_mse:.4f}\nprobe={probe}\n')


if __name__ == '__main__':
    main()
