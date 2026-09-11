"""Phase 1 (masked reconstruction): deterministic mTAND encoder +
heteroscedastic NLL decoder, trained with a genuine context/target split.

Every prior reconstruction model in this project (the plain autoencoder,
the VAE) let the encoder see the *same* points the decoder had to
reconstruct -- the model could get away with a largely trivial
copy-through-attention shortcut. Here, for each series, a random subset of
its real observations is held out entirely from the encoder (masked to
look like padding); the decoder must predict values there from context
alone, scored with Gaussian NLL only on the held-out points. This is the
actual mechanism the original mTAND paper uses for interpolation, and a
much stronger self-supervised signal than plain autoencoding.

No VAE here (no KL term) -- just a deterministic encoder, per the request
to compare against the VAE variant.
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
from src.mtand.model import MTANReconEncoder, MTANReconDecoderProbabilistic  # noqa: E402

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
MASK_RATIO = 0.3  # fraction of each series' real points held out as targets
MIN_CONTEXT = 4    # never mask a series down to fewer than this many context pts
MIN_TARGET = 1


def make_loaders():
    full = ToySineDataset(n_series=5000, seed=SEED, t_max=T_MAX)
    train, val = torch.utils.data.random_split(
        full, [4500, 500], generator=torch.Generator().manual_seed(SEED))

    def collate(batch):
        return collate_toy_sine(batch, t_max=T_MAX)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def split_context_target(mask, mask_ratio, rng, min_context=MIN_CONTEXT, min_target=MIN_TARGET):
    """mask: (batch, seq_len) 1=real, 0=padding. Returns (context_mask,
    target_mask), both (batch, seq_len), disjoint subsets of `mask`."""
    bsz, seq_len = mask.shape
    context_mask = torch.zeros_like(mask)
    target_mask = torch.zeros_like(mask)
    for b in range(bsz):
        real_idx = torch.nonzero(mask[b], as_tuple=True)[0].numpy()
        n_real = len(real_idx)
        n_target = max(min_target, int(round(n_real * mask_ratio)))
        n_target = min(n_target, n_real - min_context) if n_real > min_context else 0
        n_target = max(n_target, 0)
        if n_target > 0:
            target_idx = rng.choice(real_idx, size=n_target, replace=False)
        else:
            target_idx = np.array([], dtype=int)
        target_set = set(target_idx.tolist())
        context_idx = np.array([i for i in real_idx if i not in target_set])
        context_mask[b, context_idx] = 1.0
        if len(target_idx) > 0:
            target_mask[b, target_idx] = 1.0
    return context_mask, target_mask


def build_model():
    query = torch.linspace(0, 1, NUM_REF)
    encoder = MTANReconEncoder(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                learn_emb=True)
    decoder = MTANReconDecoderProbabilistic(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                             nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                             learn_emb=True)
    return encoder, decoder


def masked_gaussian_nll(mean, logvar, y, mask):
    nll = 0.5 * (logvar + (y - mean) ** 2 / torch.exp(logvar) + math.log(2 * math.pi))
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def forward_pass(encoder, decoder, x, t, context_mask, target_mask):
    """Encoder only sees context points (target treated as padding);
    decoder is queried at all real timestamps but scored only on targets."""
    y = x[:, :, 0]
    context_x = torch.stack([y, context_mask], dim=-1)
    z, enc_attn = encoder(context_x, t)
    mean, logvar, dec_attn = decoder(z, t)
    return mean, logvar, z, enc_attn, dec_attn


def evaluate(encoder, decoder, loader, rng):
    encoder.eval(); decoder.eval()
    total_sq_err, total_n = 0.0, 0.0
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            context_mask, target_mask = split_context_target(mask, MASK_RATIO, rng)
            mean, logvar, z, _, _ = forward_pass(encoder, decoder, x, t, context_mask, target_mask)
            total_sq_err += ((mean - y) ** 2 * target_mask).sum().item()
            total_n += target_mask.sum().item()
    return total_sq_err / max(total_n, 1)


def linear_probe_period(encoder, val_ds, rng_split, rng_mask):
    from sklearn.linear_model import LinearRegression
    encoder.eval()
    latents, periods = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            x, y, period = val_ds[i]
            xb, tb, _ = collate_toy_sine([(x, y, period)], t_max=T_MAX)
            mask = xb[:, :, 1]
            context_mask, _ = split_context_target(mask, MASK_RATIO, rng_mask)
            context_x = torch.stack([xb[:, :, 0], context_mask], dim=-1)
            z, _ = encoder(context_x, tb)
            latents.append(z[0].mean(dim=0).numpy())
            periods.append(period.item())
    latents = np.stack(latents); periods = np.array(periods)
    n = len(periods)
    idx = rng_split.permutation(n)
    fit_idx, test_idx = idx[:n // 2], idx[n // 2:]
    reg = LinearRegression().fit(latents[fit_idx], periods[fit_idx])
    pred = reg.predict(latents[test_idx])
    corr = float(np.corrcoef(pred, periods[test_idx])[0, 1])
    r2 = float(reg.score(latents[test_idx], periods[test_idx]))
    return latents, periods, {'probe_corr': corr, 'probe_r2': r2}


def main():
    torch.manual_seed(SEED)
    mask_rng = np.random.default_rng(SEED)
    train_loader, val_loader = make_loaders()
    encoder, decoder = build_model()
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    for epoch in range(EPOCHS):
        encoder.train(); decoder.train()
        epoch_nll, n = 0.0, 0
        for x, t, period in train_loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            context_mask, target_mask = split_context_target(mask, MASK_RATIO, mask_rng)
            opt.zero_grad()
            mean, logvar, z, _, _ = forward_pass(encoder, decoder, x, t, context_mask, target_mask)
            loss = masked_gaussian_nll(mean, logvar, y, target_mask)
            loss.backward()
            opt.step()
            epoch_nll += loss.item() * x.size(0)
            n += x.size(0)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            val_mse = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1))
            print(f'epoch {epoch:3d}  train_nll={epoch_nll/n:.4f}  '
                  f'val_mse(held-out targets)={val_mse:.4f}', flush=True)

    val_mse = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1))
    latents, periods, probe = linear_probe_period(
        encoder, val_loader.dataset, np.random.default_rng(SEED), np.random.default_rng(SEED + 2))
    print(f'\nFinal val_mse(held-out targets)={val_mse:.4f}')
    print('Linear probe (masked-recon latent -> period):', probe)

    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
               os.path.join(OUTDIR, 'mtand_masked_recon.pt'))
    np.savez(os.path.join(OUTDIR, 'masked_recon_latents.npz'), latents=latents, periods=periods)
    with open(os.path.join(OUTDIR, 'masked_recon_metrics.txt'), 'w') as f:
        f.write(f'val_mse_heldout={val_mse:.4f}\nprobe={probe}\n')


if __name__ == '__main__':
    main()
