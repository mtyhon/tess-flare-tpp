"""Masked reconstruction with LS-conditioned attention (MTANReconEncoderLS /
MTANReconDecoderLS): the model conditions its own attention on a spectral
hint, per the request to try this instead of just concatenating an LS
feature onto the final latent. Query/key time embeddings get two extra
dimensions -- sin(w_LS*t), cos(w_LS*t) at each series' own Lomb-Scargle
angular frequency -- so attention itself can be built around that period.

Motivated directly by the LS-vs-mTAND head-to-head (compare_ls_vs_mtand.py):
LS crushes mTAND on the aliased/short-period regime (corr 0.933 vs 0.407)
while mTAND wins on the resolvable regime -- complementary failure modes,
suggesting real value in giving the encoder access to the LS hint.

w_LS is computed every batch from context points only (never target
points -- same fairness convention as lomb_scargle_baseline.py, so there's
no leakage through the hint). A reduced frequency grid (N_GRID=300 vs. the
offline baseline's 5000) keeps this affordable inside the training loop --
the hint's job is just to point attention roughly at the right period, not
to be a maximally precise periodogram.

Same schedule/budget as train_masked_reconstruction_harder.py (600 epochs,
cosine LR, embed_time=128) for a fair comparison against the R^2=0.835
plain baseline.

Resumable: run repeatedly, bounded by wall-clock time budget per call.
"""
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import lombscargle
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset, collate_toy_sine  # noqa: E402
from src.mtand.model import MTANReconEncoderLS, MTANReconDecoderLS  # noqa: E402
from scripts.train_masked_reconstruction import split_context_target, masked_gaussian_nll, MASK_RATIO  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
CKPT_PATH = os.path.join(OUTDIR, 'mtand_masked_recon_ls.pt')

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 128
LATENT_DIM = 64
BATCH_SIZE = 64
BASE_LR = 1e-3
MIN_LR = 1e-5
SEED = 0
TOTAL_EPOCHS = 600

PERIOD_MIN, PERIOD_MAX, N_GRID = 0.01, 20.0, 300
PERIOD_GRID = np.logspace(np.log10(PERIOD_MIN), np.log10(PERIOD_MAX), N_GRID)
ANGULAR_FREQS = 2 * np.pi / PERIOD_GRID
FALLBACK_OMEGA = float(ANGULAR_FREQS[len(ANGULAR_FREQS) // 2])


def compute_ls_omega(t, y, context_mask):
    """t, y, context_mask: (batch, seq_len) torch tensors. Returns a
    (batch,) float32 tensor of each series' LS-estimated angular
    frequency, computed from context points only."""
    t_np = t.detach().numpy().astype(np.float64)
    y_np = y.detach().numpy().astype(np.float64)
    mask_np = context_mask.detach().numpy().astype(bool)
    batch = t_np.shape[0]
    omegas = np.empty(batch, dtype=np.float64)
    for b in range(batch):
        idx = mask_np[b]
        if idx.sum() < 3:
            omegas[b] = FALLBACK_OMEGA
            continue
        tb = t_np[b, idx]
        yb = y_np[b, idx] - y_np[b, idx].mean()
        power = lombscargle(tb, yb, ANGULAR_FREQS, normalize=True)
        omegas[b] = ANGULAR_FREQS[np.argmax(power)]
    return torch.from_numpy(omegas).float()


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
    encoder = MTANReconEncoderLS(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                  nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                  learn_emb=True)
    decoder = MTANReconDecoderLS(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                  nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                  learn_emb=True)
    return encoder, decoder


def cosine_lr(epoch, total_epochs, base_lr=BASE_LR, min_lr=MIN_LR):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


def forward_pass_ls(encoder, decoder, x, t, context_mask, target_mask):
    y = x[:, :, 0]
    ls_omega = compute_ls_omega(t, y, context_mask)
    context_x = torch.stack([y, context_mask], dim=-1)
    z, enc_attn = encoder(context_x, t, ls_omega)
    mean, logvar, dec_attn = decoder(z, t, ls_omega)
    return mean, logvar, z, enc_attn, dec_attn


def evaluate(encoder, decoder, loader, rng):
    encoder.eval(); decoder.eval()
    total_sq_err, total_n = 0.0, 0.0
    with torch.no_grad():
        for x, t, period in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            context_mask, target_mask = split_context_target(mask, MASK_RATIO, rng)
            mean, logvar, z, _, _ = forward_pass_ls(encoder, decoder, x, t, context_mask, target_mask)
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
            ls_omega = compute_ls_omega(tb, xb[:, :, 0], context_mask)
            context_x = torch.stack([xb[:, :, 0], context_mask], dim=-1)
            z, _ = encoder(context_x, tb, ls_omega)
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


def main(time_budget_s=520):
    start_time = time.time()
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()
    encoder, decoder = build_model()
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=BASE_LR)

    start_epoch = 0
    history = []
    mask_rng_state = None
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', [])
        mask_rng_state = ckpt.get('mask_rng_state')
        print(f'Resumed from checkpoint at epoch {start_epoch}/{TOTAL_EPOCHS}', flush=True)
    else:
        print('No checkpoint found, starting fresh', flush=True)

    mask_rng = np.random.default_rng(SEED)
    if mask_rng_state is not None:
        mask_rng.bit_generator.state = mask_rng_state

    epoch = start_epoch
    while epoch < TOTAL_EPOCHS:
        if time.time() - start_time > time_budget_s:
            print(f'Time budget reached at epoch {epoch}, checkpointing and stopping', flush=True)
            break

        lr = cosine_lr(epoch, TOTAL_EPOCHS)
        for g in opt.param_groups:
            g['lr'] = lr

        encoder.train(); decoder.train()
        epoch_nll, n = 0.0, 0
        for x, t, period in train_loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            context_mask, target_mask = split_context_target(mask, MASK_RATIO, mask_rng)
            opt.zero_grad()
            mean, logvar, z, _, _ = forward_pass_ls(encoder, decoder, x, t, context_mask, target_mask)
            loss = masked_gaussian_nll(mean, logvar, y, target_mask)
            loss.backward()
            opt.step()
            epoch_nll += loss.item() * x.size(0)
            n += x.size(0)

        if epoch % 20 == 0 or epoch == TOTAL_EPOCHS - 1:
            val_mse = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1))
            history.append((epoch, epoch_nll / n, val_mse, lr))
            print(f'epoch {epoch:4d}/{TOTAL_EPOCHS}  lr={lr:.2e}  train_nll={epoch_nll/n:.4f}  '
                  f'val_mse={val_mse:.4f}  elapsed={time.time()-start_time:.0f}s', flush=True)
        epoch += 1

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch - 1,
        'history': history,
        'mask_rng_state': mask_rng.bit_generator.state,
    }, CKPT_PATH)
    print(f'Checkpoint saved at epoch {epoch-1}/{TOTAL_EPOCHS}', flush=True)

    if epoch >= TOTAL_EPOCHS:
        val_mse = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1))
        latents, periods, probe = linear_probe_period(
            encoder, val_loader.dataset, np.random.default_rng(SEED), np.random.default_rng(SEED + 2))
        print(f'\n=== TRAINING COMPLETE === final val_mse={val_mse:.4f}', flush=True)
        print(f'Linear probe (LS-conditioned latent -> period): {probe}', flush=True)
        np.savez(os.path.join(OUTDIR, 'masked_recon_ls_latents.npz'), latents=latents, periods=periods)
        with open(os.path.join(OUTDIR, 'masked_recon_ls_metrics.txt'), 'w') as f:
            f.write(f'final_val_mse={val_mse:.4f}\nprobe={probe}\nhistory={history}\n')


if __name__ == '__main__':
    main()
