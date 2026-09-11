"""Masked reconstruction, same setup as train_masked_reconstruction_harder.py
(600-epoch target, cosine LR, embed_time=128, num_heads=1), but now the
model's learned time embedding defaults to log-spaced frequency
initialization (src/mtand/model.py) instead of nn.Linear's random default.
Isolated ablation: everything else matched to the harder-training run, so
any difference in probe R^2 for period is attributable to the frequency
init specifically -- the direct test of whether spectral bias was
limiting high-frequency (short-period) recovery.

Resumable: run this script repeatedly (each call bounded by a wall-clock
time budget, not an epoch count) and it picks up from the last checkpoint.
"""
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_sine import ToySineDataset, collate_toy_sine  # noqa: E402
from src.mtand.model import MTANReconEncoder, MTANReconDecoderProbabilistic  # noqa: E402
from scripts.train_masked_reconstruction import (split_context_target, masked_gaussian_nll,  # noqa: E402
                                                   forward_pass, MASK_RATIO)

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
CKPT_PATH = os.path.join(OUTDIR, 'mtand_masked_recon_freqinit.pt')

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 128
LATENT_DIM = 64
BATCH_SIZE = 64
BASE_LR = 1e-3
MIN_LR = 1e-5
SEED = 0
TOTAL_EPOCHS = 300  # matched to where the harder-training pure-sine run was evaluated (epoch 263)


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
                                learn_emb=True, freq_init='log_spaced', freeze_freq=True)
    decoder = MTANReconDecoderProbabilistic(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                             nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=1,
                                             learn_emb=True, freq_init='log_spaced', freeze_freq=True)
    return encoder, decoder


def cosine_lr(epoch, total_epochs, base_lr=BASE_LR, min_lr=MIN_LR):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


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
            mean, logvar, z, _, _ = forward_pass(encoder, decoder, x, t, context_mask, target_mask)
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
        print(f'\n=== TRAINING COMPLETE === final val_mse={val_mse:.4f}', flush=True)
        with open(os.path.join(OUTDIR, 'masked_recon_freqinit_metrics.txt'), 'w') as f:
            f.write(f'final_val_mse={val_mse:.4f}\nhistory={history}\n')


if __name__ == '__main__':
    main()
