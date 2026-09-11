"""Masked reconstruction with all three requested fixes together:
cos-paired time embedding (now the default in src/mtand/model.py),
more capacity (embed_time=128, num_heads=4, up from 1), and trained on
the multi-harmonic (non-pure-sine) dataset instead of pure sine.

Architecture-compatible with train_masked_reconstruction_harder.py's
approach (resumable via checkpoint + wall-clock time budget), but this is
a fresh model -- the embedding/capacity changes alter parameter shapes, so
the epoch-263 pure-sine checkpoint cannot be reused here.
"""
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.toy_complex_periodic import ToyComplexPeriodicDataset, collate_complex_periodic  # noqa: E402
from src.mtand.model import MTANReconEncoder, MTANReconDecoderProbabilistic  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
CKPT_PATH = os.path.join(OUTDIR, 'mtand_complex_masked_recon.pt')

T_MAX = 20.0
NUM_REF = 32
NHIDDEN = 32
EMBED_TIME = 128
NUM_HEADS = 4
LATENT_DIM = 64
BATCH_SIZE = 64
BASE_LR = 1e-3
MIN_LR = 1e-5
SEED = 0
TOTAL_EPOCHS = 300
MASK_RATIO = 0.3
MIN_CONTEXT = 4
MIN_TARGET = 1


def make_loaders():
    full = ToyComplexPeriodicDataset(n_series=5000, seed=SEED, t_max=T_MAX)
    train, val = torch.utils.data.random_split(
        full, [4500, 500], generator=torch.Generator().manual_seed(SEED))

    def collate(batch):
        return collate_complex_periodic(batch, t_max=T_MAX)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def build_model():
    query = torch.linspace(0, 1, NUM_REF)
    encoder = MTANReconEncoder(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=NUM_HEADS,
                                learn_emb=True)
    decoder = MTANReconDecoderProbabilistic(input_dim=1, query=query, latent_dim=LATENT_DIM,
                                             nhidden=NHIDDEN, embed_time=EMBED_TIME, num_heads=NUM_HEADS,
                                             learn_emb=True)
    return encoder, decoder


def split_context_target(mask, mask_ratio, rng, min_context=MIN_CONTEXT, min_target=MIN_TARGET):
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


def masked_gaussian_nll(mean, logvar, y, mask):
    nll = 0.5 * (logvar + (y - mean) ** 2 / torch.exp(logvar) + math.log(2 * math.pi))
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def forward_pass(encoder, decoder, x, t, context_mask, target_mask):
    y = x[:, :, 0]
    context_x = torch.stack([y, context_mask], dim=-1)
    z, enc_attn = encoder(context_x, t)
    mean, logvar, dec_attn = decoder(z, t)
    return mean, logvar, z, enc_attn, dec_attn


def cosine_lr(epoch, total_epochs, base_lr=BASE_LR, min_lr=MIN_LR):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


def evaluate(encoder, decoder, loader, rng, split_by_period=None):
    encoder.eval(); decoder.eval()
    total_sq_err, total_n = 0.0, 0.0
    resolvable_sq_err, resolvable_n = 0.0, 0.0
    aliased_sq_err, aliased_n = 0.0, 0.0
    with torch.no_grad():
        for x, t, period, n_harm in loader:
            y, mask = x[:, :, 0], x[:, :, 1]
            context_mask, target_mask = split_context_target(mask, MASK_RATIO, rng)
            mean, logvar, z, _, _ = forward_pass(encoder, decoder, x, t, context_mask, target_mask)
            sq_err = (mean - y) ** 2 * target_mask
            total_sq_err += sq_err.sum().item()
            total_n += target_mask.sum().item()
            if split_by_period is not None:
                resolvable = period > split_by_period
                res_mask = target_mask * resolvable.unsqueeze(1).float()
                ali_mask = target_mask * (~resolvable).unsqueeze(1).float()
                resolvable_sq_err += (sq_err * resolvable.unsqueeze(1).float()).sum().item()
                resolvable_n += res_mask.sum().item()
                aliased_sq_err += (sq_err * (~resolvable).unsqueeze(1).float()).sum().item()
                aliased_n += ali_mask.sum().item()
    result = {'mse': total_sq_err / max(total_n, 1)}
    if split_by_period is not None:
        result['mse_resolvable'] = resolvable_sq_err / max(resolvable_n, 1)
        result['mse_aliased'] = aliased_sq_err / max(aliased_n, 1)
    return result


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
        for x, t, period, n_harm in train_loader:
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
            metrics = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1), split_by_period=1.0)
            history.append((epoch, epoch_nll / n, metrics, lr))
            print(f'epoch {epoch:4d}/{TOTAL_EPOCHS}  lr={lr:.2e}  train_nll={epoch_nll/n:.4f}  '
                  f'val_mse={metrics["mse"]:.4f}  resolvable={metrics["mse_resolvable"]:.4f}  '
                  f'aliased={metrics["mse_aliased"]:.4f}  elapsed={time.time()-start_time:.0f}s', flush=True)
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
        metrics = evaluate(encoder, decoder, val_loader, np.random.default_rng(SEED + 1), split_by_period=1.0)
        print(f'\n=== TRAINING COMPLETE === {metrics}', flush=True)
        with open(os.path.join(OUTDIR, 'complex_masked_recon_metrics.txt'), 'w') as f:
            f.write(f'final_metrics={metrics}\nhistory={history}\n')


if __name__ == '__main__':
    main()
