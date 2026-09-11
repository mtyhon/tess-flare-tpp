"""mTAND encoder, adapted from the official reml-lab/mTAN implementation
(multiTimeAttention / enc_mtan_classif in their models.py).

Kept: the multi-time attention mechanism and learned time embedding verbatim.
Changed: the fixed 2-class classifier head is replaced with a generic head
(regression here), and attention weights are returned for inspection, since
Phase 1's goal is a re-usable representation to disentangle later, not a
one-off classifier.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_spaced_frequencies(n_freq, f_min, f_max):
    """Log-spaced frequencies (cycles per unit of normalized time) from
    f_min to f_max.

    Used to initialize the learned time embedding's frequency bank
    instead of leaving it at nn.Linear's default random init. Motivated
    by the "spectral bias" literature (Rahaman et al. 2019, "On the
    Spectral Bias of Neural Networks") -- networks systematically learn
    low-frequency functions more easily than high-frequency ones, so a
    randomly-initialized frequency bank has to *discover* short periods
    via gradient descent, fighting that bias directly. Explicit
    multi-scale Fourier features spanning low-to-high up front (Tancik et
    al., NeurIPS 2020, "Fourier Features Let Networks Learn High-Frequency
    Functions in Low-Dimensional Domains") sidesteps this by construction.
    Log-spacing (not linear) matches a period distribution like
    Gamma(1,1), which has heavy mass at both short and moderate periods
    spanning multiple orders of magnitude.
    """
    return torch.logspace(math.log10(f_min), math.log10(f_max), n_freq)


class MultiTimeAttention(nn.Module):
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([
            nn.Linear(embed_time, embed_time),
            nn.Linear(embed_time, embed_time),
            nn.Linear(input_dim * num_heads, nhidden),
        ])

    def attention(self, query, key, value, mask=None, dropout=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, attn = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x), attn


class MTANEncoder(nn.Module):
    """Encoder-only mTAND: fixed-length, reference-point representation.

    Given irregularly-sampled (value, mask) observations at arbitrary
    timestamps, attends from a fixed set of `query` reference times into the
    observed (`key`) times to produce a fixed-length re-representation,
    then a GRU pools that into a single embedding per series.
    """

    def __init__(self, input_dim, query, nhidden=16, embed_time=16,
                 num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        if learn_emb:
            # sin/cos quadrature pair per frequency, matching
            # fixed_time_embedding's convention -- a lone sin() term can't
            # represent arbitrary phase relationships as well as a paired
            # sin/cos basis does.
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            # freq_init='random' (default) is nn.Linear's ordinary random
            # init -- empirically the best-performing option so far
            # (probe R^2 0.835 vs 0.629-0.626 for log-spaced variants on
            # the masked-reconstruction pure-sine task), despite the
            # spectral-bias motivation for log-spaced init. 'log_spaced'
            # opts into that alternative explicitly; see freeze_freq to
            # control whether it stays trainable.
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))

        out, attn = self.att(query, key, x, mask)
        _, h = self.gru(out)
        return h.squeeze(0), attn  # (batch, nhidden), attention weights


class MTANReconEncoder(nn.Module):
    """mTAND encoder that keeps the full reference-point sequence rather
    than pooling to a single vector, adapted from reml-lab/mTAN's
    `enc_mtan_rnn` (the encoder half of their interpolation/VAE model).
    We drop the mean/logvar split -- this is a plain deterministic
    autoencoder, not a VAE; DIOSC's own variational framing is a Phase 2
    decision, not one to bake in here.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim))
        if learn_emb:
            # sin/cos quadrature pair per frequency, matching
            # fixed_time_embedding's convention -- a lone sin() term can't
            # represent arbitrary phase relationships as well as a paired
            # sin/cos basis does.
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            # freq_init='random' (default) is nn.Linear's ordinary random
            # init -- empirically the best-performing option so far
            # (probe R^2 0.835 vs 0.629-0.626 for log-spaced variants on
            # the masked-reconstruction pure-sine task), despite the
            # spectral-bias motivation for log-spaced init. 'log_spaced'
            # opts into that alternative explicitly; see freeze_freq to
            # control whether it stays trainable.
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))

        out, attn = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        z = self.hiddens_to_z(out)  # (batch, num_ref, latent_dim)
        return z, attn


class MTANReconEncoderVAE(nn.Module):
    """Same as MTANReconEncoder, but outputs (mean, log-variance) per
    reference point instead of a deterministic vector, matching the
    official reml-lab/mTAN `enc_mtan_rnn` exactly (we'd dropped this split
    for the plain autoencoder above). Meant to be trained with a
    reconstruction likelihood + KL-to-standard-normal (the actual VAE ELBO),
    not just reconstruction MSE -- the KL term regularizes the aggregate
    posterior toward an isotropic Gaussian, which tends to produce a much
    more separable latent geometry than an unregularized deterministic
    bottleneck. Use `reparameterize` to sample, or just take `mean` at
    eval time.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.latent_dim = latent_dim
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2))
        if learn_emb:
            # sin/cos quadrature pair per frequency, matching
            # fixed_time_embedding's convention -- a lone sin() term can't
            # represent arbitrary phase relationships as well as a paired
            # sin/cos basis does.
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            # freq_init='random' (default) is nn.Linear's ordinary random
            # init -- empirically the best-performing option so far
            # (probe R^2 0.835 vs 0.629-0.626 for log-spaced variants on
            # the masked-reconstruction pure-sine task), despite the
            # spectral-bias motivation for log-spaced init. 'log_spaced'
            # opts into that alternative explicitly; see freeze_freq to
            # control whether it stays trainable.
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))

        out, attn = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        mean, logvar = out[..., :self.latent_dim], out[..., self.latent_dim:]
        return mean, logvar, attn

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)


class MTANReconDecoder(nn.Module):
    """Cross-attends from arbitrary query timestamps back into the fixed
    reference-point latent sequence to reconstruct values there, letting us
    query any timestamp (not just the ones the encoder saw). Adapted from
    reml-lab/mTAN's `dec_mtan_rnn`.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.register_buffer('query', query)  # reference times, same grid as encoder
        self.att = MultiTimeAttention(2 * nhidden, 2 * nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim))
        if learn_emb:
            # sin/cos quadrature pair per frequency, matching
            # fixed_time_embedding's convention -- a lone sin() term can't
            # represent arbitrary phase relationships as well as a paired
            # sin/cos basis does.
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            # freq_init='random' (default) is nn.Linear's ordinary random
            # init -- empirically the best-performing option so far
            # (probe R^2 0.835 vs 0.629-0.626 for log-spaced variants on
            # the masked-reconstruction pure-sine task), despite the
            # spectral-bias motivation for log-spaced init. 'log_spaced'
            # opts into that alternative explicitly; see freeze_freq to
            # control whether it stays trainable.
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, z, time_steps):
        # z: (batch, num_ref, latent_dim); time_steps: (batch, seq_len) --
        # arbitrary timestamps to reconstruct at (need not equal `query`).
        out, _ = self.gru_rnn(z)  # (batch, num_ref, 2*nhidden)
        if self.learn_emb:
            query_emb = self.learn_time_embedding(time_steps)
            key_emb = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query_emb = self.fixed_time_embedding(time_steps)
            key_emb = self.fixed_time_embedding(self.query.unsqueeze(0))
        out, attn = self.att(query_emb, key_emb, out)
        recon = self.z_to_obs(out)
        return recon.squeeze(-1), attn


class MTANAutoencoder(nn.Module):
    """Unsupervised reconstruction: no labels anywhere in this loop."""

    def __init__(self, encoder: MTANReconEncoder, decoder: MTANReconDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, time_steps):
        z, enc_attn = self.encoder(x, time_steps)
        recon, dec_attn = self.decoder(z, time_steps)
        return recon, z, enc_attn, dec_attn


class MTANReconDecoderProbabilistic(nn.Module):
    """Same as MTANReconDecoder, but outputs a heteroscedastic Gaussian
    (mean, log-variance) at each query time instead of a point estimate,
    meant to be trained with Gaussian NLL.

    The official reml-lab/mTAN code doesn't have this: their VAE variant
    only puts a distribution over the *encoder's* latent `z` (mean/logvar
    sampled once per series), while the decoder itself is a deterministic
    point-estimate MLP -- same as our plain MTANReconDecoder. That's latent
    uncertainty, not the per-timestep predictive uncertainty Neural
    Processes are built around. This class adds that missing piece by
    doubling the decoder's output width and training with a likelihood
    loss instead of MSE.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * nhidden, 2 * nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim * 2))
        if learn_emb:
            # sin/cos quadrature pair per frequency, matching
            # fixed_time_embedding's convention -- a lone sin() term can't
            # represent arbitrary phase relationships as well as a paired
            # sin/cos basis does.
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            # freq_init='random' (default) is nn.Linear's ordinary random
            # init -- empirically the best-performing option so far
            # (probe R^2 0.835 vs 0.629-0.626 for log-spaced variants on
            # the masked-reconstruction pure-sine task), despite the
            # spectral-bias motivation for log-spaced init. 'log_spaced'
            # opts into that alternative explicitly; see freeze_freq to
            # control whether it stays trainable.
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        if self.learn_emb:
            query_emb = self.learn_time_embedding(time_steps)
            key_emb = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query_emb = self.fixed_time_embedding(time_steps)
            key_emb = self.fixed_time_embedding(self.query.unsqueeze(0))
        out, attn = self.att(query_emb, key_emb, out)
        out = self.z_to_obs(out)  # (batch, seq_len, 2*input_dim)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return mean.squeeze(-1), logvar.squeeze(-1), attn


class MTANAutoencoderProbabilistic(nn.Module):
    """Same as MTANAutoencoder, but the decoder emits (mean, log-variance)
    for Neural-Process-style predictive uncertainty bands."""

    def __init__(self, encoder: MTANReconEncoder, decoder: MTANReconDecoderProbabilistic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, time_steps):
        z, enc_attn = self.encoder(x, time_steps)
        mean, logvar, dec_attn = self.decoder(z, time_steps)
        return mean, logvar, z, enc_attn, dec_attn


class MTANReconEncoderLS(nn.Module):
    """Same as MTANReconEncoder, but attention conditions directly on a
    per-series Lomb-Scargle frequency hint, not just a concatenated final
    feature. Two extra embedding dimensions -- sin(w_LS * t), cos(w_LS * t)
    at that series' own LS-estimated angular frequency -- are appended to
    both the query and key time embeddings, giving attention a basis
    function tuned to what LS thinks the frequency is, so it can construct
    strong attention between time points separated by (multiples of) that
    period if useful for reconstruction. w_LS varies per series, so unlike
    the base embedding, `query` (normally series-independent) must be
    expanded to a per-series tensor here.

    Motivated by a head-to-head finding: LS substantially outperforms
    mTAND's own learned representation on the aliased (short-period)
    regime (corr 0.933 vs 0.407 on identical held-out series), meaning
    real period information exists there that the encoder wasn't
    extracting on its own. w_LS itself is computed outside this class
    (non-differentiable scipy call) and passed in as a plain tensor.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time + 2, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim))
        if learn_emb:
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _append_ls_dims(self, base_emb, t, ls_omega):
        # base_emb: (batch_or_1, n, embed_time); t: (batch, n); ls_omega: (batch,)
        batch = ls_omega.shape[0]
        if base_emb.shape[0] == 1:
            base_emb = base_emb.expand(batch, -1, -1)
        phase = ls_omega.unsqueeze(1) * t  # (batch, n)
        ls_sin = torch.sin(phase).unsqueeze(-1)
        ls_cos = torch.cos(phase).unsqueeze(-1)
        return torch.cat([base_emb, ls_sin, ls_cos], dim=-1)

    def forward(self, x, time_steps, ls_omega):
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key_base = self.learn_time_embedding(time_steps)
            query_base = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key_base = self.fixed_time_embedding(time_steps)
            query_base = self.fixed_time_embedding(self.query.unsqueeze(0))

        key = self._append_ls_dims(key_base, time_steps, ls_omega)
        query_times = self.query.unsqueeze(0).expand(ls_omega.shape[0], -1)
        query = self._append_ls_dims(query_base, query_times, ls_omega)

        out, attn = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        z = self.hiddens_to_z(out)
        return z, attn


class MTANReconDecoderLS(nn.Module):
    """Decoder counterpart to MTANReconEncoderLS -- same LS-conditioned
    query/key extension, applied to the cross-attention from decode-query
    times back into the reference-point latent sequence. Heteroscedastic
    output (mean, log-variance), same as MTANReconDecoderProbabilistic.
    """

    def __init__(self, input_dim, query, latent_dim=8, nhidden=32,
                 embed_time=32, num_heads=1, learn_emb=True, freq=10.,
                 freq_min=1.0, freq_max=1000.0, freeze_freq=True,
                 freq_init='random', device='cpu'):
        super().__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.freq = freq
        self.learn_emb = learn_emb
        self.register_buffer('query', query)
        self.att = MultiTimeAttention(2 * nhidden, 2 * nhidden, embed_time + 2, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z_to_obs = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, input_dim * 2))
        if learn_emb:
            n_freq = embed_time // 2
            self.n_cos = (embed_time - 1) // 2
            self.periodic = nn.Linear(1, n_freq)
            if freq_init == 'log_spaced':
                with torch.no_grad():
                    freqs = log_spaced_frequencies(n_freq, freq_min, freq_max)
                    self.periodic.weight.copy_((2 * math.pi * freqs).unsqueeze(-1))
                    self.periodic.bias.uniform_(0, 2 * math.pi)
                self.periodic.weight.requires_grad_(not freeze_freq)
            elif freq_init != 'random':
                raise ValueError(f"freq_init must be 'random' or 'log_spaced', got {freq_init!r}")
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        phase = self.periodic(tt)
        out_sin = torch.sin(phase)
        out_cos = torch.cos(phase[..., :self.n_cos])
        out_lin = self.linear(tt)
        return torch.cat([out_lin, out_sin, out_cos], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) *
                              -(math.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _append_ls_dims(self, base_emb, t, ls_omega):
        batch = ls_omega.shape[0]
        if base_emb.shape[0] == 1:
            base_emb = base_emb.expand(batch, -1, -1)
        phase = ls_omega.unsqueeze(1) * t
        ls_sin = torch.sin(phase).unsqueeze(-1)
        ls_cos = torch.cos(phase).unsqueeze(-1)
        return torch.cat([base_emb, ls_sin, ls_cos], dim=-1)

    def forward(self, z, time_steps, ls_omega):
        out, _ = self.gru_rnn(z)
        if self.learn_emb:
            query_base = self.learn_time_embedding(time_steps)
            key_base = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query_base = self.fixed_time_embedding(time_steps)
            key_base = self.fixed_time_embedding(self.query.unsqueeze(0))

        query_emb = self._append_ls_dims(query_base, time_steps, ls_omega)
        key_times = self.query.unsqueeze(0).expand(ls_omega.shape[0], -1)
        key_emb = self._append_ls_dims(key_base, key_times, ls_omega)

        out, attn = self.att(query_emb, key_emb, out)
        out = self.z_to_obs(out)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return mean.squeeze(-1), logvar.squeeze(-1), attn


class MTANRegressor(nn.Module):
    """MTANEncoder + a small MLP head, for the Phase 1 period-regression
    proxy task."""

    def __init__(self, encoder: MTANEncoder, nhidden=16):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(nhidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, time_steps):
        h, attn = self.encoder(x, time_steps)
        return self.head(h).squeeze(-1), attn
