"""FactorDiffusion: a small class-conditional DDPM on the k-dim factor scores.

What it changes relative to the CAMDA "Embedded Noisy Diffusion" (MeLoMIA-ND
AUC 0.97, high calibrated GAN-leaks TPR):

* models k factor scores (default 50), not 978 genes; per-gene residual is
  drawn independently in factor_core.decode
* small MLP denoiser (~0.2-0.4M parameters) instead of ~25M
* no SMOTE oversampling (it adds interpolated copies of minority-class members)
* no QuantileTransformer (its inverse pins every gene onto the training
  patients' order statistics); plain z-scoring of the factor scores
* early stopping on a held-out denoising loss; no dropout (dropout in the
  epsilon-network inflated sample variance by ~1.4x in our tests)
* optional moment matching of the generated factor scores (--moment_match
  mean|cov|none, default mean) to remove residual mean bias
* truncated sampling: the reverse chain stops at sigma_stop > 0 and returns
  x_t / sqrt(alpha_bar_t), i.e. the last noise level is kept instead of
  being denoised all the way onto training points
"""

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from factor_core import FactorModel, class_counts, decode, moment_match


def _timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class _Denoiser(nn.Module):
    def __init__(self, k: int, n_cls: int, hidden: int = 256, t_dim: int = 64,
                 c_dim: int = 16, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.t_dim = t_dim
        self.cls_emb = nn.Embedding(n_cls, c_dim)
        layers, d_in = [], k + t_dim + c_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout)]
            d_in = hidden
        layers.append(nn.Linear(hidden, k))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, y):
        h = torch.cat([x, _timestep_embedding(t, self.t_dim), self.cls_emb(y)], 1)
        return self.net(h)


def _cosine_alpha_bar(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-6, 1)
    return ab[1:].float()  # alpha_bar_t for t = 1..T (index 0..T-1)


class FactorDiffusion:
    def __init__(self, T: int = 200, hidden: int = 256, n_layers: int = 3, dropout: float = 0.0,
                 lr: float = 3e-4, batch_size: int = 128, max_epochs: int = 4000,
                 patience: int = 400, val_frac: float = 0.1, weight_decay: float = 0.0,
                 sigma_stop: float = 0.1, moment_match: str = "mean", resid_scale: float = 1.0,
                 seed: int = 0, device: Optional[str] = None):
        self.cfg = dict(T=T, hidden=hidden, n_layers=n_layers, dropout=dropout, lr=lr,
                        batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                        val_frac=val_frac, weight_decay=weight_decay, sigma_stop=sigma_stop)
        self.resid_scale = resid_scale
        self.moment_match = moment_match
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, model: FactorModel, verbose: bool = True):
        torch.manual_seed(self.seed)
        self.model = model
        cls = model.classes
        S = np.vstack([model.stats[c].scores for c in cls])
        y = np.concatenate([[i] * model.stats[c].n for i, c in enumerate(cls)])
        self.scale = S.std(0) + 1e-8
        S = S / self.scale
        val = np.zeros(len(y), bool)
        for i in range(len(cls)):
            idx = np.where(y == i)[0]
            nv = int(round(self.cfg["val_frac"] * len(idx)))
            if nv > 0 and len(idx) - nv >= 2:
                val[self.rng.choice(idx, nv, replace=False)] = True
        dev = self.device
        Xtr = torch.tensor(S[~val], dtype=torch.float32, device=dev)
        ytr = torch.tensor(y[~val], dtype=torch.long, device=dev)
        Xva = torch.tensor(S[val], dtype=torch.float32, device=dev)
        yva = torch.tensor(y[val], dtype=torch.long, device=dev)

        T = self.cfg["T"]
        self.ab = _cosine_alpha_bar(T).to(dev)
        net = _Denoiser(S.shape[1], len(cls), self.cfg["hidden"], n_layers=self.cfg["n_layers"],
                        dropout=self.cfg["dropout"]).to(dev)
        self.n_params = sum(p.numel() for p in net.parameters())
        if verbose:
            print(f"  [FactorDiffusion] denoiser parameters: {self.n_params:,}")
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])

        def dsm_loss(x0, yy, gen=None):
            t = torch.randint(0, T, (len(x0),), device=dev, generator=gen)
            eps = torch.randn(x0.shape, device=dev, generator=gen)
            ab = self.ab[t][:, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            return ((net(xt, t, yy) - eps) ** 2).mean()

        # fixed validation noise, repeated a few times for a stable estimate
        gen = torch.Generator(device=dev)
        best, best_state, bad = np.inf, None, 0
        bs = self.cfg["batch_size"]
        for ep in range(self.cfg["max_epochs"]):
            net.train()
            perm = torch.randperm(len(ytr), device=dev)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                loss = dsm_loss(Xtr[b], ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if len(yva) and ep % 5 == 0:
                net.eval()
                with torch.no_grad():
                    gen.manual_seed(12345)
                    vl = float(np.mean([dsm_loss(Xva, yva, gen).item() for _ in range(8)]))
                if vl < best - 1e-4:
                    best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 5
                if verbose and ep % 100 == 0:
                    print(f"  [FactorDiffusion] epoch {ep:4d}  train {loss.item():.4f}  val {vl:.4f}")
                if bad >= self.cfg["patience"]:
                    if verbose:
                        print(f"  [FactorDiffusion] early stop at epoch {ep}, best val {best:.4f}")
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        self.epochs_trained = ep + 1
        # truncation step: first t whose noise-to-signal sigma_t >= sigma_stop
        sig = ((1 - self.ab) / self.ab).sqrt()
        ss = self.cfg["sigma_stop"]
        self.t_stop = int((sig >= ss).nonzero()[0].item()) if ss > 0 else -1
        return self

    @torch.no_grad()
    def _sample_scores(self, yy: torch.Tensor) -> torch.Tensor:
        T, ab, dev = self.cfg["T"], self.ab, self.device
        x = torch.randn(len(yy), self.net.net[-1].out_features, device=dev)
        for t in range(T - 1, self.t_stop, -1):
            tt = torch.full((len(yy),), t, device=dev, dtype=torch.long)
            eps = self.net(x, tt, yy)
            a_t = ab[t]
            a_prev = ab[t - 1] if t > 0 else torch.tensor(1.0, device=dev)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-6, 6)
            # DDPM posterior q(x_{t-1} | x_t, x0)
            beta_t = 1 - a_t / a_prev
            mean = (a_prev.sqrt() * beta_t / (1 - a_t)) * x0 + \
                   ((1 - beta_t).sqrt() * (1 - a_prev) / (1 - a_t)) * x
            var = beta_t * (1 - a_prev) / (1 - a_t)
            x = mean + (var.sqrt() * torch.randn_like(x) if t > 0 else 0)
        if self.t_stop >= 0:
            x = x / ab[self.t_stop].sqrt()  # keep the remaining noise level sigma_stop
        return x

    def sample(self, n_total: int, uniform_y: bool = False):
        counts = class_counts(self.model, n_total, uniform_y)
        torch.manual_seed(self.seed + 1)
        Xs, ys = [], []
        for i, c in enumerate(self.model.classes):
            n_c = counts[c]
            if n_c <= 0:
                continue
            yy = torch.full((n_c,), i, dtype=torch.long, device=self.device)
            S = self._sample_scores(yy).cpu().numpy() * self.scale
            S = moment_match(S, self.model.stats[c].score_cov, self.moment_match)
            Xs.append(decode(self.model, c, S, self.rng, self.resid_scale))
            ys += [c] * n_c
        X, y = np.vstack(Xs), np.array(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]
