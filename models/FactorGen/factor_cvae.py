"""FactorCVAE: a small conditional VAE on the k-dim factor scores.

What it changes relative to the CAMDA CVAE baseline (which leaks under
GAN-leaks / KLeak):

* models k factor scores, not 978 genes; the per-gene residual is drawn
  independently in factor_core.decode
* beta = 1 (proper ELBO) instead of 0.001 (near-deterministic autoencoder)
* small latent (default 8) instead of 128
* Gaussian decoder with a learned per-dimension variance, and samples are
  drawn WITH that observation noise (the baseline outputs the decoder mean)
* early stopping on a held-out validation ELBO instead of a fixed 10k iterations
* sampling from the N(0, I) prior only, never from posterior codes of real rows
  (unlike CVAE-GMM)
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from factor_core import FactorModel, class_counts, decode, moment_match


class _CVAE(nn.Module):
    def __init__(self, k: int, n_cls: int, z_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.n_cls, self.z_dim = n_cls, z_dim
        self.enc = nn.Sequential(nn.Linear(k + n_cls, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim + n_cls, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, k))
        self.dec_logvar = nn.Parameter(torch.zeros(k))  # learned observation noise

    def onehot(self, y):
        return nn.functional.one_hot(y, self.n_cls).float()

    def forward(self, x, y):
        h = self.enc(torch.cat([x, self.onehot(y)], 1))
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        xm = self.dec(torch.cat([z, self.onehot(y)], 1))
        return xm, mu, logvar

    def loss(self, x, y, beta=1.0):
        xm, mu, logvar = self(x, y)
        lv = self.dec_logvar.clamp(-6, 4)
        nll = 0.5 * (((x - xm) ** 2) / lv.exp() + lv).sum(1)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        return (nll + beta * kl).mean()

    @torch.no_grad()
    def sample(self, y):
        z = torch.randn(len(y), self.z_dim, device=y.device)
        xm = self.dec(torch.cat([z, self.onehot(y)], 1))
        return xm + torch.randn_like(xm) * torch.exp(0.5 * self.dec_logvar.clamp(-6, 4))


class FactorCVAE:
    def __init__(self, z_dim: int = 8, hidden: int = 128, beta: float = 1.0,
                 lr: float = 1e-3, batch_size: int = 64, max_epochs: int = 2000,
                 patience: int = 100, val_frac: float = 0.1, weight_decay: float = 1e-4,
                 moment_match: str = "none", resid_scale: float = 1.0, seed: int = 0,
                 device: Optional[str] = None):
        self.cfg = dict(z_dim=z_dim, hidden=hidden, beta=beta, lr=lr, batch_size=batch_size,
                        max_epochs=max_epochs, patience=patience, val_frac=val_frac,
                        weight_decay=weight_decay)
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
        # z-score each factor (pooled within-class scale)
        self.scale = S.std(0) + 1e-8
        S = S / self.scale
        # stratified validation split for early stopping
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

        net = _CVAE(S.shape[1], len(cls), self.cfg["z_dim"], self.cfg["hidden"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])
        best, best_state, bad = np.inf, None, 0
        bs = self.cfg["batch_size"]
        for ep in range(self.cfg["max_epochs"]):
            net.train()
            perm = torch.randperm(len(ytr), device=dev)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                loss = net.loss(Xtr[b], ytr[b], self.cfg["beta"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            # fixed validation noise without touching the training RNG stream
            devs = [torch.device(dev).index or 0] if str(dev).startswith("cuda") else []
            with torch.no_grad(), torch.random.fork_rng(devices=devs):
                torch.manual_seed(12345)
                vl = float(np.mean([net.loss(Xva, yva, self.cfg["beta"]).item()
                                    for _ in range(4)])) if len(yva) else loss.item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
            if verbose and ep % 100 == 0:
                print(f"  [FactorCVAE] epoch {ep:4d}  train {loss.item():.3f}  val {vl:.3f}")
            if bad >= self.cfg["patience"]:
                if verbose:
                    print(f"  [FactorCVAE] early stop at epoch {ep}, best val {best:.3f}")
                break
        net.load_state_dict(best_state)
        self.net = net.eval()
        self.epochs_trained = ep + 1
        return self

    def sample(self, n_total: int, uniform_y: bool = False):
        counts = class_counts(self.model, n_total, uniform_y)
        torch.manual_seed(self.seed + 1)
        Xs, ys = [], []
        for i, c in enumerate(self.model.classes):
            n_c = counts[c]
            if n_c <= 0:
                continue
            yy = torch.full((n_c,), i, dtype=torch.long, device=self.device)
            S = self.net.sample(yy).cpu().numpy() * self.scale
            S = moment_match(S, self.model.stats[c].score_cov, self.moment_match)
            Xs.append(decode(self.model, c, S, self.rng, self.resid_scale))
            ys += [c] * n_c
        X, y = np.vstack(Xs), np.array(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]
