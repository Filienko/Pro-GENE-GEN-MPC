"""FactorGauss: Gaussian (or small Gaussian mixture) on the factor scores.

CPU only, runs in seconds.  With n_components=1 this is a per-class
low-rank-plus-diagonal Gaussian (a factor-analysis-style MVN).
"""

from typing import Dict, Optional

import numpy as np

from factor_core import FactorModel, class_counts, decode


def _fit_gmm(S: np.ndarray, M: int, base_cov: np.ndarray, rng: np.random.Generator,
             n_iter: int = 100, reg: float = 0.3):
    """Tiny EM for a full-covariance GMM in k dims.  Each component covariance is
    shrunk toward the class covariance (reg), so that small components cannot
    collapse onto a handful of training patients."""
    n, k = S.shape
    M = max(1, min(M, n // max(2 * k, 10)))  # need enough rows per component
    if M == 1:
        return np.ones(1), S.mean(0, keepdims=True), base_cov[None]
    means = S[rng.choice(n, M, replace=False)]
    covs = np.repeat(base_cov[None], M, 0)
    w = np.full(M, 1 / M)
    for _ in range(n_iter):
        logp = np.empty((n, M))
        for m in range(M):
            L = np.linalg.cholesky(covs[m] + 1e-6 * np.eye(k))
            z = np.linalg.solve(L, (S - means[m]).T)
            logp[:, m] = np.log(w[m]) - 0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum()
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        w = nk / n
        means = (r.T @ S) / nk[:, None]
        for m in range(M):
            D = S - means[m]
            Cm = (r[:, m, None] * D).T @ D / nk[m]
            covs[m] = (1 - reg) * Cm + reg * base_cov
    return w, means, covs


class FactorGauss:
    def __init__(self, n_components: int = 1, resid_scale: float = 1.0, seed: int = 0):
        self.n_components = n_components
        self.resid_scale = resid_scale
        self.rng = np.random.default_rng(seed)
        self.mix: Dict[str, tuple] = {}

    def fit(self, model: FactorModel):
        self.model = model
        for c in model.classes:
            st = model.stats[c]
            self.mix[c] = _fit_gmm(st.scores, self.n_components, st.score_cov, self.rng)
        return self

    def sample(self, n_total: int, uniform_y: bool = False):
        counts = class_counts(self.model, n_total, uniform_y)
        Xs, ys = [], []
        for c, n_c in counts.items():
            if n_c <= 0:
                continue
            w, means, covs = self.mix[c]
            comp = self.rng.choice(len(w), size=n_c, p=w)
            k = means.shape[1]
            S = np.empty((n_c, k))
            for m in range(len(w)):
                idx = comp == m
                if idx.any():
                    L = np.linalg.cholesky(covs[m] + 1e-9 * np.eye(k))
                    S[idx] = means[m] + self.rng.standard_normal((idx.sum(), k)) @ L.T
            Xs.append(decode(self.model, c, S, self.rng, self.resid_scale))
            ys += [c] * n_c
        X = np.vstack(Xs)
        y = np.array(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]
