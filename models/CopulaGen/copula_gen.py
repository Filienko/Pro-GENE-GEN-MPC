"""CopulaGen: class-conditional low-rank Gaussian copula with smoothed marginals.

For class c and gene g the marginal is a location-scale family with a
nonparametric *shape*:

    x_g | c  =  m_cg + s_cg * q_cg(u),      u ~ U(0, 1)

q_cg is a standardised quantile function on K knots (piecewise linear, Gaussian
tails beyond the outer knots), shrunk toward the pooled within-class shape q_g
with weight lambda_c = max(shape_shrink, n0 / (n0 + n_c)).  Few knots + shrinkage
+ parametric tails mean that no single training value is reproduced.

Dependence: training rows are mapped to normal scores z = Phi^-1(F_cg(x)), and
the scores follow the FactorGen model  z = W^T f + e  (k shared factors, class
factor covariances shrunk toward the pooled within-class one, diagonal residual).
Sampling draws z, then maps back with x = m + s * q(Phi(z)).

Everything the model stores is an aggregate (per-class quantiles / counts and
a k x k factor covariance), so marginal-based attacks see the same information
as for Private-PGM, but the joint structure (co-expression, manifold shape) is
kept.
"""

from typing import Dict, Optional

import numpy as np
from scipy import special

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import fit_factor_model, class_counts  # noqa: E402


def ndtri(u):
    return special.ndtri(np.clip(u, 1e-7, 1 - 1e-7))


class Marginals:
    """Per-class standardised quantile functions on fixed knot probabilities."""

    def __init__(self, knots: int = 21, tail: float = 0.05, shape_shrink: float = 0.0, n0: float = 50.0,
                 shape: str = "quantile", gauss_mix: float = 0.0):
        self.K = knots
        self.u = np.linspace(tail, 1 - tail, knots)          # knot probabilities
        self.zu = ndtri(self.u)                              # their normal quantiles
        self.shape_shrink, self.n0 = shape_shrink, n0
        self.shape = shape                                   # "quantile" | "gauss"
        self.zclip = 4.0
        self.gauss_mix = gauss_mix                           # shrink shape toward the Gaussian quantiles

    def fit(self, X, y, classes):
        self.classes = classes
        d = X.shape[1]
        self.loc, self.scale, self.q = {}, {}, {}
        Zs = np.empty_like(X)
        for c in classes:
            Xc = X[y == c]
            m, s = Xc.mean(0), Xc.std(0) + 1e-6
            self.loc[c], self.scale[c] = m, s
            Zs[y == c] = (Xc - m) / s
        pooled = np.quantile(Zs, self.u, axis=0)              # (K, d) pooled within-class shape
        self.lam = {}
        for c in classes:
            Xc = X[y == c]
            if self.shape == "gauss":
                q = np.repeat(self.zu[:, None], d, 1)
                lam = 1.0
            else:
                qc = np.quantile((Xc - self.loc[c]) / self.scale[c], self.u, axis=0)
                lam = max(self.shape_shrink, self.n0 / (self.n0 + len(Xc)))
                q = (1 - lam) * qc + lam * pooled
                q = (1 - self.gauss_mix) * q + self.gauss_mix * self.zu[:, None]
            # monotone (strictly increasing) knots
            q = np.maximum.accumulate(q + 1e-6 * np.arange(self.K)[:, None], axis=0)
            self.q[c] = q
            self.lam[c] = lam
        return self

    def _tail_slopes(self, q):
        # slope (in x per unit z) beyond the outer knots: match the last segment
        # bounded to [0.5, 2] x the median segment slope so flat (tied) or very steep
        # outer segments cannot produce extreme scores / values
        sl = np.diff(q, axis=0) / np.diff(self.zu)[:, None]
        med = np.median(sl, 0)
        lo = np.clip(sl[0], 0.5 * med, 2 * med)
        hi = np.clip(sl[-1], 0.5 * med, 2 * med)
        return lo, hi

    def to_normal(self, X, c):
        """x -> z = Phi^-1(F(x)), piecewise linear in z between knots, linear tails."""
        q = self.q[c]
        T = (X - self.loc[c]) / self.scale[c]
        Z = np.empty_like(T)
        lo, hi = self._tail_slopes(q)
        for g in range(T.shape[1]):
            Z[:, g] = np.interp(T[:, g], q[:, g], self.zu)
            below, above = T[:, g] < q[0, g], T[:, g] > q[-1, g]
            Z[below, g] = self.zu[0] + (T[below, g] - q[0, g]) / max(lo[g], 1e-6)
            Z[above, g] = self.zu[-1] + (T[above, g] - q[-1, g]) / max(hi[g], 1e-6)
        return np.clip(Z, -self.zclip, self.zclip)

    def from_normal(self, Z, c):
        Z = np.clip(Z, -self.zclip - 1, self.zclip + 1)
        q = self.q[c]
        lo, hi = self._tail_slopes(q)
        T = np.empty_like(Z)
        for g in range(Z.shape[1]):
            T[:, g] = np.interp(Z[:, g], self.zu, q[:, g])
            below, above = Z[:, g] < self.zu[0], Z[:, g] > self.zu[-1]
            T[below, g] = q[0, g] + (Z[below, g] - self.zu[0]) * lo[g]
            T[above, g] = q[-1, g] + (Z[above, g] - self.zu[-1]) * hi[g]
        return self.loc[c] + self.scale[c] * T


class CopulaGen:
    def __init__(self, k: int = 50, shrink: float = 0.3, knots: int = 21, tail: float = 0.05,
                 shape_shrink: float = 0.0, n0: float = 50.0, marginal: str = "quantile",
                 factor: str = "gauss", n_components: int = 3, kde_h: float = 0.5,
                 resid_scale: float = 1.0, seed: int = 0, external: Optional[np.ndarray] = None,
                 device: Optional[str] = None, diff_kw: Optional[dict] = None, gauss_mix: float = 0.0):
        self.k, self.shrink = k, shrink
        self.marg = Marginals(knots, tail, shape_shrink, n0, marginal, gauss_mix)
        self.factor, self.n_components, self.kde_h = factor, n_components, kde_h
        self.resid_scale = resid_scale
        self.rng = np.random.default_rng(seed)
        self.external = external
        self.device, self.diff_kw = device, diff_kw or {}

    def fit(self, X, y):
        self.classes = [str(c) for c in np.unique(y)]
        y = y.astype(str)
        self.marg.fit(X, y, self.classes)
        Z = np.empty_like(X)
        for c in self.classes:
            Z[y == c] = self.marg.to_normal(X[y == c], c)
        self.Z_train = Z
        ext = None
        if self.external is not None:
            ext = (self.external - self.external.mean(0)) / (self.external.std(0) + 1e-6)
        self.fm = fit_factor_model(Z, y, k=self.k, shrink=self.shrink, external=ext)
        if self.factor == "gmm":
            from factor_gauss import _fit_gmm
            self.mix = {c: _fit_gmm(self.fm.stats[c].scores, self.n_components, self.fm.stats[c].score_cov,
                                    self.rng) for c in self.classes}
        if self.factor == "diffusion":
            from factor_diffusion import FactorDiffusion
            self.diff = FactorDiffusion(resid_scale=self.resid_scale, seed=int(self.rng.integers(1 << 30)),
                                        device=self.device, **self.diff_kw).fit(self.fm, verbose=False)
        return self

    def _factor_scores(self, c, n):
        st = self.fm.stats[c]
        k = st.score_cov.shape[0]
        if self.factor == "gauss":
            L = np.linalg.cholesky(st.score_cov + 1e-9 * np.eye(k))
            return self.rng.standard_normal((n, k)) @ L.T
        if self.factor == "gmm":
            w, means, covs = self.mix[c]
            comp = self.rng.choice(len(w), size=n, p=w)
            S = np.empty((n, k))
            for m in range(len(w)):
                idx = comp == m
                if idx.any():
                    L = np.linalg.cholesky(covs[m] + 1e-9 * np.eye(k))
                    S[idx] = means[m] + self.rng.standard_normal((idx.sum(), k)) @ L.T
            return S
        if self.factor == "kde":
            # smoothed bootstrap with variance correction: s = (s_i + h * eps) / sqrt(1 + h^2), eps ~ N(0, C)
            L = np.linalg.cholesky(st.score_cov + 1e-9 * np.eye(k))
            idx = self.rng.integers(0, len(st.scores), n)
            h = self.kde_h
            return (st.scores[idx] + h * self.rng.standard_normal((n, k)) @ L.T) / np.sqrt(1 + h * h)
        raise ValueError(self.factor)

    def sample(self, n_total, uniform_y=False):
        if self.factor == "diffusion":
            Zs, ys = self.diff.sample(n_total, uniform_y)
            X = np.empty_like(Zs)
            for c in self.classes:
                m = ys == c
                if m.any():
                    X[m] = self.marg.from_normal(Zs[m], c)
            return X, ys
        counts = class_counts(self.fm, n_total, uniform_y)
        Xs, ys = [], []
        for c, n_c in counts.items():
            if n_c <= 0:
                continue
            st = self.fm.stats[c]
            S = self._factor_scores(c, n_c)
            Zs = st.mu + S @ self.fm.W + self.rng.standard_normal((n_c, len(st.mu))) * np.sqrt(
                st.resid_var * self.resid_scale)
            Xs.append(self.marg.from_normal(Zs, c))
            ys += [c] * n_c
        X, y = np.vstack(Xs), np.array(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]
