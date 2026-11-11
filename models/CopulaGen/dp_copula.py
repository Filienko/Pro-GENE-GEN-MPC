"""DP-CopulaGen: differentially private version of CopulaGen (zCDP, Gaussian mechanism).

Measurements (add/remove-one neighbouring datasets, all three composed in zCDP):

1. class counts           n_c + N(0, s_n^2)                      sensitivity 1
2. marginal histograms    h_cgb = #{i in c : x_ig in bin b}       one count per gene per record,
                          + N(0, s_m^2) on each cell              L2 sensitivity sqrt(d)
   (this is exactly what Private-PGM measures for its (gene, label) cliques, only with more bins)
3. copula second moment   M_c = sum_{i in c} s_i s_i^T,   s_i = clip_C(W z_i) in R^k
                          + symmetric N(0, s_c^2) noise           Frobenius sensitivity C^2
   z_i are the normal scores of record i under the *noisy* marginals (post-processing), W (k x d)
   is a public basis: learned on an external non-member cohort (--external_counts) or by DP-PCA
   of the pooled d x d second moment (costs part of the copula budget).

Bin edges: per-gene equal-width bins over [lo_g, hi_g].  With an external cohort the range is
taken from it (fully DP); otherwise from the training data, which is the convention the existing
Private-PGM pipeline uses for its quantile bins (the DP guarantee then covers the measurements only).
"""

from typing import Optional

import numpy as np
from scipy import special, stats

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FactorGen"))
from factor_core import ClassStats, FactorModel, class_counts  # noqa: E402
from copula_gen import Marginals, ndtri  # noqa: E402


def rho_from_eps(eps, delta):
    """largest rho with rho + 2 sqrt(rho log(1/delta)) <= eps"""
    if not np.isfinite(eps):
        return np.inf
    L = np.log(1 / delta)
    return (np.sqrt(L + eps) - np.sqrt(L)) ** 2


def gauss_sigma(sens, rho):
    return 0.0 if not np.isfinite(rho) else sens / np.sqrt(2 * rho)


def sym_noise(rng, k, sigma):
    A = rng.standard_normal((k, k)) * sigma
    return np.triu(A) + np.triu(A, 1).T


def psd(C, floor=0.0):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.maximum(w, floor)) @ V.T


class DPCopulaGen:
    def __init__(self, epsilon=10.0, delta=1e-5, k=20, bins=8, knots=41, tail=0.02,
                 split=(0.02, 0.49, 0.49), pca_frac=0.5, clip_mult=1.5, shrink=0.3,
                 seed=0, external: Optional[np.ndarray] = None, external_y=None, edges="data",
                 marg_kappa=1.0, eb_shrink=False, eb_reps=20):
        self.eps, self.delta, self.k, self.B = epsilon, delta, k, bins
        self.split, self.pca_frac, self.clip_mult, self.shrink = split, pca_frac, clip_mult, shrink
        self.marg = Marginals(knots, tail, 0.0, 0.0, "quantile")
        self.rng = np.random.default_rng(seed)
        self.external, self.external_y = external, external_y
        self.edges, self.marg_kappa = edges, marg_kappa
        self.eb_shrink, self.eb_reps = eb_shrink, eb_reps
        self.info = {}

    # ------------------------------------------------------------------ marginals
    @staticmethod
    def _probit_fit(H, n_hat, edges):
        """Gaussian (mu, sd) per gene from (noisy) bin counts H (B, d) via weighted least squares of
        Phi^-1(F(e_b)) = (e_b - mu) / sd on the interior edges."""
        B, d = H.shape
        cum = np.cumsum(H, 0)[:-1] / max(n_hat, 1.0)                    # (B-1, d)
        cum = np.maximum.accumulate(np.clip(cum, 0.01, 0.99), axis=0)
        z = ndtri(cum)
        e = edges[1:-1]                                                  # (B-1, d)
        b = np.arange(1, B)[:, None]
        w = stats.norm.pdf(z) ** 2 / np.minimum(b, B - b)               # signal / noise of each edge
        w = w * ((cum > 0.011) & (cum < 0.989))
        sw = w.sum(0) + 1e-12
        em, zm = (w * e).sum(0) / sw, (w * z).sum(0) / sw
        slope = (w * (e - em) * (z - zm)).sum(0) / ((w * (e - em) ** 2).sum(0) + 1e-12)
        slope = np.maximum(slope, 1e-3)
        sd = 1 / slope
        mu = em - zm * sd
        lo, hi = edges[1], edges[-2]
        mu = np.clip(mu, lo - 1, hi + 1)
        ok = (sw > 1e-3) & np.isfinite(mu) & np.isfinite(sd)
        return mu, sd, ok

    def _fit_marginals(self, X, y, rho_m, edges):
        d = X.shape[1]
        B = edges.shape[0] - 1
        sig = gauss_sigma(np.sqrt(d), rho_m)
        bidx = np.zeros(X.shape, int)
        for b in range(1, B):
            bidx += X >= edges[b]
        m = self.marg
        m.classes = self.classes
        m.loc, m.scale, m.q, m.lam = {}, {}, {}, {}
        Hs = {}
        for c in self.classes:
            bc = bidx[y == c]
            H = np.stack([(bc == b).sum(0) for b in range(B)]).astype(float)
            Hs[c] = H + self.rng.standard_normal(H.shape) * sig
        # pooled fit (sum of the noisy class histograms: post-processing, more signal)
        n_tot = sum(self.n_hat.values())
        mu_p, sd_p, ok_p = self._probit_fit(sum(Hs.values()), n_tot, edges)
        mid = (edges[1] + edges[-2]) / 2
        mu_p, sd_p = np.where(ok_p, mu_p, mid), np.where(ok_p, sd_p, 1.0)
        for c in self.classes:
            mu_c, sd_c, ok = self._probit_fit(Hs[c], self.n_hat[c], edges)
            mu_c, sd_c = np.where(ok, mu_c, mu_p), np.where(ok, sd_c, sd_p)
            # shrink toward pooled: weight ~ class signal^2 / (signal^2 + noise^2 * B)
            nc = self.n_hat[c]
            a = nc ** 2 / (nc ** 2 + self.marg_kappa * (sig ** 2) * B + 1e-12) if sig > 0 else 1.0
            a = a * nc / (nc + 10.0)
            # the class shift is well determined; the class SD needs more support
            mu = a * mu_c + (1 - a) * mu_p
            a_sd = a * nc / (nc + 50.0)
            sd = np.exp(a_sd * np.log(np.clip(sd_c, 0.05, 10)) + (1 - a_sd) * np.log(np.clip(sd_p, 0.05, 10)))
            if sig > 0 and self.eb_shrink:
                # empirical-Bayes denoising of the class shift (post-processing): parametric bootstrap
                # of the estimator's noise variance, then a non-negative garrote toward the pooled fit
                P = np.diff(stats.norm.cdf((edges - mu_c) / sd_c), axis=0)
                reps_mu, reps_lsd = [], []
                for _ in range(self.eb_reps):
                    Hb = nc * P + self.rng.standard_normal(P.shape) * sig
                    mb, sb, okb = self._probit_fit(Hb, nc, edges)
                    reps_mu.append(np.where(okb, mb, mu_c))
                    reps_lsd.append(np.log(np.clip(np.where(okb, sb, sd_c), 0.05, 10)))
                v_mu = np.var(reps_mu, 0) * a ** 2
                v_lsd = np.var(reps_lsd, 0) * a_sd ** 2
                sh = mu - mu_p
                mu = mu_p + sh * np.maximum(0, 1 - v_mu / (sh ** 2 + 1e-12))
                lsh = np.log(sd) - np.log(np.clip(sd_p, 0.05, 10))
                sd = np.clip(sd_p, 0.05, 10) * np.exp(lsh * np.maximum(0, 1 - v_lsd / (lsh ** 2 + 1e-12)))
            m.loc[c], m.scale[c] = mu, sd
            m.q[c] = np.repeat(m.zu[:, None], d, 1)
            m.lam[c] = float(1 - a)
        self.info["marg_sigma"] = sig

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        y = y.astype(str)
        self.classes = [str(c) for c in np.unique(y)]
        n, d = X.shape
        rho = rho_from_eps(self.eps, self.delta)
        rho_n, rho_m, rho_c = [s * rho for s in self.split]
        self.info["rho"] = rho
        # 1. noisy class counts
        sig_n = gauss_sigma(1.0, rho_n)
        self.n_hat = {c: max(1.0, (y == c).sum() + self.rng.normal() * sig_n) for c in self.classes}
        # bin edges: quantiles of the external cohort (fully DP) or of the pooled training data
        # (the convention of the existing Private-PGM pipeline)
        qs = np.linspace(0, 1, self.B + 1)
        ref = self.external if (self.external is not None and self.edges == "external") else X
        edges = np.quantile(ref, qs, axis=0)
        edges[0], edges[-1] = edges[0] - 10, edges[-1] + 10
        edges = np.maximum.accumulate(edges + 1e-6 * np.arange(self.B + 1)[:, None], axis=0)
        # 2. marginals
        self._fit_marginals(X, y, rho_m, edges)
        Z = np.empty_like(X)
        for c in self.classes:
            Z[y == c] = self.marg.to_normal(X[y == c], c)
        # 3. basis
        if self.external is not None:
            E = self.external.copy()
            groups = self.external_y if self.external_y is not None else np.zeros(len(E))
            for gname in np.unique(groups):          # within-group standardisation (e.g. per tissue)
                mm = groups == gname
                E[mm] = (E[mm] - E[mm].mean(0)) / (E[mm].std(0) + 1e-6)
            _, _, Vt = np.linalg.svd(E, full_matrices=False)
            W = Vt[: self.k]
            rho_cov = rho_c
            C2 = self.clip_mult * float(np.median(((E @ W.T) ** 2).sum(1)))
        else:
            rho_pca, rho_cov = self.pca_frac * rho_c, (1 - self.pca_frac) * rho_c
            Cz2 = self.clip_mult * d                                   # E||z||^2 = d for normal scores
            Zc = Z * np.minimum(1, np.sqrt(Cz2) / np.linalg.norm(Z, axis=1))[:, None]
            Mz = Zc.T @ Zc + sym_noise(self.rng, d, gauss_sigma(Cz2, rho_pca))
            w, V = np.linalg.eigh(Mz)
            W = V[:, ::-1][:, : self.k].T
            # clip bound for the k-dim projections, data independent: k-dim share of d at the
            # (noisy) spectrum is post-processing of Mz
            share = np.maximum(w[::-1][: self.k], 0).sum() / max(np.maximum(w, 0).sum(), 1e-9)
            C2 = self.clip_mult * share * d
        self.info["C2"] = C2
        S = Z @ W.T
        S = S * np.minimum(1, np.sqrt(C2) / (np.linalg.norm(S, axis=1) + 1e-12))[:, None]
        sig_c = gauss_sigma(C2, rho_cov)
        self.info["cov_sigma"] = sig_c
        k = W.shape[0]
        covs = {}
        for c in self.classes:
            Sc = S[y == c]
            covs[c] = (Sc.T @ Sc + sym_noise(self.rng, k, sig_c)) / self.n_hat[c]
        n_tot = sum(self.n_hat.values())
        pooled = psd(sum(covs[c] * self.n_hat[c] for c in self.classes) / n_tot, 1e-3)
        fm = FactorModel(W=W, classes=self.classes, pooled_score_cov=pooled)
        for c in self.classes:
            # adaptive shrinkage: more weight on pooled when the class is small or noise is large
            noise_rel = sig_c * np.sqrt(k) / self.n_hat[c] / max(np.trace(pooled) / k, 1e-9)
            lam = float(np.clip(max(self.shrink, k / self.n_hat[c], noise_rel), 0, 1))
            cov = psd((1 - lam) * covs[c] + lam * pooled, 1e-3)
            # keep total variance per gene ~1 in normal-score space; residual is the remainder
            expl = np.einsum("ji,jk,ki->i", W, cov, W)
            if expl.max() > 0.98:
                f = 0.98 / expl.max()
                cov, expl = cov * f, expl * f
            rv = np.maximum(1 - expl, 0.02)
            fm.stats[c] = ClassStats(mu=np.zeros(d), resid_var=rv, scores=np.zeros((0, k)),
                                     score_cov=cov, n=int(self.n_hat[c]))
            fm.priors[c] = self.n_hat[c] / n_tot
            fm.shrink_used[c] = lam
        self.fm = fm
        return self

    def sample(self, n_total, uniform_y=False):
        counts = class_counts(self.fm, n_total, uniform_y)
        Xs, ys = [], []
        for c, n_c in counts.items():
            if n_c <= 0:
                continue
            st = self.fm.stats[c]
            k = st.score_cov.shape[0]
            L = np.linalg.cholesky(st.score_cov + 1e-9 * np.eye(k))
            Sf = self.rng.standard_normal((n_c, k)) @ L.T
            Zs = Sf @ self.fm.W + self.rng.standard_normal((n_c, len(st.mu))) * np.sqrt(st.resid_var)
            Xs.append(self.marg.from_normal(Zs, c))
            ys += [c] * n_c
        X, y = np.vstack(Xs), np.array(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]
